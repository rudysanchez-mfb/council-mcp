import asyncio, time, json, re
from council_mcp.providers import PROVIDERS, query_provider, ProviderResponse, strip_think_tags

DEFAULT_ADVOCATES = [
    ("kimi", "kimi-k3.1-pro"),
    ("deepseek", "deepseek-v4-pro"),
    ("grok", "grok-4.20-reasoning"),
    ("gemini", "gemini-2.5-flash"),
]
DEFAULT_JUDGE = ("deepseek", "deepseek-chat")

# --- Untagged chain-of-thought (CoT) leak detection -----------------------------
# An advocate can return HTTP 200 with a LARGE blob of raw reasoning as `content`,
# with NO <think> tags and no answer structure. strip_think_tags() has nothing to
# strip, so the non-empty result used to sail past the "empty after strip" check,
# get counted as full coverage, and be handed to the judge as if it were analysis.
# Observed intermittently with deepseek-v4-pro on open-ended prompts (2026-10-04):
# ~10KB of "We need answer user asks... Let's parse..." prose.
#
# Detection is STRUCTURAL first: the advocate prompt demands VERDICT/REASONING/RISKS
# line-anchored headings, so a long response that is missing >=2 of them is almost
# certainly leaked scratchpad. A small CoT-lexicon list only CORROBORATES (it is
# brittle and must never be the primary signal). A response carrying all three
# headings is never flagged; a SHORT markerless response is a different failure
# mode (dead output, not a leak) and is never flagged here.
_COT_HEADING_RE = re.compile(r"(?im)^[\s>*#_\-|]*(VERDICT|REASONING|RISKS)\b")
_REQUIRED_HEADINGS = ("VERDICT", "REASONING", "RISKS")
_COT_MIN_CHARS = 1500          # below this, a markerless body is a separate case
_COT_MIN_MISSING = 2           # at least 2 of the 3 required headings absent
_COT_MIN_PHRASES = 1           # weak corroboration, never the sole trigger
# A legit answer opens with 'VERDICT:' (preamble ~0) and names each heading once.
# A leaked scratchpad that DOES eventually draft the answer betrays itself with a
# long PREAMBLE of reasoning prose before the first heading and/or a REQUIRED
# heading repeated as it re-drafts. Legit 3-section answers observed at 1-3.1K chars.
_COT_PREAMBLE_CHARS = 600      # prose before the first heading -> suspect
_COT_PREAMBLE_HARD = 2000      # this much preamble is a leak on its own
_COT_PHRASES = (
    "we need answer", "we need to answer", "we need produce", "need parse",
    "let's parse", "let's reason", "let's think", "user is asking", "user asks",
    "the user asks", "let me reconsider", "analyzing the query",
    "breaking this down", "thinking step by step", "scratchpad", "i need to",
    "need analyze", "let's understand", "need output exactly", "walk through",
)


def classify_degraded(content: str | None) -> tuple[bool, str | None]:
    """Return (degraded, reason) for an advocate's content.

    Callers pass content that has ALREADY been through strip_think_tags(). Catches
    the empty-body case plus untagged chain-of-thought leaks the old empty-string
    check missed. Ground truth (2026-10-04, deepseek-v4-pro): leaked 17-18KB
    scratchpads that either never emitted the headings at all, or rambled for ~5.7K
    chars and THEN drafted VERDICT/REASONING/RISKS (sometimes more than once).
    Legit answers were 1-3.1K chars and opened with 'VERDICT:' at position 0,
    each heading appearing exactly once.
    """
    text = (content or "").strip()
    if not text:
        return True, "empty response body"

    matches = list(_COT_HEADING_RE.finditer(text))
    found = {m.group(1).upper() for m in matches}
    missing = [h for h in _REQUIRED_HEADINGS if h not in found]
    preamble = matches[0].start() if matches else len(text)
    hits = sum(1 for p in _COT_PHRASES if p in text.lower())

    # (A) Long reasoning preamble before the first heading — the model reasoned
    #     first and only drafted the answer later.
    if preamble >= _COT_PREAMBLE_HARD or (preamble >= _COT_PREAMBLE_CHARS and hits >= 1):
        return True, (f"{preamble}-char preamble before the first heading "
                      f"({hits} CoT cue(s)) — likely untagged chain-of-thought leak")

    # (B) A required heading repeated — the scratchpad re-drafted its own answer.
    if hits >= 1 and any(
        sum(1 for m in matches if m.group(1).upper() == h) > 1 for h in _REQUIRED_HEADINGS
    ):
        return True, "required heading repeated (scratchpad re-drafting) — likely untagged chain-of-thought leak"

    # (C) Headings missing entirely on a long response (pure-prose leak).
    if len(missing) >= _COT_MIN_MISSING and len(text) >= _COT_MIN_CHARS and hits >= _COT_MIN_PHRASES:
        return True, (f"missing required headings {missing} in a {len(text)}-char response "
                      f"({hits} CoT cue(s)) — likely untagged chain-of-thought leak")

    return False, None

# Latency budget. The Hermes MCP client caps ONE tool call at 300s, so these
# per-request budgets must sum to well under that.
#
# History: this used timeout=180s with max_tokens=16384 on every advocate call,
# while retry/backfill chains run SEQUENTIALLY at 180s each. Worst case was
# 180s (parallel) + up to 4x180s (serial retries) + 180s (judge) = ~1100s of
# work under a 300s ceiling -> guaranteed TimeoutError on any long prompt.
# Observed twice: 2026-09-09 14:27 and 2026-09-10 18:49.
#
# The output cap is ALSO a latency cap: given a large budget, reasoning models
# (deepseek-v4-pro, grok-4.20-reasoning, Qwen3.6-Plus) spend all of it thinking.
# Measured 2026-09-10 on a long prompt: deepseek-v4-pro took ~70s at 4096 tokens
# and ~152s at 8000 — so the 8000 config cannot fit inside ANY MCP-callable budget.
# 4000 is the point where a heavy reasoner still finishes fast enough to take part.
#
# The starvation problem 8000 was meant to solve (empty `content`, whole answer in
# `reasoning_content`, finish_reason=length) is now fixed properly in providers.py,
# which falls back to reasoning_content. The token cap no longer has to double as a
# starvation workaround.
CALL_TIMEOUT = 110.0
ADVOCATE_MAX_TOKENS = 4000
JUDGE_MAX_TOKENS = 4096

# Hard wall-clock caps. asyncio.wait_for enforces these; the httpx `timeout` passed to
# query_provider is only a per-READ timeout and does NOT bound total call time (a model
# streaming reasoning tokens slowly never trips it). CALL_TIMEOUT above is therefore set
# as a loose safety net, and THESE are the real bounds. Measured 2026-09-10 on a long
# prompt: kimi/kimi-k3.1-pro (latency unmeasured; replaced gmi/Qwen3.6-Plus ~85-96s 2026-10-27), deepseek-v4-pro ~63s at 4000 tokens, grok ~30s,
# gemini ~13s. A 60s CALL_TIMEOUT cut Qwen off mid-answer; 110s lets it finish.
HARD_CAP_S = 120.0
JUDGE_HARD_CAP_S = 90.0

# The retry/backfill chains run SEQUENTIALLY (plain awaits in for-loops), so they can
# chain N x CALL_TIMEOUT past the parallel phase. This budget, measured from the start
# of council_deliberate(), cuts them off. Set just under HARD_CAP_S on purpose: when the
# parallel phase itself was slow, retrying a slow provider is not worth the wall clock.
#
# Worst case  ~= HARD_CAP_S (parallel) + RETRY_BUDGET_S + JUDGE_HARD_CAP_S  ~= 220s,
# comfortably under the 300s ceiling that killed this call twice (2026-09-09, 2026-09-10).
# For deliberations that deserve longer, invoke council_deliberate() DIRECTLY in a
# background process (the plugin's own venv) — that path has no MCP ceiling.
RETRY_BUDGET_S = 40.0

async def council_deliberate(question: str, advocates: list | None = None, judge: tuple | None = None, include_raw: bool = True) -> dict:
    """Run multi-model deliberation. Returns {synthesis, advocates, cost_total, latency_ms, ...}"""
    advocates = advocates or DEFAULT_ADVOCATES
    judge = judge or DEFAULT_JUDGE
    
    # Phase 1: Query all advocates in parallel
    advocate_messages = [{"role": "user", "content": (
        f"Analyze this question and give your verdict with reasoning:\n\n{question}\n\n"
        "Respond with the three sections whose headings appear VERBATIM at the start "
        "of a line: 'VERDICT:', 'REASONING:', 'RISKS:'. Output ONLY the final answer — "
        "no reasoning process, no preamble, no step-by-step thinking."
    )}]
    
    start = time.monotonic()
    # asyncio.wait_for gives a HARD wall-clock cap. The httpx `timeout` below cannot:
    # it is a per-READ timeout, so a model that streams reasoning tokens slowly but
    # steadily never trips it. Measured 2026-09-10: deepseek-v4-pro ran 152s under a
    # 120s httpx timeout, pushing a real council call to 323.7s and blowing the
    # 300s MCP ceiling.
    advocate_tasks = [
        asyncio.wait_for(
            query_provider(p, m, advocate_messages, timeout=CALL_TIMEOUT, max_tokens=ADVOCATE_MAX_TOKENS),
            timeout=HARD_CAP_S,
        )
        for p, m in advocates
    ]
    gathered = await asyncio.gather(*advocate_tasks, return_exceptions=True)
    # wait_for timeouts arrive as exceptions, not ProviderResponses, so they would
    # vanish from both `valid` and `errors` and silently shrink the advocate count.
    # Convert them into reported failures.
    advocate_responses: list[ProviderResponse] = [
        r if isinstance(r, ProviderResponse)
        else ProviderResponse(
            provider=advocates[i][0], model=advocates[i][1], content="", latency_ms=0,
            cost_usd=0, error=f"{type(r).__name__}: {r}",
        )
        for i, r in enumerate(gathered)
    ]
    
    # Filter out errors
    valid = [r for r in advocate_responses if isinstance(r, ProviderResponse) and not r.error]
    errors = [r for r in advocate_responses if isinstance(r, ProviderResponse) and r.error]

    # Classify degraded responses (after strip_think_tags — must be stripped first to assess substance)
    # Only classify non-error responses; error responses are already excluded from 'valid'
    for r in valid:
        stripped = strip_think_tags(r.content)
        r.content = stripped  # mutate in-place for downstream
        degraded, reason = classify_degraded(stripped)
        r.degraded = degraded
        r.degraded_reason = reason
    
    if not valid and not errors:
        return {"synthesis": "All advocates failed.", "advocates": [], "errors": [], "cost_total": 0, "latency_ms": 0, "advocates_count": 0, "advocates_failed": 0, "retried": 0, "backfill_count": 0}
    if not valid:
        return {"synthesis": "All advocates failed.", "advocates": [], "errors": [{"provider": r.provider, "error": r.error} for r in errors], "cost_total": 0, "latency_ms": 0, "advocates_count": 0, "advocates_failed": len(errors), "retried": 0, "backfill_count": 0}
    
    # Retry degraded responses once with same provider
    retried = 0
    for i, r in enumerate(valid):
        if getattr(r, 'degraded', False):
            if time.monotonic() - start > RETRY_BUDGET_S:
                break
            retry_resp = await query_provider(r.provider, r.model, advocate_messages, timeout=CALL_TIMEOUT, max_tokens=ADVOCATE_MAX_TOKENS)
            raw_content = retry_resp.content  # save BEFORE strip_think_tags mutates in-place
            retry_stripped = strip_think_tags(raw_content)
            retry_degraded, _ = classify_degraded(retry_stripped)
            if not retry_resp.error and not retry_degraded:
                retry_resp.content = retry_stripped
                retry_resp.degraded = False
                retry_resp.degraded_reason = None
                valid[i] = retry_resp
                retried += 1
    
    # Also retry error responses (HTTP failures, timeouts) with same provider
    handled_errors = set()  # track errors already recovered
    for r in errors:
        if time.monotonic() - start > RETRY_BUDGET_S:
            break
        retry_resp = await query_provider(r.provider, r.model, advocate_messages, timeout=CALL_TIMEOUT, max_tokens=ADVOCATE_MAX_TOKENS)
        raw_content = retry_resp.content  # save BEFORE strip_think_tags mutates in-place
        retry_stripped = strip_think_tags(raw_content)
        retry_degraded, _ = classify_degraded(retry_stripped)
        if not retry_resp.error and not retry_degraded:
            retry_resp.content = retry_stripped
            retry_resp.degraded = False
            retry_resp.degraded_reason = None
            valid.append(retry_resp)
            retried += 1
            handled_errors.add(id(r))  # mark this error as handled so backfill skips it
    
    # Build backfill pool from unused providers (recompute used after error retries)
    used_providers = {r.provider for r in valid} | {r.provider for r in errors} | {judge[0]}
    backfill_pool = []
    for p in PROVIDERS:
        if p not in used_providers:
            default_model = PROVIDERS[p]["models"].get("default")
            if not default_model:
                default_model = list(PROVIDERS[p]["models"].values())[0]
            backfill_pool.append((p, default_model))
    
    # Backfill remaining degraded responses from pool
    backfill_count = 0
    for i, r in enumerate(valid):
        if getattr(r, 'degraded', False):
            for bp, bm in backfill_pool:
                if bp in {x.provider for x in valid}:
                    continue  # skip if already used by another successful backfill
                if time.monotonic() - start > RETRY_BUDGET_S:
                    break
                bf_resp = await query_provider(bp, bm, advocate_messages, timeout=CALL_TIMEOUT, max_tokens=ADVOCATE_MAX_TOKENS)
                bf_stripped = strip_think_tags(bf_resp.content)
                bf_degraded, _ = classify_degraded(bf_stripped)
                if not bf_resp.error and not bf_degraded:
                    bf_resp.content = bf_stripped
                    bf_resp.degraded = False
                    bf_resp.degraded_reason = None
                    valid[i] = bf_resp
                    backfill_count += 1
                    break  # backfill succeeded for this slot
    
    # Backfill remaining error responses from pool (adds to valid)
    for r in errors:
        if id(r) in handled_errors:
            continue  # already recovered via retry
        for bp, bm in backfill_pool:
            if bp in {x.provider for x in valid}:
                continue
            if time.monotonic() - start > RETRY_BUDGET_S:
                break
            bf_resp = await query_provider(bp, bm, advocate_messages, timeout=CALL_TIMEOUT, max_tokens=ADVOCATE_MAX_TOKENS)
            bf_stripped = strip_think_tags(bf_resp.content)
            bf_degraded, _ = classify_degraded(bf_stripped)
            if not bf_resp.error and not bf_degraded:
                bf_resp.content = bf_stripped
                bf_resp.degraded = False
                bf_resp.degraded_reason = None
                valid.append(bf_resp)
                backfill_count += 1
                break
    
    # Phase 2: Judge synthesizes only substantive advocate responses. If none
    # survived retry/backfill, fail closed instead of laundering scratchpads.
    contributors = [r for r in valid if not getattr(r, "degraded", False)]
    if not contributors:
        result = {
            "synthesis": "All advocates failed: no substantive responses.",
            "synthesis_truncated": False,
            "advocates": [{"provider": r.provider, "model": r.model, "content": r.content,
                           "latency_ms": r.latency_ms, "cost_usd": r.cost_usd,
                           "degraded": r.degraded, "degraded_reason": r.degraded_reason} for r in valid],
            "errors": [{"provider": r.provider, "error": r.error} for r in errors],
            "cost_total": sum(r.cost_usd for r in valid),
            "latency_ms": (time.monotonic() - start) * 1000,
            "advocates_count": len(valid),
            "advocates_failed": len(errors),
            "degraded_count": len(valid),
            "contributors_count": 0,
            "retried": retried,
            "backfill_count": backfill_count,
        }
        if include_raw:
            result["raw_responses"] = [{"provider": r.provider, "content": r.content} for r in valid]
        return result
    advocate_summaries = "\n\n---\n\n".join([
        f"**{r.provider}/{r.model}** ({r.latency_ms:.0f}ms, ${r.cost_usd:.4f}):\n{r.content}"
        for r in contributors
    ])
    
    judge_messages = [{
        "role": "user",
        "content": f"You are synthesizing answers from {len(contributors)} AI models who deliberated on:\n\n{question}\n\nHere are their individual responses:\n\n{advocate_summaries}\n\nSynthesize these perspectives into ONE clear answer. Identify:\n1. CONSENSUS (where models agree)\n2. DISSENT (where they disagree)\n3. FINAL VERDICT (your synthesis)\n4. CONFIDENCE (low/medium/high)\n\nBe direct. No hedging."
    }]
    
    try:
        judge_response = await asyncio.wait_for(
            query_provider(judge[0], judge[1], judge_messages, timeout=CALL_TIMEOUT, max_tokens=JUDGE_MAX_TOKENS),
            timeout=JUDGE_HARD_CAP_S,
        )
    except asyncio.TimeoutError:
        judge_response = ProviderResponse(
            provider=judge[0], model=judge[1], content="",
            latency_ms=JUDGE_HARD_CAP_S * 1000, cost_usd=0,
            error=f"judge timed out after {JUDGE_HARD_CAP_S}s",
        )
    judge_truncated = bool(judge_response.finish_reason == "length")
    
    total_latency = (time.monotonic() - start) * 1000
    total_cost = sum(r.cost_usd for r in valid) + judge_response.cost_usd
    
    result = {
        "synthesis": judge_response.content if not judge_response.error else "JUDGE FAILED: " + (judge_response.error or ""),
        "synthesis_truncated": judge_truncated,
        "advocates": [{"provider": r.provider, "model": r.model, "content": r.content, "latency_ms": r.latency_ms, "cost_usd": r.cost_usd, "degraded": r.degraded, "degraded_reason": r.degraded_reason} for r in valid],
        "errors": [{"provider": r.provider, "error": r.error} for r in errors],
        "cost_total": total_cost,
        "latency_ms": total_latency,
        "advocates_count": len(valid),
        "advocates_failed": len(errors),
        # COVERAGE, not capacity. An advocate that returns HTTP 200 but contributes no
        # substantive content still appears in advocates_count while adding nothing to
        # the synthesis. Callers should treat (advocates_count - degraded_count) as the
        # real breadth of the deliberation — see the council-mcp degradation-modes note.
        "degraded_count": sum(1 for r in valid if getattr(r, "degraded", False)),
        "contributors_count": len(contributors),
        "retried": retried,
        "backfill_count": backfill_count,
    }
    
    if include_raw:
        result["raw_responses"] = [{"provider": r.provider, "content": r.content} for r in valid]
    
    return result
