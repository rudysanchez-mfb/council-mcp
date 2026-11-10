"""Untagged chain-of-thought (CoT) leak detection.

An advocate can return HTTP 200 with a large blob of RAW reasoning as `content`,
with NO <think> tags and no answer structure. strip_think_tags() leaves it intact
(nothing to strip), so the old "empty after strip" degradation check let it pass
as a healthy advocate and get fed to the judge. Observed intermittently with
deepseek-v4-pro on open-ended prompts (2026-10-04).
"""
import pytest
from unittest.mock import AsyncMock, patch

from council_mcp.council import classify_degraded
from council_mcp.providers import ProviderResponse


# A faithful stand-in for the real deepseek-v4-pro leak: long, markerless,
# starts with meta-reasoning prose, contains the prompt echoed back.
COT_LEAK = (
    "We need answer user asks. We need produce analysis with verdict reasoning risks. "
    "Need parse question. Let's parse. We need determine robust heuristic.\n"
    + ("This sounds like a troubleshooting exercise about bind mounts. " * 60)
)

STRUCTURED = (
    "VERDICT: yes, do it.\n"
    "REASONING: because the bind mount shares the inode.\n"
    "RISKS: none material.\n"
    + ("Detail line. " * 200)
)

# Missing only RISKS -> missing count is 1, below the gate; must NOT flag.
PARTIAL = "VERDICT: yes\nREASONING: because\n" + ("Detail line. " * 200)


def test_classify_degraded_empty_is_degraded():
    degraded, reason = classify_degraded("")
    assert degraded is True
    assert "empty" in reason.lower()


def test_classify_degraded_none_is_degraded():
    degraded, _ = classify_degraded(None)
    assert degraded is True


def test_classify_degraded_structured_passes():
    degraded, reason = classify_degraded(STRUCTURED)
    assert degraded is False
    assert reason is None


def test_classify_degraded_short_markerless_not_flagged():
    # Short markerless output is a different failure mode; never flag on length alone.
    degraded, _ = classify_degraded("This is just unsafe, don't do it.")
    assert degraded is False


def test_classify_degraded_long_but_one_heading_missing_not_flagged():
    degraded, _ = classify_degraded(PARTIAL)
    assert degraded is False


def test_classify_degraded_untagged_cot_leak_flagged():
    assert len(COT_LEAK) > 1500
    degraded, reason = classify_degraded(COT_LEAK)
    assert degraded is True
    assert "chain-of-thought" in reason.lower()


def test_classify_degraded_markdown_headings_recognized():
    # Models emit headings as "**VERDICT:**", "## VERDICT", "> REASONING:" etc.
    md = ("**VERDICT:** go\n## REASONING:\n> RISKS: low\n" + ("y " * 900))
    degraded, _ = classify_degraded(md)
    assert degraded is False


def test_classify_degraded_case_insensitive_headings():
    ci = ("Verdict: go\nReasoning: fine\nRisks: low\n" + ("y " * 900))
    degraded, _ = classify_degraded(ci)
    assert degraded is False


# --- integration: a leaked advocate must be replaced, not passed through -------

@pytest.mark.asyncio
async def test_untagged_cot_advocate_is_flagged_and_replaced():
    """deepseek leaks raw CoT; its retry leaks again; backfill from the unused
    pool (kimi/gemini) replaces it. The leak must never reach the judge as coverage."""
    from council_mcp.council import council_deliberate

    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        if provider == "deepseek" and model == "deepseek-v4-pro":
            return ProviderResponse(provider=provider, model=model, content=COT_LEAK,
                                    latency_ms=1, cost_usd=0, finish_reason="stop")
        return ProviderResponse(provider=provider, model=model, content=STRUCTURED,
                                latency_ms=1, cost_usd=0, finish_reason="stop")

    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate(
            "Open ended question?",
            advocates=[("deepseek", "deepseek-v4-pro"), ("grok", "grok-4.20-reasoning")],
            judge=("deepseek", "deepseek-chat"),
        )

    # No surviving advocate may still be carrying the leak.
    assert all(a["degraded"] is False for a in result["advocates"])
    assert all("We need answer user asks" not in a["content"] for a in result["advocates"])


@pytest.mark.asyncio
async def test_council_deliberate_include_raw_false_suppresses_raw():
    from council_mcp.council import council_deliberate

    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        return ProviderResponse(provider=provider, model=model, content=STRUCTURED,
                                latency_ms=1, cost_usd=0, finish_reason="stop")

    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate(
            "Q?", advocates=[("grok", "x")], judge=("deepseek", "deepseek-chat"),
            include_raw=False,
        )

    assert "raw_responses" not in result


# --- server plumbing: include_raw must actually be forwarded -------------------

@pytest.mark.asyncio
async def test_server_forwards_include_raw():
    import council_mcp.server as server_mod

    mock_result = {"synthesis": "ok", "advocates": [], "advocates_count": 0}
    with patch.object(server_mod, "council_deliberate", new=AsyncMock(return_value=mock_result)) as m:
        await server_mod.call_tool("council_deliberate", {"question": "q", "include_raw": False})

    _, kwargs = m.call_args
    assert kwargs.get("include_raw") is False


# --- headings-drafting leak (the flavour the first fix MISSED) -----------------
# Ground truth: 2026-10-04 sample-2, deepseek-v4-pro returned an 18,360-char blob
# whose first ~5.7K chars were reasoning prose, THEN it drafted VERDICT/REASONING/
# RISKS — twice. A "missing headings" gate cannot see this: the headings ARE there.

HEADINGS_DRAFT_LEAK = (
    "We need answer user. Need parse: \"Analyze this question and give your verdict "
    "with reasoning ... Respond with the three sections whose headings appear VERBATIM "
    "at the start of a line: 'VERDICT:', 'REASONING:', 'RISKS:'.\"\n\n"
    "We need provide verdict, reasoning, risks. Need think thoroughly. Need output exactly "
    "three sections headings verbatim at start lines. Ensure no preamble.\n\n"
    "Let's understand scenario. Possible root causes:\n"
    + ("1. Non-idempotent model; Airflow retries; race conditions; clock skew.\n" * 60)
    + "\nVERDICT:\nLikely a non-idempotent load step.\n"
    + "REASONING:\nSeveral candidates.\n"
    + "RISKS:\nSilent duplication.\n"
    + "\nWait, refine.\nVERDICT:\nMost likely overlapping Airflow runs.\n"
    + "REASONING:\nReconsidering the trigger.\n"
)


def test_classify_degraded_headings_drafting_leak_flagged():
    assert len(HEADINGS_DRAFT_LEAK) > 2000
    degraded, reason = classify_degraded(HEADINGS_DRAFT_LEAK)
    assert degraded is True
    assert "chain-of-thought" in reason.lower()


def test_classify_degraded_large_preamble_single_headings_flagged():
    text = ("We need reason carefully about this. " * 60) + "\nVERDICT: a\nREASONING: b\nRISKS: c\n"
    degraded, reason = classify_degraded(text)
    assert degraded is True


def test_classify_degraded_repeated_heading_flagged():
    text = "VERDICT: a\nREASONING: b\nRISKS: c\nLet's think again.\nVERDICT: d\n"
    degraded, _ = classify_degraded(text)
    assert degraded is True


def test_classify_degraded_substantial_clean_answer_passes():
    text = "VERDICT: x\nREASONING: " + ("detail " * 300) + "\nRISKS: y\n"
    assert len(text) > 1500
    degraded, _ = classify_degraded(text)
    assert degraded is False


@pytest.mark.asyncio
async def test_degraded_advocate_is_excluded_from_judge_input():
    """When a leaked advocate cannot be replaced (backfill also leaks), its
    scratchpad must NOT be handed to the judge. It stays in `advocates` flagged
    degraded, but is dropped from the synthesis input."""
    from council_mcp.council import council_deliberate

    calls = []

    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        calls.append((provider, model, messages[0]["content"]))
        if provider == "grok":
            return ProviderResponse(provider=provider, model=model, content=STRUCTURED,
                                    latency_ms=1, cost_usd=0, finish_reason="stop")
        if model == "deepseek-chat":  # the judge itself
            return ProviderResponse(provider=provider, model=model,
                                    content="VERDICT: s\nREASONING: r\nRISKS: k",
                                    latency_ms=1, cost_usd=0, finish_reason="stop")
        return ProviderResponse(provider=provider, model=model, content=COT_LEAK,
                                latency_ms=1, cost_usd=0, finish_reason="stop")

    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate(
            "Q?",
            advocates=[("deepseek", "deepseek-v4-pro"), ("grok", "grok-4.20-reasoning")],
            judge=("deepseek", "deepseek-chat"),
        )

    assert result["contributors_count"] == 1  # only grok fed the judge
    judge_inputs = [m for p, mo, m in calls if mo == "deepseek-chat"]
    assert judge_inputs
    assert all("We need answer user asks" not in m for m in judge_inputs)


@pytest.mark.asyncio
async def test_all_degraded_advocates_fail_closed_without_judge():
    from council_mcp.council import council_deliberate

    calls = []

    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        calls.append((provider, model))
        return ProviderResponse(provider=provider, model=model, content=COT_LEAK,
                                latency_ms=1, cost_usd=0, finish_reason="stop")

    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate(
            "Open-ended question?",
            advocates=[("deepseek", "deepseek-v4-pro"), ("grok", "grok-4.20-reasoning")],
            judge=("deepseek", "deepseek-chat"), include_raw=False,
        )

    assert ("deepseek", "deepseek-chat") not in calls
    assert result["advocates_count"] == 2
    assert result["degraded_count"] == 2
    assert result["contributors_count"] == 0
    assert result["synthesis"].startswith("All advocates failed")
    assert "raw_responses" not in result
    assert set(result) >= {"synthesis_truncated", "advocates", "errors", "cost_total",
                           "latency_ms", "advocates_failed", "retried", "backfill_count"}


def test_long_structured_answer_with_question_language_is_not_degraded():
    text = ("VERDICT: yes\nREASONING: "
            + "The user asks us to walk through this carefully. " * 170
            + "\nRISKS: none\n")
    assert len(text) > 6000
    degraded, reason = classify_degraded(text)
    assert degraded is False, reason
