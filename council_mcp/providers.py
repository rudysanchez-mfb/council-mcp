import os, httpx, asyncio, time, json, sys, re
from dataclasses import dataclass, field

PROVIDERS = {
    "kimi": {
        "base_url": "https://api.moonshot.ai/v1/chat/completions",
        "api_key_env": "KIMI_API_KEY",
        "models": {"default": "kimi-k3.1-pro"},
        "auth_type": "bearer",
        # Moonshot rejects any temperature other than 1 for this model.
        "extra_body": {"temperature": 1},
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1/chat/completions",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": {"default": "deepseek-v4-pro"},
        "auth_type": "bearer",
    },
    "grok": {
        "base_url": "https://api.x.ai/v1/chat/completions",
        "api_key_env": "XAI_API_KEY",
        "models": {"default": "grok-4.20-reasoning"},
        "auth_type": "bearer",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "api_key_env": "GEMINI_API_KEY",
        "models": {"default": "gemini-2.5-flash"},
        "auth_type": "bearer",
    },
}

@dataclass
class ProviderResponse:
    provider: str
    model: str
    content: str
    latency_ms: float
    cost_usd: float
    error: str | None = None
    finish_reason: str | None = None
    degraded: bool = False
    degraded_reason: str | None = None

DEEPSEEK_EOT = "<｜end▁of▁thinking｜>"

def strip_think_tags(content: str | None) -> str:
    """Remove reasoning-trace wrappers (<think>/<thinking> blocks, DeepSeek EOT marker).

    Returns only visible/declarative content, stripped. Empty string = the model
    produced NO substantive output (degradation Mode 5b)."""
    if not content:
        return ""
    if DEEPSEEK_EOT in content:
        content = content.split(DEEPSEEK_EOT, 1)[1]
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
    content = re.sub(r"<thinking>.*?</thinking>", "", content, flags=re.DOTALL)
    # Unclosed opener (token cap hit mid-reasoning): everything after is trace
    for opener in ("<think>", "<thinking>"):
        if opener in content:
            content = content.split(opener, 1)[0]
    return content.strip()

def _parse_anthropic_response(data: dict) -> str:
    """Extract text from Anthropic Messages API response."""
    blocks = data.get("content", [])
    return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")

def _build_anthropic_request(model: str, messages: list[dict], max_tokens: int) -> dict:
    """Convert OpenAI-style messages to Anthropic Messages API format.
    
    Anthropic puts system as a top-level field, not a message role.
    """
    system_parts = []
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_parts.append(msg["content"])
        else:
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})

    body = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": anthropic_messages,
    }
    if system_parts:
        body["system"] = "\n\n".join(system_parts)
    return body

async def query_provider(provider_name: str, model_name: str, messages: list[dict], timeout: float = 180.0, max_tokens: int = 16384) -> ProviderResponse:
    """Query a single provider. Handles both OpenAI-compatible and Anthropic-native APIs."""
    cfg = PROVIDERS[provider_name]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        return ProviderResponse(provider=provider_name, model=model_name, content="", latency_ms=0, cost_usd=0, error=f"Missing {cfg['api_key_env']}")

    auth_type = cfg.get("auth_type", "bearer")
    is_anthropic = (auth_type == "x-api-key")

    # Build headers
    if is_anthropic:
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
    else:
        auth_header = api_key if auth_type == "raw" else f"Bearer {api_key}"
        headers = {"Authorization": auth_header, "Content-Type": "application/json"}

    # Build request body
    if is_anthropic:
        body = _build_anthropic_request(model_name, messages, max_tokens=max_tokens)
    else:
        body = {"model": model_name, "messages": messages, "temperature": 0.7, "max_tokens": max_tokens, **(cfg.get("extra_body", {}))}

    start = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(cfg["base_url"], headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()

            # Extract content
            if is_anthropic:
                content = _parse_anthropic_response(data) or ""
                usage = data.get("usage", {})
                prompt_tokens = usage.get("input_tokens", 0)
                completion_tokens = usage.get("output_tokens", 0)
                finish_reason = data.get("stop_reason")
                # Debug: log when Anthropic returns 200 but content is empty
                if not content or not content.strip():
                    print(f"[council DEBUG] {provider_name}/{model_name}: Anthropic 200 OK but empty content. "
                          f"input_tokens={prompt_tokens} output_tokens={completion_tokens} "
                          f"stop_reason={data.get('stop_reason', 'N/A')} "
                          f"finish_reason={finish_reason} "
                          f"content_blocks={len(data.get('content', []))}",
                          file=sys.stderr)
            else:
                content = data["choices"][0]["message"].get("content") or ""
                # Reasoning models can return an EMPTY `content` while placing the
                # entire answer in the sibling `reasoning_content` field. Observed
                # 2026-09-10 with deepseek-v4-pro: completion_tokens=8000,
                # finish_reason=length, content="". Treating that as a successful
                # advocate silently drops the model's whole contribution and the
                # synthesis proceeds one model short while reporting full coverage.
                # Fall back to the reasoning text; strip_think_tags() downstream
                # already handles trace wrappers.
                if not content.strip():
                    reasoning = data["choices"][0]["message"].get("reasoning_content")
                    if reasoning and reasoning.strip():
                        content = reasoning
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                finish_reason = data["choices"][0].get("finish_reason")
                # Debug: log when API returns 200 but content is empty
                if not content or not content.strip():
                    print(f"[council DEBUG] {provider_name}/{model_name}: 200 OK but empty content. "
                          f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} "
                          f"finish_reason={finish_reason} "
                          f"raw_message_keys={list(data['choices'][0].get('message', {}).keys())}",
                          file=sys.stderr)

            latency = (time.monotonic() - start) * 1000
            cost = estimate_cost(provider_name, model_name, prompt_tokens, completion_tokens)
            return ProviderResponse(provider=provider_name, model=model_name, content=content, latency_ms=latency, cost_usd=cost, finish_reason=finish_reason)
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        # Log timeout vs other errors
        err_type = type(e).__name__
        if "timeout" in err_type.lower() or "timeout" in str(e).lower():
            print(f"[council DEBUG] {provider_name}/{model_name}: TIMEOUT after {latency:.0f}ms ({err_type}: {e})", file=sys.stderr)
        else:
            print(f"[council DEBUG] {provider_name}/{model_name}: ERROR {err_type}: {e}", file=sys.stderr)
        return ProviderResponse(provider=provider_name, model=model_name, content="", latency_ms=latency, cost_usd=0, error=str(e))

def estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Rough cost estimate. Not exact — for tracking purposes."""
    # Approximate rates per 1M tokens
    # kimi: platform.kimi.ai/docs/pricing/chat, checked 2026-10-27 (input/output, cache-miss)
    rates = {"kimi": (2.00, 10.00), "deepseek": (0.89, 2.65), "grok": (5.00, 15.00), "gemini": (0.075, 0.30)}
    pr, cr = rates.get(provider, (0.50, 0.50))
    return (prompt_tokens * pr + completion_tokens * cr) / 1_000_000
