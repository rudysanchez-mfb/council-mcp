import pytest, json, asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from council_mcp.council import council_deliberate
from council_mcp.providers import ProviderResponse, query_provider

@pytest.mark.asyncio
async def test_council_deliberate_all_advocates_succeed():
    mock_responses = [
        ProviderResponse(provider="deepseek", model="deepseek-chat", content="Verdict: YES\nReasoning: Good plan", latency_ms=100, cost_usd=0.001, finish_reason="stop"),
        ProviderResponse(provider="nous", model="hermes-3-llama-3.1-70b", content="Verdict: YES\nReasoning: Solid approach", latency_ms=150, cost_usd=0.002, finish_reason="stop"),
    ]
    mock_judge = ProviderResponse(provider="deepseek", model="deepseek-chat", content="Synthesis: Consensus YES", latency_ms=200, cost_usd=0.001, finish_reason="stop")
    
    call_count = 0
    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        nonlocal call_count
        if call_count < 2:
            result = mock_responses[call_count]
        else:
            result = mock_judge
        call_count += 1
        return result
    
    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate("Test question?", advocates=[("deepseek", "deepseek-chat"), ("nous", "hermes-3")], judge=("deepseek", "deepseek-chat"))
    
    assert "YES" in result["synthesis"]
    assert result["advocates_count"] == 2
    assert result["advocates_failed"] == 0
    assert result["cost_total"] > 0

@pytest.mark.asyncio
async def test_council_deliberate_one_advocate_fails():
    # Responses: success, failure, retry_failure (error), backfill_success, judge
    # deepseek → valid; nous(failure) → errors → retry succeeds → backfill not needed
    mock_success = ProviderResponse(provider="deepseek", model="deepseek-chat", content="Verdict: MAYBE\nReasoning: Risky", latency_ms=100, cost_usd=0.001, finish_reason="stop")
    mock_failure = ProviderResponse(provider="nous", model="hermes-3", content="", latency_ms=50, cost_usd=0, error="Timeout", finish_reason=None)
    mock_retry_success = ProviderResponse(provider="nous", model="hermes-3", content="Verdict: NO\nReasoning: Safe", latency_ms=100, cost_usd=0.001, finish_reason="stop")
    mock_judge = ProviderResponse(provider="deepseek", model="deepseek-chat", content="Synthesis: Mixed views", latency_ms=100, cost_usd=0.001, finish_reason="stop")
    
    responses = [mock_success, mock_failure, mock_retry_success, mock_judge]
    call_idx = 0
    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        nonlocal call_idx
        r = responses[call_idx]
        call_idx += 1
        return r
    
    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate("Risky question?", advocates=[("deepseek", "deepseek-chat"), ("nous", "hermes-3")], judge=("deepseek", "deepseek-chat"))
    
    assert result["advocates_count"] == 2  # deepseek + nous retry
    assert result["advocates_failed"] == 1  # original errors before recovery
    assert result["retried"] == 1  # error retry succeeded
    assert result["backfill_count"] == 0  # retry succeeded, no backfill needed
    assert len(result["errors"]) == 1

@pytest.mark.asyncio
async def test_council_all_advocates_fail():
    mock_failures = [
        ProviderResponse(provider="deepseek", model="deepseek-chat", content="", latency_ms=50, cost_usd=0, error="API down", finish_reason=None),
        ProviderResponse(provider="nous", model="hermes-3", content="", latency_ms=50, cost_usd=0, error="API down", finish_reason=None),
    ]
    
    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        return mock_failures.pop(0)
    
    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate("Question?", advocates=[("deepseek", "deepseek-chat"), ("nous", "hermes-3")], judge=("deepseek", "deepseek-chat"))
    
    assert "All advocates failed" in result["synthesis"]
    assert result["advocates_count"] == 0


@pytest.mark.asyncio
async def test_degraded_count_is_reported(monkeypatch):
    """An advocate that returns 200 but no substantive content is counted in
    advocates_count yet adds nothing to the synthesis. degraded_count exposes that
    so callers can compute real coverage."""
    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        return ProviderResponse(provider=provider, model=model, content="", latency_ms=1,
                                cost_usd=0, finish_reason="length")

    # Freeze wall-clock so the retry-budget guard fires immediately (see below).
    ticks = iter([0.0] + [9999.0] * 64)
    monkeypatch.setattr("council_mcp.council.time.monotonic", lambda: next(ticks))

    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate(
            "Question?",
            advocates=[("deepseek", "deepseek-v4-pro"), ("grok", "grok-4.20-reasoning")],
            judge=("deepseek", "deepseek-chat"),
        )

    assert result["advocates_count"] == 2
    assert result["degraded_count"] == 2


@pytest.mark.asyncio
async def test_retry_budget_guard_prevents_unbounded_retries(monkeypatch):
    """The retry/backfill chains run SEQUENTIALLY. Without a wall-clock budget they
    chain N x CALL_TIMEOUT past the parallel phase and blow the caller's deadline
    (this is the bug that made council_deliberate exceed the 300s MCP ceiling).
    Once the budget is spent, a degraded advocate must NOT be retried."""
    calls = []

    async def mock_query(provider, model, messages, timeout=60.0, max_tokens=16384):
        calls.append(provider)
        return ProviderResponse(provider=provider, model=model, content="", latency_ms=1,
                                cost_usd=0, finish_reason="length")

    # First read is `start`; every later read is far past RETRY_BUDGET_S.
    ticks = iter([0.0] + [9999.0] * 64)
    monkeypatch.setattr("council_mcp.council.time.monotonic", lambda: next(ticks))

    with patch("council_mcp.council.query_provider", side_effect=mock_query):
        result = await council_deliberate(
            "Question?",
            advocates=[("deepseek", "deepseek-v4-pro"), ("grok", "grok-4.20-reasoning")],
            judge=("deepseek", "deepseek-chat"),
        )

    assert result["retried"] == 0
    assert result["backfill_count"] == 0
    # 2 parallel advocates, zero retries, and no judge: all advocates were
    # degraded, so the council now fails closed instead of using their output.
    assert calls == ["deepseek", "grok"], calls
    assert result["contributors_count"] == 0

