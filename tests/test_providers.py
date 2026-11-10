import pytest
from unittest.mock import patch
from council_mcp.providers import strip_think_tags, ProviderResponse, query_provider

def test_strip_think_tags_closed_block():
    s = "<think>internal reasoning here</think>VERDICT: yes, do it."
    assert strip_think_tags(s) == "VERDICT: yes, do it."

def test_strip_think_tags_thinking_variant():
    s = "<thinking>internal reasoning</thinking>VERDICT: proceed."
    assert strip_think_tags(s) == "VERDICT: proceed."

def test_strip_think_tags_think_only_returns_empty():
    s = "<think>The user is asking me to analyze...</think>"
    assert strip_think_tags(s) == ""

def test_strip_think_tags_unclosed_block_stripped():
    s = "<think>reasoning that never ends because max_tokens"
    assert strip_think_tags(s) == ""

def test_strip_think_tags_deepseek_eot_marker():
    eot = '<｜end▁of▁thinking｜>'
    s = "hidden reasoning" + eot + "VERDICT: proceed."
    result = strip_think_tags(s)
    assert result == "VERDICT: proceed."

def test_strip_think_tags_plain_content_untouched():
    s = "VERDICT: plain answer with no tags."
    assert strip_think_tags(s) == s

def test_strip_think_tags_none_and_empty():
    assert strip_think_tags("") == ""
    assert strip_think_tags(None) == ""

def test_provider_response_new_fields_default():
    r = ProviderResponse(provider="x", model="y", content="z", latency_ms=1, cost_usd=0)
    assert r.finish_reason is None
    assert r.degraded is False
    assert r.degraded_reason is None


# --- reasoning_content fallback ------------------------------------------------
# A reasoning model can return HTTP 200 with an EMPTY `content` while putting the
# entire answer in the sibling `reasoning_content` field (observed 2026-09-10 with
# deepseek-v4-pro: completion_tokens=8000, finish_reason=length, content=""). Reading
# only `content` counted that as a successful advocate contributing nothing, so the
# synthesis reported N-model coverage while running on N-1. These tests pin the fix
# without making any network calls.

class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload
    def raise_for_status(self):
        return None
    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self._payload = payload
    async def __aenter__(self):
        return self
    async def __aexit__(self, *exc):
        return False
    async def post(self, *args, **kwargs):
        return _FakeResponse(self._payload)


async def _query_with_payload(payload):
    with patch("council_mcp.providers.httpx.AsyncClient", return_value=_FakeClient(payload)):
        return await query_provider(
            "deepseek", "deepseek-v4-pro", [{"role": "user", "content": "q"}]
        )


def _payload(message, finish_reason="stop"):
    return {
        "choices": [{"message": message, "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20},
    }


@pytest.mark.asyncio
async def test_query_provider_falls_back_to_reasoning_content(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    r = await _query_with_payload(_payload(
        {"role": "assistant", "content": "", "reasoning_content": "VERDICT: the real analysis"},
        finish_reason="length",
    ))
    assert r.content == "VERDICT: the real analysis"


@pytest.mark.asyncio
async def test_query_provider_prefers_content_when_present(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    r = await _query_with_payload(_payload(
        {"role": "assistant", "content": "the answer", "reasoning_content": "ignored"},
    ))
    assert r.content == "the answer"


@pytest.mark.asyncio
async def test_query_provider_empty_without_reasoning_stays_empty(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    r = await _query_with_payload(_payload({"role": "assistant", "content": ""}))
    assert r.content == ""
