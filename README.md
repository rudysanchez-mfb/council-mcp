# council-mcp

A [Model Context Protocol](https://modelcontextprotocol.io) server that runs
**multi-model deliberation**: several LLMs answer the same question in parallel,
then a judge model synthesizes their perspectives into one answer.

It is built for the cases where a single model is not enough — architecture
decisions, debugging strategy, high-risk operational changes, or any question
where hallucination risk matters. It also treats "the model returned HTTP 200"
as a *claim to verify*, not proof that the model actually said anything useful.

- **Parallel advocates** — 2–4 models queried concurrently with a hard per-call
  wall-clock cap.
- **Judge synthesis** — a separate model produces consensus, dissent, a final
  verdict, and a confidence level from the substantive advocate responses.
- **Degraded-response detection** — catches empty bodies, `<think>` / `<thinking>`
  reasoning wrappers, DeepSeek's end-of-thinking marker, and *untagged*
  chain-of-thought leaks that used to sail through as valid answers.
- **Bounded retry and backfill** — degraded or failed advocates get one retry;
  unused providers backfill remaining slots, all inside a wall-clock budget so a
  deliberation cannot run past the caller's deadline.
- **Cost and latency telemetry** — every response reports provider, model,
  latency, estimated cost, finish reason, and degradation status.

## How it works

```
question
   │
   ├──► advocate A ──┐
   ├──► advocate B ──┤  parallel, hard-capped
   ├──► advocate C ──┤
   └──► advocate D ──┘
                     │
             strip reasoning wrappers
             classify degraded output
                     │
          ┌──────────┴──────────┐
          │ retry degraded once │
          │ backfill from pool  │  wall-clock budget
          └──────────┬──────────┘
                     │
              substantive responses only
                     │
                 judge model
                     │
              synthesis + metadata
```

If no advocate produces substantive content after retry/backfill, the server
**fails closed**: it returns `All advocates failed: no substantive responses.`
instead of feeding raw scratchpad text to the judge.

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/rudysanchez-mfb/council-mcp.git
cd council-mcp
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

That installs the `council-mcp` console script. You can also run the module
directly:

```bash
python -m council_mcp.server
```

Both speak MCP over stdio and print **only** JSON-RPC to stdout; diagnostics go
to stderr.

## Configuration

Set API keys in the environment of your MCP host (not in this repository). Only
providers with a key present are usable; a missing key is reported as a
per-provider error and the rest of the council continues.

| Provider | Environment variable | Default model | Endpoint |
| --- | --- | --- | --- |
| Moonshot / Kimi | `KIMI_API_KEY` | `kimi-k3.1-pro` | `https://api.moonshot.ai/v1/chat/completions` |
| DeepSeek | `DEEPSEEK_API_KEY` | `deepseek-v4-pro` (judge: `deepseek-chat`) | `https://api.deepseek.com/v1/chat/completions` |
| xAI | `XAI_API_KEY` | `grok-4.20-reasoning` | `https://api.x.ai/v1/chat/completions` |
| Google Gemini | `GEMINI_API_KEY` | `gemini-2.5-flash` | Google OpenAI-compatible endpoint |

Default lineup: Kimi, DeepSeek, Grok, Gemini (the first `num_advocates` are
used). DeepSeek is the default judge. See `.env.example`.

The provider registry lives in `council_mcp/providers.py`, so adding a model or
an OpenAI-compatible endpoint is a small local change.

## MCP host configuration

Replace values as needed. Use an absolute path to the installed console script
if your host does not inherit your shell `PATH` (`which council-mcp`).

### Claude Desktop

`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "council": {
      "command": "council-mcp",
      "env": {
        "KIMI_API_KEY": "…",
        "DEEPSEEK_API_KEY": "…",
        "XAI_API_KEY": "…",
        "GEMINI_API_KEY": "…"
      }
    }
  }
}
```

### Cursor

`~/.cursor/mcp.json` (or project `.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "council": {
      "command": "council-mcp",
      "env": {
        "KIMI_API_KEY": "…",
        "DEEPSEEK_API_KEY": "…",
        "XAI_API_KEY": "…",
        "GEMINI_API_KEY": "…"
      }
    }
  }
}
```

### OpenCode

`opencode.json`:

```json
{
  "mcp": {
    "council": {
      "type": "local",
      "command": ["council-mcp"],
      "environment": {
        "KIMI_API_KEY": "…",
        "DEEPSEEK_API_KEY": "…",
        "XAI_API_KEY": "…",
        "GEMINI_API_KEY": "…"
      },
      "enabled": true
    }
  }
}
```

### Hermes Agent

`~/.hermes/config.yaml`:

```yaml
mcp_servers:
  council-mcp:
    command: council-mcp
    args: []
    env:
      KIMI_API_KEY: "…"
      DEEPSEEK_API_KEY: "…"
      XAI_API_KEY: "…"
      GEMINI_API_KEY: "…"
```

If your host does not support `env` in the server block, export the keys in the
environment that launches the host instead.

## Tools

### `council_deliberate`

Runs a deliberation and returns JSON.

| Argument | Type | Default | Meaning |
| --- | --- | --- | --- |
| `question` | string | required | The question for the council. |
| `num_advocates` | integer (2–4) | `3` | How many advocate models to query. |
| `include_raw` | boolean | `true` | Include each advocate's raw content in the result. |

Result fields include:

- `synthesis` — the judge's answer.
- `advocates` — per-model content, latency, cost, `degraded`, `degraded_reason`.
- `errors` — providers that failed after retry/backfill.
- `advocates_count`, `advocates_failed`, `degraded_count`, `contributors_count`.
- `retried`, `backfill_count`, `cost_total`, `latency_ms`, `synthesis_truncated`.

**Coverage, not capacity:** `advocates_count` includes degraded responses. The
real breadth of a deliberation is
`advocates_count - degraded_count` (also exposed as `contributors_count`).
Callers should read the latter before claiming "N models agreed."

### `council_list_models`

Lists the provider/model pairs currently registered.

## Degraded response detection

`classify_degraded()` runs after reasoning wrappers are stripped and returns
`(degraded, reason)`. A response is degraded when:

1. **Empty body** — nothing substantive after stripping wrappers.
2. **Long reasoning preamble** — ≥600 chars before the first required heading
   plus a chain-of-thought cue, or ≥2000 chars before it regardless.
3. **Re-drafted answer** — a required heading appears more than once alongside a
   chain-of-thought cue.
4. **Missing structure** — a ≥1500-char response missing at least two of the
   `VERDICT` / `REASONING` / `RISKS` headings, with at least one CoT cue.

Detection is deliberately **structural first**: the advocate prompt requires the
three headings, so their absence is the primary signal. A small phrase list only
corroborates; it never triggers a flag on its own. Short markerless answers and
long but clean structured answers are never flagged.

Degraded responses are retried once and then replaced from the unused-provider
pool when possible. Anything still degraded is excluded from the judge input and
reported in the result.

## Latency budget

The default budgets are tuned for hosts that cap a tool call at 300 seconds:

| Constant | Value | Purpose |
| --- | --- | --- |
| `CALL_TIMEOUT` | 110 s | Per-request read timeout. |
| `HARD_CAP_S` | 120 s | Hard wall-clock cap on the parallel advocate phase. |
| `JUDGE_HARD_CAP_S` | 90 s | Hard cap on the judge call. |
| `RETRY_BUDGET_S` | 40 s | Total budget for sequential retry/backfill. |
| `ADVOCATE_MAX_TOKENS` | 4000 | Output cap; also a latency cap for reasoners. |
| `JUDGE_MAX_TOKENS` | 4096 | Judge output cap. |

Worst case is approximately `HARD_CAP_S + RETRY_BUDGET_S + JUDGE_HARD_CAP_S`
(≈250 s). If your host allows longer calls, tune the constants in
`council_mcp/council.py`.

## Tests

The test suite makes **no network calls and needs no API keys**; provider calls
are mocked.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

Coverage includes `strip_think_tags`, the `reasoning_content` fallback, every
`classify_degraded` branch, retry/backfill accounting, the retry budget guard,
fail-closed behavior, and server argument forwarding.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 Rudy Sanchez.
