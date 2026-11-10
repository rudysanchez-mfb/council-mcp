#!/usr/bin/env python3
"""council-mcp — a stdio MCP server for multi-model deliberation.

MCP stdio transport requires pure JSON-RPC on stdout. Every log line, warning,
or traceback must go to stderr, otherwise the transport breaks.
"""
import json
import logging
import os
import sys

os.environ.setdefault("MCP_LOG_LEVEL", "ERROR")
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="%(message)s")
for _handler in list(logging.root.handlers):
    logging.root.removeHandler(_handler)

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer
except ImportError:  # MCP Python SDK 1.x
    from mcp.server.fastmcp import FastMCP as MCPServer

from mcp.types import TextContent

from council_mcp.council import DEFAULT_ADVOCATES, council_deliberate
from council_mcp.providers import PROVIDERS

server = MCPServer("council-mcp")


async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call. Kept separate from the registered tool wrappers so
    the behavior is directly testable without a running MCP session."""
    if name == "council_deliberate":
        question = arguments["question"]
        num = arguments.get("num_advocates", 3)
        if not isinstance(num, int) or not 2 <= num <= 4:
            raise ValueError("num_advocates must be an integer between 2 and 4")
        advocates = DEFAULT_ADVOCATES[:num]
        result = await council_deliberate(
            question,
            advocates=advocates,
            include_raw=arguments.get("include_raw", True),
        )
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "council_list_models":
        models = [
            {
                "provider": p,
                "model": cfg["models"].get("default", list(cfg["models"].values())[0]),
            }
            for p, cfg in PROVIDERS.items()
        ]
        return [TextContent(type="text", text=json.dumps(models, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


@server.tool(
    name="council_deliberate",
    description=(
        "Run multi-model deliberation: several models answer a question in "
        "parallel, then a judge synthesizes the best answer. Use for high-stakes "
        "decisions, architecture questions, debugging strategy, or any question "
        "where hallucination risk matters."
    ),
)
async def council_deliberate_tool(
    question: str,
    num_advocates: int = 3,
    include_raw: bool = True,
) -> str:
    """Deliberate on a question with 2-4 advocate models and a judge."""
    contents = await call_tool(
        "council_deliberate",
        {
            "question": question,
            "num_advocates": num_advocates,
            "include_raw": include_raw,
        },
    )
    return contents[0].text


@server.tool(
    name="council_list_models",
    description="List the provider/model pairs available for deliberation.",
)
async def council_list_models_tool() -> str:
    """List registered provider/model pairs."""
    contents = await call_tool("council_list_models", {})
    return contents[0].text


def main() -> None:
    """Console-script entrypoint for the stdio MCP server."""
    server.run(transport="stdio")


if __name__ == "__main__":
    main()
