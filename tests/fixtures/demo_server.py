"""Small MCP server used by the tests and the demo.

Read-only tools carry the readOnlyHint annotation, tools that change state do
not. Run with `python demo_server.py` (stdio) or
`python demo_server.py http <port>` (streamable HTTP on 127.0.0.1).
"""

import os
import sys
import time

from mcp.types import ToolAnnotations

try:  # MCP Python SDK 2.x
    from mcp.server.mcpserver import MCPServer as Server
except ImportError:  # 1.x
    from mcp.server.fastmcp import FastMCP as Server

server = Server("omega-demo")
READ = ToolAnnotations(read_only_hint=True) if "read_only_hint" in ToolAnnotations.model_fields \
    else ToolAnnotations(readOnlyHint=True)
NOTES: dict[str, str] = {}


@server.tool(annotations=READ)
def add(a: int, b: int) -> str:
    """Add two integers."""
    return str(a + b)


@server.tool(annotations=READ)
def list_notes() -> str:
    """List the titles of the saved notes."""
    return "\n".join(sorted(NOTES)) or "(no notes)"


@server.tool(annotations=READ)
def get_note(title: str) -> str:
    """Return the text of one note."""
    return NOTES.get(title, f"(no note named '{title}')")


@server.tool(annotations=READ)
def whoami() -> str:
    """Return the demo token the server was started with."""
    return os.environ.get("DEMO_TOKEN", "(none)")


@server.tool(annotations=READ)
def slow(seconds: float) -> str:
    """Sleep, then answer."""
    time.sleep(seconds)
    return "done"


@server.tool(annotations=READ)
def big() -> str:
    """Return a very long text."""
    return "x" * 20000


@server.tool(annotations=READ)
def crash() -> str:
    """Kill the server process."""
    os._exit(3)


@server.tool()
def create_note(title: str, text: str) -> str:
    """Save a note. Changes state."""
    NOTES[title] = text
    return f"saved '{title}'"


@server.tool()
def delete_notes() -> str:
    """Delete every note. Changes state."""
    NOTES.clear()
    return "deleted"


@server.tool(annotations=READ)
def fail() -> str:
    """Always raise an error."""
    raise ValueError("this tool always fails")


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "http":
        port = int(sys.argv[2])
        try:  # 2.x takes the address when it starts
            server.run(transport="streamable-http", host="127.0.0.1", port=port)
        except TypeError:  # 1.x keeps it in its settings
            server.settings.host, server.settings.port = "127.0.0.1", port
            server.run(transport="streamable-http")
    else:
        server.run()
