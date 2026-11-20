"""An MCP server over streamable HTTP that wants a bearer token.

    python http_token_server.py PORT TOKEN [paged]

Written without the MCP SDK, so that what the tests see does not depend on
the SDK being on both ends. It answers every request with plain JSON, which
the transport allows, and has no stream of its own (GET is refused). Without
`Authorization: Bearer TOKEN` it answers 401.

With `paged` the list of tools comes two at a time, each page with the cursor
for the next.

Tools: add; echo_authorization, which says back the header it received;
broken, which is answered with a JSON-RPC error; slow_write, which takes its
time; and forget_token, after which the server answers 401 to everything, as
when a token expires.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOKEN = ""
PAGED = False
FORGOTTEN = False
READ = {"readOnlyHint": True}
TOOLS = [
    {"name": "add", "description": "Add two integers.", "annotations": READ,
     "inputSchema": {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                     "required": ["a", "b"]}},
    {"name": "echo_authorization", "description": "Say back the Authorization header of this request.",
     "annotations": READ, "inputSchema": {"type": "object", "properties": {}}},
    {"name": "broken", "description": "Is answered with an error.", "annotations": READ,
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "slow_write", "description": "Write something, slowly.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number"}}, "required": ["seconds"]}},
    {"name": "forget_token", "description": "Stop accepting the token.",
     "inputSchema": {"type": "object", "properties": {}}},
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):  # quiet
        pass

    def _send(self, status: int, body: dict | None = None, **headers: str) -> None:
        data = b"" if body is None else json.dumps(body).encode("utf-8")
        self.send_response(status)
        if body is not None:
            self.send_header("Content-Type", "application/json")
        for name, value in headers.items():
            self.send_header(name.replace("_", "-"), value)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):  # no stream from the server to the client
        self._send(405, None, Allow="POST")

    def do_DELETE(self):  # nothing to end: there are no sessions
        self._send(405, None, Allow="POST")

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length)
        global FORGOTTEN
        if FORGOTTEN or self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._send(401, {"error": "a bearer token is needed"}, WWW_Authenticate="Bearer")
            return
        message = json.loads(raw)
        if "id" not in message:  # a notification
            self._send(202)
            return
        method, params = message.get("method"), message.get("params") or {}
        if method == "initialize":
            result = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                      "capabilities": {"tools": {}},
                      "serverInfo": {"name": "token-http", "version": "1"}}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            if PAGED:
                start = int(params.get("cursor") or 0)
                result = {"tools": TOOLS[start:start + 2]}
                if start + 2 < len(TOOLS):
                    result["nextCursor"] = str(start + 2)
            else:
                result = {"tools": TOOLS}
        elif method == "tools/call":
            arguments = params.get("arguments") or {}
            if params.get("name") == "add":
                text = str(int(arguments["a"]) + int(arguments["b"]))
            elif params.get("name") == "echo_authorization":
                text = self.headers.get("Authorization", "")
            elif params.get("name") == "forget_token":
                FORGOTTEN, text = True, "forgotten"
            elif params.get("name") == "slow_write":
                time.sleep(float(arguments["seconds"]))
                text = "written"
            elif params.get("name") == "broken":
                self._send(200, {"jsonrpc": "2.0", "id": message["id"],
                                 "error": {"code": -32603, "message": "out of order"}})
                return
            else:
                self._send(200, {"jsonrpc": "2.0", "id": message["id"],
                                 "error": {"code": -32602, "message": "no such tool"}})
                return
            result = {"content": [{"type": "text", "text": text}], "isError": False}
        else:
            self._send(200, {"jsonrpc": "2.0", "id": message["id"],
                             "error": {"code": -32601, "message": "method not found"}})
            return
        self._send(200, {"jsonrpc": "2.0", "id": message["id"], "result": result})


if __name__ == "__main__":
    TOKEN = sys.argv[2]
    PAGED = "paged" in sys.argv[3:]
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), Handler).serve_forever()
