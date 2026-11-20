#!/usr/bin/env python3
"""Run the real Omega loop with this plugin and check what happened.

    python tests/e2e/run_e2e.py /path/to/PeTTa

/path/to/PeTTa is a PeTTa tree set up for Omega as Omega's README describes
(run.sh, run.metta, repos/Omega). Run this with the Python that Omega runs
with, and with the `mcp` package installed in it.

No LLM key is needed. A test-only plugin (e2eharness.py) plays the LLM from a
script and plays the channel; everything else is Omega itself: its loop, its
plugin loader, its tool dispatch, its prompt. The harness speaks both of
Omega's ways of talking to an LLM, the tools API of the main branch and the
text protocol of the v0.1.x releases, and uses the one this Omega has. The
plugin is registered from this checkout, wherever it is, through Omega's
custom plugins file. It gets three servers from tests/fixtures, all given the
same token: demo_server.py and hostile_server.py over stdio, and
http_token_server.py over HTTP, which answers 401 to a request without the
token.

What is checked, on the requests that the LLM provider received:
- the three tools are offered and the plugin's section is in the prompt of
  every request (a heartbeat that wiped it would show here);
- list, describe and call return what the servers gave; JSON arguments with
  quotes, backslashes and braces arrive intact;
- a write is refused unless the servers file allows it;
- tool names outside the MCP naming rules never reach the LLM;
- the token is in no request and nowhere in Omega's log;
- when a server dies the prompt says so at the next iteration, and says it is
  back after the next call reconnects; the prompt is rewritten on those two
  occasions only.

Two short runs follow, in which the plugin must stay out: one with the plugin
left off, as it is when installed, and one with a servers file that cannot be
read. In both the agent has to run, without the plugin's tools; in the second
Omega's log has to say where the file is wrong without quoting it.

The runs are without Omega's filesystem restrictions unless --security-policy
names a policy file. With one, a line is added to the checks when Omega's log
says that the policy was applied.

Use a tree that no real agent lives in. The run keeps Omega's memory folder
and its vector store in a temporary folder, but Omega writes its history to
repos/Omega/memory/history.metta wherever the memory folder is: that file is
emptied for the run and its content put back afterwards. With
--core-plugins-file (for an Omega without customPluginsConfigPath, such as
v0.1.24) the two plugins are added to repos/Omega/config/plugins.yaml for the
run, and that file is put back as well. It also works on a tree where the
plugin is installed: the run uses this checkout in place of the entry that is
there.

--filesystem-server adds the reference filesystem MCP server, started with
npx (network needed the first time; proxy variables are passed on to it),
limited to a temporary folder. --filesystem-command starts it in another
way, for instance one version of it under one version of Node:

    --filesystem-command "npx -y -p node@24 -p @modelcontextprotocol/server-filesystem@2.4.0 mcp-server-filesystem"
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PLUGIN = HERE.parents[1]
FIXTURES = PLUGIN / "tests" / "fixtures"
TOKEN = "e2e-token-7f/Qx+91==zz"
NOTE_TEXT = 'buy milk; call "Bob" (today) & check {budget}; path C:\\temp\\x; done.'
REDACTED = "[redacted]"


# A stdio server is started with a short list of variables (HOME, PATH and a few more), not with
# the environment of Omega. These are passed on to npx when they are set, so that its first
# download also works where the network is reached through a proxy.
NETWORK_VARIABLES = ("HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
                     "NODE_EXTRA_CA_CERTS", "SSL_CERT_FILE", "NPM_CONFIG_USERCONFIG")


def servers_file(work: Path, files_command: list[str] | None, remote_port: int) -> str:
    passed_on = ", ".join(f'{name}: "${{{name}}}"' for name in NETWORK_VARIABLES if os.environ.get(name))
    files = f"""
  files:
    transport: stdio
    command: {json.dumps(files_command[0])}
    args: {json.dumps([*files_command[1:], str(work / 'sandbox')])}
    env: {{{passed_on}}}
    description: Reference filesystem server, one folder
    connect_timeout: 180
""" if files_command else ""
    return f"""servers:
  demo:
    transport: stdio
    command: {sys.executable}
    args: ["{FIXTURES / 'demo_server.py'}"]
    env: {{DEMO_TOKEN: "${{E2E_MCP_TOKEN}}"}}
    description: Demo notes server
    tools:
      deny: ["slow", "big"]
      allow_write: ["create_note"]
  hostile:
    transport: stdio
    command: {sys.executable}
    args: ["{FIXTURES / 'hostile_server.py'}"]
    env: {{HOSTILE_TOKEN: "${{E2E_MCP_TOKEN}}"}}
    description: A server that misbehaves
  remote:
    transport: http
    url: http://127.0.0.1:{remote_port}/mcp
    headers: {{Authorization: "Bearer ${{E2E_MCP_TOKEN}}"}}
    description: A server over HTTP that wants a token
    tools:
      deny: ["forget_token", "slow_write"]
{files}"""


def plugins_file() -> str:
    return f"""- name: mcpclient
  loader: metta
  location: "{PLUGIN}"

- name: e2eharness
  loader: python
  location: "{HERE}"
"""


class Failed(Exception):
    pass


def need(condition: bool, message: str) -> None:
    if not condition:
        raise Failed(message)


def steps(work: Path, with_files: bool, text_protocol: bool) -> list[tuple[str, dict, object]]:
    """The script: (tool, arguments, check of the tool's result).

    With the tools API the arguments are the tool call's. With the text protocol the harness
    writes one line, the tool's name and then the values one after the other.
    """

    def listed_demo(result):
        need(result["status"] == "ok", f"list demo: {result}")
        names = [line.split(" ")[0] for line in result["tools"]]
        need(names == ["add", "crash", "create_note", "delete_notes", "fail", "get_note", "list_notes", "whoami"],
             f"demo tools: {names}")
        lines = {line.split(" ")[0]: line for line in result["tools"]}
        need(lines["add"].startswith("add (read; args: a, b)"), lines["add"])
        need(lines["create_note"].startswith("create_note (write; args: title, text)"), lines["create_note"])
        need(lines["delete_notes"].startswith("delete_notes (write, denied by policy;"), lines["delete_notes"])

    def described(result):
        need(result["status"] == "ok" and result["kind"] == "read" and result["allowed"] is True, str(result))
        need(result["input_schema"]["required"] == ["a", "b"], str(result))

    def equals(expected):
        def check(result):
            need(result.get("status") == "ok" and result.get("result") == expected,
                 f"expected {expected!r}, got {result}")
        return check

    def error(kind=None):
        def check(result):
            need(result.get("status") == "error" and (kind is None or result.get("type") == kind),
                 f"expected an error{' of type ' + kind if kind else ''}, got {result}")
        return check

    def listed_hostile(result):
        need(result["status"] == "ok", f"list hostile: {result}")
        names = [line.split(" ")[0] for line in result["tools"][:-1]]
        need(names == ["bell_schema", "fine", "leak", "leak_json", "picture", "shout", "tells_all", "wide", "wipe"],
             f"hostile tools: {names}")
        need(result["tools"][-1].startswith("... 4 more tools are left out"), result["tools"][-1])
        need("IGNORE" not in json.dumps(result), "a tool name with a line break reached the LLM")
        told = next(line for line in result["tools"] if line.startswith("tells_all "))
        need(f"My token is {REDACTED}" in told, told)

    def listed_all(result):
        need(result["status"] == "ok", f"list all: {result}")
        need(sorted(result["servers"]) == sorted(["demo", "hostile", "remote"] + (["files"] if with_files else [])),
             f"servers: {sorted(result['servers'])}")
        need(len(result["servers"]["demo"]) == 8, f"demo tools: {result['servers']['demo']}")

    script = [
        ("mcp-list-tools", {"server": "demo"}, listed_demo),
        ("mcp-describe-tool", {"server": "demo", "tool": "add"}, described),
        ("mcp-call", {"server": "demo", "tool": "add", "arguments": json.dumps({"a": 2, "b": 40})}, equals("42")),
        ("mcp-call", {"server": "demo", "tool": "create_note",
                      "arguments": json.dumps({"title": "plan", "text": NOTE_TEXT})}, equals("saved 'plan'")),
        ("mcp-call", {"server": "demo", "tool": "get_note", "arguments": json.dumps({"title": "plan"})},
         equals(NOTE_TEXT)),
        ("mcp-call", {"server": "demo", "tool": "delete_notes", "arguments": "{}"}, error("denied")),
        ("mcp-call", {"server": "demo", "tool": "add", "arguments": "not json"}, error("invalid_arguments")),
        ("mcp-call", {"server": "demo", "tool": "whoami", "arguments": "{}"}, equals(REDACTED)),
        ("mcp-list-tools", {"server": "hostile"}, listed_hostile),
        ("mcp-call", {"server": "hostile", "tool": "leak", "arguments": json.dumps({"pad": 0})},
         equals(f"{REDACTED} tail")),
        ("mcp-call", {"server": "hostile", "tool": "say hello", "arguments": "{}"}, error("unknown_tool")),
        ("mcp-call", {"server": "hostile", "tool": "wipe", "arguments": "{}"}, error("denied")),
    ]
    # Over HTTP, to a server that answers 401 without the token: the header reached it, and the
    # token is taken out of what the server says back.
    script += [
        ("mcp-list-tools", {"server": "remote"}, lambda result: need(
            result["status"] == "ok" and [line.split(" ")[0] for line in result["tools"]]
            == ["add", "broken", "echo_authorization"], f"remote tools: {result}")),
        ("mcp-call", {"server": "remote", "tool": "add", "arguments": json.dumps({"a": 20, "b": 22})}, equals("42")),
        ("mcp-call", {"server": "remote", "tool": "echo_authorization", "arguments": "{}"},
         equals(f"Bearer {REDACTED}")),
        ("mcp-call", {"server": "remote", "tool": "broken", "arguments": "{}"}, error("server_error")),
    ]
    if text_protocol:
        # What only the text protocol lets through to a tool: arguments that the LLM left out
        # (the tools API refuses such a call itself), and arguments that it wrapped in quotes one
        # by one, which then arrive one by one.
        script += [
            ("mcp-list-tools", {}, listed_all),
            ("mcp-call", {"server": "demo", "tool": "list_notes"}, equals("plan")),
            ("mcp-call", {"line": '"demo" "add" ' + json.dumps(json.dumps({"a": 20, "b": 22}))}, equals("42")),
            ("mcp-call", {"line": '"demo" "list_notes"'}, equals("plan")),
            ("mcp-describe-tool", {"line": '"demo" "add"'}, described),
        ]
    if with_files:
        hello = str(work / "sandbox" / "hello.txt")

        def listed_files(result):
            need(result["status"] == "ok" and isinstance(result["tools"], list), f"list files: {result}")
            lines = {line.split(" ")[0]: line for line in result["tools"]}
            need("read_text_file" in lines and "write_file" in lines, f"filesystem tools: {sorted(lines)}")
            need("denied by policy" in lines["write_file"], lines["write_file"])

        script += [
            ("mcp-list-tools", {"server": "files"}, listed_files),
            ("mcp-call", {"server": "files", "tool": "read_text_file", "arguments": json.dumps({"path": hello})},
             lambda result: need(result.get("status") == "ok" and "Hello from a real MCP server" in result["result"],
                                 str(result))),
            ("mcp-call", {"server": "files", "tool": "write_file",
                          "arguments": json.dumps({"path": str(work / "sandbox" / "x.txt"), "content": "x"})},
             error("denied")),
        ]
    script += [
        ("mcp-call", {"server": "demo", "tool": "crash", "arguments": "{}"}, error()),        # the server dies
        ("mcp-list-tools", {"server": "nowhere"}, error("unknown_server")),                   # demo still down
        ("mcp-call", {"server": "demo", "tool": "add", "arguments": json.dumps({"a": 1, "b": 1})}, equals("2")),
        ("send", {"content": "done"}, None),
    ]
    return script


# The text protocol spells out these characters of the whole prompt as words.
SPELLED_OUT = (("_quote_", '"'), ("_newline_", "\n"), ("_apostrophe_", "'"))


def tool_result(request: dict, call_id: str) -> dict:
    """The plugin's answer to one tool call, as the next request carries it to the LLM."""
    if request["protocol"] == "text":
        return text_result(request["prompt"], call_id)
    for message in request["messages"]:
        if message.get("role") == "tool" and message.get("callid") == call_id:
            content = message["content"]
            marker = "RETURN: "
            need(marker in content, f"{call_id}: no tool return in {content[:300]!r}")
            returned = ast.literal_eval(content[content.index(marker) + len(marker):])
            need(isinstance(returned, list) and len(returned) == 1, f"{call_id}: {returned!r}")
            return json.loads(returned[0])
    raise Failed(f"{call_id}: the next request has no result for it")


def text_result(prompt: str, call_id: str) -> dict:
    """The same in the text protocol, where the next prompt has the answer after
    LAST_SKILL_USE_RESULTS as (RESULTS: ((COMMAND_RETURN: ((tool "what the LLM wrote") ANSWER)))).

    ANSWER is the plugin's JSON as it is; in what the LLM wrote, quotes are escaped, so the
    first unescaped {"status": is where the answer begins."""
    start = prompt.find(" LAST_SKILL_USE_RESULTS: ")
    end = prompt.find(" HISTORY: ", start)
    need(0 <= start < end, f"{call_id}: the next prompt has no results section")
    results = prompt[start:end].rstrip()
    opening = "{_quote_status_quote_: "
    need(opening in results and results.endswith("}))))"), f"{call_id}: no answer of the plugin in {results[:400]!r}")
    answer = results[results.index(opening):len(results) - 4]
    for word, character in SPELLED_OUT:
        answer = answer.replace(word, character)
    return json.loads(answer)


def check(out: Path, log: Path, script: list, with_files: bool) -> list[str]:
    requests = [json.loads(line) for line in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    need(len(requests) > len(script), f"{len(requests)} requests for {len(script)} steps: the run stopped early")
    done = []

    for request in requests[:len(script) + 1]:
        offered = [name for name in request["tools"] if name.startswith("mcp-")]
        need(offered == ["mcp-list-tools", "mcp-describe-tool", "mcp-call"], f"step {request['step']}: {offered}")
        need(request["mcp_note"].startswith("MCP servers are connected to you"),
             f"step {request['step']}: the plugin's section is missing from the system prompt")
    protocols = {request["protocol"] for request in requests}
    need(len(protocols) == 1, f"requests in more than one protocol: {sorted(protocols)}")
    spoken = "the tools API" if protocols == {"tools"} else "the text protocol"
    done.append(f"the three tools and the plugin's prompt section are in all {len(script) + 1} requests "
                f"({spoken})")

    for index, (tool, _arguments, verify) in enumerate(script):
        if verify is not None:
            try:
                verify(tool_result(requests[index + 1], f"call-{index}-0"))
            except Failed as exc:
                raise Failed(f"step {index} ({tool}): {exc}") from None
    done.append(f"{sum(1 for step in script if step[2])} tool results are as expected: list, describe, call, "
                "over stdio and over HTTP with a bearer token, arguments intact, writes refused, bad names "
                "hidden, the token scrubbed"
                + (", arguments left out or in quotes" if protocols == {"text"} else ""))

    # The demo server through the run, as each request's prompt had it: up, then down for a while
    # after the step that kills it, then up again. Omega builds an iteration's prompt before it
    # runs the heartbeat, so a change shows one request later than the heartbeat that saw it.
    crash = next(index for index, step in enumerate(script) if step[1].get("tool") == "crash")
    demo_up, demo_down = "- demo: Demo notes server - 8 tools", "- demo: Demo notes server - currently unavailable"
    states = ""
    for request in requests:
        note, step = request["mcp_note"], request["step"]
        need("- hostile: A server that misbehaves - 9 tools" in note, f"step {step}: {note[:700]!r}")
        need("- remote: A server over HTTP that wants a token - 3 tools" in note, f"step {step}: {note[:700]!r}")
        need((demo_up in note) != (demo_down in note), f"step {step}: {note[:700]!r}")
        states += "U" if demo_up in note else "D"
    down, back = states.find("D"), states.rfind("D") + 1
    need(set(states[:down]) == {"U"} and set(states[down:back]) == {"D"} and set(states[back:]) == {"U"},
         f"the demo server in the prompt, request by request: {states}")
    need(crash < down <= crash + 2, f"the server dies at step {crash}, the prompt says so at step {down}: {states}")
    text = log.read_text(encoding="utf-8", errors="replace")
    # Omega's log lines start with the date; the same words also appear where PeTTa echoes the plugin's source.
    rewrites = sum(1 for line in text.splitlines()
                   if line[:2] == "20" and "mcp-plugin" in line and "availability changed" in line)
    need(rewrites in (0, 2), f"the prompt was rewritten {rewrites} times, expected 2 (server down, server back)")
    done.append(f"the prompt says the demo server is down from request {down} (it dies at step {crash}) and back "
                f"from request {back}" + (", and was rewritten on those 2 occasions only" if rewrites else ""))

    everything = (out / "requests.jsonl").read_text(encoding="utf-8")
    for form in (TOKEN, json.dumps(TOKEN)[1:-1], TOKEN.replace("/", "\\/"), TOKEN[:10]):
        need(form not in everything, f"the token ({form!r}) is in a request to the LLM")
        need(form not in text, f"the token ({form!r}) is in Omega's log")
    done.append(f"the token is in none of the {len(requests)} requests and nowhere in Omega's log")

    sent = (out / "sent.txt").read_text(encoding="utf-8") if (out / "sent.txt").exists() else ""
    need("done" in sent.splitlines(), f"the agent's last message did not go out: {sent!r}")
    if with_files:
        done.append("the reference filesystem server was listed and read, and its write was refused")
    return done


BROKEN_SERVERS_FILE = ('servers:\n  demo:\n    transport: http\n    url: https://example.org/mcp\n'
                       '    headers: {Authorization: "Bearer typed-in-by-mistake-123456}\n')


def check_left_out(out: Path, log: Path, said: str, unsaid: str = "") -> None:
    """A run in which the plugin must not have started: the agent ran, without the plugin's
    tools and without its prompt section, and Omega's log has a line with `said` in it."""
    requests = [json.loads(line) for line in (out / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    for request in requests:
        offered = [name for name in request["tools"] if name.startswith("mcp-")]
        need(not offered and not request["mcp_note"], f"the agent was given {offered or 'the prompt section'}")
    text = log.read_text(encoding="utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line[:2] == "20"]  # Omega's own lines start with the date
    need(any(said in line for line in lines), f"Omega's log has no line with {said!r}")
    need(not any("mcpbridge |" in line and "connected" in line for line in lines), "a server was started all the same")
    if unsaid:
        need(unsaid not in text and unsaid not in json.dumps(requests), f"{unsaid!r} is in the log or in a request")


def stop_group(process: subprocess.Popen) -> None:
    """End Omega and what it started, and return when all of it is gone: run.sh ends at once,
    Omega itself a moment later, and nothing of it may be left writing when the next run starts."""
    def alive() -> bool:
        try:
            os.killpg(process.pid, 0)
            return True
        except ProcessLookupError:
            return False

    for sign in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sign)
        except ProcessLookupError:
            break
        deadline = time.time() + 15
        while alive() and time.time() < deadline:
            process.poll()  # so that run.sh, once ended, does not stay in the group as a zombie
            time.sleep(0.2)
        if not alive():
            break
    process.wait()


class Omega:
    """One PeTTa tree with Omega in it, started once per run."""

    def __init__(self, petta: Path, work: Path, embedding_provider: str, core_plugins_file: bool,
                 security_policy: str):
        self.petta, self.work, self.core_plugins_file = petta, work, core_plugins_file
        self.embedding_provider, self.security_policy = embedding_provider, security_policy
        self.omega = petta / "repos" / "Omega"

    def run(self, name: str, script: list, settings: list[str], timeout: int) -> tuple[bool, Path, Path]:
        """Start Omega with the plugin and the harness, wait until the script is used up, stop it.
        Returns whether the script ran through, the harness's folder and Omega's log."""
        out, log = self.work / f"out-{name}", self.work / f"omega-{name}.log"
        memory = self.work / f"memory-{name}"
        memory.mkdir()
        for source in (self.omega / "memory").glob("*.txt"):
            shutil.copy(source, memory / source.name)
        (memory / "history.metta").write_text("", encoding="utf-8")
        script_file = self.work / f"script-{name}.json"
        script_file.write_text(
            json.dumps([[{"name": tool, "arguments": arguments}] for tool, arguments, _ in script]), encoding="utf-8")

        command = ["sh", "run.sh", "run.metta", "provider=E2E", "commchannel=e2e",
                   f"embeddingprovider={self.embedding_provider}", f"securityPolicyPath={self.security_policy}",
                   f"memoryDirectory={memory}", *settings]
        if not self.core_plugins_file:
            command.append(f"customPluginsConfigPath={self.work / 'plugins-custom.yaml'}")
        environment = {**os.environ, "E2E_SCRIPT": str(script_file), "E2E_OUT": str(out),
                       "E2E_MCP_TOKEN": TOKEN, "CHROMA_DB_PATH": str(self.work / f"chroma_db-{name}"),
                       "OMEGA_MCP_LOG_DIR": str(self.work / "server-logs")}
        environment.setdefault("OPENAI_API_KEY", "not-a-key")

        # Omega's own list of plugins, for the run: with the two plugins added to it when that is
        # how they are registered, and in any case without an entry of the same name that is
        # already there, as it is where the plugin was installed by the README.
        core_list = self.omega / "config" / "plugins.yaml"
        original = core_list.read_text(encoding="utf-8")
        listed = yaml.safe_load(original) or []
        others = [entry for entry in listed
                  if not (isinstance(entry, dict) and entry.get("name") in ("mcpclient", "e2eharness"))]
        for_the_run = None
        if self.core_plugins_file:
            for_the_run = yaml.safe_dump(others, sort_keys=False) + "\n" + plugins_file()
        elif len(others) != len(listed):
            for_the_run = yaml.safe_dump(others, sort_keys=False)
        history = self.omega / "memory" / "history.metta"
        earlier_history = history.read_bytes() if history.exists() else None
        try:
            history.write_bytes(b"")
            if for_the_run is not None:
                core_list.write_text(for_the_run, encoding="utf-8")
            with open(log, "w", encoding="utf-8") as handle:
                process = subprocess.Popen(command, cwd=self.petta, env=environment, stdout=handle,
                                           stderr=subprocess.STDOUT, start_new_session=True)
                deadline = time.time() + timeout
                while time.time() < deadline and process.poll() is None and not (out / "done").exists():
                    time.sleep(1)
                finished = (out / "done").exists()
                time.sleep(2)  # the iteration after the last scripted step, so that its request is on record
                stop_group(process)
        finally:
            if for_the_run is not None:
                core_list.write_text(original, encoding="utf-8")
            if earlier_history is None:
                history.unlink(missing_ok=True)
            else:
                history.write_bytes(earlier_history)
        return finished, out, log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("petta", type=Path, help="PeTTa tree with repos/Omega")
    parser.add_argument("--timeout", type=int, default=240, help="seconds to wait for the script to run through")
    parser.add_argument("--embedding-provider", default="OpenAI",
                        help="Omega's embeddingprovider for the run (default OpenAI: nothing is embedded in this "
                             "run, and it avoids loading the local model)")
    parser.add_argument("--core-plugins-file", action="store_true",
                        help="register the plugins in repos/Omega/config/plugins.yaml for the run, for an Omega "
                             "without customPluginsConfigPath; the file is put back afterwards")
    parser.add_argument("--filesystem-server", action="store_true",
                        help="also use the reference filesystem MCP server (npx)")
    parser.add_argument("--filesystem-command", default="npx -y @modelcontextprotocol/server-filesystem",
                        help="the command that starts it; the folder it may use is added at the end "
                             "(default: %(default)s)")
    parser.add_argument("--security-policy", default="", metavar="FILE",
                        help="Omega's securityPolicyPath for the run: a filesystem policy that lets Omega read "
                             "its own tree, Python, this plugin and what the servers need, and write to the "
                             "temporary folder (default: none, no restrictions)")
    parser.add_argument("--keep", action="store_true", help="keep the run's folder")
    args = parser.parse_args()

    petta = args.petta.resolve()
    omega = petta / "repos" / "Omega"
    for needed in (petta / "run.sh", petta / "run.metta", omega / "src" / "plugin.py"):
        if not needed.exists():
            print(f"not a PeTTa tree set up for Omega: {needed} is missing", file=sys.stderr)
            return 2
    has_custom = "customPluginsConfigPath" in (omega / "src" / "plugin.py").read_text(encoding="utf-8")
    text_protocol = "class LLMResponse" not in (omega / "src" / "providers.py").read_text(encoding="utf-8")
    if not has_custom and not args.core_plugins_file:
        print("this Omega has no customPluginsConfigPath: run again with --core-plugins-file", file=sys.stderr)
        return 2

    work = Path(tempfile.mkdtemp(prefix="omega-mcp-e2e-"))
    (work / "sandbox").mkdir()
    (work / "sandbox" / "hello.txt").write_text("Hello from a real MCP server\n", encoding="utf-8")
    files_command = shlex.split(args.filesystem_command) if args.filesystem_server else None
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        remote_port = probe.getsockname()[1]
    (work / "mcp_servers.yaml").write_text(servers_file(work, files_command, remote_port), encoding="utf-8")
    (work / "broken.yaml").write_text(BROKEN_SERVERS_FILE, encoding="utf-8")
    (work / "plugins-custom.yaml").write_text(plugins_file(), encoding="utf-8")
    script = steps(work, args.filesystem_server, text_protocol)
    only_a_message = [("send", {"content": "done"}, None)]
    policy = str(Path(args.security_policy).resolve()) if args.security_policy else ""
    runner = Omega(petta, work, args.embedding_provider, args.core_plugins_file, policy)

    described = subprocess.run(["git", "-C", str(omega), "describe", "--tags", "--always"],
                               capture_output=True, text=True, check=False).stdout.strip()
    print(f"Omega: {omega} ({described or 'version unknown'})\nplugin: {PLUGIN}\nrun folder: {work}\n"
          f"filesystem policy: {policy or 'none'}\n"
          f"{len(script)} scripted steps, up to {args.timeout} s, then two short runs ...", flush=True)
    remote = subprocess.Popen([sys.executable, str(FIXTURES / "http_token_server.py"), str(remote_port), TOKEN],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        finished, out, log = runner.run("main", script, ["mcpEnabled=enabled",
                                                         f"mcpConfigPath={work / 'mcp_servers.yaml'}"], args.timeout)
        need(finished, f"the script did not run through in {args.timeout} s; see {log}")
        passed = check(out, log, script, args.filesystem_server)
        if policy:
            applied = [line for line in log.read_text(encoding="utf-8", errors="replace").splitlines()
                       if line[:2] == "20" and "Policy applied" in line]
            need(applied, "a policy file was given, but Omega's log does not say that it was applied")
            passed.append("all of it under Omega's filesystem policy")

        # Left as it is installed, the plugin is off: no tools, no servers started. Where this
        # Omega's own configuration switches it on, it is switched off for this run instead.
        configured = yaml.safe_load((omega / "config" / "config.yaml").read_text(encoding="utf-8")) or {}
        switch = ["mcpEnabled=disabled"] if "mcpEnabled" in configured else []
        finished, out, log = runner.run("off", only_a_message,
                                        [*switch, f"mcpConfigPath={work / 'mcp_servers.yaml'}"], 90)
        need(finished, f"with the plugin off, the agent did not run; see {log}")
        check_left_out(out, log, "MCP plugin is disabled")
        passed.append(("switched off" if switch else "not switched on, as it is when installed")
                      + ", the plugin starts no server and gives the agent nothing")

        # A servers file that cannot be read: the agent runs without the plugin, and the log says
        # where the file is wrong without quoting it.
        finished, out, log = runner.run("broken", only_a_message, ["mcpEnabled=enabled",
                                                                   f"mcpConfigPath={work / 'broken.yaml'}"], 90)
        need(finished, f"with a servers file that cannot be read, the agent did not run; see {log}")
        check_left_out(out, log, "MCP plugin is enabled but did not start: cannot read the MCP servers file",
                       unsaid="typed-in-by-mistake")
        passed.append("with a servers file that cannot be read the agent runs without the plugin, and the log "
                      "says where the file is wrong without quoting it")
    except Failed as exc:
        print(f"FAILED: {exc}\nrun folder kept: {work}", file=sys.stderr)
        return 1
    finally:
        remote.terminate()
    for line in passed:
        print(f"  ok  {line}")
    print("PASSED")
    if args.keep:
        print(f"run folder kept: {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
