"""Test-only Omega plugin: a scripted LLM provider and a scripted channel.

Lets the real Omega loop run end to end without an LLM API key, on both of
the ways Omega has of talking to an LLM:

- the tools API (Omega's main branch): a request carries the tools and the
  messages, and the answer is a list of tool calls;
- the text protocol (the v0.1.x releases): a request is one prompt, and the
  answer is text, one line per tool: `tool-name everything else`.

Provider "E2E" answers each request with the next step of a JSON script (a
list of steps, each a list of {"name": ..., "arguments": {...}} calls), in the
form this Omega expects. It writes every request it receives to a JSONL
transcript and creates the file `done` once the script is used up.

Channel "e2e" delivers one user message, then nothing, and records what the
agent sends.

Environment: E2E_SCRIPT (script path), E2E_OUT (output directory),
E2E_MESSAGE (the user message, optional).

run_e2e.py in this folder starts Omega with this plugin and checks the
transcript.
"""

import json
import os
import re
import threading

import channels
import providers

TOOLS_API = hasattr(providers, "LLMResponse")
NOTE_START = "MCP servers are connected to you"
# A skill as the text protocol lists it: "- what it does: its-name its arguments".
SKILL_LINE = re.compile(r": (mcp-[a-z-]+)(?= |'|_apostrophe_|$)")

_lock = threading.Lock()
_step = 0


def _out(name: str) -> str:
    directory = os.environ.get("E2E_OUT", "/tmp/omega-e2e")
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def _message_dict(message) -> dict:
    data = {}
    for key in ("role", "content", "tool_call_id", "callid", "name"):
        value = getattr(message, key, None)
        if value is None and isinstance(message, dict):
            value = message.get(key)
        if value is not None:
            data[key] = value if isinstance(value, (str, int, float)) else str(value)
    if not data:
        data["raw"] = str(message)
    return data


def _note(text: str) -> str:
    """The plugin's section of the prompt, as this request had it."""
    start = text.find(NOTE_START)
    return text[start:start + 1500] if start >= 0 else ""


class ScriptedProvider(providers.LLMProvider):

    def start(self) -> None:
        with open(os.environ["E2E_SCRIPT"], encoding="utf-8") as handle:
            self.script = json.load(handle)

    def stop(self) -> None:
        pass

    def chat(self, request, *_args, **_kwargs):
        global _step
        with _lock:
            step = _step
            _step += 1
        calls = self.script[step] if step < len(self.script) else None
        if calls is None:
            with open(_out("done"), "w", encoding="utf-8") as handle:
                handle.write("the script is used up\n")
        return self._tool_calls(step, request, calls) if TOOLS_API else self._text(step, request, calls)

    def _record(self, **fields) -> None:
        with open(_out("requests.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps(fields, ensure_ascii=False) + "\n")

    def _tool_calls(self, step: int, request, calls):
        messages = [_message_dict(m) for m in request.messages]
        system = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "system")
        self._record(step=step, protocol="tools", tools=[t.name for t in request.tools],
                     mcp_note=_note(system), messages=messages)
        response = providers.LLMResponse()
        for index, call in enumerate(calls if calls is not None else [{"name": "nop", "arguments": {}}]):
            response.add_tool_call(
                providers.LLMToolCall()
                .with_id(f"call-{step}-{index}")
                .with_name(call["name"])
                .with_arguments({k: str(v) for k, v in call.get("arguments", {}).items()}))
        return response

    def _text(self, step: int, prompt, calls) -> str:
        prompt = str(prompt)
        skills = prompt[:prompt.find(NOTE_START)] if NOTE_START in prompt else prompt
        self._record(step=step, protocol="text", tools=SKILL_LINE.findall(skills),
                     mcp_note=_note(prompt), prompt=prompt)
        # One line per call: the name, then the arguments as an LLM writes them, one after
        # the other with nothing around them.
        return "\n".join(" ".join([call["name"], *(str(v) for v in call.get("arguments", {}).values())])
                         for call in calls or [])


class ScriptedChannel(channels.CommChannel):

    def __init__(self):
        super().__init__()
        self._delivered = False

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def receive(self) -> str:
        if self._delivered:
            return ""
        self._delivered = True
        return os.environ.get("E2E_MESSAGE", "What can you do with the MCP servers?")

    def send(self, message: str) -> None:
        with open(_out("sent.txt"), "a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")


def loadOmegaPlugin():
    providers.registerLLMProvider("E2E", ScriptedProvider())
    channels.registerCommChannel("e2e", ScriptedChannel())
