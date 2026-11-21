# omega-mcp

An MCP client plugin for [Omega](https://github.com/singnet/Omega). It lets an
Omega agent use the tools of [Model Context Protocol](https://modelcontextprotocol.io)
servers that its operator lists in a file: servers on the same machine, started
over stdio, and remote ones over streamable HTTP.

The operator decides which servers exist and what the agent may do with each
of them. A tool that changes something is refused unless the operator allowed
it. The plugin is off until it is switched on.

It is for an Omega that runs on its own. For minds that live in an OmegaHive
there is a separate app, [omega-connect](https://github.com/N0g4D/omega-connect),
in which writes wait for the approval of a person.

## What the agent gets

Three tools, whatever the number of servers:

| Tool | Arguments | What it does |
|---|---|---|
| `mcp-list-tools` | `server`, or `*` for all | The tools of a server: name, arguments, read or write, allowed or not |
| `mcp-describe-tool` | `server tool` | The description and the JSON input schema of one tool |
| `mcp-call` | `server tool arguments` | Runs a tool. `arguments` is one JSON object, `{}` for none |

The tools of the servers are not registered one by one, because every
registered tool goes into every request to the LLM, and one server can have
dozens of them. The agent looks them up when it needs them.

A section of the prompt names the servers, says how many tools each has and
which are unavailable. It is rewritten when a server goes down or comes back.

Answers are JSON, on one line. These are from a run with the demo server of
this repository, written as the `v0.1.24` release has the calls; on `main`
they are tool calls with the same arguments:

```
mcp-list-tools demo
{"status": "ok", "server": "demo", "tools": [
  "add (read; args: a, b): Add two integers.",
  "create_note (write; args: title, text): Save a note. Changes state.",
  "delete_notes (write, denied by policy; args: no arguments): Delete every note. Changes state.",
  "get_note (read; args: title): Return the text of one note.", ...]}

mcp-call demo add {"a": 2, "b": 40}
{"status": "ok", "server": "demo", "tool": "add", "result": "42"}

mcp-call demo delete_notes {}
{"status": "error", "type": "denied", "message": "'delete_notes' can change data on 'demo' and the
operator's policy does not allow it. Do not retry; tell your owner what you wanted to do."}
```

## Requirements

- Omega, either the `main` branch or release `v0.1.24`. The two hand
  arguments to a tool in different ways (one by one on `main`, as one string
  in the release) and the plugin takes both. Releases before `v0.1.20` are
  not supported: their plugin loader, where there is one, looks for an entry
  point with another name.
- Python 3.10 or later, as Omega, with the packages in `requirements.txt`:
  `mcp` 1.30 or later and `pyyaml`.
- Whatever your servers need: Node for one started with `npx`, for instance.

## Install

Paths are from the PeTTa folder that Omega runs in, the one with `run.sh`.
Use the Python environment that Omega uses.

1. Get the plugin:

   ```sh
   git clone https://github.com/N0g4D/omega-mcp.git repos/Omega/plugins/mcpclient
   ```

2. Install its packages:

   ```sh
   python3 -m pip install -r repos/Omega/plugins/mcpclient/requirements.txt
   ```

   On `main`, `./repos/Omega/scripts/install_dependencies.sh` does this too:
   it installs the `requirements.txt` of every folder under `plugins/`.

3. Register it. On `main`, in a plugins file of your own, for instance
   `repos/Omega/config/plugins-custom.yaml`:

   ```yaml
   - name: mcpclient
     loader: metta
     location: "{REPO}/plugins/mcpclient"
   ```

   and name that file in `repos/Omega/config/config.yaml`, where the setting
   is already there with an empty value:

   ```yaml
   customPluginsConfigPath: "repos/Omega/config/plugins-custom.yaml"
   ```

   On `v0.1.24`, which has no such setting, add the same three lines at the
   end of `repos/Omega/config/plugins.yaml`.

4. Write the servers file, starting from the example:

   ```sh
   cp repos/Omega/plugins/mcpclient/mcp_servers.example.yaml repos/Omega/plugins/mcpclient/mcp_servers.yaml
   ```

   As it is, the copy has one server switched on: a demo server that comes
   with the plugin, enough for a first try. It is started with `python3`,
   which has to be the Python that has the `mcp` package, as it is when
   Omega's environment is active.

5. Switch the plugin on, with a new line in `repos/Omega/config/config.yaml`:

   ```yaml
   mcpEnabled: enabled
   ```

6. Start Omega as usual. Its log says which servers answered:

   ```
   INFO | mcpbridge | MCP plugin started: demo: connected, 6 tools (4 hidden by the rules)
   ```

   If the servers file is missing, cannot be read or lists no server, the log
   has an `ERROR` line that says so, and Omega runs without the plugin.

| Setting | Default | Meaning |
|---|---|---|
| `mcpEnabled` | `disabled` | `enabled` starts the plugin |
| `mcpConfigPath` | `mcp_servers.yaml` in the plugin's folder | The servers file |

Like every Omega setting, both can also be given on the command line
(`mcpEnabled=enabled`) or in the environment (`OMEGA_mcpEnabled=enabled`).

The plugin does not have to live under `repos/Omega/plugins`: `location` can
be any folder. Under `plugins/` it is found by Omega's dependency script and
it is inside the tree that Omega's filesystem policy makes read-only.

## The servers file

[`mcp_servers.example.yaml`](mcp_servers.example.yaml) has one example of
each kind, with comments. For each server:

| Setting | For | Meaning |
|---|---|---|
| `transport` | all | `stdio` (default) or `http` |
| `command`, `args` | stdio | The program to start and its arguments |
| `env` | stdio | Variables for that program, besides the few it gets anyway |
| `cwd` | stdio | Its working folder. A relative one starts from the servers file's folder |
| `url` | http | The address of the server |
| `headers` | http | Headers for every request, usually `Authorization` |
| `description` | all | One line that the LLM reads in its list of servers |
| `timeout` | all | Seconds for one call. Default 60 |
| `connect_timeout` | all | Seconds to wait for the server to start or answer. Default 30 |
| `tools` | all | The rules, below |
| `enabled` | all | `false` leaves the server out |

A setting that the plugin does not know stops it at start, with the name of
the setting in the log. A rule misspelt would otherwise be a rule not applied.

### Rules

A tool counts as read-only only when its server marks it with the MCP
annotation `readOnlyHint`. Every other tool counts as a write.

| Rule | Default | Meaning |
|---|---|---|
| `allow` | all | Name patterns of the tools the agent sees |
| `deny` | none | Tools it never sees, whatever `allow` says |
| `writes` | `deny` | `allow` lets every write run |
| `allow_write` | none | Writes that may run although `writes` is `deny` |

Patterns are names with `*`, such as `search_*`. A read tool that the agent
sees can always run. A write that is not allowed is still listed, marked
`denied by policy`, and a call to it is refused with a message that tells the
LLM not to retry.

`readOnlyHint` is the server's own word. For a server that you do not fully
trust, use `allow` to name the tools you have looked at. `allow` is also the
way to keep a server with hundreds of tools to the ones the agent needs.

### Tokens

`${NAME}` anywhere in a server's `command`, `args`, `env`, `url` or `headers`
is replaced with the environment variable `NAME` of the Omega process, so
that tokens stay out of the file. A server whose variable is not set is
listed as unavailable.

Every value that came in this way is taken out again, as `[redacted]`, of
everything the plugin gives to the agent or writes to the log: a server can
say its token back in a result or in a description, and an error can quote an
address or a header. Values shorter than 6 characters are not treated this
way. Put a token in a header and not in the address: the MCP SDK writes the
address to its own log at debug level.

### What a server is given

A stdio server is started with a short environment, `HOME`, `PATH` and a few
more, and with what its `env` adds. It does not inherit Omega's environment,
so it does not see Omega's API keys, and it does not see proxy settings
unless `env` names them. What it prints on stderr goes to
`omega-mcp/<server>.stderr.log` in the system's temporary folder, or in the
folder named by the environment variable `OMEGA_MCP_LOG_DIR`.

### When a server is slow or goes away

Omega's start waits for each server up to its `connect_timeout`. A server
that is not there by then is listed as unavailable and Omega starts anyway; a
slow one that arrives later is picked up, and the prompt then says so. A
server that failed is tried again each time the agent uses it.

When a server goes away during a call to a read tool, the call is made once
more on a new connection. A write is never repeated: the agent gets the
error, with a note that the call may still have taken effect.

## What it does and does not protect

Everything a server says is treated as data from outside. Tool names that
break the MCP naming rules hide their tool. Descriptions, schemas, results
and errors have control characters removed and are cut to a length: 300
characters for a description, 4,000 for a schema (a longer one is reduced to
argument names and types), 8,000 for a result. The list of one server's tools
stays within 8,000 characters too: when a server has many tools, the list
gives up descriptions first, then arguments, and `mcp-describe-tool` still
shows any one tool in full. Images and other content that is not text are
named and left out. The prompt tells the LLM to treat all of it as data and
never as instructions. None of this makes a hostile text harmless to an LLM;
it limits how much of it arrives and in what shape.

The rules bind the three tools, and none of the three can add a server or
change a rule. They are not a wall around the agent. The plugin runs inside
the Omega process, and Omega gives its agent a `shell` tool and a `metta`
tool that reach what the process reaches:

- With `shell`, the agent can read the environment of the Omega process,
  tokens included. In a test run, `printenv` put the token into the next
  request to the LLM.
- With `metta`, it can call Python functions in the process, those of this
  plugin among them.
- Without a filesystem policy it can also edit the servers file. Under
  Omega's policy the file is safe from that as long as it is in a place the
  policy makes read-only, as Omega's own tree is.

So the limit that holds is the one set outside Omega: give each server a
token that can do no more than you are willing to let the agent do. The rules
of this plugin are a second line, against a careless call and against a tool
description or a result that tries to talk the LLM into one.

There is no approval by a person before a call, and tool definitions are not
pinned: a server can offer other tools or other descriptions the next time it
connects.

## Limits

- Tools only. MCP resources, prompts, sampling and elicitation are not used.
- Static tokens only. Servers that need the OAuth sign-in of the MCP
  authorization specification cannot be used.
- Transports: stdio and streamable HTTP. The older HTTP+SSE transport is not
  supported.
- The list of a server's tools is read when the plugin connects to it, and
  again at each reconnection, not when the server announces a change.
- A change in the list of servers shows in the prompt one loop iteration
  after it happened.
- One call at a time per server.

## Docker

Not tried. From Omega's documentation and its entrypoint, it takes:

- the plugin and the plugins file inside the container, at container paths
  that the active policy lets Omega read (the default one allows `/PeTTa`);
- the `mcp` package inside the image. A plugin mounted into a built image
  gets no packages installed;
- the settings after the image name (`mcpEnabled=enabled`). The entrypoint
  starts the agent with an emptied environment, so `OMEGA_mcpEnabled` does not
  arrive, and neither do the variables that `${NAME}` would read: a server
  that takes its token from the environment is unavailable there.

## Tests

```sh
python3 -m pip install -r requirements.txt pytest
python3 -m pytest -q tests
```

97 tests of the bridge against three small servers in `tests/fixtures`: a
well-behaved one over stdio and HTTP, an HTTP one that wants a bearer token,
and a hostile one that sends what a server should not.

`tests/e2e/run_e2e.py` runs the real Omega loop with the plugin, from a PeTTa
tree set up as Omega's README describes. No LLM key is needed: a test-only
plugin plays the LLM from a script, in the way the Omega at hand talks to
one. It checks what the LLM was sent, request by request.

```sh
python3 tests/e2e/run_e2e.py /path/to/PeTTa                       # main
python3 tests/e2e/run_e2e.py /path/to/PeTTa --core-plugins-file   # v0.1.24
```

`--filesystem-server` adds the reference filesystem server, and
`--security-policy FILE` runs under an Omega filesystem policy. The top of
the script says the rest.

What was run for this version:

| | |
|---|---|
| Unit tests | On Linux: Python 3.10 and 3.13, each with MCP SDK 1.30.0 and 2.3.0. On macOS 27.2.1: Python 3.12.16 with MCP SDK 2.3.0 |
| In Omega | On Linux: `main` at `d35bfbf` (20 November 2026) and `v0.1.24`, on PeTTa v1.0.4, SWI-Prolog 10.1.20, Python 3.13 and MCP SDK 2.3.0, with and without a filesystem policy |
| Servers | The three of `tests/fixtures`, and in the Omega runs `@modelcontextprotocol/server-filesystem` 1.0.0 (Node 22) and 2.4.0 (Node 24) |

Not run: Docker, Windows, Omega itself on macOS (only the unit tests ran
there), a hosted MCP server of any vendor, and a real LLM. The model in these
runs is a script, so how well a given model writes the calls is not measured
here.

## License

Apache-2.0. See [LICENSE](LICENSE).
