; MCP client plugin for Omega.
;
; Connects the agent to Model Context Protocol servers listed in a YAML file
; (see mcp_servers.example.yaml) and gives it three tools:
;
;   mcp-list-tools    (server)                 what a server offers
;   mcp-describe-tool (server tool)            input schema of one tool
;   mcp-call          (server tool arguments)  run a tool, arguments as JSON
;
; Tools are discovered on demand instead of being registered one by one: a
; single server can expose dozens of tools and every registered tool goes
; into every LLM request. The connection, the policy and the formatting of
; results live in mcpbridge.py; this file only wires them into the agent.
;
; The plugin can live anywhere. Omega's MeTTa loader puts this folder on
; sys.path before it imports this file, so py-call finds mcpbridge.py next to
; it, and the default servers file is looked for in this folder too.
;
; Configuration (config/config.yaml, OMEGA_<key> or command line):
;   mcpEnabled     enabled | disabled (default disabled)
;   mcpConfigPath  path of the servers file
;                  (default: mcp_servers.yaml in this folder)

(= (mcpEnabled) (empty))
(= (mcpConfigPath) (empty))

(= (loadOmegaPlugin)
    (progn
        (configure mcpEnabled disabled)
        (configure mcpConfigPath (py-call (mcpbridge.default_config_path)))
        (if (== (mcpEnabled) enabled)
            ; init returns nothing when the bridge started, or one line that says
            ; why it did not. The agent gets the tools only in the first case.
            (let $problem (py-call (mcpbridge.init (mcpConfigPath)))
                (if (> (string_length $problem) 0)
                    (Error
                        (_loadOmegaPlugin)
                        (strings-concat ("MCP plugin is enabled but did not start: " $problem)))
                    (progn
                        (add-skill
                            mcp-list-tools
                            "List the tools of an MCP server with their arguments and whether the operator policy allows them. Pass a server name, or * for every server."
                            (server))
                        (add-skill
                            mcp-describe-tool
                            "Show the description and the JSON input schema of one MCP tool before calling it."
                            (server tool))
                        (add-skill
                            mcp-call
                            "Call a tool on an MCP server. arguments must be one JSON object string with the tool parameters, or {} when the tool takes none."
                            (server tool arguments))
                        (add-prompt-extension mcp_servers (py-call (mcpbridge.prompt_summary)))
                        (add-heartbeat-listener
                            mcpclient
                            (|-> ($iter) (mcp-refresh-prompt $iter))))))
            (log INFO "mcp-plugin" "MCP plugin is disabled"))))

; A server that was slow or down at startup can come up later, and one can go
; away. The bridge reports a new summary only when the picture changed, so on
; most iterations this is a single cheap call that returns an empty string.
; The test is on the length, as the core does with what py-call returns: an
; empty Python string does not compare equal to "" here, and a test with ==
; replaced the prompt section with nothing at the first heartbeat.
(= (mcp-refresh-prompt $iteration)
    (let $summary (py-call (mcpbridge.prompt_summary_if_changed))
        (if (> (string_length $summary) 0)
            (progn
                (log INFO "mcp-plugin" "MCP server availability changed, updating the prompt")
                (remove-prompt-extension mcp_servers)
                (add-prompt-extension mcp_servers $summary))
            ())))

; The tools. How their arguments arrive depends on the Omega:
;
; - with the tools API (Omega's main branch) each argument declared above
;   arrives on its own, and Omega refuses a call that leaves one out;
; - the v0.1.x releases hand a tool everything the LLM wrote after the tool's
;   name as one string: mcp-call demo add {"a": 1} arrives as one argument,
;   demo add {"a": 1}, and the bridge takes that string apart. Nothing arrives
;   when the LLM wrote the name alone, and when it wrapped each argument in
;   quotes they arrive one by one.
;
; Hence one definition for each number of arguments that can arrive.

(= (mcp-list-tools)
    (py-call (mcpbridge.list_tools "*")))

(= (mcp-list-tools $server)
    (py-call (mcpbridge.list_tools $server)))

(= (mcp-describe-tool $server_and_tool)
    (py-call (mcpbridge.describe_tool_line $server_and_tool)))

(= (mcp-describe-tool $server $tool)
    (py-call (mcpbridge.describe_tool $server $tool)))

(= (mcp-call $server_tool_and_arguments)
    (py-call (mcpbridge.call_tool_line $server_tool_and_arguments)))

(= (mcp-call $server $tool)
    (py-call (mcpbridge.call_tool $server $tool "{}")))

(= (mcp-call $server $tool $arguments)
    (py-call (mcpbridge.call_tool $server $tool $arguments)))
