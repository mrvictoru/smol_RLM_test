# RLMAgent Internal Flow

This document describes how `RLMAgent` turns a single prompt into a recursive
tree of LLM calls, and where prompt tracing is captured.

## High-level flow

```mermaid
flowchart TD
    A["User calls completion(prompt)"] --> B["Create root _CallNode at depth 0"]
    B --> C["Run _run(prompt, depth=0, node=root)"]
    C --> D["Build CodeAgent for this node"]
    D --> E["Expose rlm_call as a tool"]
    E --> F["Send agent step prompt to the OpenAI-compatible server"]
    F --> G{"Answer directly?"}
    G -->|Yes| H["Return node-level answer"]
    G -->|No, call rlm_call| I["Create child _CallNode at depth + 1"]
    I --> J{"Reached max_depth?"}
    J -->|No| K["Run child _run(sub_prompt, depth+1, child_node)"]
    J -->|Yes| L["Use _plain_completion(sub_prompt, child_node)"]
    K --> M["Return child response to the parent REPL"]
    L --> M
    M --> N["Parent aggregates child results"]
    N --> H
    H --> O["Return RLMCompletion"]
    O --> P["metadata.call_tree stores the recursive structure"]
    O --> Q["metadata.call_tree.llm_requests stores traced prompts"]
```

## Prompt trace capture points

There are two places where prompts leave the application and are sent to the
LLM server:

1. Agent steps via `CodeAgent`
2. Depth-limit fallback via `_plain_completion`

The implementation now records both.

### 1. Agent step tracing

`_build_model(node, phase="agent_step")` creates a tracing model wrapper around
`OpenAIServerModel`.

That wrapper intercepts each call to `generate()` and `generate_stream()` after
smolagents has prepared the final request payload. The captured trace includes:

- recursion depth
- step number within the current node
- phase (`agent_step`)
- the exact `messages` payload sent to the OpenAI-compatible API
- stop sequences and response format, when present
- available tool names
- the full request payload dictionary used for the API call

This means you can inspect not only the task prompt, but the full message list
as smolagents actually submitted it.

### 2. Plain completion tracing

When recursion reaches `max_depth`, `rlm_call()` switches to
`_plain_completion(sub_prompt, node=child_node)`.

Before sending the request through the OpenAI SDK, the method records a
`plain_completion` trace entry on that child node. This gives full visibility
into the terminal leaf calls too.

## Data model

Each `_CallNode` now stores:

- `prompt`: original task for that node
- `depth`: recursion depth
- `response`: node-level final answer
- `children`: recursive subcalls
- `llm_requests`: every outbound request generated while solving that node

Each `llm_requests` entry includes:

- `phase`
- `depth`
- `model_name`
- `node_step`
- `messages`
- `stop_sequences`
- `response_format`
- `tool_names`
- `request_payload`

## How to inspect traces after a run

Enable tracing:

```python
agent = RLMAgent(
    base_url="http://localhost:8080/v1",
    model_name="local-model",
    capture_prompt_traces=True,
)
result = agent.completion("Solve this by decomposition.")
```

Then inspect them in one of two ways.

### Flat view

```python
for request in result.iter_llm_requests():
    print(request["depth"], request["phase"], request["node_step"])
    for message in request["messages"]:
        print(message["role"], message.get("content", ""))
```

### Formatted notebook-friendly view

```python
print(result.format_prompt_trace())
```

## Practical interpretation

When you read the trace output, keep this distinction in mind:

- `_CallNode.prompt` is the logical task given to that recursive node
- `llm_requests[*].messages` is the actual payload sent to the LLM server

Those are related, but not identical. The second view is the one you want when
you need to debug exact prompt construction or inspect what each subagent step
actually saw.