# smol\_RLM\_test

A hands-on exploration of **Recursive Language Models (RLMs)** — the inference
paradigm introduced in
[Zhang et al., 2025](https://arxiv.org/abs/2512.24601) — re-implemented using
the [smolagents](https://github.com/huggingface/smolagents) library and a local
[llama.cpp](https://github.com/ggerganov/llama.cpp) inference server.

The whole environment runs inside a **Docker container** and exposes a
**Jupyter notebook server** so you can interactively inspect and experiment with
the recursive calling process.

---

## What is an RLM?

Traditional LLM call:

```python
response = llm.completion(prompt)   # context window is a hard limit
```

RLM call:

```python
response = rlm.completion(prompt)   # the model can split and call itself
```

Inside an RLM completion the model runs in a Python REPL.  It may decompose its
input, write helper code, and call `rlm_call(sub_prompt)` to launch child LM
instances.  This allows handling tasks that exceed the context window via
divide-and-conquer.

The implementation here uses **smolagents' `CodeAgent`** as the REPL environment
and an **OpenAI-compatible API** (llama.cpp) for inference.

```
 RLMAgent.completion(prompt)
       │
  CodeAgent REPL
       │   writes Python code, may call …
       └── rlm_call(sub_prompt_1) ──► child RLMAgent (depth+1)
       └── rlm_call(sub_prompt_2) ──► child RLMAgent (depth+1)
       │                                     │
       │                              … until max_depth
       │                              (falls back to plain LLM)
       └── aggregates results ──► final answer
```

---

## Requirements

| Component | Version |
|---|---|
| Docker | ≥ 24 |
| Docker Compose | ≥ 2 |
| llama.cpp server | any recent build |

The host machine **does not** need Python installed — everything runs inside the
container.

---

## Quick Start

### 1. Start the llama.cpp server on the host

```bash
./llama-server \
    --model /path/to/model.gguf \
    --port 8080 \
    --host 0.0.0.0   # must listen on all interfaces so Docker can reach it
```

### 2. (Optional) Create a `.env` file

```dotenv
LLAMA_BASE_URL=http://host-gateway:8080/v1   # default — usually works on Linux
LLAMA_MODEL=local-model                       # model ID returned by /v1/models
OPENAI_API_KEY=not-needed                     # llama.cpp accepts any string
```

> **macOS / Windows Docker Desktop**: replace `host-gateway` with
> `host.docker.internal` in `LLAMA_BASE_URL`.

### 3. Build and start the Jupyter server

```bash
docker compose up --build
```

### 4. Open the notebooks

Navigate to **http://localhost:8888** in your browser.

| Notebook | Description |
|---|---|
| `01_rlm_basics.ipynb` | Core concepts, architecture, simple examples |
| `02_rlm_experiments.ipynb` | Needle-in-a-Haystack, hierarchical summarisation, call-tree logging |

---

## Project Structure

```
.
├── Dockerfile              # Python 3.12 + Jupyter + smolagents
├── docker-compose.yml      # container orchestration
├── requirements.txt        # Python dependencies
├── src/
│   └── rlm_smolagent.py    # RLMAgent — core implementation
└── notebooks/
    ├── 01_rlm_basics.ipynb
    └── 02_rlm_experiments.ipynb
```

---

## Using RLMAgent directly

```python
from src.rlm_smolagent import RLMAgent

agent = RLMAgent(
    base_url="http://localhost:8080/v1",   # llama.cpp OpenAI API
    model_name="local-model",
    max_depth=3,    # maximum recursion levels
    max_steps=10,   # REPL cycles per level
    verbose=True,
)

result = agent.completion("Summarise the following 10 000-word article: ...")
print(result.response)
print(result.metadata)   # recursive call tree (JSON-serialisable)
```

### Inspecting the prompts sent to the LLM server

If you want to inspect the exact message payloads sent by smolagents at each
recursive level, enable prompt tracing:

```python
from src.rlm_smolagent import RLMAgent

agent = RLMAgent(
    base_url="http://localhost:8080/v1",
    model_name="local-model",
    capture_prompt_traces=True,
)

result = agent.completion("Break this task into subproblems and solve it.")

# Flat list of every outbound request across the full recursion tree.
for request in result.iter_llm_requests():
    print(request["depth"], request["phase"])
    for message in request["messages"]:
        print(message["role"], message.get("content", ""))

# Pretty-printed text report for notebook exploration.
print(result.format_prompt_trace())
```

Each node in `result.metadata["call_tree"]` now includes an `llm_requests`
array. For agent-driven nodes this captures each step that smolagents sends to
the OpenAI-compatible server; for depth-limit fallback nodes it captures the
single plain completion request.

For a deeper walk-through of the internal flow, see
[docs/rlm_agent_flow.md](docs/rlm_agent_flow.md).

---

## References

- [Recursive Language Models — arXiv](https://arxiv.org/abs/2512.24601)
- [RLM official codebase](https://github.com/alexzhang13/rlm)
- [RLM minimal implementation](https://github.com/alexzhang13/rlm-minimal)
- [smolagents documentation](https://huggingface.co/docs/smolagents)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
