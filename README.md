# smol\_RLM\_test

A hands-on exploration of **Recursive Language Models (RLMs)** — the inference
paradigm introduced in
[Zhang et al., 2025](https://arxiv.org/abs/2512.24601) — re-implemented using
the [smolagents](https://github.com/huggingface/smolagents) library and a local
[llama.cpp](https://github.com/ggerganov/llama.cpp) inference server.

The whole environment runs inside a **Docker container** and exposes a
**Jupyter notebook server** so you can interactively inspect and experiment with
the recursive calling process. The repo also includes a standalone HTML slide
deck for presenting the ideas before switching into the live notebook demo.

---

## What is an RLM?

Traditional LLM call:

```python
response = llm.completion(prompt)   # context window is a hard limit
```

RLM call:

```python
response = rlm.completion(task, context)   # context stays in the Python env
```

### Key design principle: context lives in the REPL, not the prompt

The critical insight from the paper is **where the context lives**.

| Approach | Where is the context? |
|---|---|
| Naive | Embedded as a string inside the prompt (hits the context-window limit) |
| **RLM** | Stored as `rlm_context` Python variable inside the REPL environment |

Because the context is a Python variable, the model can inspect and slice it
programmatically and pass those slices to child calls — without re-embedding the
full content in each prompt string.

```python
# The model writes this inside the REPL:
mid   = len(rlm_context) // 2
left  = rlm_call("Summarise first half",  rlm_context[:mid])
right = rlm_call("Summarise second half", rlm_context[mid:])
final_answer(left + " " + right)

# Never do this — it re-embeds the context and defeats the purpose:
# rlm_call(f"Summarise: {rlm_context}")
```

```
 RLMAgent.completion(task, context)
       │
       ├── rlm_context = context  ← injected into REPL state (not the prompt)
       │
  CodeAgent REPL  ←── model sees: task description + rlm_context variable
       │
       ├── rlm_call("sub-task", rlm_context[:mid]) ──► child RLMAgent (depth+1)
       └── rlm_call("sub-task", rlm_context[mid:]) ──► child RLMAgent (depth+1)
                                                              │
                                                     … until max_depth
                                                     (plain LLM call with small slice)
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
| `02_rlm_experiments.ipynb` | Session-ready demo flow: letter counting, prompt tracing, Needle-in-a-Haystack, hierarchical summarisation |
| `03_rlm_long_context_qa.ipynb` | **Showcase**: hierarchical summarization + natural-language comprehension Q&A over a multi-section corporate report — demonstrates recursive sub-agent decomposition, call-tree inspection, accuracy verification, and interactive HTML visualization |

### 5. Open the presentation slides

Open `docs/rlm_session_slides.html` directly in a browser for a lightweight
slide deck that mirrors the notebook walkthrough.

Suggested flow for a live session:

1. Start with the HTML slides to explain the motivation.
2. Move to `02_rlm_experiments.ipynb` for the letter-counting demo.
3. Continue in the notebook for the recursive examples and prompt tracing.
4. Run `03_rlm_long_context_qa.ipynb` for the recursive-power showcase (summarization + Q&A).
5. Open the generated HTML trace files in `logs/` to explore interactively.

---

## Project Structure

```
.
├── Dockerfile              # Python 3.12 + Jupyter + smolagents
├── docker-compose.yml      # container orchestration
├── requirements.txt        # Python dependencies
├── docs/
│   ├── rlm_agent_flow.md   # internal flow and prompt-tracing notes
│   └── rlm_session_slides.html
├── src/
│   ├── rlm_smolagent.py    # RLMAgent — core implementation
│   └── rlm_visualizer.py   # self-contained HTML trace visualizer
├── logs/                    # generated HTML/JSON trace files
└── notebooks/
    ├── 01_rlm_basics.ipynb
    ├── 02_rlm_experiments.ipynb
    └── 03_rlm_long_context_qa.ipynb
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

# task    — short description of what to do (no raw context content)
# context — the long input data; stored as `rlm_context` Python variable
#           in the REPL, NOT embedded in the prompt string
result = agent.completion(
    task="Summarise the article",
    context=very_long_article_text,
)
print(result.response)
print(result.metadata)   # recursive call tree (JSON-serialisable)
```

Inside the REPL the model writes code like:

```python
mid   = len(rlm_context) // 2
left  = rlm_call("Summarise first half",  rlm_context[:mid])
right = rlm_call("Summarise second half", rlm_context[mid:])
final_answer(left + " " + right)
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

### Interactive HTML visualizer

Every RLM run can be saved as a self-contained HTML file that provides an
interactive tree view of the recursive call structure, agent code steps, and
full LLM request payloads:

```python
from rlm_visualizer import save_html, save_json, load_json

# Save an interactive HTML visualization (open in any browser)
save_html(result, "trace.html")

# Or use the convenience methods on RLMCompletion directly
result.save_html("trace.html")
result.save_json("trace.json")

# Reload a saved JSON trace and re-visualize without re-running the agent
data = load_json("trace.json")
save_html(data, "trace_reloaded.html")
```

Each node in the visualization shows:
- **Task** and **response** summaries
- **Agent steps** — the Python code the model wrote and its observations
- **LLM requests** — the exact message payloads sent to the server
- **Timing** and **context size** metadata

See `logs/` for pre-generated examples.

For a deeper walk-through of the internal flow, see
[docs/rlm_agent_flow.md](docs/rlm_agent_flow.md).

For a presentation-friendly walkthrough that starts with the letter-counting
example and then transitions into RLM internals, open
[docs/rlm_session_slides.html](docs/rlm_session_slides.html).

---

## References

- [Recursive Language Models — arXiv](https://arxiv.org/abs/2512.24601)
- [RLM official codebase](https://github.com/alexzhang13/rlm)
- [RLM minimal implementation](https://github.com/alexzhang13/rlm-minimal)
- [smolagents documentation](https://huggingface.co/docs/smolagents)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
