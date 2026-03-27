"""
rlm_smolagent.py — Recursive Language Model implemented with smolagents
=======================================================================

This module recreates the fundamental functionality of a Recursive Language
Model (RLM) — originally described in https://github.com/alexzhang13/rlm —
using the smolagents library and an OpenAI-compatible inference backend (e.g.
a local llama.cpp server).

Core idea
---------
A standard LLM call is:
    response = llm.completion(prompt)

An RLM call replaces that with:
    response = rlm.completion(task, context)

Key architectural principle (from the paper)
--------------------------------------------
The context (the potentially very long input) is **never** embedded as a string
literal inside the prompt.  Instead it lives as a Python variable ``rlm_context``
inside the REPL execution environment.  The model interacts with it
programmatically — slicing, searching, splitting — and decides how to decompose
the task itself.

Two LLM sub-call tools are available inside the REPL (mirroring the official
paper's ``llm_query`` / ``rlm_query`` distinction):

    ``llm_call(sub_task, context_slice)``
        Direct, non-recursive LLM call.  Fast and lightweight — use for
        leaf-level queries on chunks that are already small enough to answer
        directly (mirrors ``llm_query`` in the reference implementation).

    ``rlm_call(sub_task, context_slice)``
        Recursive RLM sub-call.  The child agent gets its own REPL and can
        decompose the slice further (mirrors ``rlm_query``).  Falls back to a
        plain completion when ``max_depth`` is reached.

The model orchestrates freely in Python — it chooses HOW to split, what
strategy to use, and which tool to call:

    # summarise paragraph-by-paragraph with direct LLM calls
    paragraphs = [p for p in rlm_context.split("\\n\\n") if p.strip()]
    summaries = [llm_call(f"Summarise paragraph {i+1}", p)
                 for i, p in enumerate(paragraphs)]
    final_answer("\\n".join(summaries))

    # recursive binary split for very large contexts
    mid   = len(rlm_context) // 2
    left  = rlm_call("Analyse first half",  rlm_context[:mid])
    right = rlm_call("Analyse second half", rlm_context[mid:])
    final_answer(left + " " + right)

    # WRONG — embeds full context in a prompt string (always avoid this)
    result = rlm_call(f"Summarise: {rlm_context}")

smolagents provides the REPL via its CodeAgent.  ``rlm_context`` is injected
into the executor's state before the agent runs so it is available as a Python
variable without appearing in the LLM-visible prompt text.

Usage
-----
    from rlm_smolagent import RLMAgent

    agent = RLMAgent(
        base_url="http://localhost:8080/v1",  # llama.cpp OpenAI API
        model_name="local-model",
        max_depth=3,
        verbose=True,
    )
    # task  — short description of what to do
    # context — the raw input data; stored as `rlm_context` in the Python REPL
    result = agent.completion(
        task="Summarise the article",
        context=very_long_article_text,
    )
    print(result.response)
    print(result.metadata)   # recursive call tree
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

import smolagents.models as smol_models
from smolagents import CodeAgent, tool
from smolagents.models import OpenAIServerModel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

def _sanitize_for_trace(value: Any) -> Any:
    """Convert trace payloads into JSON-serialisable primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Mapping):
        return {str(key): _sanitize_for_trace(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_trace(item) for item in value]

    if hasattr(value, "model_dump"):
        return _sanitize_for_trace(value.model_dump())

    role = getattr(value, "role", None)
    content = getattr(value, "content", None)
    tool_calls = getattr(value, "tool_calls", None)
    if role is not None or content is not None or tool_calls is not None:
        message: dict[str, Any] = {}
        if role is not None:
            message["role"] = role
        if content is not None:
            message["content"] = _sanitize_for_trace(content)
        if tool_calls is not None:
            message["tool_calls"] = _sanitize_for_trace(tool_calls)
        return message

    return str(value)


def _flatten_llm_requests(call_tree: Mapping[str, Any]) -> list[dict[str, Any]]:
    requests = list(call_tree.get("llm_requests", []))
    for child in call_tree.get("children", []):
        if isinstance(child, Mapping):
            requests.extend(_flatten_llm_requests(child))
    return requests

@dataclass
class RLMCompletion:
    """Result returned by :meth:`RLMAgent.completion`."""
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def iter_llm_requests(self) -> list[dict[str, Any]]:
        """Return every captured LLM request across the recursive call tree."""
        call_tree = self.metadata.get("call_tree")
        if not isinstance(call_tree, Mapping):
            return []
        return _flatten_llm_requests(call_tree)

    def format_prompt_trace(self) -> str:
        """Render captured prompts in a human-readable text report."""
        requests = self.iter_llm_requests()
        if not requests:
            return "No LLM request traces were captured. Enable capture_prompt_traces to inspect prompts."

        blocks: list[str] = []
        for request in requests:
            header = (
                f"depth={request.get('depth', '?')} "
                f"step={request.get('node_step', '?')} "
                f"phase={request.get('phase', 'unknown')}"
            )
            lines = [header]
            for message in request.get("messages", []):
                role = message.get("role", "unknown")
                content = message.get("content", "")
                lines.append(f"[{role}] {content}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    # ------------------------------------------------------------------
    # Visualizer convenience methods
    # ------------------------------------------------------------------

    def save_html(self, path: str) -> "Path":
        """
        Generate a self-contained HTML visualizer for this trace.

        The resulting file can be opened directly in any browser — no
        server or extra dependencies required.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"trace.html"``).

        Returns
        -------
        pathlib.Path
            Resolved path of the generated file.
        """
        from rlm_visualizer import save_html as _save_html
        return _save_html(self, path)

    def save_json(self, path: str) -> "Path":
        """
        Persist the full completion payload (response + metadata) as JSON.

        The JSON file can later be reloaded with
        ``rlm_visualizer.load_json()`` and fed back into ``save_html()``
        to regenerate the visualizer without re-running the agent.

        Parameters
        ----------
        path:
            Destination file path (e.g. ``"trace.json"``).

        Returns
        -------
        pathlib.Path
            Resolved path of the generated file.
        """
        from rlm_visualizer import save_json as _save_json
        return _save_json(self, path)


@dataclass
class _LLMRequestTrace:
    """Captured payload for one outbound LLM request."""
    phase: str
    depth: int
    model_name: str
    node_step: int
    messages: list[dict[str, Any]]
    stop_sequences: list[str] | None = None
    response_format: dict[str, Any] | None = None
    tool_names: list[str] = field(default_factory=list)
    request_payload: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "depth": self.depth,
            "model_name": self.model_name,
            "node_step": self.node_step,
            "timestamp": round(self.timestamp, 3),
            "messages": self.messages,
            "stop_sequences": self.stop_sequences,
            "response_format": self.response_format,
            "tool_names": self.tool_names,
            "request_payload": self.request_payload,
        }


@dataclass
class _CallNode:
    """A node in the recursive call tree (used for metadata / logging)."""
    task: str           # short task description (never contains raw context)
    depth: int
    context_size: int = 0   # byte-length of the context slice at this level
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    response: str = ""
    llm_requests: list[_LLMRequestTrace] = field(default_factory=list)
    agent_steps: list[dict[str, Any]] = field(default_factory=list)
    children: list["_CallNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_preview": self.task[:120] + ("…" if len(self.task) > 120 else ""),
            "task": self.task,
            "depth": self.depth,
            "context_size": self.context_size,
            "duration_s": round((self.end_time or time.time()) - self.start_time, 3),
            "response_preview": self.response[:120] + ("…" if len(self.response) > 120 else ""),
            "response": self.response,
            "llm_requests": [request.to_dict() for request in self.llm_requests],
            "agent_steps": self.agent_steps,
            "children": [c.to_dict() for c in self.children],
        }


class _TracingOpenAIServerModel(OpenAIServerModel):
    """OpenAI-compatible model wrapper that records each outbound request."""

    def __init__(self, *args, trace_callback, trace_phase: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._trace_callback = trace_callback
        self._trace_phase = trace_phase

    def _emit_trace(
        self,
        *,
        messages: list[smol_models.ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None,
        response_format: dict[str, str] | None,
        tools_to_call_from: list[Any] | None,
        completion_kwargs: dict[str, Any],
    ) -> None:
        self._trace_callback(
            phase=self._trace_phase,
            messages=completion_kwargs.get("messages", messages),
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            request_payload=completion_kwargs,
        )

    def generate(
        self,
        messages: list[smol_models.ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> smol_models.ChatMessage:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._emit_trace(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            completion_kwargs=completion_kwargs,
        )
        self._apply_rate_limit()
        response = self.retryer(self.client.chat.completions.create, **completion_kwargs)

        content = response.choices[0].message.content
        if stop_sequences is not None and not self.supports_stop_parameter:
            content = smol_models.remove_content_after_stop_sequences(content, stop_sequences)

        return smol_models.ChatMessage(
            role=response.choices[0].message.role,
            content=content,
            tool_calls=response.choices[0].message.tool_calls,
            raw=response,
            token_usage=smol_models.TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    def generate_stream(
        self,
        messages: list[smol_models.ChatMessage | dict[str, Any]],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> Generator[smol_models.ChatMessageStreamDelta]:
        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )
        self._emit_trace(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            completion_kwargs=completion_kwargs,
        )
        self._apply_rate_limit()
        for event in self.retryer(
            self.client.chat.completions.create,
            **completion_kwargs,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if event.usage:
                yield smol_models.ChatMessageStreamDelta(
                    content="",
                    token_usage=smol_models.TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield smol_models.ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            smol_models.ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                elif not getattr(choice, "finish_reason", None):
                    raise ValueError(f"No content or tool calls in event: {event}")


# ---------------------------------------------------------------------------
# RLMAgent
# ---------------------------------------------------------------------------

class RLMAgent:
    """
    Recursive Language Model agent built on smolagents.

    Parameters
    ----------
    base_url:
        Base URL of the OpenAI-compatible API (e.g. ``http://localhost:8080/v1``
        for a local llama.cpp server).
    model_name:
        Model identifier as expected by the server.
    api_key:
        API key.  llama.cpp accepts any non-empty string.
    max_depth:
        Maximum recursion depth.  At ``depth == max_depth`` the agent falls back
        to a plain (non-recursive) completion so the call tree is always finite.
    max_steps:
        Maximum number of agent steps (code-execution cycles) per call.
    verbose:
        If *True*, smolagents will stream step-by-step reasoning to stdout.
    capture_prompt_traces:
        If *True*, every outbound request to the LLM server is attached to the
        call-tree metadata so prompts can be inspected after completion.
    execution_timeout_seconds:
        Maximum time allowed for each Python code execution step inside the
        smolagents executor. Set to ``None`` to disable the timeout.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model_name: str = "local-model",
        api_key: str = "not-needed",
        max_depth: int = 3,
        max_steps: int = 10,
        verbose: bool = False,
        capture_prompt_traces: bool = True,
        execution_timeout_seconds: int | None = None,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.verbose = verbose
        self.capture_prompt_traces = capture_prompt_traces
        self.execution_timeout_seconds = execution_timeout_seconds
        self._active_capture_prompt_traces = capture_prompt_traces

        # Shared call-tree root (rebuilt on every top-level completion call)
        self._root_node: _CallNode | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def completion(
        self,
        task: str | None = None,
        context: str | None = None,
        capture_prompt_traces: bool | None = None,
        **kwargs,
    ) -> RLMCompletion:
        """
        Run an RLM completion.

        This is the drop-in replacement for a plain ``llm.completion(prompt)``
        call.  Internally the model runs inside a smolagents CodeAgent REPL
        where it can freely orchestrate how to process ``rlm_context``:

        * ``llm_call(sub_task, context_slice)`` — direct, non-recursive LLM call.
          Use for leaf-level queries on chunks that are already small enough to
          answer in one shot (mirrors ``llm_query`` in the reference paper).

        * ``rlm_call(sub_task, context_slice)`` — recursive RLM sub-call.  The
          child agent gets its own REPL and can decompose the slice further
          (mirrors ``rlm_query`` in the reference paper).

        The key difference from a naive LLM call is that ``context`` is never
        embedded as a string in the prompt.  It is instead stored as the Python
        variable ``rlm_context`` inside the REPL execution environment, allowing
        the model to interact with it programmatically (slicing, searching, etc.)
        and pass portions to child calls without blowing up the prompt length.

        Parameters
        ----------
        task:
            Short description of what to do (no raw context content here).
        context:
            The raw input data (long text, document, list, etc.).  Stored as
            ``rlm_context`` inside the Python REPL — never embedded in the
            prompt string.  Pass ``None`` for tasks that need no separate data.
        capture_prompt_traces:
            Override the constructor-level prompt trace setting for this call.

        Returns
        -------
        RLMCompletion
            Object containing the final *response* and *metadata* (call tree).
        """
        if task is None:
            task = kwargs.pop("prompt", None)
        if task is None:
            raise TypeError("RLMAgent.completion() missing required argument: 'task'")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"RLMAgent.completion() got unexpected keyword argument(s): {unexpected}")

        previous_capture_setting = self._active_capture_prompt_traces
        self._active_capture_prompt_traces = (
            self.capture_prompt_traces
            if capture_prompt_traces is None
            else capture_prompt_traces
        )

        self._root_node = _CallNode(
            task=task,
            depth=0,
            context_size=len(context) if context else 0,
        )
        try:
            response = self._run(task, context=context, depth=0, node=self._root_node)
            self._root_node.response = response
            self._root_node.end_time = time.time()

            return RLMCompletion(
                response=response,
                metadata={
                    "call_tree": self._root_node.to_dict(),
                    "prompt_trace_enabled": self._active_capture_prompt_traces,
                },
            )
        finally:
            self._active_capture_prompt_traces = previous_capture_setting

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, node: _CallNode, phase: str) -> OpenAIServerModel:
        return _TracingOpenAIServerModel(
            model_id=self.model_name,
            api_base=self.base_url,
            api_key=self.api_key,
            trace_phase=phase,
            trace_callback=lambda **trace: self._record_llm_request(node=node, **trace),
        )

    def _record_llm_request(
        self,
        node: _CallNode,
        phase: str,
        messages: list[Any],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, Any] | None = None,
        tools_to_call_from: list[Any] | None = None,
        request_payload: dict[str, Any] | None = None,
    ) -> None:
        if not self._active_capture_prompt_traces:
            return

        sanitized_messages = _sanitize_for_trace(messages)
        normalized_messages: list[dict[str, Any]] = []
        if isinstance(sanitized_messages, list):
            for item in sanitized_messages:
                if isinstance(item, dict):
                    normalized_messages.append(item)
                else:
                    normalized_messages.append({"content": str(item)})

        node.llm_requests.append(
            _LLMRequestTrace(
                phase=phase,
                depth=node.depth,
                model_name=self.model_name,
                node_step=len(node.llm_requests) + 1,
                messages=normalized_messages,
                stop_sequences=_sanitize_for_trace(stop_sequences),
                response_format=_sanitize_for_trace(response_format),
                tool_names=[
                    getattr(tool_obj, "name", getattr(tool_obj, "__name__", str(tool_obj)))
                    for tool_obj in (tools_to_call_from or [])
                ],
                request_payload=_sanitize_for_trace(request_payload or {}),
            )
        )

    def _build_agent(self, depth: int, parent_node: _CallNode) -> CodeAgent:
        """
        Build a CodeAgent equipped with two LLM sub-call tools:

        ``llm_call`` — direct, non-recursive LLM call (mirrors ``llm_query``
        in the reference implementation).  Use for leaf-level queries on chunks
        that are already small enough to answer in one shot.

        ``rlm_call`` — recursive RLM sub-call (mirrors ``rlm_query``).  The
        child agent gets its own REPL and can decompose the slice further.
        Falls back to a plain completion when ``max_depth`` is reached.
        """
        rlm_self = self  # capture for closure

        @tool
        def llm_call(sub_task: str, context_slice: str = "") -> str:
            """
            Make a direct (non-recursive) LLM call on a sub-task.

            Use this for leaf-level queries on chunks that are small enough to
            answer in a single LLM call without further decomposition.

            IMPORTANT: extract the relevant portion of `rlm_context` in your
            Python code first, then pass the result as `context_slice`.
            Never embed raw context content inside `sub_task`.

            Example:
                paragraphs = [p for p in rlm_context.split("\\n\\n") if p.strip()]
                summaries = [llm_call(f"Summarise paragraph {i+1}", p)
                             for i, p in enumerate(paragraphs)]
                final_answer("\\n".join(summaries))

            Wrong (defeats the purpose of RLM, never do this):
                result = llm_call(f"Summarise: {rlm_context}")

            Args:
                sub_task: Short description of what to do (no raw content).
                context_slice: Python-extracted portion of rlm_context to process.

            Returns:
                The response string from the LLM.
            """
            child_context: str | None = context_slice or None
            ctx_size = len(child_context) if child_context else 0

            child_node = _CallNode(task=sub_task, depth=depth + 1, context_size=ctx_size)
            parent_node.children.append(child_node)
            result = rlm_self._plain_completion(sub_task, context=child_context, node=child_node)
            child_node.response = result
            child_node.end_time = time.time()
            return result

        @tool
        def rlm_call(sub_task: str, context_slice: str = "") -> str:
            """
            Recursively call the RLM on a sub-task with an optional context slice.

            The child agent gets its own Python REPL and can further decompose
            the slice using `llm_call` or `rlm_call`.  Use this for complex
            sub-tasks that themselves may require recursive processing.

            IMPORTANT: extract the relevant portion of `rlm_context` in your
            Python code first, then pass the result as `context_slice`.
            Never embed raw context content inside `sub_task`.

            Example — recursive binary split for very large contexts:
                mid   = len(rlm_context) // 2
                left  = rlm_call("Analyse first half",  rlm_context[:mid])
                right = rlm_call("Analyse second half", rlm_context[mid:])
                final_answer(left + " " + right)

            Wrong (defeats the purpose of RLM, never do this):
                result = rlm_call(f"Summarise: {rlm_context}")

            Args:
                sub_task: Short description of what to do (no raw content).
                context_slice: Python-extracted portion of rlm_context to process.

            Returns:
                The response string from the child RLM.
            """
            child_context: str | None = context_slice or None
            ctx_size = len(child_context) if child_context else 0

            child_node = _CallNode(task=sub_task, depth=depth + 1, context_size=ctx_size)
            parent_node.children.append(child_node)

            if depth >= rlm_self.max_depth:
                # Base case: plain LLM completion — context is small enough now
                result = rlm_self._plain_completion(sub_task, context=child_context, node=child_node)
            else:
                result = rlm_self._run(sub_task, context=child_context, depth=depth + 1, node=child_node)

            child_node.response = result
            child_node.end_time = time.time()
            return result

        model = self._build_model(node=parent_node, phase="agent_step")
        agent = CodeAgent(
            tools=[llm_call, rlm_call],
            model=model,
            max_steps=self.max_steps,
            verbosity_level=1 if self.verbose else 0,
            executor_kwargs={"timeout_seconds": self.execution_timeout_seconds},
        )
        return agent

    def _run(self, task: str, context: str | None, depth: int, node: _CallNode) -> str:
        """
        Run one agent REPL step and return the final answer.

        The context is injected into the agent's Python execution state as the
        variable ``rlm_context`` — it is NOT embedded in the prompt text.
        This is the fundamental difference from a plain LLM call and the core
        mechanic of an RLM: the model manipulates the context programmatically.
        """
        system_hint = textwrap.dedent(f"""\
            You are an RLM (Recursive Language Model) agent at recursion depth {depth}/{self.max_depth}.

            You run inside a Python REPL.  The input context is available as the
            Python variable `rlm_context` — treat it as a Python object.  Slice it,
            search it, split it, transform it.  Do NOT embed its raw content as a
            string literal inside any sub-call argument.

            Two tools are available for making LLM sub-calls:

            `llm_call(sub_task, context_slice)`:
                Direct, non-recursive LLM call.  Fast and lightweight.
                Use for leaf-level queries on chunks that are small enough to
                answer in a single LLM call without further decomposition.

            `rlm_call(sub_task, context_slice)`:
                Recursive RLM sub-call.  The child agent gets its own Python REPL
                and can decompose the slice further.  Use for complex sub-tasks
                that may themselves need recursive processing.

            You decide HOW to orchestrate — use any Python logic to split, filter,
            or transform `rlm_context` before passing slices to sub-calls.

            Example — summarise paragraph-by-paragraph with direct LLM calls:
                paragraphs = [p for p in rlm_context.split("\\n\\n") if p.strip()]
                summaries = [llm_call(f"Summarise paragraph {{i+1}}", p)
                             for i, p in enumerate(paragraphs)]
                final_answer("\\n".join(summaries))

            Example — recursive binary split for very large contexts:
                mid   = len(rlm_context) // 2
                left  = rlm_call("Analyse first half",  rlm_context[:mid])
                right = rlm_call("Analyse second half", rlm_context[mid:])
                final_answer(left + " " + right)

            WRONG — never embed the full context in a sub-call string:
                rlm_call(f"Summarise: {{rlm_context}}")

            If the task is simple enough to answer directly without sub-calls, just do so.
        """)
        task_desc = f"{system_hint}\n\nTask:\n{task}"
        agent = self._build_agent(depth=depth, parent_node=node)

        # Inject context as a Python variable WITHOUT embedding it in the prompt.
        # agent.state is synced to the executor by agent.run() before the first step.
        if context is not None:
            agent.state["rlm_context"] = context

        result = agent.run(task_desc)

        # Capture agent steps (code actions, observations) from the CodeAgent memory.
        self._capture_agent_steps(agent, node)

        return str(result)

    @staticmethod
    def _capture_agent_steps(agent: CodeAgent, node: _CallNode) -> None:
        """Extract intermediate step data from the CodeAgent memory into the node."""
        from smolagents.memory import ActionStep

        for step in agent.memory.steps:
            if not isinstance(step, ActionStep):
                continue
            step_data: dict[str, Any] = {
                "step_number": step.step_number,
                "model_output": step.model_output if isinstance(step.model_output, str) else None,
                "code_action": step.code_action,
                "observations": step.observations,
                "is_final_answer": step.is_final_answer,
            }
            if step.tool_calls:
                step_data["tool_calls"] = [
                    {"name": tc.name, "arguments": tc.arguments, "id": tc.id}
                    for tc in step.tool_calls
                ]
            if step.error:
                step_data["error"] = str(step.error)
            node.agent_steps.append(step_data)

    def _plain_completion(self, task: str, context: str | None = None, node: _CallNode | None = None) -> str:
        """
        Fallback leaf-node completion: a single non-recursive LLM chat call.

        At this point the context has already been decomposed into a small enough
        slice that it fits comfortably in the model's context window.
        """
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        content = f"{task}\n\nContext:\n{context}" if context else task
        messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
        if node is not None:
            self._record_llm_request(
                node=node,
                phase="plain_completion",
                messages=messages,
                request_payload={
                    "model": self.model_name,
                    "messages": messages,
                },
            )
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        return response.choices[0].message.content or ""
