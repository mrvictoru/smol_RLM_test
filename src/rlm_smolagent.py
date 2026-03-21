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
    response = rlm.completion(prompt)

Inside an RLM completion, the model runs inside a Python REPL.  It may freely
decompose its input, write helper code, and recursively call itself via:
    result = rlm_call(sub_prompt)

smolagents provides exactly this capability through its CodeAgent, which:
  - Runs LLM-generated Python code in a sandboxed environment.
  - Makes tools available to the generated code (we expose `rlm_call` as one).

Usage
-----
    from rlm_smolagent import RLMAgent

    agent = RLMAgent(
        base_url="http://localhost:8080/v1",  # llama.cpp OpenAI API
        model_name="local-model",
        max_depth=3,
        verbose=True,
    )
    result = agent.completion("Summarise the following 10 000-word article: ...")
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
    prompt: str
    depth: int
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    response: str = ""
    llm_requests: list[_LLMRequestTrace] = field(default_factory=list)
    children: list["_CallNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_preview": self.prompt[:120] + ("…" if len(self.prompt) > 120 else ""),
            "depth": self.depth,
            "duration_s": round((self.end_time or time.time()) - self.start_time, 3),
            "response_preview": self.response[:120] + ("…" if len(self.response) > 120 else ""),
            "llm_requests": [request.to_dict() for request in self.llm_requests],
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
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model_name: str = "local-model",
        api_key: str = "not-needed",
        max_depth: int = 3,
        max_steps: int = 10,
        verbose: bool = False,
        capture_prompt_traces: bool = False,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.verbose = verbose
        self.capture_prompt_traces = capture_prompt_traces
        self._active_capture_prompt_traces = capture_prompt_traces

        # Shared call-tree root (rebuilt on every top-level completion call)
        self._root_node: _CallNode | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def completion(self, prompt: str, capture_prompt_traces: bool | None = None) -> RLMCompletion:
        """
        Run an RLM completion.

        This is the drop-in replacement for a plain ``llm.completion(prompt)``
        call.  Internally the model runs inside a smolagents CodeAgent REPL
        where it can call ``rlm_call(sub_prompt)`` to decompose the task
        recursively.

        Parameters
        ----------
        prompt:
            The task or question to complete.
        capture_prompt_traces:
            Override the constructor-level prompt trace setting for this call.

        Returns
        -------
        RLMCompletion
            Object containing the final *response* and *metadata* (call tree).
        """
        previous_capture_setting = self._active_capture_prompt_traces
        self._active_capture_prompt_traces = (
            self.capture_prompt_traces
            if capture_prompt_traces is None
            else capture_prompt_traces
        )

        self._root_node = _CallNode(prompt=prompt, depth=0)
        try:
            response = self._run(prompt, depth=0, node=self._root_node)
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
        Build a CodeAgent equipped with an ``rlm_call`` tool that performs
        recursive sub-calls up to ``max_depth``.
        """
        rlm_self = self  # capture for closure

        @tool
        def rlm_call(sub_prompt: str) -> str:
            """
            Recursively call the RLM on a sub-prompt.

            Use this tool to decompose a complex or long task into smaller
            pieces and get an answer for each piece.  The answers can then be
            combined in your code to produce the final result.

            Args:
                sub_prompt: The sub-task or question to hand off to a child RLM.

            Returns:
                The response string from the child RLM call.
            """
            if depth >= rlm_self.max_depth:
                # Base case: plain LLM call without further recursion
                child_node = _CallNode(prompt=sub_prompt, depth=depth + 1)
                parent_node.children.append(child_node)
                result = rlm_self._plain_completion(sub_prompt, node=child_node)
                child_node.response = result
                child_node.end_time = time.time()
                return result

            child_node = _CallNode(prompt=sub_prompt, depth=depth + 1)
            parent_node.children.append(child_node)
            result = rlm_self._run(sub_prompt, depth=depth + 1, node=child_node)
            child_node.response = result
            child_node.end_time = time.time()
            return result

        model = self._build_model(node=parent_node, phase="agent_step")
        agent = CodeAgent(
            tools=[rlm_call],
            model=model,
            max_steps=self.max_steps,
            verbosity_level=1 if self.verbose else 0,
        )
        return agent

    def _run(self, prompt: str, depth: int, node: _CallNode) -> str:
        """Run one agent step and return the final answer."""
        system_hint = textwrap.dedent(f"""\
            You are an RLM (Recursive Language Model) agent at recursion depth {depth}/{self.max_depth}.

            You run inside a Python REPL.  For any task that is too long or complex to
            handle in a single pass, you should:
              1. Split the input into manageable chunks.
              2. Call `rlm_call(sub_prompt)` for each chunk.
              3. Aggregate the results in Python and produce the final answer.

            If the task is simple enough to answer directly, just do so.
        """)
        full_prompt = f"{system_hint}\n\nTask:\n{prompt}"
        agent = self._build_agent(depth=depth, parent_node=node)
        result = agent.run(full_prompt)
        return str(result)

    def _plain_completion(self, prompt: str, node: _CallNode | None = None) -> str:
        """Fallback: a single non-recursive LLM completion (no agent loop)."""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        messages = [{"role": "user", "content": prompt}]
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
