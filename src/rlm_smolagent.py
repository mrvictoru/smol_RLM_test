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

import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

from smolagents import CodeAgent, tool
from smolagents.models import OpenAIServerModel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RLMCompletion:
    """Result returned by :meth:`RLMAgent.completion`."""
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class _CallNode:
    """A node in the recursive call tree (used for metadata / logging)."""
    prompt: str
    depth: int
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    response: str = ""
    children: list["_CallNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_preview": self.prompt[:120] + ("…" if len(self.prompt) > 120 else ""),
            "depth": self.depth,
            "duration_s": round((self.end_time or time.time()) - self.start_time, 3),
            "response_preview": self.response[:120] + ("…" if len(self.response) > 120 else ""),
            "children": [c.to_dict() for c in self.children],
        }


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
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080/v1",
        model_name: str = "local-model",
        api_key: str = "not-needed",
        max_depth: int = 3,
        max_steps: int = 10,
        verbose: bool = False,
    ) -> None:
        self.base_url = base_url
        self.model_name = model_name
        self.api_key = api_key
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.verbose = verbose

        # Shared call-tree root (rebuilt on every top-level completion call)
        self._root_node: _CallNode | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def completion(self, prompt: str) -> RLMCompletion:
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

        Returns
        -------
        RLMCompletion
            Object containing the final *response* and *metadata* (call tree).
        """
        self._root_node = _CallNode(prompt=prompt, depth=0)
        response = self._run(prompt, depth=0, node=self._root_node)
        self._root_node.response = response
        self._root_node.end_time = time.time()

        return RLMCompletion(
            response=response,
            metadata={"call_tree": self._root_node.to_dict()},
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self) -> OpenAIServerModel:
        return OpenAIServerModel(
            model_id=self.model_name,
            api_base=self.base_url,
            api_key=self.api_key,
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
                result = rlm_self._plain_completion(sub_prompt)
                child_node.response = result
                child_node.end_time = time.time()
                return result

            child_node = _CallNode(prompt=sub_prompt, depth=depth + 1)
            parent_node.children.append(child_node)
            result = rlm_self._run(sub_prompt, depth=depth + 1, node=child_node)
            child_node.response = result
            child_node.end_time = time.time()
            return result

        model = self._build_model()
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

    def _plain_completion(self, prompt: str) -> str:
        """Fallback: a single non-recursive LLM completion (no agent loop)."""
        from openai import OpenAI

        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
