from __future__ import annotations

import asyncio
import importlib
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from calbi.core.events import EventBus
from calbi.core.policy import Autonomy, Policy
from calbi.core.registry import ToolRegistry
from calbi.core.runtime import Runtime
from calbi.core.schema import Playbook, load_playbook
from calbi.core.state import RunState

_delegate_handler: Callable[..., Any] | None = None


def set_delegate_handler(handler: Callable[..., Any]) -> None:
    """Set the handler function for autonomous delegation (used by calbi.ai)."""
    global _delegate_handler
    _delegate_handler = handler


class Calbi:
    """Main facade for the Calbi Autonomous Playbook Engine."""

    def __init__(
        self,
        llm: Any | None = None,
        autonomy: Autonomy | str = Autonomy.CONTROLLED,
        policy: Policy | None = None,
        memory: Any | None = None,
    ) -> None:
        self.llm = llm
        self.memory = memory
        self.events = EventBus()
        self.registry = ToolRegistry()

        if policy is not None:
            self.policy = policy
            if autonomy != Autonomy.CONTROLLED:
                self.policy.autonomy = Autonomy.from_str(autonomy)
        else:
            self.policy = Policy(autonomy=Autonomy.from_str(autonomy))

        self.runtime = Runtime(
            registry=self.registry,
            policy=self.policy,
            events=self.events,
        )

        # Expose tool decorator directly on the engine
        self.tool = self.registry.tool

    def register_tools(self, fns: Iterable[Callable[..., Any]]) -> None:
        """Register multiple tool functions."""
        self.registry.register_tools(fns)

    async def arun(
        self,
        playbook: Playbook | str | dict[str, Any] | Path,
        inputs: dict[str, Any] | None = None,
    ) -> RunState:
        """Asynchronously run a playbook through the deterministic runtime."""
        if not isinstance(playbook, Playbook):
            pb = load_playbook(playbook)
        else:
            pb = playbook

        return await self.runtime.run(pb, inputs=inputs)

    def run(
        self,
        playbook: Playbook | str | dict[str, Any] | Path,
        inputs: dict[str, Any] | None = None,
    ) -> RunState:
        """Synchronous wrapper for arun.

        Raises RuntimeError if called within an active asyncio event loop.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "Cannot call sync run() inside an active asyncio event loop. "
                "Use 'await engine.arun()' instead."
            )

        return asyncio.run(self.arun(playbook, inputs=inputs))

    async def adelegate(
        self,
        task: str,
        inputs: dict[str, Any] | None = None,
        *,
        constraints: list[str] | tuple[str, ...] = (),
        expected_output: str | None = None,
    ) -> Any:
        """Asynchronously delegate a high-level task to autonomous planning and execution."""
        global _delegate_handler

        if self.llm is None:
            raise RuntimeError(
                "LLM is required for delegation. Pass an LLM instance when initializing Calbi."
            )

        if _delegate_handler is None:
            try:
                mod = importlib.import_module("calbi.ai.delegate")
                _delegate_handler = mod.delegate_loop
            except ImportError as e:
                raise RuntimeError(
                    "calbi[ai] extra is required for delegation. "
                    "Install with: pip install 'calbi[ai]'"
                ) from e

        assert _delegate_handler is not None
        return await _delegate_handler(
            engine=self,
            task=task,
            inputs=inputs or {},
            constraints=list(constraints),
            expected_output=expected_output,
        )

    def delegate(
        self,
        task: str,
        inputs: dict[str, Any] | None = None,
        *,
        constraints: list[str] | tuple[str, ...] = (),
        expected_output: str | None = None,
    ) -> Any:
        """Synchronous wrapper for adelegate."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            raise RuntimeError(
                "Cannot call sync delegate() inside an active asyncio event loop. "
                "Use 'await engine.adelegate()' instead."
            )

        return asyncio.run(
            self.adelegate(
                task=task,
                inputs=inputs,
                constraints=constraints,
                expected_output=expected_output,
            )
        )
