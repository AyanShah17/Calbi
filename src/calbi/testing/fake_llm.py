from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from calbi.ai.llm import Message

__all__ = ["CallRecord", "FakeLLM", "Message"]


@dataclass
class CallRecord:
    messages: list[Message]
    response_format: Literal["text", "json"]
    temperature: float


class FakeLLM:
    """Scripted fake LLM for deterministic offline testing."""

    def __init__(self, responses: list[str | Callable[..., str]] | None = None) -> None:
        self.responses: list[str | Callable[..., str]] = list(responses or [])
        self.calls: list[CallRecord] = []
        self._index: int = 0

    async def complete(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float = 0.0,
    ) -> str:
        self.calls.append(
            CallRecord(
                messages=messages,
                response_format=response_format,
                temperature=temperature,
            )
        )
        if self._index >= len(self.responses):
            raise RuntimeError(
                f"FakeLLM exhausted: called {len(self.calls)} times, "
                f"but only {len(self.responses)} responses provided."
            )
        resp = self.responses[self._index]
        self._index += 1
        if callable(resp):
            return str(
                resp(
                    messages=messages,
                    response_format=response_format,
                    temperature=temperature,
                )
            )
        return resp
