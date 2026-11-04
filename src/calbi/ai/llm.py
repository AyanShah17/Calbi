from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from calbi.core.schema import CalbiError


class LLMError(CalbiError):
    """Raised when an LLM call or adapter fails."""


@dataclass
class Message:
    """A single chat message exchanged with an LLM."""

    role: Literal["system", "user", "assistant"]
    content: str


class LLM(Protocol):
    """Provider-agnostic interface for Language Models."""

    async def complete(
        self,
        messages: list[Message],
        *,
        response_format: Literal["text", "json"] = "text",
        temperature: float = 0.0,
    ) -> str:
        """Complete a conversation and return the model's text response."""
        ...
