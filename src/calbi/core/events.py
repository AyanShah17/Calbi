from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Event(BaseModel):
    """Structured event emitted during playbook planning and execution."""

    ts: datetime
    run_id: str
    type: str
    step_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """Pub/sub bus for Calbi runtime and planning events."""

    def __init__(self) -> None:
        self._subscribers: list[Callable[[Event], None]] = []
        self.history: list[Event] = []

    def subscribe(self, cb: Callable[[Event], None]) -> None:
        """Subscribe a callback to receive events synchronously as they occur."""
        self._subscribers.append(cb)

    def emit(self, event: Event) -> None:
        """Emit an event to all subscribers and append to in-memory history."""
        self.history.append(event)
        for cb in self._subscribers:
            try:
                cb(event)
            except Exception:
                pass

    def to_jsonl(self) -> str:
        """Convert the full event history to a JSON Lines string."""
        lines = [event.model_dump_json() for event in self.history]
        return "\n".join(lines)
