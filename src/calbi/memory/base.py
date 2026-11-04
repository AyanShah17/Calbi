from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel

from calbi.ai.verifier import Verdict
from calbi.core.schema import Playbook
from calbi.core.state import RunState


class PlaybookRecord(BaseModel):
    """Stored playbook record with validation trust level and usage stats."""

    record_id: str
    task: str
    playbook: Playbook
    status: Literal["succeeded", "validated", "reusable"]
    run_count: int = 1
    last_used: datetime


class MemoryStore(Protocol):
    """Protocol for persistent playbook memory and knowledge retrieval."""

    def save_run(
        self,
        task: str,
        playbook: Playbook,
        run: RunState,
        verdict: Verdict | None = None,
    ) -> str:
        """Save a completed playbook run."""
        ...

    def find_similar(self, task: str, limit: int = 3) -> list[PlaybookRecord]:
        """Find validated or reusable playbooks matching the task objective."""
        ...

    def record_failure(
        self,
        task: str,
        playbook_name: str,
        step_id: str | None,
        tool: str | None,
        error: str,
    ) -> None:
        """Record an execution failure or tool limitation."""
        ...

    def tool_limitations(self, tool: str) -> list[str]:
        """Retrieve known failure modes and limitations for a tool."""
        ...

    def promote(self, record_id: str, to: Literal["validated", "reusable"]) -> None:
        """Promote a playbook record's trust level."""
        ...
