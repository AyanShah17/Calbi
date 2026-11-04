from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from calbi.core.schema import Playbook


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    DENIED = "denied"


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    AWAITING_APPROVAL = "awaiting_approval"


class StepState(BaseModel):
    id: str
    status: StepStatus = StepStatus.PENDING
    attempts: int = 0
    output: Any = None
    error: str | None = None
    started_at: datetime | None = None
    ended_at: datetime | None = None


class RunState(BaseModel):
    run_id: str
    playbook: Playbook
    inputs: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus = RunStatus.PENDING
    steps: dict[str, StepState] = Field(default_factory=dict)
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
