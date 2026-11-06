from __future__ import annotations

import re
from enum import Enum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class CalbiError(Exception):
    """Base exception for all Calbi errors."""


class SchemaError(CalbiError):
    """Raised when a playbook schema fails validation or parsing."""


class _RiskRank(int):
    """Integer rank that is also callable as a method returning int."""

    def __call__(self) -> int:
        return int(self)


class Risk(str, Enum):
    """Risk levels ordered from lowest to highest risk."""

    READ = "read"
    WRITE = "write"
    DESTRUCTIVE = "destructive"
    EXTERNAL = "external"
    FINANCIAL = "financial"
    DEPLOY = "deploy"

    @property
    def rank(self) -> _RiskRank:
        """Integer rank of the risk level from 1 (READ) to 6 (DEPLOY)."""
        ranks = {
            Risk.READ: 1,
            Risk.WRITE: 2,
            Risk.DESTRUCTIVE: 3,
            Risk.EXTERNAL: 4,
            Risk.FINANCIAL: 5,
            Risk.DEPLOY: 6,
        }
        return _RiskRank(ranks[self])

    @property
    def _rank(self) -> _RiskRank:
        return self.rank

    def __lt__(self, other: Any) -> bool:
        if isinstance(other, Risk):
            return self._rank < other._rank
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if isinstance(other, Risk):
            return self._rank <= other._rank
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if isinstance(other, Risk):
            return self._rank > other._rank
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if isinstance(other, Risk):
            return self._rank >= other._rank
        return NotImplemented


class RetryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_attempts: int = Field(default=1, ge=1)
    backoff: Literal["none", "fixed", "exponential"] = "none"
    base_delay_s: float = Field(default=0.5, ge=0.0)
    max_delay_s: float = Field(default=30.0, ge=0.0)


class FailurePolicy(str, Enum):
    FAIL = "fail"
    CONTINUE = "continue"
    REPLAN = "replan"
    SKIP_DEPENDENTS = "skip_dependents"


class InputSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["string", "integer", "number", "boolean", "object", "array", "any"] = "any"
    required: bool = True
    default: Any = None
    description: str = ""


_STEP_ID_REGEX = re.compile(r"^[a-z][a-z0-9_]*$")


class Step(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    tool: str
    inputs: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    condition: str | None = None
    retry: RetryPolicy = Field(default_factory=RetryPolicy)
    timeout_s: float | None = Field(default=None, ge=0.0)
    on_failure: FailurePolicy = FailurePolicy.FAIL
    description: str = ""

    @field_validator("id")
    @classmethod
    def validate_step_id(cls, v: str) -> str:
        if not _STEP_ID_REGEX.match(v):
            raise ValueError(f"Step id '{v}' must match regex ^[a-z][a-z0-9_]*$")
        return v


class Playbook(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    version: int = 1
    objective: str = ""
    inputs: dict[str, InputSpec] = Field(default_factory=dict)
    steps: list[Step]
    outputs: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("steps")
    @classmethod
    def validate_steps_non_empty(cls, steps: list[Step]) -> list[Step]:
        if not steps:
            raise ValueError("Playbook steps must not be empty.")
        return steps

    @model_validator(mode="after")
    def validate_unique_step_ids(self) -> Playbook:
        seen = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"Duplicate step id detected: '{step.id}'")
            seen.add(step.id)
        return self


def load_playbook(source: str | dict[str, Any] | Path) -> Playbook:
    """Load a playbook from a YAML string, a dict, or a file path."""
    data: Any
    if isinstance(source, Path):
        try:
            content = source.read_text(encoding="utf-8")
            data = yaml.safe_load(content)
        except Exception as e:
            raise SchemaError(f"Failed to read playbook file '{source}': {e}") from e
    elif isinstance(source, str):
        try:
            data = yaml.safe_load(source)
        except Exception as e:
            raise SchemaError(f"Invalid YAML source: {e}") from e
    elif isinstance(source, dict):
        data = source
    else:
        raise SchemaError(f"Unsupported source type: {type(source).__name__}")

    if not isinstance(data, dict):
        raise SchemaError(f"Expected playbook dictionary, got {type(data).__name__}")

    try:
        return Playbook.model_validate(data)
    except Exception as e:
        raise SchemaError(f"Playbook validation failed: {e}") from e


def dump_playbook(pb: Playbook) -> str:
    """Dump a Playbook model to a YAML string."""
    data = pb.model_dump(mode="json", exclude_defaults=False)
    return yaml.safe_dump(data, sort_keys=False)
