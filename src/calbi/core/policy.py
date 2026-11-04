from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Literal

from calbi.core.registry import ToolSpec
from calbi.core.schema import Risk


class Autonomy(IntEnum):
    """Autonomy levels controlling execution permissions."""

    ADVISORY = 0
    SUPERVISED = 1
    CONTROLLED = 2
    AUTONOMOUS = 3
    ADAPTIVE = 4

    @classmethod
    def from_str(cls, val: str | Autonomy) -> Autonomy:
        if isinstance(val, Autonomy):
            return val
        mapping = {
            "advisory": cls.ADVISORY,
            "supervised": cls.SUPERVISED,
            "controlled": cls.CONTROLLED,
            "autonomous": cls.AUTONOMOUS,
            "adaptive": cls.ADAPTIVE,
        }
        normalized = val.strip().lower()
        if normalized not in mapping:
            raise ValueError(
                f"Unknown autonomy level: '{val}'. Expected one of {list(mapping.keys())}"
            )
        return mapping[normalized]


@dataclass
class ApprovalRequest:
    """Request submitted to a human/approver callback when approval is required."""

    run_id: str
    step_id: str
    tool_spec: ToolSpec
    inputs: dict[str, Any]
    risk: Risk


@dataclass
class Policy:
    """Security and autonomy policy controlling runtime execution."""

    autonomy: Autonomy = Autonomy.CONTROLLED
    allowed_tools: set[str] | None = None
    denied_tools: set[str] = field(default_factory=set)
    max_risk_auto: Risk = Risk.WRITE
    always_approve: set[Risk] = field(
        default_factory=lambda: {
            Risk.DESTRUCTIVE,
            Risk.EXTERNAL,
            Risk.FINANCIAL,
            Risk.DEPLOY,
        }
    )
    max_steps: int = 50
    max_replans: int = 3
    max_total_attempts: int = 200
    max_runtime_s: float = 600.0
    approver: Callable[[ApprovalRequest], Awaitable[bool]] | None = None

    def decide(
        self,
        spec: ToolSpec,
        autonomy: Autonomy | None = None,
    ) -> Literal["allow", "needs_approval", "deny"]:
        """Decide whether a tool execution is allowed, requires approval, or denied."""
        level = autonomy if autonomy is not None else self.autonomy

        # 1. Deny if explicitly blacklisted
        if spec.name in self.denied_tools:
            return "deny"

        # 2. Deny if whitelist is defined and tool not present
        if self.allowed_tools is not None and spec.name not in self.allowed_tools:
            return "deny"

        # 3. Requires approval if marked on tool or risk is in always_approve (any autonomy level)
        if spec.requires_approval or spec.risk in self.always_approve:
            return "needs_approval"

        # 4. Level 0 (Advisory): execution is never permitted
        if level == Autonomy.ADVISORY:
            return "deny"

        # 5. Level 1 (Supervised): everything requires approval
        if level == Autonomy.SUPERVISED:
            return "needs_approval"

        # 6. Risk exceeds automatic threshold
        if spec.risk > self.max_risk_auto:
            return "needs_approval"

        # 7. Otherwise allowed
        return "allow"
