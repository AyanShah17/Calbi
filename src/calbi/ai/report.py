from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from calbi.core.events import Event
from calbi.core.schema import Playbook
from calbi.core.state import RunState


@dataclass
class ExecutionReport:
    """Comprehensive execution report of an autonomous delegation lifecycle."""

    plans: list[Playbook] = field(default_factory=list)
    runs: list[RunState] = field(default_factory=list)
    replans: int = 0
    events: list[Event] = field(default_factory=list)
    total_attempts: int = 0
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert execution report to JSON-serializable dictionary."""
        return {
            "replans": self.replans,
            "total_attempts": self.total_attempts,
            "duration_s": self.duration_s,
            "plans": [p.model_dump(mode="json") for p in self.plans],
            "runs": [r.model_dump(mode="json") for r in self.runs],
            "events": [e.model_dump(mode="json") for e in self.events],
        }

    def to_markdown(self) -> str:
        """Format execution report as GitHub-flavored markdown."""
        lines = [
            "# Calbi Delegation Execution Report",
            f"- **Replans**: {self.replans}",
            f"- **Total Tool Attempts**: {self.total_attempts}",
            f"- **Duration**: {self.duration_s:.2f}s",
            "",
            "## Playbooks Planned",
        ]
        for idx, plan in enumerate(self.plans, 1):
            lines.append(f"### {idx}. {plan.name} (v{plan.version})")
            lines.append(f"Objective: {plan.objective or 'None'}")
            lines.append(f"Steps: {len(plan.steps)}")

        lines.append("")
        lines.append("## Execution Runs")
        for idx, run in enumerate(self.runs, 1):
            lines.append(f"### Run {idx}: {run.run_id} ({run.status.value})")
            if run.error:
                lines.append(f"**Error**: {run.error}")
            for step_id, st in run.steps.items():
                lines.append(f"- **{step_id}**: {st.status.value} (attempts={st.attempts})")

        return "\n".join(lines)
