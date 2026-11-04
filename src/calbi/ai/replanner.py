from __future__ import annotations

import json
from collections.abc import Sequence

from calbi.ai.planner import Planner, PlanResult
from calbi.ai.prompts import REPLANNER_USER_PROMPT
from calbi.ai.verifier import Verdict
from calbi.core.schema import CalbiError, Playbook
from calbi.core.state import RunState, StepStatus


class ReplanFailed(CalbiError):
    """Raised when the AI Replanner fails to generate a novel alternative plan."""


def _playbook_signature(pb: Playbook) -> list[tuple[str, str, str]]:
    """Produce a normalized structural signature for comparison."""
    return sorted((s.id, s.tool, json.dumps(s.inputs, sort_keys=True)) for s in pb.steps)


class Replanner:
    """Agent component responsible for diagnosing execution failure and replanning."""

    def __init__(self, planner: Planner) -> None:
        self.planner = planner

    async def replan(
        self,
        task: str,
        previous: Playbook,
        run: RunState,
        verdict_or_error: str | Verdict,
        attempts_history: Sequence[Playbook] | None = None,
    ) -> PlanResult:
        """Diagnose execution failure and generate an alternative plan."""
        history = list(attempts_history or [previous])
        failed_signatures = [_playbook_signature(pb) for pb in history]

        # Identify failed step and valid predecessor steps
        failed_step_id = None
        error_details = str(verdict_or_error)
        for s_id, s_state in run.steps.items():
            if s_state.status == StepStatus.FAILED:
                failed_step_id = s_id
                if s_state.error:
                    error_details = s_state.error
                break

        valid_succeeded_steps = {
            s_id: {"output": s_state.output}
            for s_id, s_state in run.steps.items()
            if s_state.status == StepStatus.SUCCEEDED
        }

        replan_hint = REPLANNER_USER_PROMPT.format(
            task=task,
            failed_step=failed_step_id or "Unknown",
            error_details=error_details,
            valid_steps=json.dumps(valid_succeeded_steps, indent=2),
            failed_approaches=f"{len(history)} previous attempt(s) failed.",
        )

        hints = [
            f"Previous execution failed at step '{failed_step_id}': {error_details}",
            "Reuse valid succeeded step IDs and tools unchanged where possible.",
            "Choose an alternative tool, strategy, or parameterization for the failed step.",
        ]

        result = await self.planner.plan(
            task=f"{task}\n\n[REPLAN INSTRUCTION]:\n{replan_hint}",
            inputs=run.inputs,
            hints=hints,
        )

        new_signature = _playbook_signature(result.playbook)
        for old_sig in failed_signatures:
            if new_signature == old_sig:
                raise ReplanFailed(
                    "Replanner generated a playbook identical to a previously failed approach."
                )

        return result
