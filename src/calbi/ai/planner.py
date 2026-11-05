from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from calbi.ai.llm import LLM, Message
from calbi.ai.prompts import (
    PLANNER_FEEDBACK_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    PLANNER_USER_PROMPT,
    extract_json,
)
from calbi.core.policy import Policy
from calbi.core.registry import ToolRegistry
from calbi.core.schema import CalbiError, Playbook, load_playbook
from calbi.core.validator import ValidationReport, validate


class PlanningFailed(CalbiError):
    """Raised when the AI Planner cannot generate a valid playbook within attempt limits."""


@dataclass
class PlanResult:
    """The result of a successful planning phase."""

    playbook: Playbook
    validation: ValidationReport
    attempts: int
    rationale: str = ""


class Planner:
    """AI agent component responsible for decomposing objectives into valid Playbooks."""

    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        policy: Policy,
        max_attempts: int = 3,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.policy = policy
        self.max_attempts = max_attempts

    async def plan(
        self,
        task: str,
        inputs: dict[str, Any] | None = None,
        constraints: Sequence[str] = (),
        expected_output: str | None = None,
        hints: Sequence[str] = (),
        templates: Sequence[Playbook] = (),
    ) -> PlanResult:
        """Plan and generate a validated playbook for the given task and capabilities."""
        inputs = dict(inputs or {})

        # Truncate large input values for prompt safety
        truncated_inputs: dict[str, Any] = {}
        inputs_schema: dict[str, str] = {}
        for k, v in inputs.items():
            inputs_schema[k] = type(v).__name__
            val_str = str(v)
            if len(val_str) > 200:
                truncated_inputs[k] = val_str[:200] + "... [truncated]"
            else:
                truncated_inputs[k] = v

        tools_manifest = json.dumps(self.registry.describe(), indent=2)
        combined_hints = list(hints)
        for tmpl in templates:
            combined_hints.append(
                f"Validated Playbook Template '{tmpl.name}': "
                f"{json.dumps(tmpl.model_dump(mode='json'))}"
            )
        hints_text = (
            "\n".join(f"- {h}" for h in combined_hints) if combined_hints else "None provided."
        )

        system_msg = PLANNER_SYSTEM_PROMPT.format(
            tools_manifest=tools_manifest,
            hints=hints_text,
        )

        user_msg = PLANNER_USER_PROMPT.format(
            task=task,
            inputs_schema=json.dumps(inputs_schema, indent=2),
            truncated_inputs=json.dumps(truncated_inputs, indent=2),
            constraints="\n".join(f"- {c}" for c in constraints) if constraints else "None.",
            expected_output=expected_output or "Complete all task requirements.",
        )

        messages = [
            Message(role="system", content=system_msg),
            Message(role="user", content=user_msg),
        ]

        last_report: ValidationReport | None = None
        for attempt in range(1, self.max_attempts + 1):
            raw_response = await self.llm.complete(
                messages,
                response_format="json",
                temperature=0.0,
            )

            try:
                data = extract_json(raw_response)
                pb = load_playbook(data)
                report = validate(pb, self.registry, self.policy, inputs=inputs)
                if report.ok:
                    return PlanResult(
                        playbook=pb,
                        validation=report,
                        attempts=attempt,
                        rationale=pb.objective,
                    )
                last_report = report
            except Exception as e:
                # Capture parse or schema error as a synthetic report
                issues_summary = f"Error parsing playbook JSON: {e}"
            else:
                issues_summary = "\n".join(
                    f"- [{i.code}] ({i.step_id or 'playbook'}): {i.message}"
                    for i in report.issues
                    if i.severity == "error"
                )

            # If not ok and attempts remain, build feedback
            if attempt < self.max_attempts:
                messages.append(Message(role="assistant", content=raw_response))
                feedback = PLANNER_FEEDBACK_PROMPT.format(issues=issues_summary)
                messages.append(Message(role="user", content=feedback))

        err_detail = (
            [i.code for i in last_report.issues if i.severity == "error"]
            if last_report
            else "parse error"
        )
        raise PlanningFailed(
            f"Planner exceeded max attempts ({self.max_attempts}) without valid plan. "
            f"Last issues: {err_detail}"
        )
