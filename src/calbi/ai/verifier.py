from __future__ import annotations

import json
from dataclasses import dataclass, field

from calbi.ai.llm import LLM, Message
from calbi.ai.prompts import VERIFIER_SYSTEM_PROMPT, VERIFIER_USER_PROMPT, extract_json
from calbi.core.state import RunState, RunStatus, StepStatus


@dataclass
class Verdict:
    """Evaluation verdict regarding whether a playbook run met its objective."""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    confidence: float = 1.0


class Verifier:
    """Independent quality verifier combining deterministic rules with LLM assessment."""

    def __init__(self, llm: LLM | None = None) -> None:
        self.llm = llm

    async def verify(
        self,
        task: str,
        expected_output: str | None,
        run: RunState,
    ) -> Verdict:
        """Verify execution outcome against task requirements."""
        # 1. Deterministic checks (CANNOT be overridden by an LLM)
        if run.status != RunStatus.SUCCEEDED:
            return Verdict(
                passed=False,
                reasons=[
                    f"Run failed deterministically with status: {run.status.value}. {run.error}"
                ],
                evidence=[],
                confidence=1.0,
            )

        failed_steps = [
            f"{s.id}: {s.error}" for s in run.steps.values() if s.status == StepStatus.FAILED
        ]
        if failed_steps:
            return Verdict(
                passed=False,
                reasons=[f"Playbook steps failed: {failed_steps}"],
                evidence=[],
                confidence=1.0,
            )

        # 2. If no LLM configured, deterministic success is sufficient
        if self.llm is None:
            return Verdict(
                passed=True,
                reasons=["All playbook steps completed successfully."],
                evidence=[f"Outputs: {run.outputs}"],
                confidence=1.0,
            )

        # 3. LLM semantic verification
        step_results = {
            s.id: {"status": s.status.value, "output": s.output} for s in run.steps.values()
        }
        user_prompt = VERIFIER_USER_PROMPT.format(
            task=task,
            expected_output=expected_output or "Objective successfully achieved.",
            outputs=json.dumps(run.outputs, indent=2),
            step_results=json.dumps(step_results, indent=2),
        )

        messages = [
            Message(role="system", content=VERIFIER_SYSTEM_PROMPT),
            Message(role="user", content=user_prompt),
        ]

        response = await self.llm.complete(messages, response_format="json", temperature=0.0)
        try:
            data = extract_json(response)
            passed = bool(data.get("passed", False))
            reasons = list(data.get("reasons", []))
            evidence = list(data.get("evidence", []))
            confidence = float(data.get("confidence", 0.9))
            return Verdict(
                passed=passed,
                reasons=reasons,
                evidence=evidence,
                confidence=confidence,
            )
        except Exception:
            # Fallback to deterministic verdict if LLM evaluation output fails to parse
            return Verdict(
                passed=True,
                reasons=["Deterministic steps completed successfully; LLM verdict parsing failed."],
                evidence=[f"Outputs: {run.outputs}"],
                confidence=0.8,
            )
