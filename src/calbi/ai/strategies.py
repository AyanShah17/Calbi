from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from calbi.ai.llm import LLM, Message
from calbi.ai.prompts import extract_json
from calbi.ai.verifier import Verdict, Verifier
from calbi.core.policy import Autonomy, Policy
from calbi.core.registry import ToolRegistry
from calbi.core.runtime import Runtime
from calbi.core.schema import CalbiError, Playbook, Risk, load_playbook
from calbi.core.state import RunState, RunStatus
from calbi.core.validator import ValidationReport, validate

_ORDERED_RISKS = (
    Risk.READ,
    Risk.WRITE,
    Risk.DESTRUCTIVE,
    Risk.EXTERNAL,
    Risk.FINANCIAL,
    Risk.DEPLOY,
)


class StrategyExplorationFailed(CalbiError):
    """Raised when strategy exploration fails to produce viable candidates or trials."""


@dataclass
class StrategyCandidate:
    """A generated candidate strategy playbook and its validation report."""

    name: str
    rationale: str
    playbook: Playbook
    validation: ValidationReport


@dataclass
class StrategyTrial:
    """The outcome and evaluation score of executing a candidate strategy."""

    candidate: StrategyCandidate
    run: RunState | None
    verdict: Verdict | None
    score: float  # 0.0..1.0 composite


@dataclass
class StrategyExplorationResult:
    """Aggregated outcome of an adaptive strategy exploration tournament."""

    winner: StrategyTrial | None
    trials: list[StrategyTrial]
    total_attempts: int


STRATEGY_SYSTEM_PROMPT = """\
You are the Adaptive Strategy Generator for Calbi, an autonomous workflow engine.
Your role is to propose diverse, distinct execution strategies (playbooks) to accomplish a task.
Each strategy should represent an alternative approach, tool chain, or parameter configuration.

RULES:
1. ONLY tools declared in the AVAILABLE TOOLS manifest exist. You must NEVER invent tools.
   Any playbook referencing undeclared tools will be rejected.
2. Produce only a valid JSON object matching the requested schema.
3. Steps must form a valid DAG with unique lowercase step IDs (e.g. "step_inspect").
4. If a step depends on previous steps, list them in "depends_on".
5. Use template syntax "{{{{ inputs.<key> }}}}" or "{{{{ steps.<step_id>.output.<field> }}}}".
6. Do NOT output executable code. Output JSON only.

AVAILABLE TOOLS:
{tools_manifest}

HINTS AND BEST PRACTICES:
{hints}
"""

STRATEGY_USER_PROMPT = """\
Generate up to {k} distinct, alternative strategies (playbooks) to accomplish the following task:

TASK:
{task}

DECLARED INPUTS (types and descriptions):
{inputs_schema}

INPUT VALUES:
{truncated_inputs}

CONSTRAINTS:
{constraints}
{expected_output}
Output a single JSON object with a "candidates" array containing up to {k} objects in this format:
{{
  "candidates": [
    {{
      "name": "strategy_name",
      "rationale": "Explanation of this strategy and how it differs from others",
      "playbook": {{
        "name": "playbook_name",
        "inputs": {{...}},
        "steps": [
          {{
            "id": "step_id",
            "tool": "tool.name",
            "inputs": {{...}},
            "depends_on": []
          }}
        ],
        "outputs": {{...}}
      }}
    }}
  ]
}}
"""


def _playbook_signature(pb: Playbook) -> tuple[tuple[str, str], ...]:
    """Produce a normalized structural signature of tool invocations and static inputs."""
    return tuple(sorted((s.tool, json.dumps(s.inputs, sort_keys=True)) for s in pb.steps))


def _extract_candidates_payload(raw_response: str) -> list[dict[str, Any]]:
    """Extract candidate strategy dicts from JSON or markdown-fenced LLM response."""
    # 1. Try extract_json first
    try:
        data = extract_json(raw_response)
        if isinstance(data, dict):
            if "candidates" in data and isinstance(data["candidates"], list):
                return [c for c in data["candidates"] if isinstance(c, dict)]
            if "strategies" in data and isinstance(data["strategies"], list):
                return [c for c in data["strategies"] if isinstance(c, dict)]
            if "playbook" in data or "steps" in data:
                return [data]
    except Exception:
        pass

    # 2. Try raw json.loads or code fence regex
    text = raw_response.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [c for c in parsed if isinstance(c, dict)]
        if isinstance(parsed, dict):
            if "candidates" in parsed and isinstance(parsed["candidates"], list):
                return [c for c in parsed["candidates"] if isinstance(c, dict)]
            if "strategies" in parsed and isinstance(parsed["strategies"], list):
                return [c for c in parsed["strategies"] if isinstance(c, dict)]
            if "playbook" in parsed or "steps" in parsed:
                return [parsed]
    except Exception:
        pass

    # 3. Substring heuristic for embedded list
    start_bracket = text.find("[")
    end_bracket = text.rfind("]")
    if 0 <= start_bracket < end_bracket:
        try:
            parsed = json.loads(text[start_bracket : end_bracket + 1])
            if isinstance(parsed, list):
                return [c for c in parsed if isinstance(c, dict)]
        except Exception:
            pass

    return []


class StrategyExplorer:
    """Autonomy Level 4 explorer that evaluates competing execution strategies."""

    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        policy: Policy,
        runtime: Runtime | None = None,
        verifier: Verifier | None = None,
        max_candidates: int = 3,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.policy = policy
        self.runtime = runtime or Runtime(registry=registry, policy=policy)
        self.verifier = verifier or Verifier(llm=llm)
        self.max_candidates = max_candidates

    @staticmethod
    def calculate_score(
        candidate: StrategyCandidate,
        run: RunState | None,
        verdict: Verdict | None,
    ) -> float:
        """Evaluate trial on a composite 0.0..1.0 scale combining confidence, steps, and risk."""
        if (
            run is None
            or run.status != RunStatus.SUCCEEDED
            or verdict is None
            or not verdict.passed
        ):
            return 0.0

        confidence = max(0.0, min(1.0, float(verdict.confidence)))
        step_count = max(1, len(candidate.playbook.steps))
        efficiency = 1.0 / (1.0 + 0.1 * float(step_count))
        max_risk = candidate.validation.max_risk
        rank_val = getattr(max_risk, "rank", None)
        if isinstance(rank_val, (int, float)):
            risk_rank = float(rank_val)
        else:
            risk_rank = float(sum(1 for r in _ORDERED_RISKS if max_risk >= r))
        risk_factor = (7.0 - risk_rank) / 6.0
        composite = 0.5 * confidence + 0.3 * efficiency + 0.2 * risk_factor
        return round(max(0.0, min(1.0, composite)), 4)

    async def generate_candidates(
        self,
        task: str,
        inputs: dict[str, Any] | None = None,
        constraints: Sequence[str] = (),
        expected_output: str | None = None,
        k: int = 3,
    ) -> list[StrategyCandidate]:
        """Prompt LLM for candidate playbooks, validate each, discard invalid, and deduplicate."""
        if k <= 0:
            return []

        inputs = dict(inputs or {})

        # Truncate large input values for prompt safety
        truncated_inputs: dict[str, Any] = {}
        inputs_schema: dict[str, str] = {}
        for k_in, v_in in inputs.items():
            inputs_schema[k_in] = type(v_in).__name__
            val_str = str(v_in)
            if len(val_str) > 200:
                truncated_inputs[k_in] = val_str[:200] + "... [truncated]"
            else:
                truncated_inputs[k_in] = v_in

        tools_manifest = json.dumps(self.registry.describe(), indent=2)
        hints_text = (
            "Propose distinct strategies with different tool chains, orders, or parameterizations."
        )

        system_msg = STRATEGY_SYSTEM_PROMPT.format(
            tools_manifest=tools_manifest,
            hints=hints_text,
        )

        if expected_output is not None and str(expected_output).strip():
            expected_output_str = f"\nEXPECTED OUTPUT:\n{str(expected_output).strip()}\n"
        else:
            expected_output_str = ""

        user_msg = STRATEGY_USER_PROMPT.format(
            k=k,
            task=task,
            inputs_schema=json.dumps(inputs_schema, indent=2),
            truncated_inputs=json.dumps(truncated_inputs, indent=2),
            constraints="\n".join(f"- {c}" for c in constraints) if constraints else "None.",
            expected_output=expected_output_str,
        )

        messages = [
            Message(role="system", content=system_msg),
            Message(role="user", content=user_msg),
        ]

        raw_response = await self.llm.complete(
            messages,
            response_format="json",
            temperature=0.0,
        )

        raw_candidates = _extract_candidates_payload(raw_response)
        candidates: list[StrategyCandidate] = []
        seen_signatures: set[tuple[tuple[str, str], ...]] = set()

        for idx, item in enumerate(raw_candidates, start=1):
            if not isinstance(item, dict):
                continue

            raw_pb = item.get("playbook")
            pb_data: Any = raw_pb if isinstance(raw_pb, dict) else item
            if not isinstance(pb_data, dict):
                continue

            try:
                pb = load_playbook(pb_data)
            except Exception:
                continue

            report = validate(pb, self.registry, self.policy, inputs=inputs)
            if not report.ok:
                continue

            sig = _playbook_signature(pb)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)

            name = str(item.get("name") or pb.name or f"candidate_{idx}")
            rationale = str(item.get("rationale") or pb.objective or "")

            candidates.append(
                StrategyCandidate(
                    name=name,
                    rationale=rationale,
                    playbook=pb,
                    validation=report,
                )
            )

            if len(candidates) >= k:
                break

        return candidates

    async def _run_and_evaluate(
        self,
        candidate: StrategyCandidate,
        inputs: dict[str, Any],
        task: str,
        expected_output: str | None,
    ) -> StrategyTrial:
        """Execute a candidate playbook through runtime, verify results, and compute score."""
        try:
            run = await self.runtime.run(
                candidate.playbook,
                inputs=inputs,
                autonomy=Autonomy.ADAPTIVE,
            )
        except Exception as exc:
            verdict = Verdict(
                passed=False,
                reasons=[f"Execution failed: {exc}"],
                confidence=0.0,
            )
            return StrategyTrial(
                candidate=candidate,
                run=None,
                verdict=verdict,
                score=0.0,
            )

        try:
            verdict = await self.verifier.verify(
                task=task,
                expected_output=expected_output,
                run=run,
            )
        except Exception as exc:
            verdict = Verdict(
                passed=False,
                reasons=[f"Verification failed: {exc}"],
                confidence=0.0,
            )

        score = self.calculate_score(candidate, run, verdict)
        return StrategyTrial(
            candidate=candidate,
            run=run,
            verdict=verdict,
            score=score,
        )

    async def explore(
        self,
        task: str,
        inputs: dict[str, Any] | None = None,
        constraints: Sequence[str] = (),
        expected_output: str | None = None,
        *,
        mode: Literal["sequential", "parallel_read_only"] = "sequential",
    ) -> StrategyExplorationResult:
        """Conduct strategy tournament exploration in sequential or parallel_read_only mode."""
        if mode not in ("sequential", "parallel_read_only"):
            raise ValueError(
                f"Unknown exploration mode: '{mode}'. "
                "Expected 'sequential' or 'parallel_read_only'."
            )

        inputs = dict(inputs or {})
        candidates = await self.generate_candidates(
            task=task,
            inputs=inputs,
            constraints=constraints,
            expected_output=expected_output,
            k=self.max_candidates,
        )

        if not candidates:
            raise StrategyExplorationFailed(
                f"Strategy exploration failed: no valid candidates generated for task '{task}'."
            )

        trials: list[StrategyTrial] = []

        if mode == "sequential":
            for candidate in candidates:
                trial = await self._run_and_evaluate(
                    candidate=candidate,
                    inputs=inputs,
                    task=task,
                    expected_output=expected_output,
                )
                trials.append(trial)
                # Early stopping condition: passed with confidence >= 0.8
                if (
                    trial.verdict is not None
                    and trial.verdict.passed
                    and trial.verdict.confidence >= 0.8
                ):
                    break

        elif mode == "parallel_read_only":
            eligible: list[StrategyCandidate] = []
            rejected_trials: dict[int, StrategyTrial] = {}

            for idx, candidate in enumerate(candidates):
                if candidate.validation.max_risk != Risk.READ:
                    rejected_trials[idx] = StrategyTrial(
                        candidate=candidate,
                        run=None,
                        verdict=Verdict(
                            passed=False,
                            reasons=[
                                f"Candidate '{candidate.name}' rejected: max_risk "
                                f"'{candidate.validation.max_risk.value}' > Risk.READ"
                            ],
                            confidence=0.0,
                        ),
                        score=0.0,
                    )
                else:
                    eligible.append(candidate)

            executed_trials: list[StrategyTrial] = []
            if eligible:
                executed_trials = list(
                    await asyncio.gather(
                        *(
                            self._run_and_evaluate(
                                candidate=c,
                                inputs=inputs,
                                task=task,
                                expected_output=expected_output,
                            )
                            for c in eligible
                        )
                    )
                )

            executed_iter = iter(executed_trials)
            for idx, _cand in enumerate(candidates):
                if idx in rejected_trials:
                    trials.append(rejected_trials[idx])
                else:
                    trials.append(next(executed_iter))

        passing_trials = [
            t
            for t in trials
            if t.run is not None
            and t.run.status == RunStatus.SUCCEEDED
            and t.verdict is not None
            and t.verdict.passed
        ]

        winner: StrategyTrial | None = None
        if passing_trials:
            winner = max(
                passing_trials,
                key=lambda t: (t.score, t.verdict.confidence if t.verdict else 0.0),
            )

        total_attempts = sum(
            s.attempts for t in trials if t.run is not None for s in t.run.steps.values()
        )

        return StrategyExplorationResult(
            winner=winner,
            trials=trials,
            total_attempts=total_attempts,
        )
