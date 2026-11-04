from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from calbi.ai.planner import Planner
from calbi.ai.replanner import Replanner
from calbi.ai.report import ExecutionReport
from calbi.ai.verifier import Verdict, Verifier
from calbi.core.engine import set_delegate_handler
from calbi.core.policy import ApprovalRequest, Autonomy
from calbi.core.schema import Playbook
from calbi.core.state import RunState, RunStatus

if TYPE_CHECKING:
    from calbi.core.engine import Calbi


@dataclass
class DelegationResult:
    """The final outcome of an autonomous delegation lifecycle."""

    status: Literal["succeeded", "failed", "needs_approval", "advisory"]
    output: dict[str, Any]
    playbook: Playbook | None
    run: RunState | None
    verdict: Verdict | None
    report: ExecutionReport
    pending_approvals: list[str] = field(default_factory=list)


async def delegate_loop(
    engine: Calbi,
    task: str,
    inputs: dict[str, Any],
    constraints: list[str],
    expected_output: str | None = None,
) -> DelegationResult:
    """Execute the full autonomous delegation state machine."""
    start_time = time.time()
    report = ExecutionReport(events=list(engine.events.history))

    assert engine.llm is not None

    planner = Planner(
        llm=engine.llm,
        registry=engine.registry,
        policy=engine.policy,
    )
    verifier = Verifier(llm=engine.llm)
    replanner = Replanner(planner=planner)

    # 1. UNDERSTAND / PLAN / VALIDATE
    plan_result = await planner.plan(
        task=task,
        inputs=inputs,
        constraints=constraints,
        expected_output=expected_output,
    )
    current_playbook = plan_result.playbook
    report.plans.append(current_playbook)

    # 2. Check Autonomy Level 0 (Advisory)
    if engine.policy.autonomy == Autonomy.ADVISORY:
        report.duration_s = time.time() - start_time
        return DelegationResult(
            status="advisory",
            output={},
            playbook=current_playbook,
            run=None,
            verdict=None,
            report=report,
        )

    # 3. Check Autonomy Level 1 (Supervised - approval required before run)
    if engine.policy.autonomy == Autonomy.SUPERVISED:
        if engine.policy.approver is None:
            report.duration_s = time.time() - start_time
            return DelegationResult(
                status="needs_approval",
                output={},
                playbook=current_playbook,
                run=None,
                verdict=None,
                report=report,
                pending_approvals=["plan_approval"],
            )

        # Submit plan approval request
        # Synthetic ToolSpec representing plan approval
        from calbi.core.registry import ToolSpec

        plan_spec = ToolSpec(
            name=f"plan.{current_playbook.name}",
            description="Playbook execution plan approval",
            fn=lambda: None,
            is_async=False,
            input_schema={},
            risk=plan_result.validation.max_risk,
        )
        req = ApprovalRequest(
            run_id="plan_precheck",
            step_id="plan",
            tool_spec=plan_spec,
            inputs={"plan": current_playbook.name},
            risk=plan_result.validation.max_risk,
        )
        approved = await engine.policy.approver(req)
        if not approved:
            report.duration_s = time.time() - start_time
            return DelegationResult(
                status="needs_approval",
                output={},
                playbook=current_playbook,
                run=None,
                verdict=None,
                report=report,
                pending_approvals=["plan_approval_denied"],
            )

    # 4. EXECUTION / REPLAN / VERIFY LOOP
    last_run: RunState | None = None
    resume_from: RunState | None = None
    replan_count = 0
    attempts_history: list[Playbook] = [current_playbook]
    verdict: Verdict | None = None

    while True:
        # Execute playbook through runtime
        run_state = await engine.runtime.run(
            current_playbook,
            inputs=inputs,
            resume_from=resume_from,
        )
        report.runs.append(run_state)
        report.total_attempts += sum(s.attempts for s in run_state.steps.values())
        last_run = run_state

        if run_state.status != RunStatus.SUCCEEDED:
            if replan_count < engine.policy.max_replans:
                replan_count += 1
                report.replans = replan_count
                try:
                    replan_result = await replanner.replan(
                        task=task,
                        previous=current_playbook,
                        run=run_state,
                        verdict_or_error=run_state.error or "Execution failure",
                        attempts_history=attempts_history,
                    )
                    current_playbook = replan_result.playbook
                    attempts_history.append(current_playbook)
                    report.plans.append(current_playbook)
                    resume_from = run_state
                    continue
                except Exception:
                    break
            else:
                break

        # If execution succeeded, run verification
        verdict = await verifier.verify(
            task=task,
            expected_output=expected_output,
            run=run_state,
        )
        if verdict.passed:
            report.duration_s = time.time() - start_time
            report.events = list(engine.events.history)
            return DelegationResult(
                status="succeeded",
                output=run_state.outputs,
                playbook=current_playbook,
                run=run_state,
                verdict=verdict,
                report=report,
            )

        # Verification failed -> attempt replan
        if replan_count < engine.policy.max_replans:
            replan_count += 1
            report.replans = replan_count
            try:
                replan_result = await replanner.replan(
                    task=task,
                    previous=current_playbook,
                    run=run_state,
                    verdict_or_error=verdict,
                    attempts_history=attempts_history,
                )
                current_playbook = replan_result.playbook
                attempts_history.append(current_playbook)
                report.plans.append(current_playbook)
                resume_from = run_state
                continue
            except Exception:
                break
        else:
            break

    report.duration_s = time.time() - start_time
    report.events = list(engine.events.history)
    return DelegationResult(
        status="failed",
        output={},
        playbook=current_playbook,
        run=last_run,
        verdict=verdict,
        report=report,
    )


# Automatically register the handler on engine
set_delegate_handler(delegate_loop)
