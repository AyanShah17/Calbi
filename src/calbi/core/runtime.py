from __future__ import annotations

import asyncio
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from calbi.core.events import Event, EventBus
from calbi.core.policy import ApprovalRequest, Autonomy, Policy
from calbi.core.registry import ToolRegistry
from calbi.core.schema import CalbiError, FailurePolicy, Playbook, Step
from calbi.core.state import RunState, RunStatus, StepState, StepStatus
from calbi.core.templating import render
from calbi.core.validator import ValidationReport, validate


class ValidationFailed(CalbiError):
    """Raised when playbook validation fails before execution."""

    def __init__(self, message: str, report: ValidationReport) -> None:
        super().__init__(message)
        self.report = report


class UnserializableOutput(CalbiError):
    """Raised when a step output cannot be JSON serialized."""


class BudgetExceeded(CalbiError):
    """Raised when total execution runtime or attempt limits are exceeded."""


class Runtime:
    """DAG execution engine for playbooks."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: Policy,
        events: EventBus | None = None,
        max_concurrency: int = 8,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.events = events or EventBus()
        self.max_concurrency = max_concurrency
        self._active_runs: dict[str, asyncio.Event] = {}  # run_id -> cancel_event

    async def cancel(self, run_id: str) -> None:
        """Cancel an active run by run_id."""
        if run_id in self._active_runs:
            self._active_runs[run_id].set()

    async def cancel_all(self) -> None:
        """Cancel all currently running executions."""
        for ev in self._active_runs.values():
            ev.set()

    async def run(
        self,
        pb: Playbook,
        inputs: dict[str, Any] | None = None,
        *,
        resume_from: RunState | None = None,
        autonomy: Autonomy | None = None,
    ) -> RunState:
        """Execute a playbook according to its DAG, policy, and inputs."""
        inputs = dict(inputs or {})
        autonomy_level = autonomy if autonomy is not None else self.policy.autonomy

        # 1. Validate
        report = validate(pb, self.registry, self.policy, inputs=inputs)
        if not report.ok:
            error_codes = [i.code for i in report.issues if i.severity == "error"]
            raise ValidationFailed(
                f"Playbook validation failed with errors: {error_codes}",
                report,
            )

        run_id = f"run_{uuid.uuid4().hex[:12]}"
        cancel_event = asyncio.Event()
        self._active_runs[run_id] = cancel_event

        start_time = time.perf_counter()
        total_attempts = 0

        # 2. Check resume_from for reusable succeeded steps
        steps_state: dict[str, StepState] = {s.id: StepState(id=s.id) for s in pb.steps}
        reused_steps: set[str] = set()
        if resume_from is not None:
            for s in pb.steps:
                if s.id in resume_from.steps:
                    prior_st = resume_from.steps[s.id]
                    if prior_st.status == StepStatus.SUCCEEDED:
                        # Ensure tool and inputs match
                        prior_step_spec = next(
                            (st for st in resume_from.playbook.steps if st.id == s.id), None
                        )
                        if (
                            prior_step_spec is not None
                            and prior_step_spec.tool == s.tool
                            and prior_step_spec.inputs == s.inputs
                        ):
                            steps_state[s.id] = prior_st.model_copy()
                            reused_steps.add(s.id)

        run_state = RunState(
            run_id=run_id,
            playbook=pb,
            inputs=inputs,
            status=RunStatus.RUNNING,
            steps=steps_state,
        )

        self.events.emit(
            Event(
                ts=datetime.now(timezone.utc),
                run_id=run_id,
                type="run.started",
                step_id=None,
                data={"playbook": pb.name, "inputs": inputs},
            )
        )

        step_events: dict[str, asyncio.Event] = {s.id: asyncio.Event() for s in pb.steps}
        for s_id in reused_steps:
            step_events[s_id].set()

        semaphore = asyncio.Semaphore(self.max_concurrency)
        replan_requested = False
        abort_run = False

        async def execute_step(step: Step) -> None:
            nonlocal total_attempts, replan_requested, abort_run

            if step.id in reused_steps:
                return

            # Wait for all dependencies
            for dep_id in step.depends_on:
                await step_events[dep_id].wait()

            if cancel_event.is_set() or abort_run:
                step_events[step.id].set()
                return

            # Check dependency states
            for dep_id in step.depends_on:
                dep_state = steps_state[dep_id]
                if dep_state.status in (StepStatus.FAILED, StepStatus.SKIPPED, StepStatus.DENIED):
                    steps_state[step.id].status = StepStatus.SKIPPED
                    self.events.emit(
                        Event(
                            ts=datetime.now(timezone.utc),
                            run_id=run_id,
                            type="step.skipped",
                            step_id=step.id,
                            data={"reason": f"Dependency '{dep_id}' was {dep_state.status.value}."},
                        )
                    )
                    step_events[step.id].set()
                    return

            # Build context for template evaluation
            ctx = {
                "inputs": inputs,
                "steps": {
                    k: {"output": v.output} for k, v in steps_state.items() if v.output is not None
                },
            }

            # Evaluate condition if present
            if step.condition:
                try:
                    cond_result = render(step.condition, ctx)
                except Exception as e:
                    steps_state[step.id].status = StepStatus.FAILED
                    steps_state[step.id].error = f"Condition evaluation error: {e}"
                    step_events[step.id].set()
                    return

                if not bool(cond_result):
                    steps_state[step.id].status = StepStatus.SKIPPED
                    self.events.emit(
                        Event(
                            ts=datetime.now(timezone.utc),
                            run_id=run_id,
                            type="step.skipped",
                            step_id=step.id,
                            data={"reason": "Condition evaluated to falsy."},
                        )
                    )
                    step_events[step.id].set()
                    return

            # Policy check immediately before execution
            tool_spec = self.registry.get(step.tool)
            decision = self.policy.decide(tool_spec, autonomy_level)

            if decision == "deny":
                steps_state[step.id].status = StepStatus.DENIED
                steps_state[step.id].error = "Execution denied by policy."
                self.events.emit(
                    Event(
                        ts=datetime.now(timezone.utc),
                        run_id=run_id,
                        type="step.failed",
                        step_id=step.id,
                        data={"reason": "policy_denied"},
                    )
                )
                if step.on_failure == FailurePolicy.FAIL:
                    abort_run = True
                step_events[step.id].set()
                return

            if decision == "needs_approval":
                if self.policy.approver is None:
                    steps_state[step.id].status = StepStatus.DENIED
                    steps_state[step.id].error = "Approval required, but no approver configured."
                    self.events.emit(
                        Event(
                            ts=datetime.now(timezone.utc),
                            run_id=run_id,
                            type="step.failed",
                            step_id=step.id,
                            data={"reason": "no_approver"},
                        )
                    )
                    if step.on_failure == FailurePolicy.FAIL:
                        abort_run = True
                    step_events[step.id].set()
                    return

                self.events.emit(
                    Event(
                        ts=datetime.now(timezone.utc),
                        run_id=run_id,
                        type="step.approval_requested",
                        step_id=step.id,
                        data={"tool": step.tool, "risk": tool_spec.risk.value},
                    )
                )

                # Render inputs for approval request
                rendered_inputs = render(step.inputs, ctx)
                req = ApprovalRequest(
                    run_id=run_id,
                    step_id=step.id,
                    tool_spec=tool_spec,
                    inputs=rendered_inputs,
                    risk=tool_spec.risk,
                )
                approved = await self.policy.approver(req)
                if not approved:
                    self.events.emit(
                        Event(
                            ts=datetime.now(timezone.utc),
                            run_id=run_id,
                            type="step.approval_denied",
                            step_id=step.id,
                            data={},
                        )
                    )
                    steps_state[step.id].status = StepStatus.DENIED
                    steps_state[step.id].error = "Approval denied by user."
                    if step.on_failure == FailurePolicy.FAIL:
                        abort_run = True
                    step_events[step.id].set()
                    return

                self.events.emit(
                    Event(
                        ts=datetime.now(timezone.utc),
                        run_id=run_id,
                        type="step.approval_granted",
                        step_id=step.id,
                        data={},
                    )
                )

            # Render inputs for step
            try:
                rendered_step_inputs = render(step.inputs, ctx)
            except Exception as e:
                steps_state[step.id].status = StepStatus.FAILED
                steps_state[step.id].error = f"Failed rendering inputs: {e}"
                if step.on_failure == FailurePolicy.FAIL:
                    abort_run = True
                step_events[step.id].set()
                return

            # Execution attempt loop
            max_attempts = (
                step.retry.max_attempts
                if (tool_spec.idempotent or step.retry.max_attempts > 1)
                else 1
            )
            step_state = steps_state[step.id]
            step_state.status = StepStatus.RUNNING
            step_state.started_at = datetime.now(timezone.utc)

            async with semaphore:
                attempt = 0
                while attempt < max_attempts:
                    if cancel_event.is_set() or abort_run:
                        break

                    # Budget check
                    if (time.perf_counter() - start_time) > self.policy.max_runtime_s:
                        step_state.status = StepStatus.FAILED
                        step_state.error = "Runtime budget exceeded."
                        abort_run = True
                        break

                    if total_attempts >= self.policy.max_total_attempts:
                        step_state.status = StepStatus.FAILED
                        step_state.error = "Total attempt budget exceeded."
                        abort_run = True
                        break

                    attempt += 1
                    total_attempts += 1
                    step_state.attempts = attempt

                    self.events.emit(
                        Event(
                            ts=datetime.now(timezone.utc),
                            run_id=run_id,
                            type="step.started",
                            step_id=step.id,
                            data={"attempt": attempt},
                        )
                    )

                    try:
                        if tool_spec.is_async:
                            coro = tool_spec.fn(**rendered_step_inputs)
                        else:
                            coro = asyncio.to_thread(tool_spec.fn, **rendered_step_inputs)

                        now = time.perf_counter()
                        rem_budget = self.policy.max_runtime_s - (now - start_time)
                        if rem_budget <= 0:
                            step_state.status = StepStatus.FAILED
                            step_state.error = "Runtime budget exceeded."
                            abort_run = True
                            break

                        if step.timeout_s is not None:
                            effective_timeout = min(step.timeout_s, rem_budget)
                        else:
                            effective_timeout = rem_budget

                        output = await asyncio.wait_for(coro, timeout=effective_timeout)

                        # Verify JSON serializability
                        try:
                            json.dumps(output)
                        except Exception as se:
                            raise UnserializableOutput(
                                f"Output from step '{step.id}' is not JSON serializable: {se}"
                            ) from se

                        step_state.status = StepStatus.SUCCEEDED
                        step_state.output = output
                        step_state.ended_at = datetime.now(timezone.utc)

                        self.events.emit(
                            Event(
                                ts=datetime.now(timezone.utc),
                                run_id=run_id,
                                type="step.completed",
                                step_id=step.id,
                                data={"output": output},
                            )
                        )
                        break

                    except (TimeoutError, asyncio.TimeoutError):  # noqa: UP041 (py3.10 compat)
                        now = time.perf_counter()
                        if (
                            step.timeout_s is None
                            or (now - start_time) >= self.policy.max_runtime_s
                        ):
                            err_msg = "Runtime budget exceeded."
                            abort_run = True
                        else:
                            err_msg = f"Timeout: step timed out after {step.timeout_s}s."
                        step_state.error = err_msg

                        if attempt < max_attempts and not abort_run:
                            delay = 0.0
                            if step.retry.backoff == "fixed":
                                delay = step.retry.base_delay_s
                            elif step.retry.backoff == "exponential":
                                delay = min(
                                    step.retry.base_delay_s * (2 ** (attempt - 1)),
                                    step.retry.max_delay_s,
                                )

                            self.events.emit(
                                Event(
                                    ts=datetime.now(timezone.utc),
                                    run_id=run_id,
                                    type="step.retrying",
                                    step_id=step.id,
                                    data={"attempt": attempt, "error": err_msg, "delay_s": delay},
                                )
                            )
                            if delay > 0:
                                await asyncio.sleep(delay)
                        else:
                            step_state.status = StepStatus.FAILED
                            step_state.ended_at = datetime.now(timezone.utc)
                            self.events.emit(
                                Event(
                                    ts=datetime.now(timezone.utc),
                                    run_id=run_id,
                                    type="step.failed",
                                    step_id=step.id,
                                    data={"error": err_msg, "attempts": attempt},
                                )
                            )
                            if step.on_failure == FailurePolicy.FAIL:
                                abort_run = True
                            elif step.on_failure == FailurePolicy.REPLAN:
                                abort_run = True
                                replan_requested = True

                    except Exception as exc:
                        step_state.error = str(exc)
                        if attempt < max_attempts:
                            delay = 0.0
                            if step.retry.backoff == "fixed":
                                delay = step.retry.base_delay_s
                            elif step.retry.backoff == "exponential":
                                delay = min(
                                    step.retry.base_delay_s * (2 ** (attempt - 1)),
                                    step.retry.max_delay_s,
                                )

                            self.events.emit(
                                Event(
                                    ts=datetime.now(timezone.utc),
                                    run_id=run_id,
                                    type="step.retrying",
                                    step_id=step.id,
                                    data={"attempt": attempt, "error": str(exc), "delay_s": delay},
                                )
                            )
                            if delay > 0:
                                await asyncio.sleep(delay)
                        else:
                            step_state.status = StepStatus.FAILED
                            step_state.ended_at = datetime.now(timezone.utc)
                            self.events.emit(
                                Event(
                                    ts=datetime.now(timezone.utc),
                                    run_id=run_id,
                                    type="step.failed",
                                    step_id=step.id,
                                    data={"error": str(exc), "attempts": attempt},
                                )
                            )
                            if step.on_failure == FailurePolicy.FAIL:
                                abort_run = True
                            elif step.on_failure == FailurePolicy.REPLAN:
                                abort_run = True
                                replan_requested = True

            step_events[step.id].set()

        # Run all step tasks concurrently
        tasks = [asyncio.create_task(execute_step(step)) for step in pb.steps]
        await asyncio.gather(*tasks, return_exceptions=True)

        self._active_runs.pop(run_id, None)

        # 3. Determine final RunState
        elapsed_time = time.perf_counter() - start_time
        budget_errors = [
            s.error
            for s in steps_state.values()
            if s.error in ("Runtime budget exceeded.", "Total attempt budget exceeded.")
        ]
        if cancel_event.is_set():
            run_state.status = RunStatus.CANCELLED
            run_state.error = "Run was cancelled."
        elif budget_errors or elapsed_time > self.policy.max_runtime_s:
            run_state.status = RunStatus.FAILED
            run_state.error = (
                budget_errors[0]
                if budget_errors
                else f"Runtime budget exceeded: {elapsed_time:.3f}s > {self.policy.max_runtime_s}s."
            )
        elif abort_run or any(s.status == StepStatus.FAILED for s in steps_state.values()):
            failed_steps = [s.id for s in steps_state.values() if s.status == StepStatus.FAILED]
            denied_steps = [s.id for s in steps_state.values() if s.status == StepStatus.DENIED]

            if any(
                steps_state[s.id].status == StepStatus.FAILED and s.on_failure == FailurePolicy.FAIL
                for s in pb.steps
            ):
                run_state.status = RunStatus.FAILED
                run_state.error = f"Step execution failed: {failed_steps or denied_steps}"
            elif replan_requested:
                run_state.status = RunStatus.FAILED
                run_state.error = f"Replan requested by step failure: {failed_steps}"
            elif denied_steps and any(
                s.status == StepStatus.DENIED
                and next(st.on_failure for st in pb.steps if st.id == s.id) == FailurePolicy.FAIL
                for s in steps_state.values()
            ):
                run_state.status = RunStatus.FAILED
                run_state.error = f"Steps denied by policy: {denied_steps}"
            else:
                # All failures were CONTINUE or SKIP_DEPENDENTS!
                run_state.status = RunStatus.SUCCEEDED
        else:
            run_state.status = RunStatus.SUCCEEDED

        # Render outputs if succeeded
        if run_state.status == RunStatus.SUCCEEDED:
            final_ctx = {
                "inputs": inputs,
                "steps": {
                    k: {"output": v.output} for k, v in steps_state.items() if v.output is not None
                },
            }
            for out_name, out_template in pb.outputs.items():
                run_state.outputs[out_name] = render(out_template, final_ctx)

        # Final event
        event_type = "run.completed" if run_state.status == RunStatus.SUCCEEDED else "run.failed"
        self.events.emit(
            Event(
                ts=datetime.now(timezone.utc),
                run_id=run_id,
                type=event_type,
                step_id=None,
                data={
                    "status": run_state.status.value,
                    "error": run_state.error,
                    "replan_requested": replan_requested,
                    "duration_s": time.perf_counter() - start_time,
                    "total_attempts": total_attempts,
                },
            )
        )

        return run_state
