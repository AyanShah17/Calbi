import asyncio

import pytest

from calbi.core.events import EventBus
from calbi.core.policy import ApprovalRequest, Autonomy, Policy
from calbi.core.registry import ToolRegistry
from calbi.core.runtime import Runtime
from calbi.core.schema import FailurePolicy, Playbook, Risk, Step
from calbi.core.state import RunStatus, StepStatus


@pytest.mark.asyncio
async def test_policy_denial_without_approver() -> None:
    reg = ToolRegistry()

    @reg.tool(name="delete_prod", risk=Risk.DESTRUCTIVE)
    def delete_prod() -> dict[str, str]:
        return {"deleted": "yes"}

    pb = Playbook(
        name="test_deny",
        steps=[Step(id="del", tool="delete_prod", on_failure=FailurePolicy.FAIL)],
    )

    # Autonomy 3 with no approver
    policy = Policy(autonomy=Autonomy.AUTONOMOUS, approver=None)
    rt = Runtime(reg, policy)
    state = await rt.run(pb)

    assert state.status == RunStatus.FAILED
    assert state.steps["del"].status == StepStatus.DENIED


@pytest.mark.asyncio
async def test_policy_approval_granted() -> None:
    reg = ToolRegistry()
    approval_calls = 0

    @reg.tool(name="delete_prod", risk=Risk.DESTRUCTIVE)
    def delete_prod() -> dict[str, str]:
        return {"deleted": "yes"}

    async def approver(req: ApprovalRequest) -> bool:
        nonlocal approval_calls
        approval_calls += 1
        return True

    pb = Playbook(
        name="test_approve",
        steps=[Step(id="del", tool="delete_prod")],
    )

    policy = Policy(autonomy=Autonomy.AUTONOMOUS, approver=approver)
    rt = Runtime(reg, policy)
    state = await rt.run(pb)

    assert state.status == RunStatus.SUCCEEDED
    assert state.steps["del"].status == StepStatus.SUCCEEDED
    assert approval_calls == 1


@pytest.mark.asyncio
async def test_policy_approval_denied() -> None:
    reg = ToolRegistry()

    @reg.tool(name="delete_prod", risk=Risk.DESTRUCTIVE)
    def delete_prod() -> dict[str, str]:
        return {"deleted": "yes"}

    async def approver(req: ApprovalRequest) -> bool:
        return False

    pb = Playbook(
        name="test_deny_by_user",
        steps=[Step(id="del", tool="delete_prod", on_failure=FailurePolicy.CONTINUE)],
    )

    policy = Policy(autonomy=Autonomy.AUTONOMOUS, approver=approver)
    rt = Runtime(reg, policy)
    state = await rt.run(pb)

    assert state.status == RunStatus.SUCCEEDED
    assert state.steps["del"].status == StepStatus.DENIED


@pytest.mark.asyncio
async def test_on_failure_modes() -> None:
    reg = ToolRegistry()

    @reg.tool(name="failing_tool")
    def failing_tool() -> None:
        raise ValueError("Intentional crash")

    @reg.tool(name="success_tool")
    def success_tool() -> str:
        return "ok"

    # 1. CONTINUE: independent branches continue
    pb_cont = Playbook(
        name="cont",
        steps=[
            Step(id="fail_step", tool="failing_tool", on_failure=FailurePolicy.CONTINUE),
            Step(id="ok_step", tool="success_tool"),
        ],
    )
    s1 = await Runtime(reg, Policy()).run(pb_cont)
    assert s1.status == RunStatus.SUCCEEDED
    assert s1.steps["fail_step"].status == StepStatus.FAILED
    assert s1.steps["ok_step"].status == StepStatus.SUCCEEDED

    # 2. SKIP_DEPENDENTS: dependents are skipped, independents continue
    pb_skip = Playbook(
        name="skip",
        steps=[
            Step(id="fail_step", tool="failing_tool", on_failure=FailurePolicy.SKIP_DEPENDENTS),
            Step(id="dep_step", tool="success_tool", depends_on=["fail_step"]),
            Step(id="ok_step", tool="success_tool"),
        ],
    )
    s2 = await Runtime(reg, Policy()).run(pb_skip)
    assert s2.status == RunStatus.SUCCEEDED
    assert s2.steps["fail_step"].status == StepStatus.FAILED
    assert s2.steps["dep_step"].status == StepStatus.SKIPPED
    assert s2.steps["ok_step"].status == StepStatus.SUCCEEDED

    # 3. REPLAN: marks run FAILED with error and replan event
    pb_replan = Playbook(
        name="replan",
        steps=[
            Step(id="fail_step", tool="failing_tool", on_failure=FailurePolicy.REPLAN),
        ],
    )
    bus = EventBus()
    s3 = await Runtime(reg, Policy(), events=bus).run(pb_replan)
    assert s3.status == RunStatus.FAILED
    assert any(e.type == "run.failed" and e.data.get("replan_requested") for e in bus.history)


@pytest.mark.asyncio
async def test_budget_exceeded() -> None:
    reg = ToolRegistry()

    @reg.tool(name="slow_task")
    async def slow_task() -> str:
        await asyncio.sleep(0.1)
        return "done"

    pb = Playbook(
        name="budget",
        steps=[Step(id="s1", tool="slow_task")],
    )

    policy = Policy(max_runtime_s=0.01)
    rt = Runtime(reg, policy)
    state = await rt.run(pb)
    assert state.status == RunStatus.FAILED
    assert "budget" in str(state.error).lower() or "runtime" in str(state.error).lower()
