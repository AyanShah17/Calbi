from typing import Any

import pytest

from calbi.core.policy import Policy
from calbi.core.registry import ToolRegistry
from calbi.core.runtime import Runtime
from calbi.core.schema import Playbook, Step
from calbi.core.state import RunState, RunStatus, StepState, StepStatus


@pytest.mark.asyncio
async def test_resume_reuses_succeeded_steps() -> None:
    reg = ToolRegistry()
    step1_calls = 0
    step2_calls = 0

    @reg.tool(name="step_one")
    def step_one() -> dict[str, int]:
        nonlocal step1_calls
        step1_calls += 1
        return {"val": 10}

    @reg.tool(name="step_two")
    def step_two(x: int) -> dict[str, int]:
        nonlocal step2_calls
        step2_calls += 1
        return {"val": x + 5}

    pb = Playbook(
        name="resume_test",
        steps=[
            Step(id="s1", tool="step_one"),
            Step(
                id="s2",
                tool="step_two",
                depends_on=["s1"],
                inputs={"x": "{{ steps.s1.output.val }}"},
            ),
        ],
    )

    prior_state = RunState(
        run_id="run_old",
        playbook=pb,
        inputs={},
        status=RunStatus.FAILED,
        steps={
            "s1": StepState(
                id="s1",
                status=StepStatus.SUCCEEDED,
                attempts=1,
                output={"val": 10},
            )
        },
    )

    rt = Runtime(reg, Policy())
    new_state = await rt.run(pb, resume_from=prior_state)

    assert new_state.status == RunStatus.SUCCEEDED
    assert new_state.steps["s1"].status == StepStatus.SUCCEEDED
    assert new_state.steps["s2"].status == StepStatus.SUCCEEDED
    # Step 1 was not called again! Reused from prior state!
    assert step1_calls == 0
    assert step2_calls == 1


@pytest.mark.asyncio
async def test_unserializable_output_fails_step() -> None:
    reg = ToolRegistry()

    class CustomObject:
        pass

    @reg.tool(name="bad_output_tool")
    def bad_output_tool() -> Any:
        return CustomObject()

    pb = Playbook(
        name="bad_output",
        steps=[Step(id="bad", tool="bad_output_tool")],
    )

    rt = Runtime(reg, Policy())
    state = await rt.run(pb)
    assert state.status == RunStatus.FAILED
    assert state.steps["bad"].status == StepStatus.FAILED
    assert "serializ" in str(state.steps["bad"].error).lower()
