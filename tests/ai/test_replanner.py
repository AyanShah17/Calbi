import json

import pytest

from calbi.ai.planner import Planner
from calbi.ai.replanner import ReplanFailed, Replanner
from calbi.core.policy import Policy
from calbi.core.registry import ToolRegistry
from calbi.core.schema import Playbook, Step
from calbi.core.state import RunState, RunStatus, StepState, StepStatus
from calbi.testing.fake_llm import FakeLLM


def setup_tools() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(name="step_a")
    def step_a() -> dict[str, str]:
        return {"data": "from_a"}

    @reg.tool(name="step_b1")
    def step_b1(d: str) -> dict[str, str]:
        raise RuntimeError("b1 failed")

    @reg.tool(name="step_b2")
    def step_b2(d: str) -> dict[str, str]:
        return {"b2": "ok"}

    return reg


@pytest.mark.asyncio
async def test_replanner_reuses_valid_step_ids() -> None:
    reg = setup_tools()
    policy = Policy()

    pb = Playbook(
        name="initial",
        steps=[
            Step(id="s_a", tool="step_a"),
            Step(
                id="s_b",
                tool="step_b1",
                depends_on=["s_a"],
                inputs={"d": "{{ steps.s_a.output.data }}"},
            ),
        ],
    )

    failed_run = RunState(
        run_id="r1",
        playbook=pb,
        inputs={},
        status=RunStatus.FAILED,
        steps={
            "s_a": StepState(id="s_a", status=StepStatus.SUCCEEDED, output={"data": "from_a"}),
            "s_b": StepState(id="s_b", status=StepStatus.FAILED, error="b1 failed"),
        },
    )

    # Replanner returns an alternative playbook that keeps s_a intact and replaces s_b with step_b2
    alternative_pb = {
        "name": "replan_alternative",
        "steps": [
            {"id": "s_a", "tool": "step_a"},
            {
                "id": "s_b",
                "tool": "step_b2",
                "depends_on": ["s_a"],
                "inputs": {"d": "{{ steps.s_a.output.data }}"},
            },
        ],
    }

    llm = FakeLLM(responses=[json.dumps(alternative_pb)])
    planner = Planner(llm=llm, registry=reg, policy=policy)
    replanner = Replanner(planner=planner)

    res = await replanner.replan(
        task="Do pipeline",
        previous=pb,
        run=failed_run,
        verdict_or_error="b1 failed",
    )
    assert res.playbook.name == "replan_alternative"
    assert res.validation.ok is True
    # Asserts that s_a is preserved
    step_ids = [s.id for s in res.playbook.steps]
    assert "s_a" in step_ids


@pytest.mark.asyncio
async def test_replanner_rejects_identical_failed_playbook() -> None:
    reg = setup_tools()
    policy = Policy()

    pb = Playbook(
        name="initial",
        steps=[Step(id="s_a", tool="step_a")],
    )
    failed_run = RunState(
        run_id="r1",
        playbook=pb,
        inputs={},
        status=RunStatus.FAILED,
        steps={"s_a": StepState(id="s_a", status=StepStatus.FAILED, error="err")},
    )

    # LLM foolishly outputs the identical failed playbook
    llm = FakeLLM(responses=[json.dumps(pb.model_dump())])
    planner = Planner(llm=llm, registry=reg, policy=policy)
    replanner = Replanner(planner=planner)

    with pytest.raises(ReplanFailed, match="identical to a previously failed approach"):
        await replanner.replan(
            task="Do pipeline",
            previous=pb,
            run=failed_run,
            verdict_or_error="err",
            attempts_history=[pb],
        )
