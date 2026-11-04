import asyncio

import pytest

from calbi.core.events import EventBus
from calbi.core.policy import Policy
from calbi.core.registry import ToolRegistry
from calbi.core.runtime import Runtime, ValidationFailed
from calbi.core.schema import InputSpec, Playbook, RetryPolicy, Step
from calbi.core.state import RunState, RunStatus, StepStatus


@pytest.mark.asyncio
async def test_linear_chain_execution() -> None:
    reg = ToolRegistry()

    @reg.tool(name="step_a")
    def step_a(x: int) -> dict[str, int]:
        return {"val": x + 1}

    @reg.tool(name="step_b")
    def step_b(y: int) -> dict[str, int]:
        return {"val": y * 2}

    pb = Playbook(
        name="linear",
        inputs={"num": InputSpec(type="integer")},
        steps=[
            Step(id="a", tool="step_a", inputs={"x": "{{ inputs.num }}"}),
            Step(id="b", tool="step_b", depends_on=["a"], inputs={"y": "{{ steps.a.output.val }}"}),
        ],
        outputs={"final": "{{ steps.b.output.val }}"},
    )

    rt = Runtime(reg, Policy())
    state = await rt.run(pb, inputs={"num": 5})

    assert state.status == RunStatus.SUCCEEDED
    assert state.steps["a"].status == StepStatus.SUCCEEDED
    assert state.steps["a"].output == {"val": 6}
    assert state.steps["b"].status == StepStatus.SUCCEEDED
    assert state.steps["b"].output == {"val": 12}
    assert state.outputs["final"] == 12


@pytest.mark.asyncio
async def test_diamond_dag_concurrent_execution() -> None:
    reg = ToolRegistry()
    b_started = asyncio.Event()
    c_started = asyncio.Event()

    @reg.tool(name="step_root")
    def step_root() -> dict[str, str]:
        return {"data": "ok"}

    @reg.tool(name="step_b")
    async def step_b() -> dict[str, str]:
        b_started.set()
        await c_started.wait()  # Wait until C also starts, proving concurrent execution
        return {"b": "done"}

    @reg.tool(name="step_c")
    async def step_c() -> dict[str, str]:
        c_started.set()
        await b_started.wait()  # Wait until B also starts
        return {"c": "done"}

    @reg.tool(name="step_join")
    def step_join(b: str, c: str) -> dict[str, str]:
        return {"res": f"{b}+{c}"}

    pb = Playbook(
        name="diamond",
        steps=[
            Step(id="root", tool="step_root"),
            Step(id="b", tool="step_b", depends_on=["root"]),
            Step(id="c", tool="step_c", depends_on=["root"]),
            Step(
                id="join",
                tool="step_join",
                depends_on=["b", "c"],
                inputs={"b": "{{ steps.b.output.b }}", "c": "{{ steps.c.output.c }}"},
            ),
        ],
        outputs={"res": "{{ steps.join.output.res }}"},
    )

    rt = Runtime(reg, Policy())
    state = await asyncio.wait_for(rt.run(pb), timeout=2.0)
    assert state.status == RunStatus.SUCCEEDED
    assert state.outputs["res"] == "done+done"


@pytest.mark.asyncio
async def test_condition_falsy_skips_step_and_dependents() -> None:
    reg = ToolRegistry()

    @reg.tool(name="check_flag")
    def check_flag(flag: bool) -> dict[str, bool]:
        return {"proceed": flag}

    @reg.tool(name="action")
    def action() -> dict[str, str]:
        return {"done": "yes"}

    @reg.tool(name="dependent_on_action")
    def dependent_on_action(val: str) -> dict[str, str]:
        return {"result": val}

    @reg.tool(name="independent")
    def independent() -> dict[str, str]:
        return {"indep": "yes"}

    pb = Playbook(
        name="branching",
        inputs={"flag": InputSpec(type="boolean")},
        steps=[
            Step(id="check", tool="check_flag", inputs={"flag": "{{ inputs.flag }}"}),
            Step(
                id="act",
                tool="action",
                depends_on=["check"],
                condition="{{ steps.check.output.proceed }}",
            ),
            Step(
                id="dep",
                tool="dependent_on_action",
                depends_on=["act"],
                inputs={"val": "{{ steps.act.output.done }}"},
            ),
            Step(id="indep", tool="independent"),
        ],
    )

    rt = Runtime(reg, Policy())
    state = await rt.run(pb, inputs={"flag": False})

    assert state.status == RunStatus.SUCCEEDED
    assert state.steps["check"].status == StepStatus.SUCCEEDED
    assert state.steps["act"].status == StepStatus.SKIPPED
    assert state.steps["dep"].status == StepStatus.SKIPPED
    assert state.steps["indep"].status == StepStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_retry_with_backoff_succeeds_on_third_attempt() -> None:
    reg = ToolRegistry()
    attempts = 0

    @reg.tool(name="flaky_tool", idempotent=True)
    def flaky_tool() -> dict[str, str]:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError(f"Temporary failure {attempts}")
        return {"status": "success"}

    pb = Playbook(
        name="retry_test",
        steps=[
            Step(
                id="flaky",
                tool="flaky_tool",
                retry=RetryPolicy(max_attempts=3, backoff="fixed", base_delay_s=0.01),
            ),
        ],
    )

    bus = EventBus()
    rt = Runtime(reg, Policy(), events=bus)
    state = await rt.run(pb)

    assert state.status == RunStatus.SUCCEEDED
    assert state.steps["flaky"].status == StepStatus.SUCCEEDED
    assert state.steps["flaky"].attempts == 3
    retry_events = [e for e in bus.history if e.type == "step.retrying"]
    assert len(retry_events) == 2


@pytest.mark.asyncio
async def test_timeout_kills_step() -> None:
    reg = ToolRegistry()

    @reg.tool(name="slow_tool")
    async def slow_tool() -> str:
        await asyncio.sleep(1.0)
        return "done"

    pb = Playbook(
        name="timeout_test",
        steps=[
            Step(id="slow", tool="slow_tool", timeout_s=0.05),
        ],
    )

    rt = Runtime(reg, Policy())
    state = await rt.run(pb)

    assert state.status == RunStatus.FAILED
    assert state.steps["slow"].status == StepStatus.FAILED
    assert "Timeout" in str(state.steps["slow"].error)


@pytest.mark.asyncio
async def test_validation_failure_prevents_run() -> None:
    reg = ToolRegistry()
    pb = Playbook(
        name="invalid",
        steps=[Step(id="bad", tool="non_existent")],
    )
    rt = Runtime(reg, Policy())
    with pytest.raises(ValidationFailed):
        await rt.run(pb)


@pytest.mark.asyncio
async def test_cancel_mid_run() -> None:
    reg = ToolRegistry()
    step_started = asyncio.Event()

    @reg.tool(name="long_task")
    async def long_task() -> str:
        step_started.set()
        await asyncio.sleep(5.0)
        return "done"

    pb = Playbook(
        name="cancel_test",
        steps=[Step(id="long", tool="long_task")],
    )

    rt = Runtime(reg, Policy())

    async def run_and_cancel() -> RunState:
        run_task = asyncio.create_task(rt.run(pb))
        await step_started.wait()
        # Find run_id or cancel first active run
        await rt.cancel_all()
        return await run_task

    state = await run_and_cancel()
    assert state.status == RunStatus.CANCELLED
