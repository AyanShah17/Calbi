import pytest

from calbi import Autonomy, Calbi, Risk, RunState, RunStatus


def test_calbi_initialization_with_defaults_and_strings() -> None:
    engine1 = Calbi()
    assert engine1.policy.autonomy == Autonomy.CONTROLLED

    engine2 = Calbi(autonomy="autonomous")
    assert engine2.policy.autonomy == Autonomy.AUTONOMOUS

    with pytest.raises(ValueError, match="Unknown autonomy level"):
        Calbi(autonomy="invalid_level")


@pytest.mark.asyncio
async def test_calbi_deterministic_arun() -> None:
    engine = Calbi(autonomy=Autonomy.CONTROLLED)

    @engine.tool(name="greet", description="Greet someone", risk=Risk.READ)
    def greet(name: str) -> dict[str, str]:
        return {"msg": f"Hello, {name}!"}

    pb_yaml = """
name: hello
inputs:
  user: {type: string}
steps:
  - id: step_greet
    tool: greet
    inputs: {name: "{{ inputs.user }}"}
outputs:
  greeting: "{{ steps.step_greet.output.msg }}"
"""
    state = await engine.arun(pb_yaml, inputs={"user": "Alice"})
    assert isinstance(state, RunState)
    assert state.status == RunStatus.SUCCEEDED
    assert state.outputs["greeting"] == "Hello, Alice!"


def test_calbi_sync_run_outside_event_loop() -> None:
    engine = Calbi()

    @engine.tool(name="calc")
    def calc(n: int) -> dict[str, int]:
        return {"ans": n * 2}

    pb = {
        "name": "sync_test",
        "inputs": {"x": {"type": "integer"}},
        "steps": [{"id": "s1", "tool": "calc", "inputs": {"n": "{{ inputs.x }}"}}],
        "outputs": {"res": "{{ steps.s1.output.ans }}"},
    }
    state = engine.run(pb, inputs={"x": 21})
    assert state.status == RunStatus.SUCCEEDED
    assert state.outputs["res"] == 42


@pytest.mark.asyncio
async def test_calbi_sync_run_inside_event_loop_raises_error() -> None:
    engine = Calbi()
    with pytest.raises(RuntimeError, match="Cannot call sync run"):
        engine.run({"name": "test", "steps": []})


@pytest.mark.asyncio
async def test_calbi_adelegate_without_llm_raises_error() -> None:
    engine = Calbi()  # No llm provided
    with pytest.raises(RuntimeError, match="LLM is required for delegation"):
        await engine.adelegate(task="Do something")
