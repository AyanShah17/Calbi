import json

import pytest

from calbi import Autonomy, Calbi, Risk
from calbi.ai.delegate import DelegationResult
from calbi.core.policy import Policy
from calbi.testing.fake_llm import FakeLLM


@pytest.mark.asyncio
async def test_delegate_autonomy_0_advisory_does_not_execute() -> None:
    tool_called = False

    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "name": "advisory_plan",
                    "steps": [{"id": "s1", "tool": "do_task", "inputs": {}}],
                }
            )
        ]
    )

    engine = Calbi(llm=llm, autonomy=Autonomy.ADVISORY)

    @engine.tool(name="do_task")
    def do_task() -> dict[str, str]:
        nonlocal tool_called
        tool_called = True
        return {"done": "yes"}

    res: DelegationResult = await engine.adelegate(task="Perform task")
    assert res.status == "advisory"
    assert res.playbook is not None
    assert res.playbook.name == "advisory_plan"
    assert res.run is None
    assert tool_called is False


@pytest.mark.asyncio
async def test_delegate_happy_path() -> None:
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "name": "happy_path",
                    "steps": [
                        {"id": "step1", "tool": "get_data", "inputs": {}},
                        {
                            "id": "step2",
                            "tool": "process_data",
                            "depends_on": ["step1"],
                            "inputs": {"raw": "{{ steps.step1.output.val }}"},
                        },
                    ],
                    "outputs": {"result": "{{ steps.step2.output.processed }}"},
                }
            ),
            # Verifier response
            json.dumps(
                {
                    "passed": True,
                    "reasons": ["Task complete"],
                    "evidence": ["Data was processed properly"],
                    "confidence": 0.99,
                }
            ),
        ]
    )

    engine = Calbi(llm=llm, autonomy=Autonomy.AUTONOMOUS)

    @engine.tool(name="get_data")
    def get_data() -> dict[str, str]:
        return {"val": "source"}

    @engine.tool(name="process_data")
    def process_data(raw: str) -> dict[str, str]:
        return {"processed": f"clean_{raw}"}

    res: DelegationResult = await engine.adelegate(task="Fetch and process data")
    assert res.status == "succeeded"
    assert res.output["result"] == "clean_source"
    assert res.verdict is not None
    assert res.verdict.passed is True
    assert res.report.replans == 0
    assert "happy_path" in res.report.to_markdown()


@pytest.mark.asyncio
async def test_delegate_replan_and_resume_on_failure() -> None:
    step1_calls = 0
    step2_strategy_a_calls = 0
    step2_strategy_b_calls = 0

    initial_pb = {
        "name": "data_pipeline",
        "steps": [
            {"id": "step1", "tool": "extract_tool", "inputs": {}},
            {
                "id": "step2",
                "tool": "strat_a",
                "depends_on": ["step1"],
                "inputs": {"data": "{{ steps.step1.output.v }}"},
            },
        ],
        "outputs": {"final": "{{ steps.step2.output.res }}"},
    }

    alternative_pb = {
        "name": "data_pipeline_alt",
        "steps": [
            {"id": "step1", "tool": "extract_tool", "inputs": {}},
            {
                "id": "step2",
                "tool": "strat_b",
                "depends_on": ["step1"],
                "inputs": {"data": "{{ steps.step1.output.v }}"},
            },
        ],
        "outputs": {"final": "{{ steps.step2.output.res }}"},
    }

    verifier_response = json.dumps(
        {"passed": True, "reasons": ["Success on second strategy"], "confidence": 1.0}
    )

    llm = FakeLLM(
        responses=[
            json.dumps(initial_pb),
            json.dumps(alternative_pb),
            verifier_response,
        ]
    )

    engine = Calbi(llm=llm, autonomy=Autonomy.AUTONOMOUS)

    @engine.tool(name="extract_tool")
    def extract_tool() -> dict[str, str]:
        nonlocal step1_calls
        step1_calls += 1
        return {"v": "extracted"}

    @engine.tool(name="strat_a")
    def strat_a(data: str) -> dict[str, str]:
        nonlocal step2_strategy_a_calls
        step2_strategy_a_calls += 1
        raise ValueError("Strategy A failed!")

    @engine.tool(name="strat_b")
    def strat_b(data: str) -> dict[str, str]:
        nonlocal step2_strategy_b_calls
        step2_strategy_b_calls += 1
        return {"res": f"handled_{data}"}

    res: DelegationResult = await engine.adelegate(task="Process document")
    assert res.status == "succeeded"
    assert res.output["final"] == "handled_extracted"
    assert res.report.replans == 1

    # Verify that step1 was reused and called ONLY ONCE!
    assert step1_calls == 1
    assert step2_strategy_a_calls == 1
    assert step2_strategy_b_calls == 1


@pytest.mark.asyncio
async def test_delegate_exhausted_max_replans() -> None:
    pb1 = {
        "name": "doomed_1",
        "steps": [{"id": "s1", "tool": "fail_always", "inputs": {"attempt": 1}}],
    }
    pb2 = {
        "name": "doomed_2",
        "steps": [{"id": "s1", "tool": "fail_always", "inputs": {"attempt": 2}}],
    }
    pb3 = {
        "name": "doomed_3",
        "steps": [{"id": "s1", "tool": "fail_always", "inputs": {"attempt": 3}}],
    }

    # 1 initial + 2 replans (max_replans=2)
    llm = FakeLLM(responses=[json.dumps(pb1), json.dumps(pb2), json.dumps(pb3)])

    policy = Policy(autonomy=Autonomy.AUTONOMOUS, max_replans=2)
    engine = Calbi(llm=llm, policy=policy)

    @engine.tool(name="fail_always")
    def fail_always(attempt: int = 1) -> None:
        raise RuntimeError("Persistent crash")

    res: DelegationResult = await engine.adelegate(task="Always fails")
    assert res.status == "failed"
    assert res.report.replans == 2


@pytest.mark.asyncio
async def test_delegate_destructive_tool_requires_approval_at_autonomy_3() -> None:
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "name": "danger",
                    "steps": [{"id": "del", "tool": "format_disk", "inputs": {}}],
                }
            )
        ]
    )

    engine = Calbi(llm=llm, autonomy=Autonomy.AUTONOMOUS)

    @engine.tool(name="format_disk", risk=Risk.DESTRUCTIVE)
    def format_disk() -> None:
        pass

    # No approver configured
    res: DelegationResult = await engine.adelegate(task="Format disk")
    assert res.status == "failed"
    assert res.run is not None
    assert res.run.steps["del"].status.value == "denied"
