import json

import pytest

from calbi.ai.verifier import Verifier
from calbi.core.schema import Playbook, Step
from calbi.core.state import RunState, RunStatus, StepState, StepStatus
from calbi.testing.fake_llm import FakeLLM


@pytest.mark.asyncio
async def test_verifier_deterministic_failure_cannot_be_overridden_by_llm() -> None:
    # LLM tries to claim the run passed
    llm = FakeLLM(
        responses=[json.dumps({"passed": True, "reasons": ["looks good to me"], "confidence": 1.0})]
    )
    verifier = Verifier(llm=llm)

    pb = Playbook(name="test", steps=[Step(id="s1", tool="t1")])
    failed_run = RunState(
        run_id="r1",
        playbook=pb,
        inputs={},
        status=RunStatus.FAILED,
        steps={"s1": StepState(id="s1", status=StepStatus.FAILED, error="Crash")},
    )

    verdict = await verifier.verify(task="Do work", expected_output="success", run=failed_run)
    assert verdict.passed is False
    assert any("failed" in r.lower() for r in verdict.reasons)
    # LLM was not even trusted or called to override deterministic failure
    assert len(llm.calls) == 0


@pytest.mark.asyncio
async def test_verifier_successful_run_evaluated_by_llm() -> None:
    llm = FakeLLM(
        responses=[
            json.dumps(
                {
                    "passed": True,
                    "reasons": ["Report generated with all required sections"],
                    "evidence": ["Final report found at /tmp/rep.pdf"],
                    "confidence": 0.95,
                }
            )
        ]
    )
    verifier = Verifier(llm=llm)

    pb = Playbook(name="test", steps=[Step(id="s1", tool="t1")])
    succ_run = RunState(
        run_id="r1",
        playbook=pb,
        inputs={},
        status=RunStatus.SUCCEEDED,
        steps={
            "s1": StepState(id="s1", status=StepStatus.SUCCEEDED, output={"path": "/tmp/rep.pdf"})
        },
        outputs={"report": "/tmp/rep.pdf"},
    )

    verdict = await verifier.verify(
        task="Generate report", expected_output="A pdf report", run=succ_run
    )
    assert verdict.passed is True
    assert verdict.confidence == 0.95
    assert len(llm.calls) == 1
