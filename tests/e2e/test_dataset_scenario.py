import json
from typing import Any

import pytest

from calbi import Autonomy, Calbi, Risk
from calbi.testing.fake_llm import FakeLLM


@pytest.mark.asyncio
async def test_dataset_e2e_replan_and_resume() -> None:
    inspect_calls = 0
    validate_calls = 0
    strat_a_calls = 0
    strat_b_calls = 0
    report_calls = 0

    # Initial playbook attempts Strategy A for analyze
    initial_plan = {
        "name": "analyze_dataset",
        "inputs": {"dataset": {"type": "string"}},
        "steps": [
            {
                "id": "inspect",
                "tool": "dataset.inspect",
                "inputs": {"file": "{{ inputs.dataset }}"},
            },
            {
                "id": "validate",
                "tool": "dataset.validate",
                "depends_on": ["inspect"],
                "inputs": {"file": "{{ inputs.dataset }}"},
            },
            {
                "id": "analyze",
                "tool": "dataset.analyze_strat_a",
                "depends_on": ["validate"],
                "inputs": {"file": "{{ inputs.dataset }}"},
            },
            {
                "id": "report",
                "tool": "report.create",
                "depends_on": ["analyze"],
                "inputs": {"data": "{{ steps.analyze.output.summary }}"},
            },
        ],
        "outputs": {"report_path": "{{ steps.report.output.path }}"},
    }

    # Alternative playbook reuses inspect and validate, replaces analyze with Strategy B
    replan = {
        "name": "analyze_dataset_v2",
        "inputs": {"dataset": {"type": "string"}},
        "steps": [
            {
                "id": "inspect",
                "tool": "dataset.inspect",
                "inputs": {"file": "{{ inputs.dataset }}"},
            },
            {
                "id": "validate",
                "tool": "dataset.validate",
                "depends_on": ["inspect"],
                "inputs": {"file": "{{ inputs.dataset }}"},
            },
            {
                "id": "analyze",
                "tool": "dataset.analyze_strat_b",
                "depends_on": ["validate"],
                "inputs": {"file": "{{ inputs.dataset }}"},
            },
            {
                "id": "report",
                "tool": "report.create",
                "depends_on": ["analyze"],
                "inputs": {"data": "{{ steps.analyze.output.summary }}"},
            },
        ],
        "outputs": {"report_path": "{{ steps.report.output.path }}"},
    }

    verifier_verdict = {
        "passed": True,
        "reasons": ["Dataset report generated and verified."],
        "evidence": ["Report at /tmp/sales_report.md"],
        "confidence": 0.98,
    }

    llm = FakeLLM(
        responses=[
            json.dumps(initial_plan),
            json.dumps(replan),
            json.dumps(verifier_verdict),
        ]
    )

    engine = Calbi(llm=llm, autonomy=Autonomy.AUTONOMOUS)

    @engine.tool(name="dataset.inspect", description="Inspect dataset metadata", risk=Risk.READ)
    def inspect_dataset(file: str) -> dict[str, Any]:
        nonlocal inspect_calls
        inspect_calls += 1
        return {"rows": 1000, "cols": 10}

    @engine.tool(
        name="dataset.validate", description="Validate dataset data quality", risk=Risk.READ
    )
    def validate_dataset(file: str) -> dict[str, Any]:
        nonlocal validate_calls
        validate_calls += 1
        return {"valid": True, "null_count": 0}

    @engine.tool(
        name="dataset.analyze_strat_a",
        description="Analyze dataset with strategy A",
        risk=Risk.READ,
    )
    def analyze_a(file: str) -> dict[str, Any]:
        nonlocal strat_a_calls
        strat_a_calls += 1
        raise ValueError("Strategy A encountered unsupported date formats.")

    @engine.tool(
        name="dataset.analyze_strat_b",
        description="Analyze dataset with strategy B",
        risk=Risk.READ,
    )
    def analyze_b(file: str) -> dict[str, Any]:
        nonlocal strat_b_calls
        strat_b_calls += 1
        return {"summary": "Total sales: $500,000"}

    @engine.tool(name="report.create", description="Generate analysis report", risk=Risk.WRITE)
    def create_report(data: str) -> dict[str, Any]:
        nonlocal report_calls
        report_calls += 1
        return {"path": "/tmp/sales_report.md"}

    result = await engine.adelegate(
        task="Analyze this dataset and generate a verified report.",
        inputs={"dataset": "sales.csv"},
    )

    assert result.status == "succeeded"
    assert result.output["report_path"] == "/tmp/sales_report.md"
    assert result.verdict is not None
    assert result.verdict.passed is True
    assert result.report.replans == 1

    # KEY ACCEPTANCE CHECK: Inspect and Validate ran ONLY ONCE, resumed on replan!
    assert inspect_calls == 1
    assert validate_calls == 1
    assert strat_a_calls == 1
    assert strat_b_calls == 1
    assert report_calls == 1
