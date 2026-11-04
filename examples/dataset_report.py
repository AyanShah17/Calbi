"""Runnable example of autonomous playbook creation, execution, replanning, and verification."""

import asyncio
import json
from typing import Any

from calbi import Autonomy, Calbi, Risk
from calbi.testing.fake_llm import FakeLLM


async def main() -> None:
    # 1. Prepare simulated LLM responses for the scenario
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
                "tool": "dataset.analyze_fast",
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
        "outputs": {"report": "{{ steps.report.output.path }}"},
    }

    replan = {
        "name": "analyze_dataset_robust",
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
                "tool": "dataset.analyze_robust",
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
        "outputs": {"report": "{{ steps.report.output.path }}"},
    }

    verifier = {
        "passed": True,
        "reasons": ["Report verified with valid data."],
        "evidence": ["Generated at /tmp/sales_report.md"],
        "confidence": 0.99,
    }

    scripted_llm = FakeLLM(
        responses=[
            json.dumps(initial_plan),
            json.dumps(replan),
            json.dumps(verifier),
        ]
    )

    # 2. Initialize engine
    engine = Calbi(llm=scripted_llm, autonomy=Autonomy.AUTONOMOUS)

    # 3. Register application capabilities
    @engine.tool(name="dataset.inspect", description="Inspect dataset metadata", risk=Risk.READ)
    def inspect_dataset(file: str) -> dict[str, Any]:
        print(f"-> [dataset.inspect] Inspecting {file}...")
        return {"rows": 5000, "cols": 12}

    @engine.tool(
        name="dataset.validate", description="Validate dataset data quality", risk=Risk.READ
    )
    def validate_dataset(file: str) -> dict[str, Any]:
        print(f"-> [dataset.validate] Validating {file}...")
        return {"valid": True}

    @engine.tool(name="dataset.analyze_fast", description="Fast heuristic analysis", risk=Risk.READ)
    def analyze_fast(file: str) -> dict[str, Any]:
        print("-> [dataset.analyze_fast] Trying fast analysis...")
        raise ValueError("Fast parser encountered unexpected corrupted characters.")

    @engine.tool(
        name="dataset.analyze_robust", description="Robust fallback analysis", risk=Risk.READ
    )
    def analyze_robust(file: str) -> dict[str, Any]:
        print("-> [dataset.analyze_robust] Running robust analysis...")
        return {"summary": "5,000 transactions processed cleanly."}

    @engine.tool(
        name="report.create", description="Generate markdown summary report", risk=Risk.WRITE
    )
    def create_report(data: str) -> dict[str, Any]:
        print(f"-> [report.create] Creating report with data: '{data}'...")
        return {"path": "/tmp/sales_report.md"}

    # 4. Delegate objective
    print("Delegating objective to Calbi...\n")
    result = await engine.adelegate(
        task="Inspect the dataset, validate quality, analyze figures, and create report.",
        inputs={"dataset": "sales_q3.csv"},
    )

    print("\n=== Result ===")
    print(f"Status: {result.status}")
    print(f"Output: {result.output}")
    print(f"Replans: {result.report.replans}")
    print("\n=== Execution Report Markdown ===")
    print(result.report.to_markdown())


if __name__ == "__main__":
    asyncio.run(main())
