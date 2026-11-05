from __future__ import annotations

import os
from typing import Any

import pytest

from calbi import Autonomy, Calbi, Risk
from calbi.ai.adapters.openai_compat import NvidiaNimLLM, OpenRouterLLM, llm_from_env
from calbi.ai.llm import LLM
from calbi.ai.planner import Planner
from calbi.core.policy import Policy
from calbi.core.registry import ToolRegistry

pytestmark = pytest.mark.live


def _get_live_llm() -> LLM:
    """Return live LLM instance configured from environment, or skip test cleanly."""
    if os.getenv("CALBI_LLM_PROVIDER") and os.getenv("CALBI_LLM_MODEL"):
        return llm_from_env()

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        model = os.getenv("CALBI_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
        return OpenRouterLLM(model=model, api_key=openrouter_key)

    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        model = os.getenv("CALBI_LLM_MODEL", "meta/llama-3.3-70b-instruct")
        return NvidiaNimLLM(model=model, api_key=nvidia_key)

    pytest.skip("Neither OPENROUTER_API_KEY nor NVIDIA_API_KEY is configured.")


@pytest.mark.asyncio
async def test_live_planner_valid_plan_within_3_attempts() -> None:
    """Live LLM decomposes dataset objective into a valid playbook within 3 attempts."""
    llm = _get_live_llm()
    reg = ToolRegistry()

    @reg.tool(
        name="dataset.inspect",
        description="Inspect dataset structure, columns, and row counts",
        risk=Risk.READ,
    )
    def inspect(file: str) -> dict[str, Any]:
        return {"rows": 1000, "columns": ["id", "amount", "timestamp"]}

    @reg.tool(
        name="dataset.analyze",
        description="Analyze dataset metrics and aggregate transaction values",
        risk=Risk.READ,
    )
    def analyze(file: str) -> dict[str, Any]:
        return {"total_transactions": 1000, "net_revenue": 142500.0}

    @reg.tool(
        name="report.create",
        description="Create markdown analysis report from aggregated summary data",
        risk=Risk.WRITE,
    )
    def report(data: str) -> dict[str, Any]:
        return {"path": "/tmp/live_dataset_summary.md"}

    planner = Planner(llm=llm, registry=reg, policy=Policy(autonomy=Autonomy.AUTONOMOUS))
    plan_result = await planner.plan(
        task="Inspect dataset file, analyze transactions, and generate a final summary report.",
        inputs={"file": "sales.csv"},
    )

    assert plan_result.validation.ok is True
    assert plan_result.attempts <= 3
    assert plan_result.playbook is not None
    assert len(plan_result.playbook.steps) >= 2


@pytest.mark.asyncio
async def test_live_adelegate_e2e_with_replanning() -> None:
    """Full autonomous delegation loop with live LLM recovering from tool failure via replanning."""
    llm = _get_live_llm()
    engine = Calbi(llm=llm, autonomy=Autonomy.AUTONOMOUS)

    primary_calls = 0
    fallback_calls = 0
    report_calls = 0

    @engine.tool(
        name="dataset.inspect",
        description="Inspect dataset schema and metadata",
        risk=Risk.READ,
    )
    def inspect_dataset(file: str) -> dict[str, Any]:
        return {"rows": 500, "format": "csv", "valid": True}

    @engine.tool(
        name="dataset.analyze_primary",
        description="Primary dataset analysis tool. Default preferred analyzer. Use first.",
        risk=Risk.READ,
    )
    def analyze_primary(file: str) -> dict[str, Any]:
        nonlocal primary_calls
        primary_calls += 1
        raise ValueError("Primary analyzer failed: unexpected legacy timestamp encoding.")

    @engine.tool(
        name="dataset.analyze_fallback",
        description="Fallback dataset analysis tool. Robust fallback parser to use on error.",
        risk=Risk.READ,
    )
    def analyze_fallback(file: str) -> dict[str, Any]:
        nonlocal fallback_calls
        fallback_calls += 1
        return {"summary": "500 records successfully processed via fallback analyzer."}

    @engine.tool(
        name="report.create",
        description="Generate formatted markdown report from analysis summary data",
        risk=Risk.WRITE,
    )
    def create_report(data: str) -> dict[str, Any]:
        nonlocal report_calls
        report_calls += 1
        return {"report_path": "/tmp/live_e2e_report.md"}

    result = await engine.adelegate(
        task=(
            "Inspect dataset file, analyze figures using dataset.analyze_primary, "
            "and create the final report. Note: dataset.analyze_fallback is available "
            "as a fallback tool if the primary tool fails."
        ),
        inputs={"file": "financial_records.csv"},
    )

    # In live LLM tests, assert on outcome validity, tool execution, and replanning recovery
    assert result.status == "succeeded"
    assert result.report.replans >= 1
    assert result.verdict is not None
    assert result.verdict.passed is True
    assert primary_calls >= 1
    assert fallback_calls >= 1
    assert report_calls >= 1
