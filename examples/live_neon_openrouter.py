"""Live and offline demonstration of Calbi delegation with OpenRouter LLM and Neon Postgres.

This script detects available cloud credentials in the environment:
- If OPENROUTER_API_KEY is configured, it connects to OpenRouter (defaulting to
  meta-llama/llama-3.3-70b-instruct).
- If DATABASE_URL is configured, it connects to Neon Postgres and applies idempotent migrations.
- When credentials are not provided, it falls back gracefully to FakeLLM and SQLite (:memory:),
  guaranteeing clean offline execution with return code 0.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from calbi import Autonomy, Calbi, Risk
from calbi.ai.adapters.openai_compat import OpenRouterLLM
from calbi.ai.llm import LLM
from calbi.memory.base import MemoryStore
from calbi.memory.postgres import PostgresMemory
from calbi.memory.sqlite import SQLiteMemory
from calbi.testing.fake_llm import FakeLLM


def build_llm() -> tuple[LLM, str]:
    """Instantiate live OpenRouter LLM or offline FakeLLM based on environment."""
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        model = os.getenv("CALBI_LLM_MODEL", "meta-llama/llama-3.3-70b-instruct")
        return OpenRouterLLM(model=model, api_key=openrouter_key), f"OpenRouter ({model})"

    simulated_plan = {
        "name": "sales_analysis_pipeline",
        "inputs": {"file": {"type": "string"}},
        "steps": [
            {
                "id": "inspect",
                "tool": "dataset.inspect",
                "inputs": {"file": "{{ inputs.file }}"},
            },
            {
                "id": "validate",
                "tool": "dataset.validate",
                "depends_on": ["inspect"],
                "inputs": {"file": "{{ inputs.file }}"},
            },
            {
                "id": "analyze",
                "tool": "dataset.analyze",
                "depends_on": ["validate"],
                "inputs": {"file": "{{ inputs.file }}"},
            },
            {
                "id": "report",
                "tool": "report.create",
                "depends_on": ["analyze"],
                "inputs": {"summary": "{{ steps.analyze.output.summary }}"},
            },
        ],
        "outputs": {"report_path": "{{ steps.report.output.path }}"},
    }
    simulated_verdict = {
        "passed": True,
        "reasons": ["Sales figures verified and report cleanly generated."],
        "evidence": ["Generated report at /tmp/sales_q3_report.md"],
        "confidence": 0.99,
    }

    scripted = FakeLLM(
        responses=[
            json.dumps(simulated_plan),
            json.dumps(simulated_verdict),
        ]
    )
    return scripted, "FakeLLM (offline simulation fallback)"


def build_memory() -> tuple[MemoryStore, str]:
    """Instantiate Neon Postgres or offline SQLite memory based on environment."""
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        schema = os.getenv("CALBI_SCHEMA", "calbi_live_demo")
        pg_mem = PostgresMemory(dsn=database_url, schema=schema)
        pg_mem.migrate()
        return pg_mem, f"Neon Postgres (schema={schema})"

    sqlite_mem = SQLiteMemory(":memory:")
    return sqlite_mem, "SQLite in-memory (:memory: fallback)"


async def main() -> None:
    """Execute autonomous delegation scenario and display execution report."""
    llm, llm_desc = build_llm()
    memory, mem_desc = build_memory()

    print("# Calbi Live / Offline Delegation Demonstration")
    print(f"- **LLM Backend**: {llm_desc}")
    print(f"- **Memory Backend**: {mem_desc}")
    print(f"- **Autonomy Level**: {Autonomy.AUTONOMOUS.name}")
    print()

    engine = Calbi(llm=llm, memory=memory, autonomy=Autonomy.AUTONOMOUS)

    @engine.tool(
        name="dataset.inspect",
        description="Inspect dataset schema, row counts, and column names",
        risk=Risk.READ,
    )
    def inspect(file: str) -> dict[str, Any]:
        return {"file": file, "rows": 2400, "columns": ["id", "amount", "date"], "valid": True}

    @engine.tool(
        name="dataset.validate",
        description="Validate dataset data cleanliness and absence of corrupted rows",
        risk=Risk.READ,
    )
    def validate(file: str) -> dict[str, Any]:
        return {"file": file, "null_count": 0, "status": "clean"}

    @engine.tool(
        name="dataset.analyze",
        description="Compute aggregate revenue metrics and sales transaction figures",
        risk=Risk.READ,
    )
    def analyze(file: str) -> dict[str, Any]:
        return {"summary": "Processed 2,400 transactions. Total gross revenue: $1,450,200."}

    @engine.tool(
        name="report.create",
        description="Generate formatted markdown executive report from analysis figures",
        risk=Risk.WRITE,
    )
    def create_report(summary: str) -> dict[str, Any]:
        return {"path": "/tmp/sales_q3_report.md", "bytes": len(summary)}

    result = await engine.adelegate(
        task="Inspect dataset, validate data quality, analyze sales figures, and generate report.",
        inputs={"file": "sales_q3.csv"},
    )

    print(result.report.to_markdown())
    print()
    print("## Final Result")
    print(f"- **Status**: `{result.status}`")
    print(f"- **Output**: `{json.dumps(result.output)}`")
    if result.verdict:
        print(f"- **Verdict**: `passed={result.verdict.passed}` (conf={result.verdict.confidence})")


if __name__ == "__main__":
    asyncio.run(main())
