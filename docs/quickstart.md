# Calbi Quickstart

Calbi allows Python applications to delegate high-level objectives to an AI agent that plans, validates, executes, and verifies structured workflows (Playbooks) using tools you register.

## Installation

```bash
# Core deterministic engine
pip install calbi

# With AI autonomous delegation (OpenRouter, NVIDIA NIM)
pip install "calbi[ai]"

# With Neon serverless Postgres persistent memory
pip install "calbi[postgres]"
```

## Basic Example

```python
import asyncio
from calbi import Calbi, Autonomy, Risk
from calbi.ai.adapters import OpenRouterLLM


async def main():
    # 1. Initialize engine
    engine = Calbi(
        llm=OpenRouterLLM(model="meta-llama/llama-3.3-70b-instruct"),
        autonomy=Autonomy.AUTONOMOUS,
    )

    # 2. Register tools
    @engine.tool(name="dataset.inspect", description="Inspect dataset header", risk=Risk.READ)
    def inspect_file(file: str) -> dict:
        return {"rows": 100, "cols": 5}

    @engine.tool(name="report.generate", description="Generate report", risk=Risk.WRITE)
    def make_report(rows: int) -> dict:
        return {"path": "/tmp/report.md"}

    # 3. Delegate task
    result = await engine.adelegate(
        task="Inspect dataset and generate report",
        inputs={"file": "data.csv"},
    )
    print("Result:", result.output)
    print(result.report.to_markdown())


if __name__ == "__main__":
    asyncio.run(main())
```
