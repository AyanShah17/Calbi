# Calbi — Autonomous Playbook Engine

Calbi is a Python library that lets an AI agent take a task, work out how to do it, write its own **playbook**, execute it, check the result, and change course when something fails. It is meant to be a dependency of any Python application.

> **The AI proposes. The runtime disposes.**
> The model only ever produces *data* (a structured playbook). The runtime validates that data and is the only thing that executes anything, and only through tools the host application has registered.

Status: **v0.2.0-dev (Phase 2 Wave 1 completed & merged into `main`).** See [PLAN.md](PLAN.md) for the build plan, architecture details, and milestone specifications.

---

## Why

Workflow engines need a developer to write every workflow ahead of time. Calbi removes that step. The host app registers *capabilities* (tools) and *permissions*. The AI composes those capabilities into a workflow for each task.

## Architecture

Two systems with a hard boundary between them:

| System | Role | Trusted? |
| --- | --- | --- |
| **AI Playbook Creator** (`calbi.ai`) | Understand, decompose, plan, generate, review, replan, explore strategies | No. Its output is untrusted input. |
| **Playbook Runtime** (`calbi.core`) | Validate, authorize, execute, track state, isolate subprocesses, retry, recover | Yes. Enforces all policy and permissions. |

```
Task + Inputs + Constraints
        │
        ▼
 Capability Discovery ──► AI Playbook Creator ──► Playbook (YAML/JSON, untrusted)
                                  ▲                          │
                                  │                          ▼
                          Diagnose / Replan       Validator (schema, deps, permissions, risk)
                                  ▲                          │
                                  │                          ▼
                              Failure ◄────────── Runtime (DAG, retries, branches, parallel, sandboxing)
                                                             │
                                                          Success ──► Verifier ──► Result + Evidence + Report
                                                             │
                                                             ▼
                                                    Self-Improvement Loop (proposals & gated promotion)
```

### Layered Packages

Delivered as one repo with modular layers:

| Layer | Install | Contents | Required |
| --- | --- | --- | --- |
| `calbi-core` | `pip install calbi` | Schema, tool registry, validator, runtime, executors, composition, state, permissions, events, redaction | Yes |
| `calbi-ai` | `pip install calbi[ai]` | Planner, replanner, verifier, strategy exploration tournament, OpenAI-compatible LLM adapters | No |
| `calbi-memory` | `pip install calbi[memory]` (SQLite) or `calbi[postgres]` (Neon) | Store for playbooks, runs, failure logs, and self-improvement proposals | No |

`calbi.core` has zero AI or memory dependencies (`tests/test_architecture.py` enforces this boundary). You can use it as a deterministic, sandboxed workflow engine or add `ai` and `memory` for fully autonomous delegation.

---

## Features & Implementation Status

### V0.1 — Core Engine & Autonomous Delegation (Complete)
- **Declarative Playbook DAG**: Structured JSON/YAML playbooks with typed inputs/outputs, topological execution, conditional branching, retries, and failure routing.
- **Strict Capability & Policy Gate**: Capabilities declared with explicit risks (`READ`, `WRITE`, `EXTERNAL`, `DESTRUCTIVE`, `ADMIN`). Runtime enforces policies and approval thresholds independently of LLM outputs.
- **Autonomous Delegation Loop (`adelegate`)**: Planner produces valid DAG plans; runtime executes them; verifier checks outcomes; replanner diagnoses failures and resumes execution from the failure point.
- **Provider Adapters**: Shared async HTTP adapter (`OpenAICompatLLM`) powering **OpenRouter** and **NVIDIA NIM API** with retry backoff and fallback JSON parsing.
- **Persistent Memory Stores**: SQLite with FTS5 keyword retrieval and serverless **Neon Postgres** (`PostgresMemory`) with cold-start retry protection.

### Phase 2 (V0.2 Wave 1) — Advanced Autonomy & Isolation (Merged into `main`)
- **Autonomy Level 4: Adaptive Strategies (`calbi.ai.strategies`)**:
  - `StrategyExplorer` generates $k$ distinct candidate playbooks and deduplicates structural signatures.
  - Multi-strategy exploration tournament supporting `sequential` early exit (stops once a candidate passes with $\ge 0.8$ confidence) and `parallel_read_only` concurrent execution (strictly rejecting side-effecting tools from dry runs).
  - *Documentation*: [docs/adaptive-strategies.md](docs/adaptive-strategies.md)
- **Self-Improvement Loop & Human Approval Gating (`calbi.memory.improvement`)**:
  - `ImprovementManager` evaluates successful runs against baselines to calculate `steps_saved` and `risk_delta`.
  - Enforces safety invariant: a single run **never** auto-promotes to `reusable`. Playbooks only achieve reusable promotion via explicit human approval (`approve(proposal_id)`).
  - *Documentation*: [docs/self-improvement.md](docs/self-improvement.md)
- **Sandboxed Step Execution (`calbi.core.executor`)**:
  - `StepExecutor` protocol with `InProcessExecutor` and isolated `SubprocessExecutor`.
  - Child process isolation with `os.environ` secret scrubbing (preventing host API keys and database DSNs from leaking to child tools), hard process kill on timeout, and output byte limits.
  - *Documentation*: [docs/sandboxing.md](docs/sandboxing.md)
- **Credential Redaction, JSONL Sinks & OTel Tracing (`calbi.core.observability`)**:
  - `Redactor` recursively sanitizes sensitive keys, scrubs inline Bearer tokens and database passwords (`postgres://user:[REDACTED]@...`), and caps string payloads at 2 KB.
  - Thread-safe `JSONLEventSink` and OpenTelemetry-compatible `TraceCollector` mapping lifecycle events into hierarchical trace spans.
  - *Documentation*: [docs/observability.md](docs/observability.md)
- **Sub-Playbooks & Concurrent For-Each Mapping (`calbi.core.composition`)**:
  - `register_subplaybook`: Exposes playbooks as modular, reusable tools within other playbooks, propagating maximum risk and approval requirements with recursion depth limits (`max_depth`).
  - `register_map_tool`: Wraps existing tools into bounded concurrent mappers (`max_items`, `asyncio.Semaphore(max_concurrency)`).
  - *Documentation*: [docs/composition.md](docs/composition.md)
- **Unified Memory Contract & Live Benchmarking**:
  - Shared contract test suite (`tests/memory/test_memory_contract.py`) ensuring identical behavior between SQLite and Neon Postgres.
  - Live E2E delegation benchmark suite for OpenRouter and NVIDIA NIM.
  - Runnable live/offline demonstration script (`examples/live_neon_openrouter.py`).
  - *Documentation*: [docs/model-compat.md](docs/model-compat.md)

---

## Autonomy Levels

| Level | Name | Behaviour |
| :---: | :--- | :--- |
| **0** | **Advisory** | Generates a declarative playbook, does not execute |
| **1** | **Supervised** | Generates playbook; executes only after operator approval |
| **2** | **Controlled** | Executes pre-approved operations autonomously; halts on escalation |
| **3** | **Autonomous** | Plans, executes, retries, and replans within safety limits |
| **4** | **Adaptive** | Compares competing strategies, executes tournaments, and proposes self-improvements |

Permissions are strictly enforced by the runtime at **every** level. A generated playbook cannot widen them.

---

## Developer Experience Example

```python
import asyncio
from calbi import Calbi, Autonomy, Risk

engine = Calbi(autonomy=Autonomy.AUTONOMOUS)


@engine.tool(name="dataset.inspect", description="Inspect dataset structure", risk=Risk.READ)
def inspect_dataset(file: str) -> dict:
    return {"file": file, "rows": 1000, "clean": True}


@engine.tool(
    name="report.send",
    description="Email executive report",
    risk=Risk.EXTERNAL,
    requires_approval=True,
)
def send_report(path: str, to: str) -> dict:
    return {"status": "sent", "recipient": to}


async def main() -> None:
    result = await engine.adelegate(
        task="Analyze this dataset and email a verified executive report to team@company.com.",
        inputs={"file": "sales_q3.csv"},
    )
    print("Execution Status:", result.status)
    print("Output:", result.output)
    print(result.report.to_markdown())


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Services & Cloud Adapters

| Need | Service | Notes |
| --- | --- | --- |
| Persistent memory | [Neon](https://neon.tech) (serverless Postgres) | `pip install calbi[postgres]`, set `DATABASE_URL` |
| LLM (hosted models) | [OpenRouter](https://openrouter.ai) | Set `OPENROUTER_API_KEY` (e.g. `meta-llama/llama-3.3-70b-instruct`) |
| LLM (NVIDIA-hosted) | [NVIDIA NIM API](https://build.nvidia.com) | Set `NVIDIA_API_KEY` (e.g. `meta/llama-3.3-70b-instruct`) |

All cloud integrations are optional. The test suite and offline engines run with zero credentials using `FakeLLM` and in-memory SQLite:

```bash
cp .env.example .env      # optional: add live keys for OpenRouter, NVIDIA NIM, or Neon
pytest -m "not live"      # runs 235 offline tests (default, zero network required)
pytest -m live            # runs live cloud integration tests
```

---

## Development & Verification

```bash
# Set up virtual environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run full test suite (235 passed offline)
pytest -v -m "not live"

# Code quality & strict typing
ruff check .
ruff format --check .
mypy src
```

Tested and verified against Python **3.10**, **3.11**, **3.12**, and **3.13**.

---

## Documentation Index

- [PLAN.md](PLAN.md) — Architectural roadmap, milestone specs, and parallel branching plan
- [docs/adaptive-strategies.md](docs/adaptive-strategies.md) — Autonomy Level 4 & strategy tournaments
- [docs/self-improvement.md](docs/self-improvement.md) — Self-improvement loops & proposal gating
- [docs/sandboxing.md](docs/sandboxing.md) — In-process & subprocess execution sandboxing
- [docs/observability.md](docs/observability.md) — Credential redaction, JSONL sinks & OpenTelemetry tracing
- [docs/composition.md](docs/composition.md) — Modular sub-playbooks & concurrent for-each tools
- [docs/model-compat.md](docs/model-compat.md) — LLM backend benchmarks & Neon Postgres guide

---

## License

See [LICENSE](LICENSE).
