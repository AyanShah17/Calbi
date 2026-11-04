# Calbi — Autonomous Playbook Engine

Calbi is a Python library that lets an AI agent take a task, work out how to do it, write its own **playbook**, execute it, check the result, and change course when something fails. It is meant to be a dependency of any Python application.

> **The AI proposes. The runtime disposes.**
> The model only ever produces *data* (a structured playbook). The runtime validates that data and is the only thing that executes anything, and only through tools the host application has registered.

Status: **pre-alpha, planning stage.** See [PLAN.md](PLAN.md) for the build plan. Agents implementing the work should start there.

---

## Why

Workflow engines need a developer to write every workflow ahead of time. Calbi removes that step. The host app registers *capabilities* (tools) and *permissions*. The AI composes those capabilities into a workflow for each task.

## Architecture

Two systems with a hard boundary between them:

| System | Role | Trusted? |
| --- | --- | --- |
| **AI Playbook Creator** (`calbi.ai`) | Understand, decompose, plan, generate, review, replan | No. Its output is untrusted input. |
| **Playbook Runtime** (`calbi.core`) | Validate, authorize, execute, track state, retry, recover | Yes. Enforces all policy. |

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
                              Failure ◄────────── Runtime (DAG, retries, branches, parallel)
                                                             │
                                                          Success ──► Verifier ──► Result + Evidence + Report
```

### Packages

Delivered as one repo, three installable layers (extras of a single distribution at first, split later if needed):

| Layer | Install | Contents | Required |
| --- | --- | --- | --- |
| `calbi-core` | `pip install calbi` | Schema, tool registry, validator, runtime, state, permissions, events | Yes |
| `calbi-ai` | `pip install calbi[ai]` | Planner, replanner, verifier, strategy selection, LLM adapter protocol | No |
| `calbi-memory` | `pip install calbi[memory]` (SQLite) or `calbi[postgres]` (Neon) | Store for playbooks, runs, strategies, tool limits | No |

Core has no AI dependency. You can use it as a deterministic workflow engine, or add `ai` for autonomous delegation.

## Autonomy levels

| Level | Name | Behaviour |
| --- | --- | --- |
| 0 | Advisory | Generates a playbook, does not execute |
| 1 | Supervised | Executes only after approval |
| 2 | Controlled | Executes pre-approved operations on its own |
| 3 | Autonomous | Plans, executes, retries, replans within limits |
| 4 | Adaptive | Also tries alternative strategies and proposes improvements |

Permissions are enforced by the runtime at **every** level. A generated playbook cannot widen them.

## Intended developer experience

```python
from calbi import Calbi, Autonomy, Risk

engine = Calbi(llm=my_llm, autonomy=Autonomy.AUTONOMOUS)


@engine.tool(name="dataset.inspect", description="Inspect dataset structure", risk=Risk.READ)
def inspect_dataset(file: str) -> dict: ...


@engine.tool(
    name="report.send", description="Email a report", risk=Risk.EXTERNAL, requires_approval=True
)
def send_report(path: str, to: str) -> None: ...


result = await engine.adelegate(
    task="Analyze this dataset and produce a verified report.",
    inputs={"file": "sales.csv"},
)
print(result.output)
print(result.report)  # steps, retries, replans, evidence
```

Sync wrapper: `engine.delegate(...)`.

## Playbook format (illustrative)

```yaml
name: analyze_dataset
version: 1
objective: Analyze an uploaded dataset.
inputs:
  dataset: {type: string, required: true}
steps:
  - id: inspect
    tool: dataset.inspect
    inputs: {file: "{{ inputs.dataset }}"}
    retry: {max_attempts: 3, backoff: exponential}
    timeout_s: 60
  - id: analyze
    tool: dataset.analyze
    depends_on: [inspect]
    condition: "{{ steps.inspect.output.valid }}"
    on_failure: replan
outputs:
  report: "{{ steps.analyze.output.path }}"
```

The authoritative schema lives in `src/calbi/core/schema.py` once built (see PLAN.md, WP-1).

## Services used for development and testing

| Need | Service | Notes |
| --- | --- | --- |
| Persistent memory | [Neon](https://neon.tech) (serverless Postgres) | `pip install calbi[postgres]`, set `DATABASE_URL` |
| LLM (hosted, many models) | [OpenRouter](https://openrouter.ai) | `OPENROUTER_API_KEY` |
| LLM (NVIDIA-hosted models) | [NVIDIA NIM API](https://build.nvidia.com) | `NVIDIA_API_KEY` (optional `NVIDIA_NIM_BASE_URL` for self-hosted NIM) |

All three are optional adapters. The core engine and the default test suite need none of them: tests use a scripted `FakeLLM` and in-memory SQLite. Tests that hit real services are marked `live` and skip automatically when credentials are absent.

```bash
cp .env.example .env      # fill in your own keys; .env is gitignored, never commit it
pytest -m "not live"      # offline, default
pytest -m live            # real OpenRouter / NIM / Neon
```

```python
from calbi import Calbi
from calbi.ai.adapters import OpenRouterLLM, NvidiaNimLLM
from calbi.memory import PostgresMemory

engine = Calbi(
    llm=OpenRouterLLM(model="<model-id>"),  # or NvidiaNimLLM(model="<model-id>")
    memory=PostgresMemory(),  # reads DATABASE_URL
    autonomy="autonomous",
)
```

Environment variables: `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `NVIDIA_NIM_BASE_URL`, `DATABASE_URL` (Neon pooled host, `sslmode=require`), `DATABASE_URL_DIRECT` (optional, for migrations), `CALBI_LLM_PROVIDER`, `CALBI_LLM_MODEL`.

## Design principles

1. **Data, not code.** Playbooks are declarative. No `eval`, no model-written Python executed.
2. **Capabilities are the only door.** If a tool is not registered and permitted, it cannot run.
3. **Small footprint.** Core depends on `pydantic`, `pyyaml` only. No Redis, Celery, Temporal, or LangGraph.
4. **Provider-agnostic.** The LLM is a small `Protocol`; adapters are optional.
5. **Observable.** Every state change emits a structured JSON event.
6. **Deterministic core.** Everything except the LLM is testable with no network.
7. **One run is not a rule.** Memory distinguishes *succeeded*, *validated*, and *generally reusable*.

## Repository layout (target)

```
src/calbi/
  __init__.py          # public API
  core/                # schema, registry, validator, runtime, state, policy, events, templating
  ai/                  # llm protocol, planner, replanner, verifier, strategies, prompts
  ai/adapters/         # OpenRouter, NVIDIA NIM (OpenAI-compatible, httpx)
  memory/              # sqlite + postgres (Neon) stores, retrieval, promotion
  testing/             # FakeLLM, fixtures
tests/
  core/  ai/  memory/  e2e/
examples/
docs/
pyproject.toml
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check . && mypy src
```

Requires Python 3.10+.

## Contributing / agent workflow

Read [PLAN.md](PLAN.md). Work is split into work packages with defined interfaces, file ownership, and acceptance tests. Do not change another package's public interface without recording it in the PLAN.md decision log.

## License

See [LICENSE](LICENSE).
