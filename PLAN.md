# Calbi Build Plan (V0.1)

Audience: smaller implementation agents. Read this whole file, then do **only** the work package you were assigned. Overview and rationale are in [README.md](README.md).

## 0. Ground rules (apply to every work package)

1. **Language/tooling:** Python ≥3.10, `asyncio`, `pydantic` v2, `pyyaml`. Dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy --strict` on `src/`.
2. **No network in tests.** The LLM is always `FakeLLM` (WP-0) in tests.
3. **Never execute model output as code.** No `eval`/`exec`/`compile`/`pickle` on playbook or LLM data. Template expressions use the restricted evaluator from WP-2.
4. **Core must not import `calbi.ai` or `calbi.memory`.** Dependency direction: `ai → core`, `memory → core`. Enforce with a test (WP-9).
5. **Public interfaces in §3 are contracts.** Implement them exactly. If one must change, stop, record it in §9 Decision Log, and flag it. Don't silently diverge.
6. **Each WP is done only when:** its acceptance tests pass, `ruff` and `mypy` are clean for its files, and public functions have docstrings and type hints.
7. **Stay in your files.** Only edit files listed under your WP's "Owns". Add tests under your WP's test directory.
8. **Errors:** raise subclasses of `CalbiError` (defined in WP-1). No bare `Exception`, no swallowed errors.
9. **Events:** every state transition emits an `Event` (WP-3). No `print`.

## 1. Scope

**In V0.1:** playbook schema, tool registry, validator, DAG runtime (parallel, retry, timeout, branching, resume), permission/approval policy, autonomy levels 0–3, LLM planner, verifier, replanner, in-memory state, event log, SQLite memory (store + retrieve), e2e tests with a FakeLLM.

**Deferred (design for, don't build):** autonomy level 4 strategy experimentation beyond a stub interface, process/container sandboxing (interface hook only), loops/sub-playbooks, distributed execution, provider adapters beyond one reference adapter.

## 2. Dependency graph / order

```
WP-0 scaffold ─► WP-1 schema ─► WP-2 templating ─┐
                    │                            ├─► WP-4 validator ─► WP-5 runtime ─► WP-6 engine facade (core only)
                    └─► WP-3 registry+events+policy ┘                                   │
                                                                                        ▼
                               WP-7 AI layer (planner/verifier/replanner) ─► WP-8 delegate loop
                               WP-10 memory (parallel after WP-1)                          │
                                                                                        ▼
                                                            WP-9 e2e + safety + docs/examples
```

Parallelizable: WP-2 ∥ WP-3 after WP-1; WP-7 ∥ WP-10 after WP-6. WP-11 after WP-7; WP-10b after WP-10. WP-9 live e2e runs last, after WP-8, WP-10b and WP-11.

## 3. Contracts (exact interfaces)

### 3.1 Schema (`calbi/core/schema.py`) — WP-1

```python
class CalbiError(Exception): ...
class SchemaError(CalbiError): ...

class Risk(str, Enum):      # ordered low→high
    READ="read"; WRITE="write"; DESTRUCTIVE="destructive"; EXTERNAL="external"; FINANCIAL="financial"; DEPLOY="deploy"

class RetryPolicy(BaseModel):
    max_attempts: int = 1            # >=1
    backoff: Literal["none","fixed","exponential"] = "none"
    base_delay_s: float = 0.5
    max_delay_s: float = 30.0

class FailurePolicy(str, Enum):  FAIL="fail"; CONTINUE="continue"; REPLAN="replan"; SKIP_DEPENDENTS="skip_dependents"

class InputSpec(BaseModel):  type: Literal["string","integer","number","boolean","object","array","any"]="any"; required: bool=True; default: Any=None; description: str=""

class Step(BaseModel):
    id: str                          # ^[a-z][a-z0-9_]*$
    tool: str
    inputs: dict[str, Any] = {}      # values may contain "{{ ... }}" templates
    depends_on: list[str] = []
    condition: str | None = None     # template expression -> truthy/falsy
    retry: RetryPolicy = RetryPolicy()
    timeout_s: float | None = None
    on_failure: FailurePolicy = FailurePolicy.FAIL
    description: str = ""

class Playbook(BaseModel):
    name: str; version: int = 1; objective: str = ""
    inputs: dict[str, InputSpec] = {}
    steps: list[Step]                # non-empty, unique ids
    outputs: dict[str, str] = {}     # name -> template
    metadata: dict[str, Any] = {}

def load_playbook(source: str | dict | Path) -> Playbook   # YAML text, dict, or path; raises SchemaError
def dump_playbook(pb: Playbook) -> str                       # YAML, round-trips
```

### 3.2 Templating (`calbi/core/templating.py`) — WP-2

```python
def render(value: Any, ctx: dict) -> Any
# Recursively renders dicts/lists/strings. A string that is exactly one "{{ expr }}" returns the raw
# typed value; a string with embedded templates returns str. Context keys: inputs, steps, env (optional, caller-provided).
def evaluate(expr: str, ctx: dict) -> Any
def referenced_steps(value: Any) -> set[str]     # step ids referenced via steps.<id>...
class TemplateError(CalbiError): ...
```
Expression grammar (whitelist, parsed with `ast` and walked manually): attribute/key access (`a.b`, `a["b"]`), literals, `== != < <= > >= and or not in`, `+ - * /`, parentheses, and `len()`. Everything else (calls except `len`, comprehensions, lambdas, dunder attributes, imports) raises `TemplateError`. Missing keys raise `TemplateError` with the path.

### 3.3 Registry, events, policy (`calbi/core/registry.py`, `events.py`, `policy.py`) — WP-3

```python
@dataclass
class ToolSpec:
    name: str; description: str; fn: Callable; is_async: bool
    input_schema: dict      # JSON-schema derived from type hints / pydantic
    output_schema: dict | None
    risk: Risk = Risk.READ
    requires_approval: bool = False
    idempotent: bool = False
    tags: list[str] = []
    limitations: str = ""

class ToolRegistry:
    def register(self, fn=None, *, name=None, description=None, risk=Risk.READ, requires_approval=False, idempotent=False, tags=(), limitations="") -> ...  # usable as decorator or call
    def register_tools(self, fns: Iterable[Callable]) -> None
    def get(self, name) -> ToolSpec          # raises UnknownToolError
    def list(self) -> list[ToolSpec]
    def describe(self) -> list[dict]         # LLM-safe manifest: name, description, input_schema, output_schema, risk, requires_approval, limitations. NEVER includes fn.
```
Duplicate names raise `DuplicateToolError`. Sync tools run in a thread executor in the runtime.

```python
class Event(BaseModel): ts: datetime; run_id: str; type: str; step_id: str|None; data: dict
# types: run.started run.completed run.failed step.started step.completed step.failed step.retrying
#        step.skipped step.approval_requested step.approval_granted step.approval_denied
#        plan.generated plan.validated plan.rejected replan.started replan.completed verify.passed verify.failed
class EventBus:
    def subscribe(self, cb: Callable[[Event], None]) -> None
    def emit(self, event: Event) -> None
    history: list[Event]
    def to_jsonl(self) -> str

class Autonomy(IntEnum): ADVISORY=0; SUPERVISED=1; CONTROLLED=2; AUTONOMOUS=3; ADAPTIVE=4

@dataclass
class Policy:
    autonomy: Autonomy = Autonomy.CONTROLLED
    allowed_tools: set[str] | None = None     # None = all registered
    denied_tools: set[str] = set()
    max_risk_auto: Risk = Risk.WRITE          # highest risk executable without approval
    always_approve: set[Risk] = {Risk.DESTRUCTIVE, Risk.EXTERNAL, Risk.FINANCIAL, Risk.DEPLOY}
    max_steps: int = 50; max_replans: int = 3; max_total_attempts: int = 200; max_runtime_s: float = 600
    approver: Callable[[ApprovalRequest], Awaitable[bool]] | None = None
    def decide(self, spec: ToolSpec, autonomy: Autonomy) -> Literal["allow","needs_approval","deny"]
```
Decision rules (tested exhaustively): denied → deny; not in allowed → deny; `spec.requires_approval` or risk in `always_approve` → needs_approval at every autonomy level, including 3 and 4; autonomy 0 → deny execution; autonomy 1 → needs_approval for everything; risk > `max_risk_auto` → needs_approval; else allow. If approval is needed and no `approver` is set → treated as denied.

### 3.4 Validator (`calbi/core/validator.py`) — WP-4

```python
class Issue(BaseModel): severity: Literal["error","warning"]; code: str; message: str; step_id: str|None
class ValidationReport(BaseModel): ok: bool; issues: list[Issue]; max_risk: Risk; approvals_needed: list[str]; step_count: int
def validate(pb: Playbook, registry: ToolRegistry, policy: Policy, inputs: dict|None=None) -> ValidationReport
```
Checks, each with a stable `code`: `unknown_tool`, `tool_denied`, `duplicate_step`, `unknown_dependency`, `cycle`, `unknown_step_ref` (template references a step not in `depends_on` transitively), `unknown_input_ref`, `missing_required_input`, `bad_input_type`, `tool_arg_missing`/`tool_arg_unknown` (against `input_schema`), `too_many_steps`, `bad_template`, `needs_approval` (warning), `unreachable_output`.

### 3.5 Runtime (`calbi/core/runtime.py`, `state.py`) — WP-5

```python
class StepStatus(str, Enum): PENDING; RUNNING; SUCCEEDED; FAILED; SKIPPED; DENIED
class RunStatus(str, Enum): PENDING; RUNNING; SUCCEEDED; FAILED; CANCELLED; AWAITING_APPROVAL

class StepState(BaseModel): id; status; attempts:int; output:Any; error:str|None; started_at; ended_at
class RunState(BaseModel): run_id; playbook: Playbook; inputs: dict; status; steps: dict[str, StepState]; outputs: dict; error:str|None

class Runtime:
    def __init__(self, registry, policy, events: EventBus|None=None, max_concurrency: int = 8)
    async def run(self, pb: Playbook, inputs: dict, *, resume_from: RunState|None=None, autonomy: Autonomy|None=None) -> RunState
    async def cancel(self, run_id: str) -> None
```
Behaviour:
- Validates first; refuses to run if `report.ok` is false (`ValidationFailed`).
- Executes ready steps (all deps SUCCEEDED/SKIPPED-allowed) concurrently, bounded by `max_concurrency`.
- `condition` falsy → SKIPPED (dependents that need its output are skipped too; others proceed).
- Per-step retry/backoff/timeout; only retries steps with `idempotent=True` or when the step's own `retry.max_attempts>1` was explicitly set (explicit playbook retry on a non-idempotent tool is a validator warning `retry_non_idempotent`).
- Policy check immediately before each step executes (not only at validation). Approval is awaited via `policy.approver`; denial → step DENIED, treated per `on_failure`.
- `on_failure`: `fail` ends run; `continue` marks failed and continues independent branches; `skip_dependents`; `replan` ends run with status FAILED and `RunState.error` set and a `replan_requested` flag in the final event, for the AI layer to act on.
- Enforces `max_total_attempts` and `max_runtime_s`.
- `resume_from`: reuses SUCCEEDED step outputs from a prior `RunState` whose step ids and `tool`+rendered inputs are unchanged; re-executes the rest.
- Step outputs must be JSON-serializable or the step fails with `UnserializableOutput`.
- Cancellation cancels running tasks and sets CANCELLED.

### 3.6 Engine facade (`calbi/__init__.py`, `calbi/core/engine.py`) — WP-6

```python
class Calbi:
    def __init__(self, llm: LLM | None=None, autonomy: Autonomy | str = Autonomy.CONTROLLED, policy: Policy | None=None, memory: MemoryStore | None=None)
    tool = ...  # decorator -> registry.register
    def register_tools(self, fns) -> None
    events: EventBus
    # deterministic path (no LLM needed)
    async def arun(self, playbook: Playbook | str | dict | Path, inputs: dict|None=None) -> RunState
    def run(...)  # sync wrapper
    # autonomous path (requires llm; defined in WP-8)
    async def adelegate(self, task: str, inputs: dict|None=None, *, constraints: list[str]=(), expected_output: str|None=None) -> DelegationResult
    def delegate(...)  # sync wrapper
```
`autonomy` accepts strings `"advisory"|"supervised"|"controlled"|"autonomous"|"adaptive"`. Sync wrappers raise a clear error if called inside a running event loop.

### 3.7 AI layer (`calbi/ai/`) — WP-7, WP-8

```python
class LLM(Protocol):
    async def complete(self, messages: list[Message], *, response_format: Literal["text","json"]="text", temperature: float=0.0) -> str
@dataclass class Message: role: Literal["system","user","assistant"]; content: str

class Planner:
    def __init__(self, llm, registry, policy, max_attempts: int = 3)
    async def plan(self, task, inputs, constraints, expected_output, hints: list[str]=()) -> PlanResult  # PlanResult(playbook, validation, attempts, rationale)
# Loop: build prompt (tool manifest via registry.describe() only) -> LLM JSON -> parse -> validate -> if errors, feed Issues back to LLM and retry up to max_attempts -> else raise PlanningFailed.

class Verifier:
    async def verify(self, task, expected_output, run: RunState) -> Verdict   # Verdict(passed: bool, reasons: list[str], evidence: list[str], confidence: float)
# Combines deterministic checks (outputs present, no failed required steps) with an LLM judgement. Deterministic failure can't be overridden by the LLM.

class Replanner:
    async def replan(self, task, previous: Playbook, run: RunState, verdict_or_error, attempts_history) -> PlanResult
# Gets failure diagnosis; marks which SUCCEEDED steps are still valid; MUST produce a playbook that reuses valid step ids unchanged where possible so runtime resume_from works. Must not re-propose an approach already recorded as failed in attempts_history.

@dataclass class DelegationResult: status; output: dict; playbook: Playbook|None; run: RunState|None; verdict: Verdict|None; report: ExecutionReport; pending_approvals: list
@dataclass class ExecutionReport: plans: list[Playbook]; runs: list[RunState]; replans: int; events: list[Event]; total_attempts: int; duration_s: float; to_dict(); to_markdown()
```

**Delegation loop (WP-8), exact state machine:**
```
UNDERSTAND/DISCOVER → PLAN → VALIDATE → [autonomy 0: return plan] → [autonomy 1: await approval of plan]
→ EXECUTE → (RunState FAILED w/ replan flag or any failure within max_replans) → REPLAN → VALIDATE → EXECUTE(resume)
→ VERIFY → pass: DELIVER | fail: REPLAN (counts toward max_replans)
Terminal: SUCCEEDED | FAILED(exhausted) | NEEDS_APPROVAL | NEEDS_INPUT | ADVISORY
```
Stop conditions: `max_replans`, `max_runtime_s`, `max_total_attempts`, PlanningFailed, approval denied. Never loops unbounded.

### 3.8 Memory (`calbi/memory/`) — WP-10

```python
class MemoryStore(Protocol):
    def save_run(self, task: str, playbook: Playbook, run: RunState, verdict: Verdict|None) -> str
    def find_similar(self, task: str, limit: int = 3) -> list[PlaybookRecord]   # keyword/BM25-ish via SQLite FTS5, no embeddings required
    def record_failure(self, task: str, playbook_name: str, step_id: str|None, tool: str|None, error: str) -> None
    def tool_limitations(self, tool: str) -> list[str]
    def promote(self, record_id: str, to: Literal["validated","reusable"]) -> None
class SQLiteMemory(MemoryStore): __init__(path: str = ":memory:")
```
Record trust levels: `succeeded` (default after one passing run) → `validated` (≥N passing runs or explicit approval) → `reusable`. **A single run never auto-promotes beyond `succeeded`.** Planner receives only `validated`/`reusable` records as templates plus all recorded failures/limitations as `hints`. Memory is optional; the engine works without it.

### 3.9 Live services (Neon, OpenRouter, NVIDIA NIM) — WP-10b, WP-11

These are **optional adapters outside core**. Offline tests (FakeLLM, in-memory SQLite) remain the default and must always pass without any credentials.

```python
# calbi/ai/adapters/openai_compat.py  (shared base; uses httpx, no vendor SDK)
class OpenAICompatLLM:      # implements LLM
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout_s: float = 60,
                 max_retries: int = 3, extra_headers: dict[str,str]|None = None, json_mode: bool = True)
    async def complete(self, messages, *, response_format="text", temperature=0.0) -> str
# Retries on 429/5xx with exponential backoff + jitter, honours Retry-After. Never logs api_key or full prompts.
# If json_mode is rejected by the model (HTTP 400), falls back to plain text + the planner's JSON extraction.

class OpenRouterLLM(OpenAICompatLLM):   # base_url https://openrouter.ai/api/v1
    def __init__(self, model: str, api_key: str|None=None, **kw)   # key from OPENROUTER_API_KEY
    # optional headers: HTTP-Referer, X-Title from CALBI_APP_URL / CALBI_APP_NAME
class NvidiaNimLLM(OpenAICompatLLM):    # base_url https://integrate.api.nvidia.com/v1
    def __init__(self, model: str, api_key: str|None=None, base_url: str|None=None, **kw)  # key from NVIDIA_API_KEY; base_url override for self-hosted NIM (NVIDIA_NIM_BASE_URL)

def llm_from_env() -> LLM   # CALBI_LLM_PROVIDER=openrouter|nvidia_nim, CALBI_LLM_MODEL=<model id>; raises ConfigError listing missing vars

# calbi/memory/postgres.py  (Neon)
class PostgresMemory(MemoryStore):
    def __init__(self, dsn: str | None = None, *, schema: str = "calbi", pool_size: int = 5)  # dsn from DATABASE_URL
    def migrate(self) -> None        # idempotent, versioned migrations (schema_version table)
```
Rules:
- Neon connection strings need `sslmode=require`; reject DSNs without TLS unless host is `localhost`.
- Driver: `psycopg[binary,pool]` v3 (extra `[postgres]`). Use Neon's **pooled** host (`-pooler`) for the app; migrations may use the direct host (`DATABASE_URL_DIRECT`, optional). Don't depend on session state (no `SET`, no advisory locks held across requests) so PgBouncer transaction pooling works. Handle Neon scale-to-zero cold starts with connect retry (3 tries, backoff).
- Postgres full-text search (`tsvector` + GIN) replaces SQLite FTS5. Same `MemoryStore` contract and same trust-level behaviour.
- Use a dedicated schema (`calbi`) and parameterized queries only. Table names are never built from user input.
- Credentials come only from environment variables or constructor arguments. Never hardcode, never write to events/logs/reports, never commit. Provide `.env.example` with placeholders only.

**Live test policy:** tests marked `@pytest.mark.live` are skipped unless the relevant env vars exist. They never run in the default CI job. A separate manual/scheduled CI job runs them with secrets. Live tests must be cheap (small prompts, `max_tokens` capped, one tiny model) and idempotent (Neon tests use a throwaway schema `calbi_test_<random>` dropped in teardown).

## 4. Work packages

Each WP lists: Owns · Depends on · Tasks · Acceptance tests (must exist and pass).

### WP-0 Scaffold
**Owns:** `pyproject.toml`, `src/calbi/__init__.py` (stub), `src/calbi/testing/fake_llm.py`, `tests/conftest.py`, `.gitignore`, `.github/workflows/ci.yml`, `Makefile`
- Packaging with `hatchling`; extras `ai`, `memory`, `dev`. Core deps: `pydantic>=2`, `pyyaml`.
- `FakeLLM(responses: list[str|Callable])`: returns scripted outputs in order, records all calls in `.calls`, raises if exhausted.
- CI: ruff, mypy, pytest on 3.10–3.13.
- **Accept:** `pip install -e ".[dev]"` works; `pytest` runs (0 tests OK); `import calbi` works; FakeLLM unit test.

### WP-1 Schema
**Owns:** `core/schema.py`, `tests/core/test_schema.py`
- Implement §3.1. Validate step id regex, unique ids, non-empty steps, `max_attempts>=1`.
- **Accept:** YAML round-trip equality; the README example parses; invalid id/duplicate id/empty steps/unknown field each raise `SchemaError`; extra fields forbidden (`extra="forbid"`).

### WP-2 Templating
**Owns:** `core/templating.py`, `tests/core/test_templating.py`
- Implement §3.2 with `ast`-walk whitelist.
- **Accept:** typed passthrough (`"{{ inputs.n }}"` → int); embedded → str; nested dict/list rendering; each forbidden construct (`__import__`, `().__class__`, lambda, comprehension, arbitrary call, `open`) raises `TemplateError`; missing key error names the path; `referenced_steps` correct; fuzz test of 50 hostile strings never executes anything.

### WP-3 Registry, events, policy
**Owns:** `core/registry.py`, `core/events.py`, `core/policy.py`, `tests/core/test_registry.py`, `test_policy.py`, `test_events.py`
- Implement §3.3. Derive `input_schema` from type hints (via `pydantic.TypeAdapter`/`create_model`); support sync and async.
- **Accept:** decorator and call forms; duplicate error; `describe()` JSON-serializable and contains no callables; policy decision table tested for every (autonomy × risk × requires_approval × allow/deny list) combination; approval-required at autonomy 3 and 4 asserted explicitly; `to_jsonl` round-trips.

### WP-4 Validator
**Owns:** `core/validator.py`, `tests/core/test_validator.py`
- Implement §3.4. Cycle detection via Kahn's algorithm with the cycle path in the message.
- **Accept:** one test per issue `code`; valid README-style playbook returns `ok=True`; `max_risk` and `approvals_needed` computed correctly; step referencing an output without a dependency is `unknown_step_ref`.

### WP-5 Runtime
**Owns:** `core/runtime.py`, `core/state.py`, `tests/core/test_runtime_*.py`
- Implement §3.5.
- **Accept (each is a test):** linear chain passes outputs; diamond DAG runs the two middle steps concurrently (assert via timing/overlap with `asyncio.Event`); concurrency cap respected; condition false → SKIPPED and dependents handled; retry with backoff succeeds on the 3rd attempt (fake clock/`sleep` injected); timeout kills a step; `on_failure` each of 4 modes; policy denial of a destructive tool at autonomy 3 without approver → DENIED; with approver returning True → runs; approver called exactly once per step; `max_total_attempts` and `max_runtime_s` enforced; cancel mid-run; resume reuses unchanged succeeded steps (assert the tool call count) and re-runs changed ones; non-serializable output fails the step; sync tools do not block the event loop.

### WP-6 Engine facade (core path)
**Owns:** `core/engine.py`, `src/calbi/__init__.py`, `tests/core/test_engine.py`
- Implement the §3.6 deterministic path and exports: `Calbi, Autonomy, Risk, Policy, Playbook, load_playbook, RunState`.
- **Accept:** the README snippet minus `delegate` works as a test; string autonomy parsing; sync wrapper inside a running loop raises a clear error; `calbi.core` imports with `ai`/`memory` extras not installed.

### WP-7 AI layer: planner, verifier, replanner
**Owns:** `ai/llm.py`, `ai/planner.py`, `ai/verifier.py`, `ai/replanner.py`, `ai/prompts.py`, `tests/ai/`
- Implement §3.7. Prompts are plain-text templates in `prompts.py` with a documented JSON output contract (the playbook object). JSON extraction must tolerate code fences and surrounding prose.
- Planner prompt includes: task, inputs (names and types, **values truncated**), constraints, tool manifest, the schema description, template syntax rules, hints. It must tell the model that only listed tools exist and that playbooks that use others are rejected.
- **Accept (FakeLLM):** valid plan accepted first try; invalid-then-valid triggers one feedback retry and the feedback contains the validator issue codes; three invalid → `PlanningFailed`; plan using an unregistered tool never reaches runtime; a prompt-injection string inside `inputs` does not change the tool manifest; verifier: deterministic failure cannot be overridden by an LLM "pass"; replanner output reuses valid step ids; replanner given a failed-approach history does not return an identical playbook (else raises `ReplanFailed`).

### WP-8 Delegation loop
**Owns:** `ai/delegate.py`, `ai/report.py`, `core/engine.py` (add `adelegate`/`delegate` only), `tests/ai/test_delegate.py`
- Implement the §3.7 state machine and `DelegationResult`/`ExecutionReport`.
- **Accept:** autonomy 0 returns plan, nothing executed (assert tool not called); autonomy 1 pauses with `NEEDS_APPROVAL` until approved; happy path end to end; step 3 of 5 fails → replan → resume re-uses steps 1–2 (assert call counts) → succeeds; verify fail → replan → succeeds; `max_replans` exhausted → `FAILED` with a full report; destructive tool at autonomy 3 → `NEEDS_APPROVAL`, never executed without the approver; total LLM calls bounded; report `to_markdown()` includes plans, replans, and evidence.

### WP-9 End-to-end, safety, docs
**Owns:** `tests/e2e/`, `tests/test_architecture.py`, `examples/`, `docs/`
- E2E: the dataset scenario (inspect→validate→analyze→report→verify) with scripted FakeLLM, including a failed first strategy and a successful alternative.
- Safety suite: (a) model emits a playbook referencing `os.system`, a shell tool name, or a path-traversal input → rejected; (b) template injection attempts; (c) an attempt to set `requires_approval`/`policy` fields inside the playbook → rejected by `extra="forbid"`; (d) hostile tool output containing instructions doesn't alter the policy; (e) architecture test: static import check that `calbi.core` imports nothing from `calbi.ai` or `calbi.memory`.
- Docs: `docs/quickstart.md`, `docs/writing-tools.md`, `docs/safety-model.md`, `docs/playbook-reference.md`; runnable `examples/dataset_report.py` using FakeLLM.
- **Accept:** all suites green on CI; examples run offline.

### WP-10 Memory (parallel after WP-6)
**Owns:** `memory/`, `tests/memory/`, integration hook in `ai/planner.py` via `hints`/`templates` params only
- Implement §3.8 using stdlib `sqlite3` + FTS5 (fallback to `LIKE` if FTS5 is unavailable).
- **Accept:** save/find round-trip; single success stays `succeeded` and is **not** returned as a template; promoting with the N-run rule works; failures and tool limitations are appended to planner hints in an integration test; the engine works with `memory=None`; DB schema has a `schema_version` table.

### WP-10b Neon (Postgres) memory backend (after WP-10)
**Owns:** `memory/postgres.py`, `memory/migrations/`, `tests/memory/test_postgres_*.py`, `.env.example` (DB lines)
- Implement §3.9 `PostgresMemory`, sharing a contract test suite with `SQLiteMemory` (parametrize one suite over both backends).
- **Accept (offline):** contract suite passes on SQLite; DSN validation (no `sslmode`, wrong scheme) raises `ConfigError`; migration SQL is idempotent (lint by running twice in the live test); no f-string SQL (grep test).
- **Accept (`live`, needs `DATABASE_URL`):** same contract suite passes on Neon in a throwaway schema; migrate twice is a no-op; cold-start retry exercised by unit test with a fake failing connection; trust-level promotion rule identical to SQLite.

### WP-11 LLM adapters: OpenRouter and NVIDIA NIM (after WP-7)
**Owns:** `ai/adapters/`, `tests/ai/test_adapters.py`, `tests/live/`, `.env.example` (LLM lines)
- Implement §3.9 with `httpx.AsyncClient`. No vendor SDKs.
- **Accept (offline, using `httpx.MockTransport`):** request has correct URL, `Authorization: Bearer`, model, messages, and `response_format` when `json_mode`; 429 then 200 retries and honours `Retry-After`; 400 on json_mode falls back to text; 5xx exhausted raises `LLMError`; the api key never appears in exceptions, logs or `repr()`; `llm_from_env` errors name the missing variables; NIM `base_url` override works.
- **Accept (`live`):** with `OPENROUTER_API_KEY` or `NVIDIA_API_KEY` set, (1) a "reply with the word OK" smoke call returns text; (2) the real planner produces a **valid** playbook for the dataset scenario using registered stub tools within 3 attempts; (3) the full `adelegate` e2e passes against a real model. Because real models are nondeterministic, assert on **validity and outcome** (validator ok, tool calls happened, verifier passed), never on exact text. Record the model id and pass rate in `docs/model-compat.md`.

## 5. Safety model (non-negotiable)

1. The model never receives function objects, only `registry.describe()`.
2. The model's output is parsed as data into `Playbook` (`extra="forbid"`); it has no fields that grant permission.
3. The validator runs before execution **and** the policy runs again immediately before each step.
4. `always_approve` risks need approval at every autonomy level, including 4.
5. Hard budgets: steps, replans, attempts, runtime. All are enforced in code, not in the prompt.
6. Untrusted data (inputs, tool outputs) is placed in delimited data blocks in prompts and never in the system role.
7. Secrets: tools receive their own credentials from the host. Event logs record step inputs/outputs through an optional `redact` hook (default: truncate long values at 2 KB).
8. Sandboxing hook: `Runtime(executor=...)` is a documented extension point for process or container isolation. It is not built in V0.1.

## 6. Quality gates (CI)

- `ruff check`, `ruff format --check`, `mypy --strict src`
- Default job: `pytest -q -m "not live"` (no secrets, no network)
- Live job (manual dispatch/nightly, uses repo secrets `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `DATABASE_URL`): `pytest -m live`
- Secret scan (e.g. `gitleaks`) on every push; `.env` is gitignored
- `pytest -q` with coverage ≥ 90% on `core`, ≥ 85% on `ai` and `memory`
- Architecture test (core independence)
- Examples executed in CI

## 7. V0.1 definition of done

- [x] `pip install calbi` gives the deterministic engine with no AI dependency
- [x] The README developer-experience snippet works offline with FakeLLM
- [x] Dataset e2e scenario passes including replan and resume
- [x] Safety suite passes
- [x] Autonomy levels 0–3 behave per the table
- [x] Docs and examples present, CI green

## 8. Phase 2 (V0.2) — Parallel Feature Branch Plan

V0.1 is complete on `main`. Phase 2 (V0.2) is architected so **6 agents can work concurrently on separate `feat/*` branches with zero file collisions**.

> **Status (2026-11-06): Wave 1 DONE.** WP-12..WP-17 merged to `main` via PRs #1–#6 (`db7f512` → `c8cf8f7`), zero conflicts. `main`: 235 offline tests pass, `ruff check`, `ruff format --check`, `mypy src` clean, CI green on Python 3.10–3.13. The new modules are **not yet wired** into `Calbi` / `Runtime` / `delegate_loop` — see §8.5–§8.7 for what's next.

### 8.1 Branching & Collision-Free Rules

1. **One Branch per Feature**: Every agent must branch directly off `main`:
   ```bash
   git checkout main && git pull origin main
   git checkout -b <branch-name>
   ```
2. **Strict Disjoint File Ownership (Wave 1)**:
   - In **Wave 1**, no two branches touch the same file.
   - Wave 1 branches **MUST NOT** edit shared wiring files (`src/calbi/__init__.py`, `src/calbi/core/engine.py`, `src/calbi/core/runtime.py`, `src/calbi/core/schema.py`, `src/calbi/ai/delegate.py`, `pyproject.toml`).
   - Instead, each Wave 1 branch builds its module as a self-contained unit tested against existing V0.1 public interfaces (`Playbook`, `Runtime`, `ToolRegistry`, `EventBus`, `MemoryStore`, `Planner`, `Verifier`).
3. **Wave 2 Wiring (`feat/v0.2-integration`)**:
   - Once the Wave 1 PRs are merged into `main`, a single short Wave 2 branch wires the new modules into `Calbi`, `Runtime`, `delegate_loop`, and `__init__.py`.

```
                         ┌─► feat/adaptive-strategies     (WP-12) ─┐
                         ├─► feat/self-improvement        (WP-13) ─┤
                         ├─► feat/sandbox-executor        (WP-14) ─┤
main (v0.1.0 complete) ──┼─► feat/observability-redaction (WP-15) ─┼──► Merge to main ──► feat/v0.2-integration (WP-18)
                         ├─► feat/subplaybooks-foreach    (WP-16) ─┤
                         └─► feat/live-benchmarks         (WP-17) ─┘
```

### 8.2 File Ownership Matrix (Wave 1 — Zero Overlap)

| Branch | WP | Exclusive Code Files Owned | Exclusive Test & Doc Files Owned | Shared Files Touched |
| --- | --- | --- | --- | --- |
| `feat/adaptive-strategies` | WP-12 | `src/calbi/ai/strategies.py` | `tests/ai/test_strategies.py`, `docs/adaptive-strategies.md` | **None** |
| `feat/self-improvement` | WP-13 | `src/calbi/memory/improvement.py` | `tests/memory/test_improvement.py`, `docs/self-improvement.md` | **None** |
| `feat/sandbox-executor` | WP-14 | `src/calbi/core/executor.py` | `tests/core/test_executor.py`, `docs/sandboxing.md` | **None** |
| `feat/observability-redaction` | WP-15 | `src/calbi/core/observability.py` | `tests/core/test_observability.py`, `docs/observability.md` | **None** |
| `feat/subplaybooks-foreach` | WP-16 | `src/calbi/core/composition.py` | `tests/core/test_composition.py`, `docs/composition.md` | **None** |
| `feat/live-benchmarks` | WP-17 | `examples/live_neon_openrouter.py` | `tests/live/test_live_e2e_delegation.py`, `tests/memory/test_memory_contract.py`, `docs/model-compat.md` | **None** |

---

### 8.3 Wave 1 Work Packages & Contracts

#### WP-12 — Autonomy Level 4: Adaptive Strategy Experimentation
- **Branch**: `feat/adaptive-strategies`
- **Owns**: `src/calbi/ai/strategies.py`, `tests/ai/test_strategies.py`, `docs/adaptive-strategies.md`
- **Contract**:
  ```python
  @dataclass
  class StrategyCandidate:
      name: str
      rationale: str
      playbook: Playbook
      validation: ValidationReport


  @dataclass
  class StrategyTrial:
      candidate: StrategyCandidate
      run: RunState | None
      verdict: Verdict | None
      score: float  # 0.0..1.0 composite of verdict.confidence, step efficiency, risk


  @dataclass
  class StrategyExplorationResult:
      winner: StrategyTrial | None
      trials: list[StrategyTrial]
      total_attempts: int


  class StrategyExplorationFailed(CalbiError): ...


  class StrategyExplorer:
      def __init__(
          self,
          llm: LLM,
          registry: ToolRegistry,
          policy: Policy,
          runtime: Runtime | None = None,
          verifier: Verifier | None = None,
          max_candidates: int = 3,
      ) -> None: ...

      async def generate_candidates(
          self,
          task: str,
          inputs: dict[str, Any] | None = None,
          constraints: Sequence[str] = (),
          k: int = 3,
      ) -> list[StrategyCandidate]: ...

      async def explore(
          self,
          task: str,
          inputs: dict[str, Any] | None = None,
          constraints: Sequence[str] = (),
          expected_output: str | None = None,
          *,
          mode: Literal["sequential", "parallel_read_only"] = "sequential",
      ) -> StrategyExplorationResult: ...
  ```
- **Behaviour**:
  - `generate_candidates` prompts the LLM to produce `k` distinct strategies (different tool chains or parameters; structural signatures must differ). Each candidate is validated via `validate(...)`; invalid candidates are discarded.
  - `explore(..., mode="sequential")` tries candidate 1, verifies it, and stops early if `verdict.passed` with `confidence >= 0.8`; otherwise tries candidate 2, etc., and picks the highest-scoring passing trial.
  - `explore(..., mode="parallel_read_only")` runs candidates concurrently **only** for candidates whose `validation.max_risk == Risk.READ` (any candidate with `max_risk > Risk.READ` is rejected from parallel dry-run to prevent side-effects).
  - Never bypasses `Policy`: all candidate runs execute through `Runtime` with `policy` enforced.
- **Accept (`tests/ai/test_strategies.py`)**:
  - Generates 3 distinct valid candidates with `FakeLLM`; duplicate candidate signatures are deduplicated.
  - Sequential mode: strategy A fails, strategy B passes → winner is B, strategy C is not executed.
  - Parallel read-only mode: runs two `Risk.READ` candidates concurrently and picks the higher confidence winner.
  - Candidate with `Risk.WRITE` or `Risk.DESTRUCTIVE` in `parallel_read_only` mode is skipped/refused before execution.

---

#### WP-13 — Self-Improvement & Playbook Promotion Proposals
- **Branch**: `feat/self-improvement`
- **Owns**: `src/calbi/memory/improvement.py`, `tests/memory/test_improvement.py`, `docs/self-improvement.md`
- **Contract**:
  ```python
  class ProposalStatus(str, Enum):
      PENDING = "pending"
      APPROVED = "approved"
      REJECTED = "rejected"


  class ImprovementProposal(BaseModel):
      proposal_id: str
      task: str
      baseline_playbook: Playbook | None
      proposed_playbook: Playbook
      rationale: str
      evidence_run_ids: list[str]
      steps_saved: int
      risk_delta: int  # negative = lower risk, 0 = same, positive = higher risk
      status: ProposalStatus = ProposalStatus.PENDING
      created_at: datetime


  class ImprovementManager:
      def __init__(
          self, memory: MemoryStore, llm: LLM | None = None, min_runs_for_reusable: int = 5
      ) -> None: ...
      def evaluate_improvement(
          self,
          task: str,
          baseline: Playbook | None,
          candidate: Playbook,
          run: RunState,
          verdict: Verdict,
      ) -> ImprovementProposal | None: ...
      def list_proposals(self, status: ProposalStatus | None = None) -> list[ImprovementProposal]: ...
      def approve(self, proposal_id: str) -> PlaybookRecord: ...
      def reject(self, proposal_id: str, reason: str = "") -> ImprovementProposal: ...
  ```
- **Behaviour**:
  - Distinguishes **succeeded** (1 run) vs **validated** (≥3 runs) vs **reusable** (approved proposal or ≥`min_runs_for_reusable` with 0 failures).
  - `evaluate_improvement` creates a `PENDING` `ImprovementProposal` when a succeeded+verified run either: (a) succeeded after a replan where the initial baseline failed, or (b) achieved the objective in fewer steps or strictly lower `max_risk` than an existing `validated` playbook for the same task.
  - **One run never auto-promotes to `reusable`** without an explicit `approve(proposal_id)` call.
  - `approve(proposal_id)` saves/promotes the playbook to `"reusable"` in `MemoryStore` and marks the proposal `APPROVED`.
- **Accept (`tests/memory/test_improvement.py`)**:
  - Single successful run does not become `reusable` automatically.
  - Replanned run that succeeded creates a `PENDING` proposal recording the baseline and proposed playbook.
  - Faster/lower-risk playbook creates a proposal with accurate `steps_saved` and `risk_delta`.
  - `approve()` promotes the record in `SQLiteMemory` to `"reusable"` so `find_similar()` returns it; `reject()` leaves it unpromoted.

---

#### WP-14 — Sandboxed Step Executor (In-Process & Subprocess Isolation)
- **Branch**: `feat/sandbox-executor`
- **Owns**: `src/calbi/core/executor.py`, `tests/core/test_executor.py`, `docs/sandboxing.md`
- **Contract**:
  ```python
  class ExecutionIsolationError(CalbiError): ...


  class StepExecutor(Protocol):
      async def execute(
          self,
          spec: ToolSpec,
          inputs: dict[str, Any],
          *,
          timeout_s: float | None = None,
      ) -> Any: ...


  class InProcessExecutor:
      """Default executor running async tools on the loop and sync tools via asyncio.to_thread."""

      async def execute(
          self, spec: ToolSpec, inputs: dict[str, Any], *, timeout_s: float | None = None
      ) -> Any: ...


  class SubprocessExecutor:
      """Runs top-level importable tool functions in an isolated child Python process."""

      def __init__(
          self,
          *,
          allowed_env_keys: set[str] = frozenset({"PATH", "LANG", "LC_ALL", "SYSTEMROOT"}),
          max_output_bytes: int = 1_048_576,
      ) -> None: ...
      async def execute(
          self, spec: ToolSpec, inputs: dict[str, Any], *, timeout_s: float | None = None
      ) -> Any: ...
  ```
- **Behaviour**:
  - `calbi.core` rule: must not import `calbi.ai` or `calbi.memory`.
  - `InProcessExecutor` matches current runtime behavior (async coroutine or `asyncio.to_thread`), enforcing `timeout_s` and JSON serializability.
  - `SubprocessExecutor` spawns `sys.executable -c ...` via `asyncio.create_subprocess_exec`, passes JSON-serialized `inputs` over `stdin`, reads JSON result from `stdout`, and **scrubs `os.environ`** so secrets (`OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `DATABASE_URL`, etc.) are NOT leaked to child tool processes unless listed in `allowed_env_keys`.
  - If the child process exceeds `timeout_s`, kills the process (`proc.kill()`) and raises `TimeoutError`.
  - If the tool is a local closure/lambda (not importable by module+qualname), raises `ExecutionIsolationError` with a clear message.
- **Accept (`tests/core/test_executor.py`)**:
  - `InProcessExecutor` runs sync and async tools and enforces timeout.
  - `SubprocessExecutor` executes a top-level tool function in a separate PID (assert `os.getpid()` returned by tool != parent `os.getpid()`).
  - Secret env var set in parent process (`os.environ["SECRET_KEY"] = "xyz"`) is absent inside `SubprocessExecutor` child process.
  - Infinite sleep in `SubprocessExecutor` is killed on timeout without hanging test suite.
  - Non-JSON output or non-zero exit raises `ExecutionIsolationError`.

---

#### WP-15 — Event Redaction, Secret Scrubbing & Observability Sinks
- **Branch**: `feat/observability-redaction`
- **Owns**: `src/calbi/core/observability.py`, `tests/core/test_observability.py`, `docs/observability.md`
- **Contract**:
  ```python
  class Redactor:
      def __init__(
          self,
          *,
          max_string_bytes: int = 2048,
          sensitive_keys: set[str] = frozenset(
              {"api_key", "password", "secret", "token", "authorization", "dsn", "database_url"}
          ),
          extra_patterns: Sequence[re.Pattern[str]] = (),
      ) -> None: ...
      def redact(self, value: Any) -> Any: ...
      def redact_event(self, event: Event) -> Event: ...


  class JSONLEventSink:
      """Subscriber that writes redacted JSONL events to a file path or TextIO stream."""

      def __init__(self, target: Path | TextIO, redactor: Redactor | None = None) -> None: ...
      def __call__(self, event: Event) -> None: ...


  class SpanRecord(BaseModel):
      trace_id: str
      span_id: str
      parent_span_id: str | None
      name: str
      start_ts: datetime
      end_ts: datetime | None
      status: Literal["ok", "error", "in_progress"]
      attributes: dict[str, Any]


  class TraceCollector:
      """Builds hierarchical run -> step spans from EventBus events (OTel-compatible structure, zero extra deps)."""

      def __init__(self, redactor: Redactor | None = None) -> None: ...
      def __call__(self, event: Event) -> None: ...
      @property
      def spans(self) -> list[SpanRecord]: ...
  ```
- **Behaviour**:
  - `Redactor.redact` recursively walks dicts/lists/strings:
    - Any dict key matching `sensitive_keys` (case-insensitive substring match for `secret`, `token`, `password`, `api_key`, `dsn`) has its value replaced with `"[REDACTED]"`.
    - Scrubs known inline patterns in strings: Bearer tokens (`Bearer [A-Za-z0-9._-]+`), Postgres DSN passwords (`postgresql://user:pass@` → `postgresql://user:[REDACTED]@`), `sk-...`, `nvapi-...`.
    - Truncates strings longer than `max_string_bytes` (default 2048 bytes per §5.7) with `"... [truncated]"`.
  - `JSONLEventSink` and `TraceCollector` attach directly to `EventBus.subscribe(...)`.
- **Accept (`tests/core/test_observability.py`)**:
  - Dicts with `api_key`, `DATABASE_URL`, Bearer headers, and `postgresql://user:secret@host/db` are scrubbed in `redact_event`.
  - Strings > 2 KB are truncated to 2 KB + suffix.
  - `JSONLEventSink` subscribed to `EventBus` writes valid redacted JSONL lines to `tmp_path`.
  - `TraceCollector` pairs `run.started`/`run.completed` and `step.started`/`step.completed` into parent-child `SpanRecord` objects.

---

#### WP-16 — Sub-Playbooks & Bounded `for_each` Composition
- **Branch**: `feat/subplaybooks-foreach`
- **Owns**: `src/calbi/core/composition.py`, `tests/core/test_composition.py`, `docs/composition.md`
- **Contract**:
  ```python
  class SubPlaybookError(CalbiError): ...


  def register_subplaybook(
      registry: ToolRegistry,
      playbook: Playbook,
      runtime: Runtime,
      *,
      name: str | None = None,
      description: str | None = None,
      max_depth: int = 3,
  ) -> ToolSpec:
      """Register a Playbook as a reusable tool in ToolRegistry."""


  def register_map_tool(
      registry: ToolRegistry,
      base_tool_name: str,
      *,
      name: str | None = None,
      items_param: str = "items",
      item_arg: str,
      max_items: int = 50,
      max_concurrency: int = 8,
  ) -> ToolSpec:
      """Register a batched for-each tool that maps base_tool over a list of items concurrently."""
  ```
- **Behaviour**:
  - `register_subplaybook` computes the effective `risk` of the sub-playbook as the maximum `Risk` of all steps in `playbook`, derives `input_schema` from `playbook.inputs`, and registers an async tool that runs `runtime.run(playbook, inputs=...)` and returns `run_state.outputs` (raising `SubPlaybookError` if the sub-playbook fails or recursion exceeds `max_depth`).
  - `register_map_tool` wraps an existing registered tool (`base_tool_name`) into a new tool that accepts `items: list[Any]` (plus any static kwargs), enforces `len(items) <= max_items`, runs `base_tool` concurrently bounded by `max_concurrency`, and returns `{"results": [...], "count": len(items)}`. Inherits `risk` and `requires_approval` from `base_tool`.
- **Accept (`tests/core/test_composition.py`)**:
  - Sub-playbook registered as a tool can be invoked as a step inside a parent playbook; outputs pass cleanly to downstream parent steps.
  - Sub-playbook containing a `Risk.DESTRUCTIVE` step automatically gets `risk=Risk.DESTRUCTIVE` on its registered `ToolSpec`.
  - Recursive sub-playbook exceeding `max_depth` raises `SubPlaybookError`.
  - `register_map_tool` processes a list of 5 items concurrently and rejects lists exceeding `max_items`.

---

#### WP-17 — Live Service E2E Suite & Shared Memory Contract Tests
- **Branch**: `feat/live-benchmarks`
- **Owns**: `tests/memory/test_memory_contract.py`, `tests/live/test_live_e2e_delegation.py`, `examples/live_neon_openrouter.py`, `docs/model-compat.md`
- **Tasks**:
  1. Create a parametrized contract test suite in `tests/memory/test_memory_contract.py` that runs all `MemoryStore` invariants (`save_run`, single-run stays `succeeded`, 3-run auto-promotion to `validated`, `promote` to `reusable`, `find_similar`, `record_failure`, `tool_limitations`) against `SQLiteMemory` offline and against `PostgresMemory` (in a throwaway schema `calbi_test_<uuid>`) when `DATABASE_URL` is set.
  2. Create `tests/live/test_live_e2e_delegation.py` (marked `@pytest.mark.live`):
     - Tests real `Planner.plan` on OpenRouter and/or NVIDIA NIM for the dataset scenario, asserting `report.ok is True` within 3 attempts.
     - Tests full `engine.adelegate` end-to-end with replanning (first tool raises an error, model replans to fallback tool and verifies).
  3. Create `examples/live_neon_openrouter.py` demonstrating `llm_from_env()` + `PostgresMemory` with graceful fallback to `FakeLLM` + `SQLiteMemory` when env vars are unset (so `python examples/live_neon_openrouter.py` works both offline in CI and live with `.env`).
- **Accept**:
  - `pytest -m "not live"` runs the full `test_memory_contract.py` against SQLite and passes.
  - `python examples/live_neon_openrouter.py` runs cleanly offline without env vars.
  - Live tests skip cleanly when `OPENROUTER_API_KEY` / `NVIDIA_API_KEY` / `DATABASE_URL` are absent, and pass when present.

---

### 8.4 Wave 1 Post-Merge Review (2026-11-06)

Tests pass, but a review of the merged code found gaps that the Wave 1 tests don't cover. Fix these **before** wiring the modules into the engine (§8.6), so integration builds on hardened pieces.

| # | Module | Finding | Severity |
| --- | --- | --- | --- |
| R1 | `core/composition.py` | `register_subplaybook` silently skips steps whose tool isn't registered yet (`except UnknownToolError: pass`), so the computed risk can be **lower than the real risk** → a destructive step can be laundered behind a `READ` wrapper. Must fail closed. | High |
| R2 | `core/composition.py` | `register_map_tool` calls `base_spec.fn` directly, bypassing runtime policy, per-step timeout, and (later) the executor. | High |
| R3 | `core/composition.py` | Writes to private `registry._tools`. Needs a public `ToolRegistry` API. | Medium |
| R4 | `core/executor.py` | `SubprocessExecutor` uses `proc.communicate()`, which buffers **all** stdout before the `max_output_bytes` check → a misbehaving tool can exhaust parent memory. Tool `print()` output also shares the JSON result channel and corrupts it. | High |
| R5 | `core/executor.py`, `docs/sandboxing.md` | Docs should say clearly this is *process + env isolation*, not a security sandbox (same user, filesystem, network). | Medium |
| R6 | `core/observability.py` | Key matching is substring-based, so `max_tokens`, `tokens_used`, `token_count` get redacted (false positives that hide useful telemetry). | Medium |
| R7 | `core/observability.py` | `TraceCollector` keys step spans by `(run_id, step_id)`; on retries the earlier attempt's span is orphaned in `in_progress` forever. | Low |
| R8 | `memory/improvement.py` | Proposals live only in an in-process dict → lost on restart; cannot be approved from another process. | High |
| R9 | `memory/improvement.py` | Optimization trigger compares against `find_similar(task)[0]`, which may be a *different* task's playbook (FTS similarity), producing misleading `steps_saved`. | Medium |
| R10 | `ai/strategies.py` | `generate_candidates` hard-codes `expected_output` in the prompt instead of passing the caller's value; uses private `Risk._rank`. | Low |
| R11 | CI | No `fail-fast: false` (one failed matrix job cancels others, as seen on PR #3); Node 20 deprecation for `actions/checkout@v4` / `setup-python@v5`; the "separate live CI job" from the decision log doesn't exist; `mypy` runs on `src` only; `asyncio_default_fixture_loop_scope` unset (pytest-asyncio warning). | Medium |
| R12 | Repo | Merged `feat/*` branches still on `origin`; version is still `0.1.0`. | Low |

### 8.5 Wave 2A — Hardening (parallel, zero file overlap)

Same rules as §8.1: branch off `main`, touch only owned files, one PR per branch, CI green before merge. Hub files (`__init__.py`, `engine.py`, `runtime.py`, `schema.py`, `delegate.py`) stay untouched until Wave 2B.

| Branch | WP | Fixes | Owns |
| --- | --- | --- | --- |
| `fix/composition-hardening` | WP-19 | R1, R2, R3 | `src/calbi/core/composition.py`, `src/calbi/core/registry.py`, `tests/core/test_composition.py`, `tests/core/test_registry.py`, `docs/composition.md` |
| `fix/executor-hardening` | WP-20 | R4, R5 | `src/calbi/core/executor.py`, `tests/core/test_executor.py`, `docs/sandboxing.md` |
| `fix/observability-hardening` | WP-21 | R6, R7 | `src/calbi/core/observability.py`, `tests/core/test_observability.py`, `docs/observability.md` |
| `fix/improvement-persistence` | WP-22 | R8, R9 | `src/calbi/memory/improvement.py`, `src/calbi/memory/base.py`, `src/calbi/memory/sqlite.py`, `src/calbi/memory/postgres.py`, `tests/memory/test_improvement.py`, `tests/memory/test_memory_contract.py`, `docs/self-improvement.md` |
| `fix/strategies-polish` | WP-23 | R10 | `src/calbi/ai/strategies.py`, `tests/ai/test_strategies.py` |
| `chore/ci-hygiene` | WP-24 | R11 | `.github/workflows/ci.yml`, `pyproject.toml` (`[tool.pytest]` / `[tool.mypy]` sections only), `tests/memory/test_postgres_memory.py` |

**Acceptance per WP:**
- **WP-19**: Add public `ToolRegistry.add_spec(spec)` (raises `DuplicateToolError`); composition never touches `_tools`. Registering a sub-playbook that references an unregistered tool raises `SubPlaybookError` (fail closed). The map tool runs each item through policy (`Policy.decide`) and honors the base tool's timeout; a `DESTRUCTIVE` base tool under `Autonomy.AUTONOMOUS` is refused, same as a direct step.
- **WP-20**: Read child stdout incrementally; kill the child and raise `ExecutionIsolationError` as soon as `max_output_bytes` is exceeded (test: a tool emitting 50 MB is killed; parent RSS stays bounded). The result travels on a dedicated channel (e.g. a length-prefixed frame or a separate pipe), so `print()` inside a tool doesn't break parsing. `docs/sandboxing.md` gets a "Threat model / non-goals" section.
- **WP-21**: Sensitive-key matching uses exact names plus delimiter-aware tokens (`api_key`, `x-api-key`, `db_password` match; `max_tokens`, `tokens_used` don't). Retry attempts produce one span per attempt; superseded attempts close with `status="error"`.
- **WP-22**: Add `save_proposal` / `get_proposal` / `list_proposals` / `update_proposal_status` to `MemoryStore` (SQLite + Postgres, with migration); `ImprovementManager` persists through it. Contract tests cover proposal round-trip on both backends. The optimization trigger only compares against records whose normalized task matches.
- **WP-23**: The caller's `expected_output` reaches the prompt; scoring uses public `Risk` comparisons only.
- **WP-24**: `fail-fast: false`; actions bumped to Node 24-compatible majors; a `live` job runs on `workflow_dispatch` / nightly using repo secrets (`OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `DATABASE_URL`) and skips cleanly when absent; `mypy src tests` clean; `asyncio_default_fixture_loop_scope = "function"` set.

> [!NOTE]
> WP-22 is the only package that touches the memory backends, and WP-24 fixes the `postgres.time` mypy issue from the **test** side, so the two don't collide.

### 8.6 Wave 2B — Integration (after all of Wave 2A merges)

#### WP-18 — V0.2 Facade & Runtime Wiring
- **Branch**: `feat/v0.2-integration` (branch from `main` **only after** WP-19..WP-24 are merged)
- **Owns**: `src/calbi/__init__.py`, `src/calbi/core/runtime.py`, `src/calbi/core/engine.py`, `src/calbi/core/events.py`, `src/calbi/ai/delegate.py`, `tests/e2e/test_v02_integrated.py`, `README.md`
- **Tasks**:
  1. Add `executor: StepExecutor | None = None` to `Runtime.__init__` (default `InProcessExecutor()`); runtime delegates tool invocation (including timeout) to it.
  2. Add `redactor: Redactor | None = None` to `EventBus.__init__` so emitted events are redacted before reaching any subscriber.
  3. In `delegate_loop`: when `autonomy == Autonomy.ADAPTIVE`, use `StrategyExplorer` for initial planning and replanning; after a verified run with `engine.memory` present, call `ImprovementManager.evaluate_improvement`.
  4. Expose `Calbi(executor=..., redactor=..., trace=...)` and `engine.proposals` (list / approve / reject).
  5. Export new public symbols from `calbi` (`StepExecutor`, `InProcessExecutor`, `SubprocessExecutor`, `Redactor`, `JSONLEventSink`, `TraceCollector`, `register_subplaybook`, `register_map_tool`, `StrategyExplorer`, `ImprovementManager`).
- **Accept (`tests/e2e/test_v02_integrated.py`)**:
  - Full e2e combining `Autonomy.ADAPTIVE`, `SubprocessExecutor`, `Redactor`, `SQLiteMemory`, and `ImprovementManager`: strategy tournament picks a winner, run executes in a child process, events on the bus contain no secrets, a proposal is persisted, and `approve()` makes it retrievable by `find_similar`.
  - Existing 235 tests still pass unchanged.

### 8.7 Release & Later

1. **V0.2.0 release** (`chore/release-0.2.0`, after WP-18): bump `pyproject.toml` version, add `CHANGELOG.md`, tag `v0.2.0`, delete merged `feat/*` / `fix/*` branches.
2. **V0.3 candidates** (unscheduled; each needs a plan entry before work starts):
   - Embedding-based retrieval in memory (optional `[embeddings]` extra), keeping FTS5 as default.
   - Token / cost budgets per run, enforced by the runtime alongside step limits.
   - Approval callbacks (webhook / async hook) so `requires_approval` steps can pause and resume across processes.
   - Optional OpenTelemetry exporter extra (`[otel]`) on top of `TraceCollector`.
   - Stronger sandbox backend (container / nsjail) behind the same `StepExecutor` protocol.

## 9. Decision log

Append entries here when a contract changes. Format: `YYYY-MM-DD · WP-n · change · reason · affected WPs`.

| Date | WP | Decision | Reason |
| --- | --- | --- | --- |
| 2026-11-05 | — | Single repo, extras `[ai]`/`[memory]` instead of three distributions | Simpler to build and test; split later |
| 2026-11-05 | — | Template language is a whitelisted `ast` subset, not Jinja | Jinja can reach Python objects; whitelist is safer and has no extra dependency |
| 2026-11-05 | — | SQLite FTS5 retrieval, no embeddings in V0.1 | Keeps the footprint small and tests offline |
| 2026-11-05 | WP-10b | Neon (Postgres) is the production memory backend; SQLite stays as the offline/default test backend | User chose Neon; offline tests must not need credentials |
| 2026-11-05 | WP-11 | OpenRouter and NVIDIA NIM via one shared OpenAI-compatible httpx adapter | Both expose OpenAI-style chat completions; avoids SDK dependencies |
| 2026-11-05 | — | Live tests are marked `live`, skipped without env vars, run in a separate CI job | Keeps default CI deterministic and free |

## 10. Instructions for implementing agents

1. Pick the next unassigned WP whose dependencies are done (see §2). Say which one you took.
2. Write the tests from "Acceptance" first, then implement.
3. Run `pytest`, `ruff`, `mypy` for your files before reporting done.
4. Report: files changed, tests added, any contract deviations (and add a Decision Log row).
5. If a contract is ambiguous, choose the safer, stricter reading and note it. Don't expand scope.
