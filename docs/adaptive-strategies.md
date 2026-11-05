# Autonomy Level 4: Adaptive Strategies (`calbi.ai.strategies`)

## Overview

Calbi introduces **Autonomy Level 4 (`Autonomy.ADAPTIVE`)** to elevate playbook execution from reactive replanning to proactive, multi-strategy tournament exploration. In complex or ambiguous problem domains, a single initial plan may be sub-optimal or fragile. Adaptive strategies enable Calbi to:

1. **Synthesize Alternative Strategies**: Prompt the language model to generate multiple structurally distinct playbooks exploring different tool chains, execution paths, and parameter configurations.
2. **Pre-Validate and Deduplicate**: Discard malformed, cycle-inducing, or policy-violating playbooks immediately, and prune redundant candidates sharing identical structural signatures.
3. **Conduct Strategy Tournaments**: Evaluate candidate playbooks under either **Sequential Early-Stopping** or **Parallel Read-Only Concurrent** exploration modes.
4. **Multi-Objective Composite Scoring**: Rank trials using an objective composite function balancing verification confidence (50%), step efficiency (30%), and execution safety/risk (20%).
5. **Enforce Policy Invariants**: Maintain strict permission checks and policy gates throughout every stage of exploration.

---

## Architecture & Data Contracts

```
               ┌────────────────────────────────────────────────────────┐
               │                    StrategyExplorer                    │
               │   - llm: LLM                                           │
               │   - registry: ToolRegistry                             │
               │   - policy: Policy                                     │
               │   - runtime: Runtime                                   │
               │   - verifier: Verifier                                 │
               └───────────────────────────┬────────────────────────────┘
                                           │
             ┌─────────────────────────────┴────────────────────────────┐
             ▼                                                          ▼
┌──────────────────────────┐                               ┌──────────────────────────┐
│   generate_candidates()  │                               │        explore()         │
│  - Prompts LLM for plans │                               │  - sequential mode       │
│  - Validates schema/DAG  │                               │  - parallel_read_only    │
│  - Deduplicates by sig   │                               │  - computes scores       │
└────────────┬─────────────┘                               └────────────┬─────────────┘
             │                                                          │
             ▼                                                          ▼
┌──────────────────────────┐                               ┌──────────────────────────┐
│    StrategyCandidate     │                               │ StrategyExplorationResult│
│  - name: str             │                               │  - winner: StrategyTrial │
│  - rationale: str        │                               │  - trials: list[Trial]   │
│  - playbook: Playbook    │                               │  - total_attempts: int   │
│  - validation: Report    │                               └──────────────────────────┘
└──────────────────────────┘
```

### Core Data Models

#### `StrategyCandidate`
Represents an individual proposed execution strategy before execution:
```python
@dataclass
class StrategyCandidate:
    name: str
    rationale: str
    playbook: Playbook
    validation: ValidationReport
```

#### `StrategyTrial`
Represents the result of executing and verifying a candidate strategy:
```python
@dataclass
class StrategyTrial:
    candidate: StrategyCandidate
    run: RunState | None
    verdict: Verdict | None
    score: float  # 0.0..1.0 composite score
```

#### `StrategyExplorationResult`
The final outcome of a strategy exploration tournament:
```python
@dataclass
class StrategyExplorationResult:
    winner: StrategyTrial | None
    trials: list[StrategyTrial]
    total_attempts: int
```

#### `StrategyExplorationFailed`
Exception raised when strategy exploration cannot produce viable candidates:
```python
class StrategyExplorationFailed(CalbiError):
    """Raised when strategy exploration fails to produce viable candidates or trials."""
```

---

## Structural Deduplication & Candidate Filtering

When prompted for alternative strategies, language models may generate functionally identical plans with renamed step identifiers or trivially shuffled steps. Calbi guarantees strategy diversity via deterministic structural signatures:

```python
def _playbook_signature(pb: Playbook) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((s.tool, json.dumps(s.inputs, sort_keys=True)) for s in pb.steps))
```

### Invariants:
1. **Tool & Parameter Normalization**: Playbooks utilizing the exact same tool chain and input arguments yield identical signatures, regardless of step naming or DAG declaration order.
2. **Pre-Validation**: Every candidate is evaluated with `validate(pb, registry, policy, inputs=inputs)`. Playbooks referencing unregistered tools, cycles, or missing inputs are discarded before execution.
3. **Budget Guard**: Exploration terminates candidate generation as soon as `k` valid, structurally unique strategies have been collected.

---

## Exploration Modes

### 1. Sequential Mode (`mode="sequential"`)
Iteratively executes candidate playbooks one by one, checking for early exit criteria.

- **Early Stopping Optimization**: If a candidate succeeds and the verifier reports `verdict.passed is True` with `verdict.confidence >= 0.8`, exploration terminates immediately. Subsequent candidates are never executed, saving tokens, time, and computational budget.
- **Winner Selection**: If no candidate reaches the 0.8 confidence threshold, all candidates are evaluated and the highest scoring passing trial is selected as the winner.

### 2. Parallel Read-Only Mode (`mode="parallel_read_only"`)
Executes eligible read-only candidates concurrently to rapidly discover the optimal path.

- **Safety Invariant**: Only candidates whose validated `max_risk == Risk.READ` are permitted to execute concurrently.
- **Pre-Execution Rejection**: Any candidate containing `Risk.WRITE`, `Risk.DESTRUCTIVE`, `Risk.EXTERNAL`, `Risk.FINANCIAL`, or `Risk.DEPLOY` tools is immediately rejected before execution (`trial.run = None`, `trial.score = 0.0`, `trial.verdict.passed = False`).
- **Concurrent Tournament**: Eligible candidates run concurrently via `asyncio.gather` through `Runtime.run()`, verified, and scored. The passing candidate with the highest composite score is selected as the winner.

---

## Multi-Objective Composite Scoring

Trials are evaluated on a normalized `0.0..1.0` scale combining confidence, step efficiency, and execution risk.

$$\text{Score} = 0.5 \times \text{Confidence} + 0.3 \times \text{Efficiency} + 0.2 \times \text{RiskFactor}$$

### Scoring Components:
1. **Outcome Hard Gate**:
   If `run is None`, `run.status != RunStatus.SUCCEEDED`, or `verdict.passed is False`, the score is strictly `0.0`.
2. **Verification Confidence (Weight: 0.5)**:
   $$\text{Confidence} = \text{clamp}(\text{verdict.confidence}, 0.0, 1.0)$$
3. **Step Efficiency (Weight: 0.3)**:
   Fewer steps are rewarded according to:
   $$\text{Efficiency} = \frac{1.0}{1.0 + 0.1 \times \text{step\_count}}$$
4. **Risk Factor (Weight: 0.2)**:
   Safer execution profiles (lower risk) receive higher scores based on `Risk._rank`:
   $$\text{RiskFactor} = \frac{7 - \text{rank}}{6.0}$$
   - `READ` (Rank 1): 1.000
   - `WRITE` (Rank 2): 0.833
   - `DESTRUCTIVE` (Rank 3): 0.667
   - `EXTERNAL` (Rank 4): 0.500
   - `FINANCIAL` (Rank 5): 0.333
   - `DEPLOY` (Rank 6): 0.167

The composite score is clamped to `[0.0, 1.0]` and rounded to 4 decimal places.

---

## Usage Examples

### Example 1: Sequential Exploration with Early Stopping

```python
import asyncio
from calbi.ai.llm import LLM
from calbi.ai.strategies import StrategyExplorer
from calbi.core.policy import Autonomy, Policy
from calbi.core.registry import ToolRegistry

async def run_sequential_adaptive(llm: LLM, registry: ToolRegistry) -> None:
    policy = Policy(autonomy=Autonomy.ADAPTIVE)
    explorer = StrategyExplorer(
        llm=llm,
        registry=registry,
        policy=policy,
        max_candidates=3,
    )

    result = await explorer.explore(
        task="Diagnose and repair network latency issue",
        inputs={"host": "api.production.internal"},
        mode="sequential",
    )

    if result.winner:
        print(f"Winning Strategy: {result.winner.candidate.name}")
        print(f"Composite Score: {result.winner.score}")
        print(f"Total Step Attempts: {result.total_attempts}")
    else:
        print("No viable strategy succeeded.")
```

### Example 2: Parallel Read-Only Dry-Run Exploration

```python
async def run_parallel_read_only(llm: LLM, registry: ToolRegistry) -> None:
    policy = Policy(autonomy=Autonomy.ADAPTIVE)
    explorer = StrategyExplorer(
        llm=llm,
        registry=registry,
        policy=policy,
        max_candidates=3,
    )

    result = await explorer.explore(
        task="Analyze customer churn signals across datasets",
        inputs={"quarter": "2026-Q3"},
        mode="parallel_read_only",
    )

    for trial in result.trials:
        status = "PASSED" if trial.verdict and trial.verdict.passed else "REJECTED/FAILED"
        print(f"Trial '{trial.candidate.name}': {status} (score={trial.score})")

    if result.winner:
        print(f"Best Strategy: {result.winner.candidate.name} (score={result.winner.score})")
```

---

## Architectural & Safety Guarantees

1. **No Policy Bypass**: All candidate playbooks execute through `Runtime.run(..., autonomy=Autonomy.ADAPTIVE)`. Policy rules (denied tools, max risk thresholds, approvals) cannot be subverted.
2. **Zero Side-Effects in Parallel Mode**: Non-READ strategies are filtered out prior to execution in `parallel_read_only` mode.
3. **Total Attempt Accountability**: `StrategyExplorationResult.total_attempts` tracks cumulative attempts across all executed candidate steps, ensuring complete visibility into resource utilization.
