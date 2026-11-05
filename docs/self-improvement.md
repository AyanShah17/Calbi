# Calbi Self-Improvement Loop

Calbi provides an autonomous self-improvement loop with human approval gating for playbook evolution, optimization, and trust promotion.

---

## 1. Trust Hierarchy & Guarantees

Playbooks stored in `calbi.memory` advance through three distinct trust tiers:

| Trust Tier | Definition | Retrieval via `find_similar()` | Promotion Mechanism |
|---|---|---|---|
| `succeeded` | Executed successfully at least 1 time. | ❌ No | Automatic upon successful run. |
| `validated` | Executed successfully $\ge 3$ times with consistent results. | ✅ Yes | Automatic promotion by `MemoryStore` after 3 runs. |
| `reusable` | Production-vetted golden template for task synthesis. | ✅ Yes | **Explicit human approval** via `ImprovementManager.approve()`. |

### Non-Negotiable Safety Invariants
1. **Single Run Never Auto-Promotes to `reusable`**: A single successful run is never automatically granted `reusable` status.
2. **Human Approval Gating**: Any elevation of an improved, recovered, or optimized playbook to `reusable` requires explicit human review (`approve()`).
3. **Audit Trail**: Every proposal retains its baseline playbook, candidate playbook, rationale, step savings, risk delta, and evidence run IDs.

---

## 2. Architecture & Core Classes

All self-improvement components reside in `calbi.memory.improvement`:

### `ProposalStatus`
Enum tracking the lifecycle of an improvement proposal:
- `PENDING = "pending"`: Created and awaiting operator review.
- `APPROVED = "approved"`: Approved by an operator; playbook promoted to `reusable` in `MemoryStore`.
- `REJECTED = "rejected"`: Rejected by an operator; leaves `MemoryStore` unpromoted.

### `ImprovementProposal`
Pydantic model encapsulating an improvement hypothesis:
- `proposal_id`: Unique identifier (`prop_<uuid>`).
- `task`: Task objective description.
- `baseline_playbook`: Previous baseline playbook (or `None` if brand new).
- `proposed_playbook`: Optimized or recovered candidate playbook.
- `rationale`: Human-readable explanation of why this proposal was generated.
- `evidence_run_ids`: List of execution run IDs evidencing this improvement.
- `steps_saved`: Integer step count delta (`len(baseline.steps) - len(candidate.steps)`).
- `risk_delta`: Integer risk rank delta (`candidate_risk._rank - baseline_risk._rank`). Negative indicates lower risk.
- `status`: Current `ProposalStatus` (`PENDING`, `APPROVED`, or `REJECTED`).
- `created_at`: UTC timestamp of proposal creation.
- `rejection_reason`: Optional reason if the proposal was rejected.

### `ImprovementManager`
Manages the evaluation, listing, approval, and rejection of improvement proposals:
- `__init__(memory: MemoryStore, llm: LLM | None = None, min_runs_for_reusable: int = 5, registry: ToolRegistry | None = None)`
- `evaluate_improvement(task, baseline, candidate, run, verdict) -> ImprovementProposal | None`
- `list_proposals(status: ProposalStatus | None = None) -> list[ImprovementProposal]`
- `get_proposal(proposal_id: str) -> ImprovementProposal`
- `approve(proposal_id: str) -> PlaybookRecord`
- `reject(proposal_id: str, reason: str = "") -> ImprovementProposal`

---

## 3. Evaluation Triggers

`ImprovementManager.evaluate_improvement` evaluates only runs where `run.status == RunStatus.SUCCEEDED` and `verdict.passed is True`.

### Trigger A: Replan Recovery
When a task fails its initial plan, replans, and subsequently succeeds (`baseline is not None and baseline != candidate`):
- `steps_saved = len(baseline.steps) - len(candidate.steps)`
- `risk_delta = candidate_risk._rank - baseline_risk._rank`
- A `PENDING` proposal is registered documenting the recovery from the failed baseline.

### Trigger B: Optimization Over Existing Playbook
When an existing validated or reusable playbook exists in `MemoryStore` for `task`:
- `steps_saved = len(existing.steps) - len(candidate.steps)`
- `risk_delta = candidate_risk._rank - existing_risk._rank`
- If `steps_saved > 0` (fewer steps) or `risk_delta < 0` (lower risk level), a `PENDING` proposal is registered.

### Trigger C: High-Confidence Reusable Candidate
When an existing playbook reaches $\ge \text{min\_runs\_for\_reusable}$ (default `5`) with zero recorded tool limitations or failures, an eligible `PENDING` proposal is generated for promotion to `reusable`.

---

## 4. Usage Example

```python
from calbi.core.schema import Playbook, Step, Risk
from calbi.core.state import RunState, RunStatus
from calbi.ai.verifier import Verdict
from calbi.memory.sqlite import SQLiteMemory
from calbi.memory.improvement import ImprovementManager, ProposalStatus

# Initialize store and improvement manager
memory = SQLiteMemory(path=":memory:")
manager = ImprovementManager(memory=memory)

# Existing baseline playbook that failed or is slower
baseline_pb = Playbook(
    name="data_sync_v1",
    steps=[Step(id="s1", tool="fetch"), Step(id="s2", tool="transform"), Step(id="s3", tool="save")]
)

# Optimized candidate playbook that succeeded
candidate_pb = Playbook(
    name="data_sync_v2",
    steps=[Step(id="s1", tool="stream_sync")]
)
run = RunState(run_id="run_101", playbook=candidate_pb, status=RunStatus.SUCCEEDED)
verdict = Verdict(passed=True, confidence=1.0)

# Evaluate improvement
proposal = manager.evaluate_improvement(
    task="sync remote data stream",
    baseline=baseline_pb,
    candidate=candidate_pb,
    run=run,
    verdict=verdict,
)

assert proposal is not None
print(f"Proposal ID: {proposal.proposal_id}")
print(f"Steps Saved: {proposal.steps_saved}") # 2
print(f"Status: {proposal.status.value}")    # pending

# Operator approves proposal
record = manager.approve(proposal.proposal_id)
print(f"Promoted to: {record.status}")       # reusable

# Now memory.find_similar() retrieves the approved reusable playbook
similar = memory.find_similar("sync remote data stream")
assert similar[0].record_id == record.record_id
```
