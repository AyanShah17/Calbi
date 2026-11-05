from __future__ import annotations

import pytest

from calbi.ai.verifier import Verdict
from calbi.core.registry import ToolRegistry
from calbi.core.schema import Playbook, Risk, Step
from calbi.core.state import RunState, RunStatus
from calbi.memory.improvement import ImprovementManager, ProposalStatus
from calbi.memory.sqlite import SQLiteMemory


def _make_playbook(
    name: str, step_tools: list[str], metadata: dict[str, object] | None = None
) -> Playbook:
    steps = [Step(id=f"step_{idx}", tool=tool) for idx, tool in enumerate(step_tools)]
    return Playbook(
        name=name,
        steps=steps,
        metadata=metadata or {},
    )


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()

    def read_fn(x: str) -> str:
        return x

    def write_fn(x: str) -> str:
        return x

    def delete_fn(x: str) -> str:
        return x

    reg.register(read_fn, name="tool_read", risk=Risk.READ)
    reg.register(write_fn, name="tool_write", risk=Risk.WRITE)
    reg.register(delete_fn, name="tool_delete", risk=Risk.DESTRUCTIVE)
    return reg


def test_single_successful_run_does_not_become_reusable() -> None:
    """A single successful run must never auto-promote to 'reusable'."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "summarize monthly quarterly revenues"

    pb = _make_playbook("revenue_summary", ["tool_read", "tool_read"])
    run = RunState(run_id="run_101", playbook=pb, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True, confidence=1.0)

    # 1. Saving run into memory stores it with status 'succeeded'
    rec_id = mem.save_run(task=task, playbook=pb, run=run, verdict=verdict)
    assert rec_id.startswith("rec_")

    # 2. find_similar only returns 'validated' or 'reusable', so it should return empty
    similar = mem.find_similar(task)
    assert similar == []

    # 3. Direct evaluation of a first successful run without replan or prior baseline returns None
    proposal = manager.evaluate_improvement(
        task=task,
        baseline=None,
        candidate=pb,
        run=run,
        verdict=verdict,
    )
    assert proposal is None
    assert len(manager.list_proposals()) == 0


def test_replanned_run_succeeded_creates_pending_proposal() -> None:
    """A replanned run that succeeded creates a PENDING proposal with baseline and candidate."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "process client account migration"

    baseline_pb = _make_playbook("migration_v1", ["tool_read", "tool_write", "tool_read"])
    candidate_pb = _make_playbook("migration_v2", ["tool_read", "tool_write"])
    run = RunState(run_id="run_replan_1", playbook=candidate_pb, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True, confidence=0.95)

    proposal = manager.evaluate_improvement(
        task=task,
        baseline=baseline_pb,
        candidate=candidate_pb,
        run=run,
        verdict=verdict,
    )

    assert proposal is not None
    assert proposal.status == ProposalStatus.PENDING
    assert proposal.task == task
    assert proposal.baseline_playbook == baseline_pb
    assert proposal.proposed_playbook == candidate_pb
    assert proposal.evidence_run_ids == ["run_replan_1"]
    assert proposal.steps_saved == 1  # 3 - 2 = 1
    assert proposal.risk_delta == 0
    assert "Replan recovery" in proposal.rationale

    pending_list = manager.list_proposals(ProposalStatus.PENDING)
    assert len(pending_list) == 1
    assert pending_list[0].proposal_id == proposal.proposal_id


def test_replanned_run_with_more_steps_records_accurate_steps_saved() -> None:
    """Replan recovery with more steps accurately records negative steps_saved."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "generate consolidated transaction report"

    baseline_pb = _make_playbook("tx_fast", ["tool_read"])
    candidate_pb = _make_playbook("tx_robust", ["tool_read", "tool_read", "tool_read"])
    run = RunState(run_id="run_202", playbook=candidate_pb, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True)

    proposal = manager.evaluate_improvement(
        task=task,
        baseline=baseline_pb,
        candidate=candidate_pb,
        run=run,
        verdict=verdict,
    )

    assert proposal is not None
    assert proposal.steps_saved == -2  # 1 - 3 = -2
    assert proposal.baseline_playbook == baseline_pb
    assert proposal.proposed_playbook == candidate_pb


def test_faster_playbook_creates_proposal_with_accurate_steps_saved() -> None:
    """Faster playbook compared to existing validated playbook creates an optimization proposal."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "clean and analyze survey data"

    # Seed an existing validated playbook in memory
    old_pb = _make_playbook("old_survey", ["tool_read", "tool_read", "tool_read", "tool_read"])
    dummy_run = RunState(run_id="seed_run", playbook=old_pb, status=RunStatus.SUCCEEDED)
    rec_id = mem.save_run(task, old_pb, dummy_run)
    mem.promote(rec_id, to="validated")

    # New candidate playbook achieves the task in 2 steps
    new_pb = _make_playbook("fast_survey", ["tool_read", "tool_read"])
    run = RunState(run_id="run_fast_1", playbook=new_pb, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True)

    proposal = manager.evaluate_improvement(
        task=task,
        baseline=None,
        candidate=new_pb,
        run=run,
        verdict=verdict,
    )

    assert proposal is not None
    assert proposal.status == ProposalStatus.PENDING
    assert proposal.steps_saved == 2  # 4 - 2 = 2
    assert proposal.risk_delta == 0
    assert proposal.baseline_playbook == old_pb
    assert proposal.proposed_playbook == new_pb
    assert "saves 2 step(s)" in proposal.rationale


def test_lower_risk_playbook_creates_proposal_with_accurate_risk_delta() -> None:
    """Lower risk playbook creates an optimization proposal with accurate negative risk_delta."""
    mem = SQLiteMemory(path=":memory:")
    reg = _make_registry()
    manager = ImprovementManager(memory=mem, registry=reg)
    task = "archive customer documents secure"

    # Seed an existing validated playbook using tool_delete (Risk.DESTRUCTIVE, rank 3)
    old_pb = _make_playbook("archive_v1", ["tool_delete"])
    dummy_run = RunState(run_id="old_run", playbook=old_pb, status=RunStatus.SUCCEEDED)
    rec_id = mem.save_run(task, old_pb, dummy_run)
    mem.promote(rec_id, to="validated")

    # Candidate uses tool_read (Risk.READ, rank 1) instead
    new_pb = _make_playbook("archive_v2", ["tool_read"])
    run = RunState(run_id="run_safe", playbook=new_pb, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True)

    proposal = manager.evaluate_improvement(
        task=task,
        baseline=None,
        candidate=new_pb,
        run=run,
        verdict=verdict,
    )

    assert proposal is not None
    assert proposal.status == ProposalStatus.PENDING
    assert proposal.steps_saved == 0  # 1 - 1
    assert proposal.risk_delta == -2  # 1 (READ) - 3 (DESTRUCTIVE) = -2
    assert "reduces risk" in proposal.rationale


def test_both_faster_and_lower_risk_playbook() -> None:
    """Candidate that is both faster and lower risk produces accurate delta metrics."""
    mem = SQLiteMemory(path=":memory:")
    reg = _make_registry()
    manager = ImprovementManager(memory=mem, registry=reg)
    task = "sync inventory warehouse stocks"

    # Existing: 3 steps, highest risk is tool_write (rank 2)
    old_pb = _make_playbook("sync_v1", ["tool_read", "tool_write", "tool_read"])
    rec_id = mem.save_run(
        task, old_pb, RunState(run_id="run_old", playbook=old_pb, status=RunStatus.SUCCEEDED)
    )
    mem.promote(rec_id, to="validated")

    # Candidate: 1 step, tool_read (rank 1)
    new_pb = _make_playbook("sync_v2", ["tool_read"])
    proposal = manager.evaluate_improvement(
        task=task,
        baseline=None,
        candidate=new_pb,
        run=RunState(run_id="run_new", playbook=new_pb, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )

    assert proposal is not None
    assert proposal.steps_saved == 2  # 3 - 1 = 2
    assert proposal.risk_delta == -1  # 1 - 2 = -1


def test_slower_higher_risk_playbook_produces_no_proposal() -> None:
    """Candidate that is slower and higher risk than existing validated playbook is not proposed."""
    mem = SQLiteMemory(path=":memory:")
    reg = _make_registry()
    manager = ImprovementManager(memory=mem, registry=reg)
    task = "compute aggregate portfolio metrics"

    # Existing: 1 step, tool_read (rank 1)
    old_pb = _make_playbook("metric_v1", ["tool_read"])
    rec_id = mem.save_run(
        task, old_pb, RunState(run_id="r0", playbook=old_pb, status=RunStatus.SUCCEEDED)
    )
    mem.promote(rec_id, to="validated")

    # Candidate: 2 steps, tool_write (rank 2)
    candidate_pb = _make_playbook("metric_v2", ["tool_read", "tool_write"])
    proposal = manager.evaluate_improvement(
        task=task,
        baseline=None,
        candidate=candidate_pb,
        run=RunState(run_id="r1", playbook=candidate_pb, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )

    assert proposal is None


def test_failed_or_unverified_run_produces_no_proposal() -> None:
    """Failed runs or failed verdicts never produce improvement proposals."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "download daily market updates"

    pb_base = _make_playbook("pb_base", ["tool_read", "tool_read"])
    pb_cand = _make_playbook("pb_cand", ["tool_read"])

    # 1. RunStatus is FAILED
    failed_run = RunState(run_id="f1", playbook=pb_cand, status=RunStatus.FAILED)
    assert (
        manager.evaluate_improvement(
            task=task,
            baseline=pb_base,
            candidate=pb_cand,
            run=failed_run,
            verdict=Verdict(passed=True),
        )
        is None
    )

    # 2. Verdict passed is False
    succ_run = RunState(run_id="s1", playbook=pb_cand, status=RunStatus.SUCCEEDED)
    assert (
        manager.evaluate_improvement(
            task=task,
            baseline=pb_base,
            candidate=pb_cand,
            run=succ_run,
            verdict=Verdict(passed=False, reasons=["Output mismatch"]),
        )
        is None
    )


def test_approve_promotes_record_to_reusable_and_find_similar_retrieves_it() -> None:
    """Approving a proposal promotes the playbook to 'reusable' in MemoryStore."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "automate database backup schedule"

    pb_base = _make_playbook("backup_base", ["tool_read", "tool_write", "tool_read"])
    pb_cand = _make_playbook("backup_cand", ["tool_read", "tool_write"])
    run = RunState(run_id="run_backup_1", playbook=pb_cand, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True)

    proposal = manager.evaluate_improvement(
        task=task,
        baseline=pb_base,
        candidate=pb_cand,
        run=run,
        verdict=verdict,
    )
    assert proposal is not None
    prop_id = proposal.proposal_id

    # Memory does not have candidate as reusable yet
    similar_before = mem.find_similar(task)
    assert not any(r.playbook == pb_cand and r.status == "reusable" for r in similar_before)

    # Approve the proposal
    record = manager.approve(prop_id)

    assert record.status == "reusable"
    assert record.playbook == pb_cand
    assert proposal.status == ProposalStatus.APPROVED

    # Now find_similar retrieves the promoted reusable record
    similar_after = mem.find_similar(task)
    reusable_records = [r for r in similar_after if r.status == "reusable"]
    assert len(reusable_records) >= 1
    assert reusable_records[0].record_id == record.record_id

    # List proposals by status
    assert len(manager.list_proposals(ProposalStatus.APPROVED)) == 1
    assert len(manager.list_proposals(ProposalStatus.PENDING)) == 0


def test_reject_leaves_memory_unpromoted() -> None:
    """Rejecting a proposal marks it REJECTED and leaves memory unpromoted."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "deploy internal metrics dashboard"

    pb_base = _make_playbook("dash_v1", ["tool_read", "tool_write"])
    pb_cand = _make_playbook("dash_v2", ["tool_read"])
    run = RunState(run_id="run_dash_1", playbook=pb_cand, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True)

    # Save initial run in memory (which creates status='succeeded')
    mem.save_run(task, pb_cand, run, verdict)

    proposal = manager.evaluate_improvement(
        task=task,
        baseline=pb_base,
        candidate=pb_cand,
        run=run,
        verdict=verdict,
    )
    assert proposal is not None

    rejected = manager.reject(proposal.proposal_id, reason="Security review requested changes")
    assert rejected.status == ProposalStatus.REJECTED
    assert rejected.rejection_reason == "Security review requested changes"

    # Memory must still NOT return it in find_similar
    similar = mem.find_similar(task)
    assert similar == []

    # Status in proposal lists
    assert len(manager.list_proposals(ProposalStatus.REJECTED)) == 1
    assert len(manager.list_proposals(ProposalStatus.PENDING)) == 0


def test_approve_and_reject_nonexistent_proposal_raises_key_error() -> None:
    """Operating on an unknown proposal_id raises KeyError."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)

    with pytest.raises(KeyError, match="Improvement proposal 'missing' not found."):
        manager.approve("missing")

    with pytest.raises(KeyError, match="Improvement proposal 'missing' not found."):
        manager.reject("missing")

    with pytest.raises(KeyError, match="Improvement proposal 'missing' not found."):
        manager.get_proposal("missing")


def test_duplicate_evaluation_aggregates_evidence_runs() -> None:
    """Evaluating the same candidate run again aggregates evidence run IDs."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)
    task = "export financial quarterly reports"

    pb_base = _make_playbook("exp_v1", ["tool_read", "tool_write"])
    pb_cand = _make_playbook("exp_v2", ["tool_read"])

    run1 = RunState(run_id="run_ev_1", playbook=pb_cand, status=RunStatus.SUCCEEDED)
    prop1 = manager.evaluate_improvement(
        task=task,
        baseline=pb_base,
        candidate=pb_cand,
        run=run1,
        verdict=Verdict(passed=True),
    )
    assert prop1 is not None
    assert prop1.evidence_run_ids == ["run_ev_1"]

    run2 = RunState(run_id="run_ev_2", playbook=pb_cand, status=RunStatus.SUCCEEDED)
    prop2 = manager.evaluate_improvement(
        task=task,
        baseline=pb_base,
        candidate=pb_cand,
        run=run2,
        verdict=Verdict(passed=True),
    )
    assert prop2 is not None
    assert prop2.proposal_id == prop1.proposal_id
    assert "run_ev_1" in prop2.evidence_run_ids
    assert "run_ev_2" in prop2.evidence_run_ids
    assert len(manager.list_proposals()) == 1


def test_reusable_qualification_after_min_runs_with_no_failures() -> None:
    """A validated playbook reaching min_runs_for_reusable creates proposal if zero failures."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem, min_runs_for_reusable=5)
    task = "query production system telemetry"

    pb = _make_playbook("telemetry_query", ["tool_read"])

    # Simulate 5 successful runs in SQLite memory
    for idx in range(5):
        run = RunState(run_id=f"run_tel_{idx}", playbook=pb, status=RunStatus.SUCCEEDED)
        mem.save_run(task, pb, run, Verdict(passed=True))

    similar = mem.find_similar(task)
    assert len(similar) == 1
    assert similar[0].status == "validated"
    assert similar[0].run_count >= 5

    # 6th run evaluates improvement
    curr_run = RunState(run_id="run_tel_5", playbook=pb, status=RunStatus.SUCCEEDED)
    proposal = manager.evaluate_improvement(
        task=task,
        baseline=None,
        candidate=pb,
        run=curr_run,
        verdict=Verdict(passed=True),
    )

    assert proposal is not None
    assert proposal.status == ProposalStatus.PENDING
    assert "Reusable promotion qualification" in proposal.rationale
