from __future__ import annotations

from pathlib import Path

import pytest

from calbi.ai.verifier import Verdict
from calbi.core.registry import ToolRegistry
from calbi.core.schema import Playbook, Risk, Step
from calbi.core.state import RunState, RunStatus
from calbi.memory.improvement import (
    ImprovementManager,
    ProposalStatus,
    normalize_task,
    task_similarity,
)
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


def test_normalize_task_and_similarity_scoring() -> None:
    """Task normalization case-folds, collapses whitespace, and strips punctuation."""
    t1 = "  Deploy   Internal Metrics Dashboard!  "
    t2 = "deploy-internal-metrics-dashboard"
    t3 = "deploy internal metrics dashboard"
    t4 = "deploy external metrics dashboard"

    assert normalize_task(t1) == "deploy internal metrics dashboard"
    assert normalize_task(t2) == "deploy internal metrics dashboard"
    assert normalize_task(t3) == "deploy internal metrics dashboard"

    # Exact normalized matches return 1.0
    assert task_similarity(t1, t2) == 1.0
    assert task_similarity(t1, t3) == 1.0

    # Partial similarity
    sim = task_similarity(t3, t4)
    assert 0.70 < sim < 1.0

    # Disjoint strings
    assert task_similarity("abc", "xyz") == 0.0
    assert task_similarity("", "") == 1.0
    assert task_similarity("", "something") == 0.0


def test_exact_normalized_task_match_aggregates_evidence_without_duplicate() -> None:
    """Exact normalized task match prefers existing proposal and aggregates evidence."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem)

    pb_base = _make_playbook("dep_base", ["tool_read", "tool_write"])
    pb_cand = _make_playbook("dep_cand", ["tool_read"])

    # 1. First run with messy casing and punctuation
    prop1 = manager.evaluate_improvement(
        task="  Deploy   Internal Metrics Dashboard!  ",
        baseline=pb_base,
        candidate=pb_cand,
        run=RunState(run_id="run_d1", playbook=pb_cand, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert prop1 is not None
    assert prop1.evidence_run_ids == ["run_d1"]

    # 2. Second run with normalized task string
    prop2 = manager.evaluate_improvement(
        task="deploy internal metrics dashboard",
        baseline=pb_base,
        candidate=pb_cand,
        run=RunState(run_id="run_d2", playbook=pb_cand, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert prop2 is not None
    assert prop2.proposal_id == prop1.proposal_id
    assert "run_d1" in prop2.evidence_run_ids
    assert "run_d2" in prop2.evidence_run_ids
    assert len(manager.list_proposals()) == 1


def test_weak_similarity_memory_match_does_not_create_optimization_proposal() -> None:
    """A weak similarity match from memory (< threshold) does not trigger optimization proposal."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem, similarity_threshold=0.85)

    # Seed an existing validated playbook for survey data
    old_pb = _make_playbook("survey_v1", ["tool_read", "tool_read", "tool_read", "tool_read"])
    rec_id = mem.save_run(
        "clean and analyze survey data",
        old_pb,
        RunState(run_id="seed_survey", playbook=old_pb, status=RunStatus.SUCCEEDED),
    )
    mem.promote(rec_id, to="validated")

    # New run for customer feedback data (shares 'clean', 'and', 'analyze' with survey data)
    cand_pb = _make_playbook("feedback_v1", ["tool_read"])
    run = RunState(run_id="run_feedback", playbook=cand_pb, status=RunStatus.SUCCEEDED)

    # Memory find_similar may return survey record, but similarity is below 0.85
    proposal = manager.evaluate_improvement(
        task="clean and analyze customer feedback data",
        baseline=None,
        candidate=cand_pb,
        run=run,
        verdict=Verdict(passed=True),
    )

    # Must NOT propose optimization against unrelated survey baseline
    assert proposal is None
    assert len(manager.list_proposals()) == 0


def test_configurable_similarity_threshold_enforcement() -> None:
    """Configurable similarity threshold controls whether near-matches trigger optimization."""
    mem = SQLiteMemory(path=":memory:")

    old_pb = _make_playbook("survey_v1", ["tool_read", "tool_read", "tool_read", "tool_read"])
    rec_id = mem.save_run(
        "clean and analyze survey data",
        old_pb,
        RunState(run_id="seed_survey", playbook=old_pb, status=RunStatus.SUCCEEDED),
    )
    mem.promote(rec_id, to="validated")

    cand_pb = _make_playbook("feedback_v1", ["tool_read"])
    run = RunState(run_id="run_fb", playbook=cand_pb, status=RunStatus.SUCCEEDED)

    # 1. Permissive threshold: 0.70 allows ~0.75 match to trigger optimization
    permissive_manager = ImprovementManager(memory=mem, similarity_threshold=0.70)
    prop_permissive = permissive_manager.evaluate_improvement(
        task="clean and analyze customer feedback data",
        baseline=None,
        candidate=cand_pb,
        run=run,
        verdict=Verdict(passed=True),
    )
    assert prop_permissive is not None
    assert prop_permissive.steps_saved == 3

    # 2. Strict threshold: 0.90 rejects ~0.75 match
    strict_manager = ImprovementManager(memory=mem, similarity_threshold=0.90)
    prop_strict = strict_manager.evaluate_improvement(
        task="clean and analyze customer feedback data",
        baseline=None,
        candidate=cand_pb,
        run=run,
        verdict=Verdict(passed=True),
    )
    assert prop_strict is None


def test_pending_proposal_overwrite_requires_normalized_match_or_threshold() -> None:
    """Pending proposal is updated only if task matches and candidate is better."""
    mem = SQLiteMemory(path=":memory:")
    manager = ImprovementManager(memory=mem, similarity_threshold=0.85)

    pb_base = _make_playbook("tax_base", ["tool_read", "tool_write", "tool_read"])
    pb_cand1 = _make_playbook("tax_v1", ["tool_read", "tool_write"])  # saves 1 step

    # 1. Create initial proposal P1
    prop1 = manager.evaluate_improvement(
        task="process quarterly tax returns",
        baseline=pb_base,
        candidate=pb_cand1,
        run=RunState(run_id="r_tax_1", playbook=pb_cand1, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert prop1 is not None
    assert prop1.steps_saved == 1
    p1_id = prop1.proposal_id

    # 2. Candidate for a weakly similar task ("process annual tax returns", sim < 0.85)
    # Even if cand_other saves 2 steps, it must NOT overwrite P1; it becomes a distinct proposal!
    pb_cand_other = _make_playbook("annual_tax", ["tool_read"])
    pb_base_other = _make_playbook("annual_base", ["tool_read", "tool_read", "tool_read"])
    prop_distinct = manager.evaluate_improvement(
        task="process annual tax returns",
        baseline=pb_base_other,
        candidate=pb_cand_other,
        run=RunState(run_id="r_ann_1", playbook=pb_cand_other, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert prop_distinct is not None
    assert prop_distinct.proposal_id != p1_id
    assert len(manager.list_proposals()) == 2

    # P1 remains intact
    assert manager.get_proposal(p1_id).steps_saved == 1

    # 3. Candidate for exact normalized match ("Process Quarterly Tax Returns!") with BETTER savings
    pb_cand_better = _make_playbook("tax_v2", ["tool_read"])  # saves 2 steps
    prop_updated = manager.evaluate_improvement(
        task="Process Quarterly Tax Returns!",
        baseline=pb_base,
        candidate=pb_cand_better,
        run=RunState(run_id="r_tax_better", playbook=pb_cand_better, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert prop_updated is not None
    assert prop_updated.proposal_id == p1_id
    assert prop_updated.steps_saved == 2
    assert prop_updated.proposed_playbook == pb_cand_better
    assert "r_tax_better" in prop_updated.evidence_run_ids
    assert len(manager.list_proposals()) == 2

    # 4. Candidate for exact normalized match with WORSE savings (e.g. saves 0 steps)
    # Must NOT overwrite P1 with worse candidate
    pb_cand_worse = _make_playbook("tax_v3", ["tool_read", "tool_write", "tool_read"])
    prop_not_overwritten = manager.evaluate_improvement(
        task="Process Quarterly Tax Returns!",
        baseline=pb_base,
        candidate=pb_cand_worse,
        run=RunState(run_id="r_tax_worse", playbook=pb_cand_worse, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert prop_not_overwritten is not None
    assert prop_not_overwritten.proposal_id == p1_id
    assert prop_not_overwritten.steps_saved == 2  # Still keeps the better candidate!


def test_sqlite_persistence_across_process_restarts(tmp_path: Path) -> None:
    """Improvement proposals persist in SQLite across ImprovementManager restarts."""
    db_file = tmp_path / "proposals.sqlite"
    mem = SQLiteMemory(path=":memory:")

    pb_base = _make_playbook("persist_base", ["tool_read", "tool_write", "tool_read"])
    pb_cand1 = _make_playbook("persist_cand1", ["tool_read", "tool_write"])
    pb_cand2 = _make_playbook("persist_cand2", ["tool_read"])

    # Session 1: Create two proposals
    mgr1 = ImprovementManager(memory=mem, db_path=db_file)
    p1 = mgr1.evaluate_improvement(
        task="optimize database indices",
        baseline=pb_base,
        candidate=pb_cand1,
        run=RunState(run_id="run_p1", playbook=pb_cand1, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    p2 = mgr1.evaluate_improvement(
        task="prune stale session tokens",
        baseline=pb_base,
        candidate=pb_cand2,
        run=RunState(run_id="run_p2", playbook=pb_cand2, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert p1 is not None and p2 is not None
    p1_id = p1.proposal_id
    p2_id = p2.proposal_id
    mgr1.close()

    # Session 2: Re-instantiate from the same SQLite DB file
    mgr2 = ImprovementManager(memory=mem, storage_path=db_file)
    assert len(mgr2.list_proposals()) == 2
    restored_p1 = mgr2.get_proposal(p1_id)
    assert restored_p1.task == "optimize database indices"
    assert restored_p1.status == ProposalStatus.PENDING
    assert restored_p1.steps_saved == 1
    assert restored_p1.evidence_run_ids == ["run_p1"]
    assert restored_p1.proposed_playbook.name == "persist_cand1"

    # Approve proposal 1 in session 2
    record = mgr2.approve(p1_id)
    assert record.status == "reusable"
    assert restored_p1.status == ProposalStatus.APPROVED
    mgr2.close()

    # Session 3: Re-instantiate and verify proposal 1 remains APPROVED
    mgr3 = ImprovementManager(memory=mem, db_path=db_file)
    assert mgr3.get_proposal(p1_id).status == ProposalStatus.APPROVED
    assert len(mgr3.list_proposals(ProposalStatus.APPROVED)) == 1
    assert len(mgr3.list_proposals(ProposalStatus.PENDING)) == 1

    # Reject proposal 2 in session 3
    mgr3.reject(p2_id, reason="Requires architectural review")
    assert mgr3.get_proposal(p2_id).status == ProposalStatus.REJECTED
    mgr3.close()

    # Session 4: Re-instantiate and verify rejection persisted
    mgr4 = ImprovementManager(memory=mem, storage_path=db_file)
    restored_p2 = mgr4.get_proposal(p2_id)
    assert restored_p2.status == ProposalStatus.REJECTED
    assert restored_p2.rejection_reason == "Requires architectural review"
    mgr4.close()


def test_file_backed_json_persistence_across_restarts(tmp_path: Path) -> None:
    """Improvement proposals persist in a JSON file across ImprovementManager restarts."""
    json_file = tmp_path / "proposals.json"
    mem = SQLiteMemory(path=":memory:")

    pb_base = _make_playbook("json_base", ["tool_read", "tool_write"])
    pb_cand = _make_playbook("json_cand", ["tool_read"])

    # Session 1: Create proposal in JSON-backed manager
    mgr1 = ImprovementManager(memory=mem, storage_path=json_file)
    prop = mgr1.evaluate_improvement(
        task="compress archived customer logs",
        baseline=pb_base,
        candidate=pb_cand,
        run=RunState(run_id="run_json_1", playbook=pb_cand, status=RunStatus.SUCCEEDED),
        verdict=Verdict(passed=True),
    )
    assert prop is not None
    assert json_file.exists()
    prop_id = prop.proposal_id

    # Session 2: Re-instantiate from the same JSON file
    mgr2 = ImprovementManager(memory=mem, storage_path=json_file)
    assert len(mgr2.list_proposals()) == 1
    restored = mgr2.get_proposal(prop_id)
    assert restored.task == "compress archived customer logs"
    assert restored.steps_saved == 1
    assert restored.evidence_run_ids == ["run_json_1"]

    # Reject and verify persistence in JSON
    mgr2.reject(prop_id, reason="Not needed right now")
    mgr3 = ImprovementManager(memory=mem, storage_path=json_file)
    assert mgr3.get_proposal(prop_id).status == ProposalStatus.REJECTED
    assert mgr3.get_proposal(prop_id).rejection_reason == "Not needed right now"
