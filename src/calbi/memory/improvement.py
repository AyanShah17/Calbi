from __future__ import annotations

import dataclasses
import difflib
import json
import re
import sqlite3
import string
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from calbi.ai.llm import LLM
from calbi.ai.verifier import Verdict
from calbi.core.registry import ToolRegistry, UnknownToolError
from calbi.core.schema import Playbook, Risk
from calbi.core.state import RunState, RunStatus
from calbi.memory.base import MemoryStore, PlaybookRecord


class ProposalStatus(str, Enum):
    """Lifecycle status of a playbook improvement proposal."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class ImprovementProposal(BaseModel):
    """Proposal to promote or replace a playbook based on execution performance."""

    proposal_id: str
    task: str
    baseline_playbook: Playbook | None
    proposed_playbook: Playbook
    rationale: str
    evidence_run_ids: list[str] = Field(default_factory=list)
    steps_saved: int
    risk_delta: int  # negative = lower risk, 0 = same, positive = higher risk
    status: ProposalStatus = ProposalStatus.PENDING
    created_at: datetime
    rejection_reason: str | None = None


def _compute_playbook_risk(playbook: Playbook, registry: ToolRegistry | None = None) -> Risk:
    """Determine the maximum risk rank across all steps of a playbook."""
    max_risk = Risk.READ

    # Check playbook-level metadata overrides
    for key in ("max_risk", "risk"):
        if key in playbook.metadata:
            val = playbook.metadata[key]
            if isinstance(val, Risk) and val > max_risk:
                max_risk = val
            elif isinstance(val, str):
                try:
                    r = Risk(val.lower())
                    if r > max_risk:
                        max_risk = r
                except ValueError:
                    pass

    # Check step-level tools in registry
    for step in playbook.steps:
        if registry is not None:
            try:
                spec = registry.get(step.tool)
                if spec.risk > max_risk:
                    max_risk = spec.risk
            except (UnknownToolError, Exception):
                pass

    return max_risk


def normalize_task(task: str) -> str:
    """Normalize a task string: case-folded, punctuation normalized, whitespace collapsed."""
    text = task.casefold().strip()
    text = re.sub(r"[" + re.escape(string.punctuation) + r"]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def task_similarity(task1: str, task2: str) -> float:
    """Compute normalized similarity ratio in [0.0, 1.0] between two tasks."""
    norm1 = normalize_task(task1)
    norm2 = normalize_task(task2)
    if norm1 == norm2:
        return 1.0
    if not norm1 or not norm2:
        return 0.0
    return difflib.SequenceMatcher(None, norm1, norm2).ratio()


class ImprovementManager:
    """Manages playbook evolution proposals, evaluating optimizations and gating promotion."""

    def __init__(
        self,
        memory: MemoryStore,
        llm: LLM | None = None,
        min_runs_for_reusable: int = 5,
        registry: ToolRegistry | None = None,
        storage_path: Path | str | None = None,
        db_path: Path | str | None = None,
        similarity_threshold: float = 0.85,
        min_similarity_threshold: float | None = None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.min_runs_for_reusable = min_runs_for_reusable
        self.registry = registry
        self.similarity_threshold = (
            min_similarity_threshold
            if min_similarity_threshold is not None
            else similarity_threshold
        )
        resolved_path = db_path if db_path is not None else storage_path
        if isinstance(resolved_path, str) and resolved_path != ":memory:":
            self.storage_path: Path | str | None = Path(resolved_path)
        else:
            self.storage_path = resolved_path

        self._proposals: dict[str, ImprovementProposal] = {}
        self._proposal_runs: dict[str, tuple[RunState, Verdict]] = {}
        self._conn: sqlite3.Connection | None = None
        self._storage_type: str | None = None

        if self.storage_path is not None:
            if isinstance(self.storage_path, Path) and self.storage_path.suffix == ".json":
                self._storage_type = "json"
                self._load_from_json()
            else:
                self._storage_type = "sqlite"
                if isinstance(self.storage_path, Path):
                    self.storage_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(
                    str(self.storage_path),
                    isolation_level=None,
                )
                self._init_sqlite_db()
                self._load_from_sqlite()

    def _init_sqlite_db(self) -> None:
        if self._conn is None:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS improvement_proposals (
                proposal_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                run_json TEXT,
                verdict_json TEXT
            );
            """
        )

    def _load_from_sqlite(self) -> None:
        if self._conn is None:
            return
        cur = self._conn.cursor()
        cur.execute(
            """
            SELECT proposal_id, proposal_json, run_json, verdict_json
            FROM improvement_proposals
            """
        )
        for row in cur.fetchall():
            pid, prop_json, run_json, verd_json = row
            proposal = ImprovementProposal.model_validate_json(prop_json)
            self._proposals[pid] = proposal
            if run_json and verd_json:
                try:
                    run = RunState.model_validate_json(run_json)
                    verdict_data = json.loads(verd_json)
                    verdict = Verdict(**verdict_data)
                    self._proposal_runs[pid] = (run, verdict)
                except Exception:
                    pass

    def _load_from_json(self) -> None:
        if not isinstance(self.storage_path, Path) or not self.storage_path.exists():
            return
        try:
            content = self.storage_path.read_text(encoding="utf-8")
            if not content.strip():
                return
            data = json.loads(content)
            for item in data.get("proposals", []):
                prop = ImprovementProposal.model_validate(item)
                self._proposals[prop.proposal_id] = prop
            for pid, r_info in data.get("runs", {}).items():
                run = RunState.model_validate(r_info["run"])
                verdict = Verdict(**r_info["verdict"])
                self._proposal_runs[pid] = (run, verdict)
        except Exception:
            pass

    def _save_to_json(self) -> None:
        if not isinstance(self.storage_path, Path):
            return
        data = {
            "proposals": [p.model_dump(mode="json") for p in self._proposals.values()],
            "runs": {
                pid: {
                    "run": run.model_dump(mode="json"),
                    "verdict": dataclasses.asdict(verd),
                }
                for pid, (run, verd) in self._proposal_runs.items()
            },
        }
        tmp_file = self.storage_path.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_file.replace(self.storage_path)

    def _save_proposal(
        self,
        proposal: ImprovementProposal,
        run: RunState | None = None,
        verdict: Verdict | None = None,
    ) -> None:
        self._proposals[proposal.proposal_id] = proposal
        if run is not None and verdict is not None:
            self._proposal_runs[proposal.proposal_id] = (run, verdict)

        if self.storage_path is None:
            return

        if self._storage_type == "sqlite" and self._conn is not None:
            run_info = self._proposal_runs.get(proposal.proposal_id)
            run_json = run_info[0].model_dump_json() if run_info else None
            verd_json = json.dumps(dataclasses.asdict(run_info[1])) if run_info else None
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT OR REPLACE INTO improvement_proposals (
                    proposal_id, task, status, created_at, proposal_json, run_json, verdict_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    proposal.proposal_id,
                    proposal.task,
                    proposal.status.value,
                    proposal.created_at.isoformat(),
                    proposal.model_dump_json(),
                    run_json,
                    verd_json,
                ),
            )
        elif self._storage_type == "json":
            self._save_to_json()

    def _find_best_memory_match(
        self,
        task: str,
        limit: int = 10,
    ) -> tuple[PlaybookRecord | None, float]:
        """Find best matching validated or reusable playbook record from memory.

        Prefers exact normalized task match. If none, falls back to records meeting
        the similarity threshold.
        """
        candidates = self.memory.find_similar(task, limit=limit)
        if not candidates:
            return None, 0.0

        norm_task = normalize_task(task)
        # 1. Exact normalized match
        for rec in candidates:
            if normalize_task(rec.task) == norm_task:
                return rec, 1.0

        # 2. Similarity threshold match
        best_rec: PlaybookRecord | None = None
        best_sim = 0.0
        for rec in candidates:
            sim = task_similarity(rec.task, task)
            if sim > best_sim:
                best_sim = sim
                best_rec = rec

        if best_sim >= self.similarity_threshold:
            return best_rec, best_sim

        return None, best_sim

    def _find_matching_proposal(
        self,
        task: str,
        status: ProposalStatus = ProposalStatus.PENDING,
    ) -> tuple[ImprovementProposal | None, float]:
        """Find existing proposal matching task by normalized match or similarity threshold."""
        norm_task = normalize_task(task)
        # 1. Prefer exact normalized match
        for prop in self._proposals.values():
            if prop.status != status:
                continue
            if normalize_task(prop.task) == norm_task:
                return prop, 1.0

        # 2. Minimum similarity threshold
        best_prop: ImprovementProposal | None = None
        best_sim = 0.0
        for prop in self._proposals.values():
            if prop.status != status:
                continue
            sim = task_similarity(prop.task, task)
            if sim > best_sim:
                best_sim = sim
                best_prop = prop

        if best_sim >= self.similarity_threshold:
            return best_prop, best_sim

        return None, best_sim

    def evaluate_improvement(
        self,
        task: str,
        baseline: Playbook | None,
        candidate: Playbook,
        run: RunState,
        verdict: Verdict,
    ) -> ImprovementProposal | None:
        """Evaluate a successful execution run for potential self-improvement proposals.

        A proposal is created when:
        1. Replan Recovery: baseline is not None and baseline != candidate, representing
           a successful recovery from a failed initial plan.
        2. Optimization: candidate achieves the objective with fewer steps (steps_saved > 0)
           or lower risk (risk_delta < 0) compared to an existing validated/reusable playbook.
        3. High-Confidence Reusable Candidate: a validated playbook matches candidate with
           >= min_runs_for_reusable and zero recorded tool failures.

        A single run NEVER auto-promotes to 'reusable'.
        """
        # Only evaluate runs that succeeded deterministically and passed verification
        if run.status != RunStatus.SUCCEEDED or not verdict.passed:
            return None

        candidate_risk = _compute_playbook_risk(candidate, self.registry)
        evidence_ids = [run.run_id] if run.run_id else []

        # Trigger A: Replan recovery (baseline != candidate)
        if baseline is not None and baseline != candidate:
            baseline_risk = _compute_playbook_risk(baseline, self.registry)
            steps_saved = len(baseline.steps) - len(candidate.steps)
            risk_delta = candidate_risk._rank - baseline_risk._rank

            # Check if a matching pending proposal already exists
            existing_prop, _ = self._find_matching_proposal(task, ProposalStatus.PENDING)
            if existing_prop is not None:
                # Same candidate playbook: aggregate evidence
                if existing_prop.proposed_playbook == candidate:
                    if run.run_id and run.run_id not in existing_prop.evidence_run_ids:
                        existing_prop.evidence_run_ids.append(run.run_id)
                        self._save_proposal(existing_prop)
                    return existing_prop

                # Different candidate: update/overwrite only if new candidate improves steps or risk
                if steps_saved > existing_prop.steps_saved or (
                    steps_saved == existing_prop.steps_saved
                    and risk_delta < existing_prop.risk_delta
                ):
                    existing_prop.proposed_playbook = candidate
                    existing_prop.baseline_playbook = baseline
                    existing_prop.steps_saved = steps_saved
                    existing_prop.risk_delta = risk_delta
                    existing_prop.rationale = (
                        f"Replan recovery (updated): candidate playbook succeeded with "
                        f"{len(candidate.steps)} steps (risk: {candidate_risk.value}) after "
                        f"baseline failed ({len(baseline.steps)} steps, "
                        f"risk: {baseline_risk.value}). "
                        f"Steps saved: {steps_saved}, Risk delta: {risk_delta}."
                    )
                    if run.run_id and run.run_id not in existing_prop.evidence_run_ids:
                        existing_prop.evidence_run_ids.append(run.run_id)
                    self._save_proposal(existing_prop, run=run, verdict=verdict)
                    return existing_prop
                else:
                    return existing_prop

            proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
            rationale = (
                f"Replan recovery: candidate playbook succeeded with {len(candidate.steps)} steps "
                f"(risk: {candidate_risk.value}) after baseline failed "
                f"({len(baseline.steps)} steps, risk: {baseline_risk.value}). "
                f"Steps saved: {steps_saved}, Risk delta: {risk_delta}."
            )
            proposal = ImprovementProposal(
                proposal_id=proposal_id,
                task=task,
                baseline_playbook=baseline,
                proposed_playbook=candidate,
                rationale=rationale,
                evidence_run_ids=evidence_ids,
                steps_saved=steps_saved,
                risk_delta=risk_delta,
                status=ProposalStatus.PENDING,
                created_at=datetime.now(timezone.utc),
            )
            self._save_proposal(proposal, run=run, verdict=verdict)
            return proposal

        # Trigger B: Optimization against existing validated or reusable playbooks in memory
        existing_rec, _ = self._find_best_memory_match(task, limit=10)
        if existing_rec is not None:
            existing_pb = existing_rec.playbook
            existing_risk = _compute_playbook_risk(existing_pb, self.registry)

            steps_saved = len(existing_pb.steps) - len(candidate.steps)
            risk_delta = candidate_risk._rank - existing_risk._rank

            # Optimization condition: fewer steps OR lower risk
            if steps_saved > 0 or risk_delta < 0:
                existing_prop, _ = self._find_matching_proposal(task, ProposalStatus.PENDING)
                if existing_prop is not None:
                    if existing_prop.proposed_playbook == candidate:
                        if run.run_id and run.run_id not in existing_prop.evidence_run_ids:
                            existing_prop.evidence_run_ids.append(run.run_id)
                            self._save_proposal(existing_prop)
                        return existing_prop

                    if steps_saved > existing_prop.steps_saved or (
                        steps_saved == existing_prop.steps_saved
                        and risk_delta < existing_prop.risk_delta
                    ):
                        existing_prop.proposed_playbook = candidate
                        existing_prop.baseline_playbook = existing_pb
                        existing_prop.steps_saved = steps_saved
                        existing_prop.risk_delta = risk_delta
                        rationale_parts: list[str] = []
                        if steps_saved > 0:
                            rationale_parts.append(
                                f"saves {steps_saved} step(s) "
                                f"({len(existing_pb.steps)} -> {len(candidate.steps)})"
                            )
                        if risk_delta < 0:
                            rationale_parts.append(
                                f"reduces risk from {existing_risk.value} to "
                                f"{candidate_risk.value} (delta: {risk_delta})"
                            )

                        existing_prop.rationale = (
                            f"Playbook optimization: candidate achieves task with "
                            f"{', '.join(rationale_parts)} compared to existing "
                            f"{existing_rec.status} playbook (record {existing_rec.record_id})."
                        )
                        if run.run_id and run.run_id not in existing_prop.evidence_run_ids:
                            existing_prop.evidence_run_ids.append(run.run_id)
                        self._save_proposal(existing_prop, run=run, verdict=verdict)
                        return existing_prop
                    else:
                        return existing_prop

                proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
                rationale_parts = []
                if steps_saved > 0:
                    rationale_parts.append(
                        f"saves {steps_saved} step(s) "
                        f"({len(existing_pb.steps)} -> {len(candidate.steps)})"
                    )
                if risk_delta < 0:
                    rationale_parts.append(
                        f"reduces risk from {existing_risk.value} to "
                        f"{candidate_risk.value} (delta: {risk_delta})"
                    )

                rationale = (
                    f"Playbook optimization: candidate achieves task with "
                    f"{', '.join(rationale_parts)} compared to existing "
                    f"{existing_rec.status} playbook (record {existing_rec.record_id})."
                )
                proposal = ImprovementProposal(
                    proposal_id=proposal_id,
                    task=task,
                    baseline_playbook=existing_pb,
                    proposed_playbook=candidate,
                    rationale=rationale,
                    evidence_run_ids=evidence_ids,
                    steps_saved=steps_saved,
                    risk_delta=risk_delta,
                    status=ProposalStatus.PENDING,
                    created_at=datetime.now(timezone.utc),
                )
                self._save_proposal(proposal, run=run, verdict=verdict)
                return proposal

            # Trigger C: Reusable qualification based on min_runs_for_reusable
            if (
                existing_rec.status == "validated"
                and existing_rec.run_count >= self.min_runs_for_reusable
            ):
                # Check for tool failures
                tool_errors: list[str] = []
                for step in candidate.steps:
                    tool_errors.extend(self.memory.tool_limitations(step.tool))

                if not tool_errors:
                    existing_prop, _ = self._find_matching_proposal(task, ProposalStatus.PENDING)
                    if existing_prop is not None:
                        if existing_prop.proposed_playbook == candidate:
                            if run.run_id and run.run_id not in existing_prop.evidence_run_ids:
                                existing_prop.evidence_run_ids.append(run.run_id)
                                self._save_proposal(existing_prop)
                            return existing_prop
                        return existing_prop

                    proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
                    rationale = (
                        f"Reusable promotion qualification: playbook reached "
                        f"{existing_rec.run_count} successful runs "
                        f"(threshold: {self.min_runs_for_reusable}) with zero tool limitations."
                    )
                    proposal = ImprovementProposal(
                        proposal_id=proposal_id,
                        task=task,
                        baseline_playbook=existing_pb,
                        proposed_playbook=candidate,
                        rationale=rationale,
                        evidence_run_ids=evidence_ids,
                        steps_saved=steps_saved,
                        risk_delta=risk_delta,
                        status=ProposalStatus.PENDING,
                        created_at=datetime.now(timezone.utc),
                    )
                    self._save_proposal(proposal, run=run, verdict=verdict)
                    return proposal

        # No improvement or promotion criteria met; single successful run does not auto-promote
        return None

    def list_proposals(self, status: ProposalStatus | None = None) -> list[ImprovementProposal]:
        """List proposals, optionally filtered by status."""
        if status is None:
            return list(self._proposals.values())
        return [p for p in self._proposals.values() if p.status == status]

    def get_proposal(self, proposal_id: str) -> ImprovementProposal:
        """Retrieve a specific proposal by ID."""
        if proposal_id not in self._proposals:
            raise KeyError(f"Improvement proposal '{proposal_id}' not found.")
        return self._proposals[proposal_id]

    def approve(self, proposal_id: str) -> PlaybookRecord:
        """Approve an improvement proposal, promoting the proposed playbook to 'reusable'.

        Updates proposal status to APPROVED, ensures the playbook is recorded in MemoryStore,
        promotes it to 'reusable', and returns the resulting PlaybookRecord.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Improvement proposal '{proposal_id}' not found.")

        proposal.status = ProposalStatus.APPROVED
        self._save_proposal(proposal)

        # Ensure run state is available for memory.save_run
        run_info = self._proposal_runs.get(proposal_id)
        if run_info is not None:
            run, verdict = run_info
        else:
            evidence_id = (
                proposal.evidence_run_ids[0]
                if proposal.evidence_run_ids
                else f"run_{uuid.uuid4().hex[:8]}"
            )
            run = RunState(
                run_id=evidence_id,
                playbook=proposal.proposed_playbook,
                status=RunStatus.SUCCEEDED,
            )
            verdict = Verdict(passed=True)

        record_id = self.memory.save_run(
            task=proposal.task,
            playbook=proposal.proposed_playbook,
            run=run,
            verdict=verdict,
        )
        self.memory.promote(record_id, to="reusable")

        # Retrieve promoted record from MemoryStore
        records = self.memory.find_similar(proposal.task, limit=10)
        for rec in records:
            if rec.record_id == record_id:
                return rec

        return PlaybookRecord(
            record_id=record_id,
            task=proposal.task,
            playbook=proposal.proposed_playbook,
            status="reusable",
            run_count=1,
            last_used=datetime.now(timezone.utc),
        )

    def reject(self, proposal_id: str, reason: str = "") -> ImprovementProposal:
        """Reject an improvement proposal.

        Updates proposal status to REJECTED, records rejection reason, and leaves
        MemoryStore unpromoted.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"Improvement proposal '{proposal_id}' not found.")

        proposal.status = ProposalStatus.REJECTED
        proposal.rejection_reason = reason
        self._save_proposal(proposal)
        return proposal

    def close(self) -> None:
        """Close any open database connections."""
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> ImprovementManager:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
