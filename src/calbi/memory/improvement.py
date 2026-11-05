from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

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


def _compute_playbook_risk(
    playbook: Playbook, registry: ToolRegistry | None = None
) -> Risk:
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


class ImprovementManager:
    """Manages playbook evolution proposals, evaluating optimizations and gating promotion."""

    def __init__(
        self,
        memory: MemoryStore,
        llm: LLM | None = None,
        min_runs_for_reusable: int = 5,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.min_runs_for_reusable = min_runs_for_reusable
        self.registry = registry
        self._proposals: dict[str, ImprovementProposal] = {}
        self._proposal_runs: dict[str, tuple[RunState, Verdict]] = {}

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

            # Check if an identical pending proposal already exists to aggregate evidence
            for prop in self._proposals.values():
                if (
                    prop.status == ProposalStatus.PENDING
                    and prop.task == task
                    and prop.proposed_playbook == candidate
                    and prop.baseline_playbook == baseline
                ):
                    if run.run_id and run.run_id not in prop.evidence_run_ids:
                        prop.evidence_run_ids.append(run.run_id)
                    return prop

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
            self._proposals[proposal_id] = proposal
            self._proposal_runs[proposal_id] = (run, verdict)
            return proposal

        # Trigger B: Optimization against existing validated or reusable playbooks in memory
        similar_records = self.memory.find_similar(task, limit=1)
        if similar_records:
            existing_rec = similar_records[0]
            existing_pb = existing_rec.playbook
            existing_risk = _compute_playbook_risk(existing_pb, self.registry)

            steps_saved = len(existing_pb.steps) - len(candidate.steps)
            risk_delta = candidate_risk._rank - existing_risk._rank

            # Optimization condition: fewer steps OR lower risk
            if steps_saved > 0 or risk_delta < 0:
                for prop in self._proposals.values():
                    if (
                        prop.status == ProposalStatus.PENDING
                        and prop.task == task
                        and prop.proposed_playbook == candidate
                    ):
                        if run.run_id and run.run_id not in prop.evidence_run_ids:
                            prop.evidence_run_ids.append(run.run_id)
                        return prop

                proposal_id = f"prop_{uuid.uuid4().hex[:12]}"
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
                self._proposals[proposal_id] = proposal
                self._proposal_runs[proposal_id] = (run, verdict)
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
                    for prop in self._proposals.values():
                        if (
                            prop.status == ProposalStatus.PENDING
                            and prop.task == task
                            and prop.proposed_playbook == candidate
                        ):
                            if run.run_id and run.run_id not in prop.evidence_run_ids:
                                prop.evidence_run_ids.append(run.run_id)
                            return prop

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
                    self._proposals[proposal_id] = proposal
                    self._proposal_runs[proposal_id] = (run, verdict)
                    return proposal

        # No improvement or promotion criteria met; single successful run does not auto-promote
        return None

    def list_proposals(
        self, status: ProposalStatus | None = None
    ) -> list[ImprovementProposal]:
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
        return proposal
