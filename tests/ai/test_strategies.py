from __future__ import annotations

import json
from typing import Any

import pytest

from calbi.ai.strategies import (
    StrategyCandidate,
    StrategyExplorationFailed,
    StrategyExplorer,
    _playbook_signature,
)
from calbi.ai.verifier import Verdict, Verifier
from calbi.core.policy import Autonomy, Policy
from calbi.core.registry import ToolRegistry
from calbi.core.schema import Playbook, Risk, Step
from calbi.core.state import RunState, RunStatus
from calbi.core.validator import ValidationReport
from calbi.testing.fake_llm import FakeLLM


def setup_tools() -> tuple[ToolRegistry, dict[str, int]]:
    reg = ToolRegistry()
    calls: dict[str, int] = {
        "data.read_a": 0,
        "data.read_b": 0,
        "data.read_c": 0,
        "data.fail": 0,
        "data.write": 0,
        "data.purge": 0,
    }

    @reg.tool(name="data.read_a", description="Read source via method A", risk=Risk.READ)
    def read_a(source: str) -> dict[str, Any]:
        calls["data.read_a"] += 1
        return {"data": f"content_a from {source}", "count": 10}

    @reg.tool(name="data.read_b", description="Read source via method B", risk=Risk.READ)
    def read_b(source: str) -> dict[str, Any]:
        calls["data.read_b"] += 1
        return {"data": f"content_b from {source}", "count": 15}

    @reg.tool(name="data.read_c", description="Read source via method C", risk=Risk.READ)
    def read_c(source: str) -> dict[str, Any]:
        calls["data.read_c"] += 1
        return {"data": f"content_c from {source}", "count": 20}

    @reg.tool(name="data.fail", description="Always failing tool", risk=Risk.READ)
    def failing_tool(source: str) -> dict[str, Any]:
        calls["data.fail"] += 1
        raise RuntimeError("Deterministic tool failure")

    @reg.tool(name="data.write", description="Write output data", risk=Risk.WRITE)
    def write_tool(data: str, destination: str) -> dict[str, Any]:
        calls["data.write"] += 1
        return {"written": True, "destination": destination}

    @reg.tool(name="data.purge", description="Purge database", risk=Risk.DESTRUCTIVE)
    def purge_tool(target: str) -> dict[str, Any]:
        calls["data.purge"] += 1
        return {"purged": True, "target": target}

    return reg, calls


def _make_candidate_json(
    name: str,
    tool: str,
    inputs: dict[str, Any],
    rationale: str = "Test rationale",
    playbook_inputs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "rationale": rationale,
        "playbook": {
            "name": name,
            "inputs": playbook_inputs or {},
            "steps": [
                {
                    "id": "step_1",
                    "tool": tool,
                    "inputs": inputs,
                }
            ],
            "outputs": {"res": "{{ steps.step_1.output.data }}"},
        },
    }


# ---------------------------------------------------------------------------
# Candidate Generation & Deduplication Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_candidates_with_deduplication() -> None:
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    # 3 candidates: strategy 1 and strategy 2 have identical tool and inputs (duplicate signature)
    # strategy 3 has a different tool
    cand1 = _make_candidate_json(
        "strat_1",
        "data.read_a",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )
    cand2 = _make_candidate_json(
        "strat_2_dup",
        "data.read_a",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )
    cand3 = _make_candidate_json(
        "strat_3_unique",
        "data.read_b",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )

    llm_response = json.dumps({"candidates": [cand1, cand2, cand3]})
    llm = FakeLLM(responses=[llm_response])

    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy, max_candidates=3)
    candidates = await explorer.generate_candidates(task="Read dataset", inputs={"source": "db"})

    assert len(candidates) == 2
    assert candidates[0].name == "strat_1"
    assert candidates[1].name == "strat_3_unique"
    assert candidates[0].validation.ok is True
    assert candidates[1].validation.ok is True


@pytest.mark.asyncio
async def test_generate_candidates_filters_invalid_playbooks() -> None:
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    valid_cand = _make_candidate_json(
        "valid_strat",
        "data.read_a",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )
    invalid_cand = {
        "name": "invalid_strat",
        "rationale": "Uses undeclared tool",
        "playbook": {
            "name": "bad",
            "steps": [{"id": "s1", "tool": "nonexistent.tool", "inputs": {}}],
        },
    }

    llm_response = json.dumps({"candidates": [invalid_cand, valid_cand]})
    llm = FakeLLM(responses=[llm_response])

    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)
    candidates = await explorer.generate_candidates(task="Read data", inputs={"source": "x"})

    assert len(candidates) == 1
    assert candidates[0].name == "valid_strat"


@pytest.mark.asyncio
async def test_generate_candidates_empty_or_negative_k() -> None:
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)
    llm = FakeLLM(responses=[])

    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)
    assert await explorer.generate_candidates(task="Task", k=0) == []
    assert await explorer.generate_candidates(task="Task", k=-1) == []


# ---------------------------------------------------------------------------
# Sequential Mode Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_exploration_early_stopping() -> None:
    """Strategy A fails, Strategy B passes with high confidence -> Strategy C is not executed."""
    reg, calls = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand_a = _make_candidate_json(
        "strat_fail",
        "data.fail",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )
    cand_b = _make_candidate_json(
        "strat_pass_b",
        "data.read_b",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )
    cand_c = _make_candidate_json(
        "strat_pass_c",
        "data.read_c",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )

    candidates_json = json.dumps({"candidates": [cand_a, cand_b, cand_c]})

    # Verifier will evaluate:
    # 1. Candidate A: deterministic run failure -> Verdict(passed=False)
    # 2. Candidate B: succeeds -> LLM verifier returns confidence=0.9, passed=True -> early stop!
    verifier_resp = json.dumps(
        {
            "passed": True,
            "reasons": ["High quality data returned"],
            "evidence": ["15 rows"],
            "confidence": 0.9,
        }
    )

    llm = FakeLLM(responses=[candidates_json, verifier_resp])
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy, max_candidates=3)

    result = await explorer.explore(
        task="Read data safely",
        inputs={"source": "prod_db"},
        mode="sequential",
    )

    assert result.winner is not None
    assert result.winner.candidate.name == "strat_pass_b"
    assert result.winner.verdict is not None
    assert result.winner.verdict.passed is True
    assert result.winner.verdict.confidence == 0.9
    assert result.winner.score > 0.0

    # Early stopping: only 2 trials executed
    assert len(result.trials) == 2
    assert result.trials[0].candidate.name == "strat_fail"
    assert result.trials[0].verdict is not None
    assert result.trials[0].verdict.passed is False
    assert result.trials[0].score == 0.0

    # Ensure cand_c was NEVER executed
    assert calls["data.read_c"] == 0
    assert calls["data.read_b"] == 1
    assert calls["data.fail"] == 1


@pytest.mark.asyncio
async def test_sequential_exploration_without_early_stopping() -> None:
    """Strategy A passes with low confidence (< 0.8), Strategy B with high confidence -> B wins."""
    reg, calls = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand_a = _make_candidate_json(
        "strat_low",
        "data.read_a",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )
    cand_b = _make_candidate_json(
        "strat_high",
        "data.read_b",
        {"source": "{{ inputs.source }}"},
        playbook_inputs={"source": {"type": "string"}},
    )

    candidates_json = json.dumps({"candidates": [cand_a, cand_b]})
    verdict_low = json.dumps({"passed": True, "reasons": ["Acceptable"], "confidence": 0.5})
    verdict_high = json.dumps({"passed": True, "reasons": ["Excellent"], "confidence": 0.95})

    llm = FakeLLM(responses=[candidates_json, verdict_low, verdict_high])
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy, max_candidates=2)

    result = await explorer.explore(
        task="Read data",
        inputs={"source": "db"},
        mode="sequential",
    )

    assert result.winner is not None
    assert result.winner.candidate.name == "strat_high"
    assert len(result.trials) == 2
    assert calls["data.read_a"] == 1
    assert calls["data.read_b"] == 1
    assert result.winner.score > result.trials[0].score


@pytest.mark.asyncio
async def test_sequential_exploration_all_fail() -> None:
    reg, calls = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand_a = _make_candidate_json("strat_1", "data.fail", {"source": "x"})
    cand_b = _make_candidate_json("strat_2", "data.fail", {"source": "y"})

    llm = FakeLLM(responses=[json.dumps({"candidates": [cand_a, cand_b]})])
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)

    result = await explorer.explore(task="Read data", inputs={}, mode="sequential")

    assert result.winner is None
    assert len(result.trials) == 2
    assert all(t.score == 0.0 for t in result.trials)
    assert calls["data.fail"] == 2


# ---------------------------------------------------------------------------
# Parallel Read-Only Mode Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parallel_read_only_exploration_picking_best_winner() -> None:
    reg, calls = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand_a = _make_candidate_json("strat_a", "data.read_a", {"source": "test"})
    cand_b = _make_candidate_json("strat_b", "data.read_b", {"source": "test"})

    candidates_json = json.dumps({"candidates": [cand_a, cand_b]})

    # Use Verifier with custom confidence
    class CustomVerifier(Verifier):
        async def verify(
            self,
            task: str,
            expected_output: str | None,
            run: RunState,
        ) -> Verdict:
            step = run.steps.get("step_1")
            conf = 0.95 if step and "content_b" in str(step.output) else 0.70
            return Verdict(passed=True, reasons=["Success"], confidence=conf)

    llm = FakeLLM(responses=[candidates_json])
    explorer = StrategyExplorer(
        llm=llm,
        registry=reg,
        policy=policy,
        verifier=CustomVerifier(),
        max_candidates=2,
    )

    result = await explorer.explore(
        task="Read concurrently",
        inputs={},
        mode="parallel_read_only",
    )

    assert result.winner is not None
    assert result.winner.candidate.name == "strat_b"
    assert len(result.trials) == 2
    assert calls["data.read_a"] == 1
    assert calls["data.read_b"] == 1
    assert result.trials[1].score > result.trials[0].score


@pytest.mark.asyncio
async def test_parallel_read_only_rejection_of_non_read() -> None:
    """Candidates with Risk.WRITE or Risk.DESTRUCTIVE are rejected before execution."""
    reg, calls = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand_read = _make_candidate_json("read_strat", "data.read_a", {"source": "s"})
    cand_write = {
        "name": "write_strat",
        "rationale": "Writes file",
        "playbook": {
            "name": "write_pb",
            "steps": [
                {
                    "id": "step_write",
                    "tool": "data.write",
                    "inputs": {"data": "d", "destination": "out.txt"},
                }
            ],
        },
    }
    cand_destr = {
        "name": "purge_strat",
        "rationale": "Purges file",
        "playbook": {
            "name": "purge_pb",
            "steps": [
                {
                    "id": "step_purge",
                    "tool": "data.purge",
                    "inputs": {"target": "table"},
                }
            ],
        },
    }

    candidates_json = json.dumps({"candidates": [cand_read, cand_write, cand_destr]})
    verifier = Verifier(llm=None)
    llm = FakeLLM(responses=[candidates_json])

    explorer = StrategyExplorer(
        llm=llm,
        registry=reg,
        policy=policy,
        verifier=verifier,
        max_candidates=3,
    )

    result = await explorer.explore(
        task="Explore options",
        inputs={},
        mode="parallel_read_only",
    )

    assert result.winner is not None
    assert result.winner.candidate.name == "read_strat"
    assert len(result.trials) == 3

    # Trial 1: READ -> executed
    assert result.trials[0].run is not None
    assert result.trials[0].run.status == RunStatus.SUCCEEDED
    assert calls["data.read_a"] == 1

    # Trial 2: WRITE -> rejected before execution
    assert result.trials[1].candidate.name == "write_strat"
    assert result.trials[1].run is None
    assert result.trials[1].verdict is not None
    assert result.trials[1].verdict.passed is False
    assert result.trials[1].score == 0.0
    assert calls["data.write"] == 0

    # Trial 3: DESTRUCTIVE -> rejected before execution
    assert result.trials[2].candidate.name == "purge_strat"
    assert result.trials[2].run is None
    assert result.trials[2].verdict is not None
    assert result.trials[2].verdict.passed is False
    assert result.trials[2].score == 0.0
    assert calls["data.purge"] == 0


@pytest.mark.asyncio
async def test_parallel_read_only_all_rejected_returns_no_winner() -> None:
    reg, calls = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand_write = {
        "name": "write_strat",
        "playbook": {
            "name": "write_pb",
            "steps": [
                {
                    "id": "s1",
                    "tool": "data.write",
                    "inputs": {"data": "d", "destination": "x"},
                }
            ],
        },
    }

    llm = FakeLLM(responses=[json.dumps({"candidates": [cand_write]})])
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)

    result = await explorer.explore(task="Write", inputs={}, mode="parallel_read_only")

    assert result.winner is None
    assert len(result.trials) == 1
    assert result.trials[0].score == 0.0
    assert calls["data.write"] == 0


# ---------------------------------------------------------------------------
# Error Handling & Edge Cases
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_explore_raises_when_no_valid_candidates() -> None:
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    # Empty candidate list from LLM
    llm = FakeLLM(responses=[json.dumps({"candidates": []})])
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)

    with pytest.raises(StrategyExplorationFailed, match="no valid candidates generated"):
        await explorer.explore(task="Do something impossible")


@pytest.mark.asyncio
async def test_invalid_mode_raises_value_error() -> None:
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)
    llm = FakeLLM(responses=[])
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)

    with pytest.raises(ValueError, match="Unknown exploration mode"):
        await explorer.explore(task="Test", mode="invalid_mode")  # type: ignore[arg-type]


def test_calculate_score_logic() -> None:
    step = Step(id="s1", tool="data.read_a", inputs={})
    dummy_pb = Playbook(name="dummy", steps=[step])
    dummy_report = ValidationReport(
        ok=True, issues=[], max_risk=Risk.READ, approvals_needed=[], step_count=1
    )
    cand = StrategyCandidate(
        name="test",
        rationale="r",
        playbook=dummy_pb,
        validation=dummy_report,
    )

    # 1. Null / Failed run -> 0.0
    assert StrategyExplorer.calculate_score(cand, None, Verdict(passed=True)) == 0.0
    failed_run = RunState(run_id="r1", playbook=dummy_pb, status=RunStatus.FAILED)
    assert StrategyExplorer.calculate_score(cand, failed_run, Verdict(passed=True)) == 0.0

    # 2. Failed verdict -> 0.0
    succeeded_run = RunState(run_id="r1", playbook=dummy_pb, status=RunStatus.SUCCEEDED)
    assert StrategyExplorer.calculate_score(cand, succeeded_run, Verdict(passed=False)) == 0.0

    # 3. Passed run composite:
    # 1 step, max_risk=Risk.READ (rank 1), confidence=1.0
    # confidence: 1.0 (weight 0.5 -> 0.5)
    # efficiency: 1.0 / (1.0 + 0.1 * 1) = 1.0 / 1.1 = 0.9091 (weight 0.3 -> 0.2727)
    # risk: (7 - 1) / 6 = 1.0 (weight 0.2 -> 0.20)
    # sum: 0.5 + 0.2727 + 0.20 = 0.9727
    score = StrategyExplorer.calculate_score(
        cand, succeeded_run, Verdict(passed=True, confidence=1.0)
    )
    assert 0.97 <= score <= 0.98


@pytest.mark.asyncio
async def test_total_attempts_calculation() -> None:
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand = _make_candidate_json("strat", "data.read_a", {"source": "s"})
    llm = FakeLLM(responses=[json.dumps({"candidates": [cand]})])
    verifier = Verifier(llm=None)

    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy, verifier=verifier)
    result = await explorer.explore(task="Read", mode="sequential")

    assert result.winner is not None
    assert result.total_attempts == 1


def test_playbook_signature_determinism() -> None:
    pb1 = Playbook(
        name="p1",
        steps=[
            Step(id="b", tool="tool_b", inputs={"x": 1}),
            Step(id="a", tool="tool_a", inputs={"y": 2}),
        ],
    )
    pb2 = Playbook(
        name="p2",
        steps=[
            Step(id="x", tool="tool_a", inputs={"y": 2}),
            Step(id="y", tool="tool_b", inputs={"x": 1}),
        ],
    )

    # Different step IDs and different insertion order, but same tools and inputs
    assert _playbook_signature(pb1) == _playbook_signature(pb2)


# ---------------------------------------------------------------------------
# Expected Output Forwarding & Scoring Tests (WP-23 / R10)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_candidates_with_expected_output_includes_section_in_prompt() -> None:
    """When expected_output is provided, prompt must explicitly include section and content."""
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand = _make_candidate_json("strat_1", "data.read_a", {"source": "db"})
    llm = FakeLLM(responses=[json.dumps({"candidates": [cand]})])
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)

    target_output = "A Markdown table containing rows with columns: name, count, and status"
    candidates = await explorer.generate_candidates(
        task="Generate report",
        inputs={"source": "db"},
        expected_output=target_output,
    )

    assert len(candidates) == 1
    assert candidates[0].name == "strat_1"
    assert len(llm.calls) == 1

    user_msg = [m for m in llm.calls[0].messages if m.role == "user"][0].content
    assert "EXPECTED OUTPUT:" in user_msg
    assert target_output in user_msg


@pytest.mark.asyncio
async def test_generate_candidates_omitted_or_none_expected_output_omits_section() -> None:
    """When expected_output is None or omitted, EXPECTED OUTPUT header is omitted gracefully."""
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand = _make_candidate_json("strat_1", "data.read_a", {"source": "db"})
    llm = FakeLLM(
        responses=[
            json.dumps({"candidates": [cand]}),
            json.dumps({"candidates": [cand]}),
            json.dumps({"candidates": [cand]}),
        ]
    )
    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)

    # 1. Completely omitted
    await explorer.generate_candidates(task="Task 1", inputs={"source": "db"})
    user_msg_omitted = [m for m in llm.calls[0].messages if m.role == "user"][0].content
    assert "EXPECTED OUTPUT" not in user_msg_omitted

    # 2. Explicitly None
    await explorer.generate_candidates(task="Task 2", inputs={"source": "db"}, expected_output=None)
    user_msg_none = [m for m in llm.calls[1].messages if m.role == "user"][0].content
    assert "EXPECTED OUTPUT" not in user_msg_none

    # 3. Blank / whitespace string
    await explorer.generate_candidates(
        task="Task 3", inputs={"source": "db"}, expected_output="   \n  "
    )
    user_msg_blank = [m for m in llm.calls[2].messages if m.role == "user"][0].content
    assert "EXPECTED OUTPUT" not in user_msg_blank


@pytest.mark.asyncio
async def test_explore_forwards_expected_output_to_candidate_generation() -> None:
    """StrategyExplorer.explore forwards expected_output to generate_candidates prompt."""
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand = _make_candidate_json("strat_1", "data.read_a", {"source": "db"})
    cand_resp = json.dumps({"candidates": [cand]})
    verifier_resp = json.dumps({"passed": True, "reasons": ["Success"], "confidence": 0.95})
    llm = FakeLLM(responses=[cand_resp, verifier_resp])

    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)
    target_output = "Normalized records in JSON schema"

    result = await explorer.explore(
        task="Process dataset",
        inputs={"source": "db"},
        expected_output=target_output,
        mode="sequential",
    )

    assert result.winner is not None
    # Verify generate_candidates prompt received expected_output
    gen_user_msg = [m for m in llm.calls[0].messages if m.role == "user"][0].content
    assert "EXPECTED OUTPUT:" in gen_user_msg
    assert target_output in gen_user_msg

    # Verify verifier prompt also received expected_output
    verifier_user_msg = [m for m in llm.calls[1].messages if m.role == "user"][0].content
    assert target_output in verifier_user_msg


@pytest.mark.asyncio
async def test_explore_without_expected_output_omits_section_in_prompt() -> None:
    """StrategyExplorer.explore without expected_output omits EXPECTED OUTPUT section in prompt."""
    reg, _ = setup_tools()
    policy = Policy(autonomy=Autonomy.ADAPTIVE)

    cand = _make_candidate_json("strat_1", "data.read_a", {"source": "db"})
    cand_resp = json.dumps({"candidates": [cand]})
    verifier_resp = json.dumps({"passed": True, "reasons": ["Success"], "confidence": 0.95})
    llm = FakeLLM(responses=[cand_resp, verifier_resp])

    explorer = StrategyExplorer(llm=llm, registry=reg, policy=policy)

    result = await explorer.explore(
        task="Process dataset",
        inputs={"source": "db"},
        mode="sequential",
    )

    assert result.winner is not None
    gen_user_msg = [m for m in llm.calls[0].messages if m.role == "user"][0].content
    assert "EXPECTED OUTPUT" not in gen_user_msg


def test_calculate_score_public_risk_comparison_all_ranks() -> None:
    """calculate_score computes correct risk factor using public comparisons across Risk levels."""
    step = Step(id="s1", tool="data.read_a", inputs={})
    dummy_pb = Playbook(name="dummy", steps=[step])
    succeeded_run = RunState(run_id="r1", playbook=dummy_pb, status=RunStatus.SUCCEEDED)
    verdict = Verdict(passed=True, confidence=1.0)

    # Risk ranking order: READ (1), WRITE (2), DESTRUCTIVE (3), EXTERNAL (4), FINANCIAL (5), DEPLOY
    expected_risk_factors = {
        Risk.READ: (7.0 - 1.0) / 6.0,  # 1.0
        Risk.WRITE: (7.0 - 2.0) / 6.0,  # ~0.8333
        Risk.DESTRUCTIVE: (7.0 - 3.0) / 6.0,  # ~0.6667
        Risk.EXTERNAL: (7.0 - 4.0) / 6.0,  # 0.5
        Risk.FINANCIAL: (7.0 - 5.0) / 6.0,  # ~0.3333
        Risk.DEPLOY: (7.0 - 6.0) / 6.0,  # ~0.1667
    }

    efficiency = 1.0 / (1.0 + 0.1 * 1.0)  # 1/1.1 = ~0.90909
    confidence_term = 0.5 * 1.0
    efficiency_term = 0.3 * efficiency

    previous_score = 1.0
    for risk, factor in expected_risk_factors.items():
        report = ValidationReport(
            ok=True, issues=[], max_risk=risk, approvals_needed=[], step_count=1
        )
        cand = StrategyCandidate(name="t", rationale="r", playbook=dummy_pb, validation=report)
        score = StrategyExplorer.calculate_score(cand, succeeded_run, verdict)
        expected_score = round(confidence_term + efficiency_term + 0.2 * factor, 4)
        assert score == expected_score
        assert score <= previous_score
        previous_score = score
