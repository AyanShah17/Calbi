"""End-to-end integration tests for Calbi V0.2 Facade & Runtime Integration (WP-18)."""

from __future__ import annotations

import io
import json
from typing import Any

import pytest

import calbi
from calbi import (
    Calbi,
    DuplicateToolError,
    Event,
    ImprovementManager,
    InProcessExecutor,
    JSONLEventSink,
    Playbook,
    ProposalStatus,
    Redactor,
    Risk,
    Runner,
    RunState,
    RunStatus,
    Runtime,
    Step,
    StrategyExplorer,
    ToolRegistry,
    ToolSpec,
    TraceCollector,
)
from calbi.ai.verifier import Verdict
from calbi.memory.sqlite import SQLiteMemory
from calbi.testing.fake_llm import FakeLLM

# ---------------------------------------------------------------------------
# Helpers & Mocks
# ---------------------------------------------------------------------------


class RecordingExecutor:
    """StepExecutor that records execution calls while executing tools genuine logic."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []
        self._inner = InProcessExecutor()

    async def execute(
        self,
        spec: ToolSpec,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        self.calls.append((spec.name, dict(inputs), timeout_s))
        return await self._inner.execute(spec, inputs, timeout_s=timeout_s)


class CustomListEventSink:
    """EventSink storing received events into an in-memory list."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def __call__(self, event: Event) -> None:
        self.events.append(event)


# ---------------------------------------------------------------------------
# 1. Public Types and Re-Exports
# ---------------------------------------------------------------------------


def test_v02_public_exports_and_all() -> None:
    """All required V0.2 public types must be cleanly re-exported in calbi.__all__."""
    required_symbols = [
        "Calbi",
        "Runner",
        "Runtime",
        "ToolRegistry",
        "ToolSpec",
        "Risk",
        "Policy",
        "Autonomy",
        "StepExecutor",
        "InProcessExecutor",
        "SubprocessExecutor",
        "TraceCollector",
        "Redactor",
        "StrategyExplorer",
        "ImprovementManager",
        "ImprovementProposal",
        "ProposalStatus",
        "EventSink",
        "JSONLEventSink",
        "SpanRecord",
        "register_subplaybook",
        "register_map_tool",
    ]
    for sym in required_symbols:
        assert hasattr(calbi, sym), f"calbi missing expected public symbol '{sym}'"
        assert sym in calbi.__all__, f"calbi.__all__ missing '{sym}'"

    assert Runner is Runtime


# ---------------------------------------------------------------------------
# 2. Risk.rank and ToolRegistry.add_spec
# ---------------------------------------------------------------------------


def test_risk_rank_properties_and_ordering() -> None:
    """Risk.rank must return integer rank from 1 to 6 and support comparisons."""
    assert Risk.READ.rank == 1
    assert Risk.WRITE.rank == 2
    assert Risk.DESTRUCTIVE.rank == 3
    assert Risk.EXTERNAL.rank == 4
    assert Risk.FINANCIAL.rank == 5
    assert Risk.DEPLOY.rank == 6

    # Verify method call equivalence if called as a method
    assert Risk.READ.rank() == 1
    assert Risk.DEPLOY.rank() == 6

    # Verify comparisons respect integer ranks
    assert Risk.READ < Risk.WRITE < Risk.DESTRUCTIVE < Risk.EXTERNAL < Risk.FINANCIAL < Risk.DEPLOY
    assert Risk.DEPLOY > Risk.READ
    assert Risk.WRITE >= Risk.READ
    assert Risk.FINANCIAL <= Risk.DEPLOY

    # Verify backwards compatibility with _rank
    assert Risk.READ._rank == 1
    assert Risk.DEPLOY._rank == 6


def test_tool_registry_add_spec() -> None:
    """ToolRegistry.add_spec must register a pre-constructed ToolSpec and reject duplicates."""
    reg = ToolRegistry()

    def multiply(a: int, b: int) -> int:
        return a * b

    spec = ToolSpec(
        name="math.multiply",
        description="Multiply two integers",
        fn=multiply,
        is_async=False,
        input_schema={
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        },
        risk=Risk.READ,
    )

    reg.add_spec(spec)
    assert reg.get("math.multiply") is spec
    assert "math.multiply" in [s.name for s in reg.list()]

    # Duplicate registration must raise DuplicateToolError
    with pytest.raises(DuplicateToolError):
        reg.add_spec(spec)


# ---------------------------------------------------------------------------
# 3. Calbi Facade with Custom StepExecutor
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calbi_with_custom_executor_executes_steps() -> None:
    """Calbi initialized with a StepExecutor executes playbook steps through that executor."""
    recording_executor = RecordingExecutor()
    engine = Calbi(executor=recording_executor)

    @engine.tool(name="greet", risk=Risk.READ)
    def greet(name: str) -> dict[str, str]:
        return {"greeting": f"Hello, {name}!"}

    pb = Playbook(
        name="test_executor_pb",
        steps=[
            Step(
                id="s1",
                tool="greet",
                inputs={"name": "Alice"},
            )
        ],
        outputs={"res": "{{ steps.s1.output.greeting }}"},
    )

    state = await engine.arun(pb)
    assert state.status == RunStatus.SUCCEEDED
    assert state.outputs["res"] == "Hello, Alice!"

    # Verify custom executor received execution call
    assert len(recording_executor.calls) == 1
    tool_name, inputs, timeout = recording_executor.calls[0]
    assert tool_name == "greet"
    assert inputs == {"name": "Alice"}


# ---------------------------------------------------------------------------
# 4. Calbi with EventSinks and TraceCollector
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calbi_with_event_sinks_and_trace_collector() -> None:
    """Calbi captures events across event_sinks and hierarchical spans in trace_collector."""
    custom_sink = CustomListEventSink()
    jsonl_stream = io.StringIO()
    redactor = Redactor(sensitive_keys={"secret_token"})
    jsonl_sink = JSONLEventSink(target=jsonl_stream, redactor=redactor)
    trace_collector = TraceCollector(redactor=redactor)

    engine = Calbi(
        event_sinks=[custom_sink, jsonl_sink],
        trace_collector=trace_collector,
    )

    @engine.tool(name="step_a", risk=Risk.READ)
    def step_a() -> dict[str, Any]:
        return {"data": "foo", "secret_token": "super_secret_value"}

    @engine.tool(name="step_b", risk=Risk.WRITE)
    def step_b(in_data: str) -> dict[str, str]:
        return {"summary": f"processed_{in_data}"}

    pb = Playbook(
        name="observability_pb",
        steps=[
            Step(id="step_1", tool="step_a"),
            Step(
                id="step_2",
                tool="step_b",
                inputs={"in_data": "{{ steps.step_1.output.data }}"},
                depends_on=["step_1"],
            ),
        ],
        outputs={"final": "{{ steps.step_2.output.summary }}"},
    )

    state = await engine.arun(pb)
    assert state.status == RunStatus.SUCCEEDED
    assert state.outputs["final"] == "processed_foo"

    # 1. Custom list event sink verification
    event_types = [e.type for e in custom_sink.events]
    assert "run.started" in event_types
    assert "step.started" in event_types
    assert "step.completed" in event_types
    assert "run.completed" in event_types

    # 2. JSONL sink with Redactor verification
    jsonl_content = jsonl_stream.getvalue()
    lines = [json.loads(line) for line in jsonl_content.strip().split("\n") if line.strip()]
    assert len(lines) >= 4
    # Ensure sensitive key was redacted
    assert "super_secret_value" not in jsonl_content
    step1_completed = next(
        line
        for line in lines
        if line.get("step_id") == "step_1" and line["type"] == "step.completed"
    )
    assert step1_completed["data"]["output"]["secret_token"] == "[REDACTED]"

    # 3. TraceCollector spans verification
    spans = trace_collector.spans
    assert len(spans) == 3  # 1 root run span + 2 step spans

    root_span = next(s for s in spans if s.name.startswith("run:"))
    assert root_span.parent_span_id is None
    assert root_span.status == "ok"

    step_spans = [s for s in spans if s.name.startswith("step:")]
    assert len(step_spans) == 2
    for s in step_spans:
        assert s.parent_span_id == root_span.span_id
        assert s.status == "ok"


# ---------------------------------------------------------------------------
# 5. Calbi with explore=True and auto_propose_improvements=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_calbi_explore_and_auto_propose_improvements_delegation() -> None:
    """Calbi with explore=True and auto_propose_improvements=True
    delegates through strategies and proposes improvements.
    """
    memory = SQLiteMemory(":memory:")

    cand_playbook_json = {
        "name": "data_pipeline",
        "steps": [
            {
                "id": "fetch",
                "tool": "fetch_data",
                "inputs": {"source": "db"},
            }
        ],
        "outputs": {"result": "{{ steps.fetch.output.rows }}"},
    }

    # Responses:
    # 1. Strategy candidate generation
    strat_resp = json.dumps(
        {
            "candidates": [
                {
                    "name": "fast_fetch_strategy",
                    "rationale": "Direct single-step query",
                    "playbook": cand_playbook_json,
                }
            ]
        }
    )
    # 2. Trial verification response for explore
    trial_verdict_resp = json.dumps(
        {
            "passed": True,
            "reasons": ["Fetched expected rows"],
            "confidence": 0.95,
        }
    )
    # 3. Main delegation loop verifier response
    final_verdict_resp = json.dumps(
        {
            "passed": True,
            "reasons": ["Pipeline executed successfully"],
            "confidence": 0.98,
        }
    )

    llm = FakeLLM(responses=[strat_resp, trial_verdict_resp, final_verdict_resp])

    engine = Calbi(
        llm=llm,
        memory=memory,
        explore=True,
        auto_propose_improvements=True,
    )

    assert engine.explore is True
    assert engine.auto_propose_improvements is True
    assert isinstance(engine.strategy_explorer, StrategyExplorer)
    assert isinstance(engine.improvement_manager, ImprovementManager)

    @engine.tool(name="fetch_data", risk=Risk.READ)
    def fetch_data(source: str) -> dict[str, Any]:
        return {"rows": [1, 2, 3], "source": source}

    # Pre-seed memory with an existing baseline for task to trigger Trigger B (Optimization)
    baseline_pb = Playbook(
        name="legacy_pipeline",
        steps=[
            Step(id="s1", tool="fetch_data", inputs={"source": "db"}),
            Step(id="s2", tool="fetch_data", inputs={"source": "db"}, depends_on=["s1"]),
        ],
        outputs={"result": "{{ steps.s2.output.rows }}"},
    )
    # Save 3 successful runs of baseline to establish a validated baseline in memory
    for i in range(3):
        memory.save_run(
            task="Fetch and prepare dataset",
            playbook=baseline_pb,
            run=RunState(run_id=f"run_base_{i}", playbook=baseline_pb, status=RunStatus.SUCCEEDED),
            verdict=Verdict(passed=True),
        )

    del_res = await engine.adelegate(
        task="Fetch and prepare dataset",
        inputs={"source": "db"},
    )

    assert del_res.status == "succeeded"
    assert del_res.playbook is not None
    assert len(del_res.playbook.steps) == 1

    # Verify improvement manager recorded proposal (Trigger B: fewer steps than baseline of 2 steps)
    proposals = engine.proposals
    assert len(proposals) >= 1
    proposal = proposals[0]
    assert proposal.status == ProposalStatus.PENDING
    assert proposal.steps_saved == 1  # 2 baseline steps - 1 candidate step

    # Approve proposal and ensure it promotes in memory
    approved_rec = engine.improvement_manager.approve(proposal.proposal_id)
    assert approved_rec.status == "reusable"
    updated_proposal = engine.improvement_manager.get_proposal(proposal.proposal_id)
    assert updated_proposal.status == ProposalStatus.APPROVED
