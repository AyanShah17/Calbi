from calbi.ai.verifier import Verdict
from calbi.core.schema import Playbook, Step
from calbi.core.state import RunState, RunStatus, StepState, StepStatus
from calbi.memory.sqlite import SQLiteMemory


def test_sqlite_memory_save_and_find_and_promote() -> None:
    mem = SQLiteMemory(":memory:")
    pb = Playbook(name="test_pb", steps=[Step(id="s1", tool="t1")])
    run = RunState(
        run_id="r1",
        playbook=pb,
        inputs={"file": "test.txt"},
        status=RunStatus.SUCCEEDED,
        steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
    )
    verdict = Verdict(passed=True, reasons=["ok"], confidence=1.0)

    # Save run
    rec_id = mem.save_run(
        task="Analyze financial sales data", playbook=pb, run=run, verdict=verdict
    )
    assert rec_id is not None

    # Initially trust level is "succeeded" -> NOT returned as reusable template by find_similar!
    similar_init = mem.find_similar("financial sales")
    assert len(similar_init) == 0

    # Promote to "validated"
    mem.promote(rec_id, to="validated")
    similar_validated = mem.find_similar("financial sales")
    assert len(similar_validated) == 1
    assert similar_validated[0].record_id == rec_id
    assert similar_validated[0].status == "validated"
    assert similar_validated[0].playbook.name == "test_pb"


def test_sqlite_memory_record_failure_and_limitations() -> None:
    mem = SQLiteMemory(":memory:")

    mem.record_failure(
        task="Extract tables",
        playbook_name="pdf_pipeline",
        step_id="step_ocr",
        tool="ocr.extract",
        error="Fails on scanned pages without text layer",
    )

    limits = mem.tool_limitations("ocr.extract")
    assert len(limits) == 1
    assert "text layer" in limits[0]

    empty_limits = mem.tool_limitations("unrelated.tool")
    assert len(empty_limits) == 0


def test_sqlite_memory_auto_promotion_with_n_runs() -> None:
    mem = SQLiteMemory(":memory:")
    pb = Playbook(name="reliable_pb", steps=[Step(id="s1", tool="t1")])
    run = RunState(
        run_id="r1",
        playbook=pb,
        inputs={},
        status=RunStatus.SUCCEEDED,
        steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
    )

    # Run 1: saved as succeeded
    mem.save_run(task="Parse logs", playbook=pb, run=run, verdict=Verdict(passed=True))
    assert len(mem.find_similar("Parse logs")) == 0

    # Save same task+playbook 2 more times (total 3 runs) -> auto promotes to validated!
    mem.save_run(task="Parse logs", playbook=pb, run=run, verdict=Verdict(passed=True))
    mem.save_run(task="Parse logs", playbook=pb, run=run, verdict=Verdict(passed=True))

    similar = mem.find_similar("Parse logs")
    assert len(similar) == 1
    assert similar[0].status == "validated"
    assert similar[0].run_count == 3
