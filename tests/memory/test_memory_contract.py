from __future__ import annotations

import os
import uuid
from collections.abc import Generator

import pytest

from calbi.ai.verifier import Verdict
from calbi.core.schema import Playbook, Step
from calbi.core.state import RunState, RunStatus, StepState, StepStatus
from calbi.memory.base import MemoryStore
from calbi.memory.postgres import PostgresMemory
from calbi.memory.sqlite import SQLiteMemory


@pytest.fixture(params=["sqlite", pytest.param("postgres", marks=pytest.mark.live)])
def memory_store(request: pytest.FixtureRequest) -> Generator[MemoryStore, None, None]:
    """Parametrized fixture providing SQLite and Postgres MemoryStore implementations.

    When running against Postgres, requires DATABASE_URL to be set; otherwise skips cleanly.
    Creates a dedicated temporary test schema and drops it completely upon teardown.
    """
    if request.param == "sqlite":
        yield SQLiteMemory(":memory:")
    elif request.param == "postgres":
        dsn = os.getenv("DATABASE_URL")
        if not dsn:
            pytest.skip("DATABASE_URL environment variable is not set.")
        test_schema = f"calbi_test_{uuid.uuid4().hex[:8]}"
        mem = PostgresMemory(dsn=dsn, schema=test_schema)
        mem.migrate()
        try:
            yield mem
        finally:
            from psycopg import sql

            with mem._get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE;").format(
                            sql.Identifier(test_schema)
                        )
                    )


def test_save_run_initial_status_is_succeeded(memory_store: MemoryStore) -> None:
    """A newly saved run has status 'succeeded' and is NOT returned by find_similar."""
    pb = Playbook(name="initial_status_pb", steps=[Step(id="s1", tool="t1")])
    run = RunState(
        run_id="run_init",
        playbook=pb,
        inputs={},
        status=RunStatus.SUCCEEDED,
        steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
    )
    verdict = Verdict(passed=True, reasons=["Initial test run ok"], confidence=0.95)

    rec_id = memory_store.save_run(
        task="Process quarterly tax filings and compliance logs",
        playbook=pb,
        run=run,
        verdict=verdict,
    )
    assert isinstance(rec_id, str)
    assert len(rec_id) > 0

    # Initial status is 'succeeded' -> MUST NOT be returned by find_similar as a template
    matches = memory_store.find_similar("tax filings compliance")
    assert len(matches) == 0


def test_auto_promotion_after_3_runs(memory_store: MemoryStore) -> None:
    """Saving the exact same task and playbook 3 times auto-promotes status to 'validated'."""
    pb = Playbook(name="auto_promote_pb", steps=[Step(id="s1", tool="t1")])
    run = RunState(
        run_id="run_ap",
        playbook=pb,
        inputs={},
        status=RunStatus.SUCCEEDED,
        steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
    )
    task_name = "Automate daily database backup snapshot archiving"

    # Run 1: initial save
    rec_id1 = memory_store.save_run(
        task=task_name,
        playbook=pb,
        run=run,
        verdict=Verdict(passed=True),
    )
    assert len(memory_store.find_similar("database backup snapshot")) == 0

    # Run 2: second run of same task and playbook
    rec_id2 = memory_store.save_run(
        task=task_name,
        playbook=pb,
        run=run,
        verdict=Verdict(passed=True),
    )
    assert rec_id2 == rec_id1
    assert len(memory_store.find_similar("database backup snapshot")) == 0

    # Run 3: third run triggers auto-promotion to 'validated'
    rec_id3 = memory_store.save_run(
        task=task_name,
        playbook=pb,
        run=run,
        verdict=Verdict(passed=True),
    )
    assert rec_id3 == rec_id1

    matches = memory_store.find_similar("database backup snapshot")
    assert len(matches) == 1
    assert matches[0].record_id == rec_id1
    assert matches[0].status == "validated"
    assert matches[0].run_count == 3
    assert matches[0].playbook.name == "auto_promote_pb"


def test_manual_promote_to_validated_and_reusable(memory_store: MemoryStore) -> None:
    """Manual promote transitions record to 'validated' then 'reusable'."""
    pb = Playbook(name="manual_promote_pb", steps=[Step(id="s1", tool="t1")])
    run = RunState(
        run_id="run_mp",
        playbook=pb,
        inputs={},
        status=RunStatus.SUCCEEDED,
        steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
    )
    task_name = "Synchronize inventory catalogs across multiple cloud regions"

    rec_id = memory_store.save_run(
        task=task_name,
        playbook=pb,
        run=run,
        verdict=Verdict(passed=True),
    )
    assert len(memory_store.find_similar("inventory catalogs cloud")) == 0

    # Promote to 'validated'
    memory_store.promote(rec_id, to="validated")
    validated_matches = memory_store.find_similar("inventory catalogs cloud")
    assert len(validated_matches) == 1
    assert validated_matches[0].status == "validated"
    assert validated_matches[0].record_id == rec_id

    # Promote to 'reusable'
    memory_store.promote(rec_id, to="reusable")
    reusable_matches = memory_store.find_similar("inventory catalogs cloud")
    assert len(reusable_matches) == 1
    assert reusable_matches[0].status == "reusable"
    assert reusable_matches[0].record_id == rec_id


def test_find_similar_ranking_and_limits(memory_store: MemoryStore) -> None:
    """find_similar respects limit parameter and filters by task relevance."""
    tasks = [
        "Generate quarterly financial sales report",
        "Generate monthly financial sales summary",
        "Generate annual financial sales forecast",
        "Analyze server infrastructure performance logs",
        "Restart kubernetes ingress controller deployment",
    ]
    rec_ids: list[str] = []
    for idx, task_name in enumerate(tasks, 1):
        pb = Playbook(name=f"pb_{idx}", steps=[Step(id="s1", tool="t1")])
        run = RunState(
            run_id=f"r_{idx}",
            playbook=pb,
            inputs={},
            status=RunStatus.SUCCEEDED,
            steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
        )
        rec_id = memory_store.save_run(
            task=task_name,
            playbook=pb,
            run=run,
            verdict=Verdict(passed=True),
        )
        # Promote records so find_similar will consider them
        memory_store.promote(rec_id, to="validated" if idx % 2 == 0 else "reusable")
        rec_ids.append(rec_id)

    # Query with limit of 2
    similar = memory_store.find_similar("financial sales report", limit=2)
    assert len(similar) <= 2
    for r in similar:
        assert r.status in ("validated", "reusable")
        assert any(kw in r.task.lower() for kw in ("financial", "sales", "report"))

    # Unrelated task search yields no matches
    unrelated = memory_store.find_similar("blockchain smart contracts")
    assert len(unrelated) == 0


def test_record_failure_and_tool_limitations(memory_store: MemoryStore) -> None:
    """record_failure appends failure modes, retrieved in reverse chronological order."""
    memory_store.record_failure(
        task="Extract scanned PDF documents",
        playbook_name="pdf_ocr_pipeline",
        step_id="step_ocr",
        tool="ocr.extract",
        error="Fails on low-contrast scanned receipts",
    )
    memory_store.record_failure(
        task="Extract scanned PDF documents",
        playbook_name="pdf_ocr_pipeline",
        step_id="step_ocr",
        tool="ocr.extract",
        error="Cannot parse password-protected files",
    )

    limits = memory_store.tool_limitations("ocr.extract")
    assert len(limits) == 2
    # Newest failure appears first
    assert "password-protected" in limits[0]
    assert "low-contrast" in limits[1]

    unknown = memory_store.tool_limitations("unregistered.nonexistent.tool")
    assert unknown == []


def test_idempotent_migration(memory_store: MemoryStore) -> None:
    """Applying migration multiple times is idempotent and preserves operability."""
    if isinstance(memory_store, PostgresMemory):
        memory_store.migrate()
        memory_store.migrate()
    elif isinstance(memory_store, SQLiteMemory):
        memory_store._init_db()
        memory_store._init_db()

    pb = Playbook(name="migrated_pb", steps=[Step(id="s1", tool="t1")])
    run = RunState(
        run_id="mig_r1",
        playbook=pb,
        inputs={},
        status=RunStatus.SUCCEEDED,
        steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
    )
    rec_id = memory_store.save_run(
        task="Test database functionality post idempotent migration",
        playbook=pb,
        run=run,
        verdict=Verdict(passed=True),
    )
    assert isinstance(rec_id, str)
    assert len(rec_id) > 0
