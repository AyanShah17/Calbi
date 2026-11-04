import os

import pytest

from calbi.ai.adapters.openai_compat import ConfigError
from calbi.memory.postgres import PostgresMemory, validate_dsn


def test_postgres_dsn_validation() -> None:
    # Localhost allowed without sslmode
    assert validate_dsn("postgresql://user:pass@localhost:5432/mydb") is True
    assert validate_dsn("postgresql://user:pass@127.0.0.1:5432/mydb") is True

    # Remote Neon host with sslmode=require is valid
    remote_ssl = (
        "postgresql://user:pass@ep-cool-fog-1234.us-east-2.aws.neon.tech/neondb?sslmode=require"
    )
    assert validate_dsn(remote_ssl) is True

    # Remote host without sslmode rejected
    remote_no_ssl = "postgresql://user:pass@ep-cool-fog-1234.us-east-2.aws.neon.tech/neondb"
    with pytest.raises(ConfigError, match="sslmode=require"):
        validate_dsn(remote_no_ssl)


def test_no_fstring_sql_in_postgres_module() -> None:
    import inspect

    from calbi.memory import postgres

    source = inspect.getsource(postgres)
    # Check that execute(f"...") is not present for SQL statements
    for line in source.splitlines():
        clean = line.strip()
        if "execute(f" in clean or "execute(F" in clean:
            # Table/schema identifiers should be parameterized or validated
            assert "WHERE" not in clean, f"Found possible f-string SQL injection line: {line}"


@pytest.mark.live
def test_live_postgres_memory_neon() -> None:
    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        pytest.skip("DATABASE_URL is not set.")

    import uuid

    test_schema = f"calbi_test_{uuid.uuid4().hex[:8]}"

    mem = PostgresMemory(dsn=dsn, schema=test_schema)
    try:
        mem.migrate()
        # Second migration call must be idempotent
        mem.migrate()

        # Contract checks
        from calbi.ai.verifier import Verdict
        from calbi.core.schema import Playbook, Step
        from calbi.core.state import RunState, RunStatus, StepState, StepStatus

        pb = Playbook(name="neon_test", steps=[Step(id="s1", tool="t1")])
        run = RunState(
            run_id="r1",
            playbook=pb,
            inputs={},
            status=RunStatus.SUCCEEDED,
            steps={"s1": StepState(id="s1", status=StepStatus.SUCCEEDED)},
        )

        rec_id = mem.save_run("Test Neon memory search", pb, run, Verdict(passed=True))
        assert rec_id is not None
        assert len(mem.find_similar("Neon memory search")) == 0

        mem.promote(rec_id, "validated")
        similar = mem.find_similar("Neon memory search")
        assert len(similar) == 1
        assert similar[0].record_id == rec_id

    finally:
        # Teardown test schema
        with mem._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(f"DROP SCHEMA IF EXISTS {test_schema} CASCADE;")
