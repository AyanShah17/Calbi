from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import parse_qs, urlparse

try:
    import psycopg
    from psycopg import sql
except ImportError:  # pragma: no cover
    psycopg = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]

from calbi.ai.adapters.openai_compat import ConfigError
from calbi.ai.verifier import Verdict
from calbi.core.schema import Playbook, load_playbook
from calbi.core.state import RunState
from calbi.memory.base import PlaybookRecord


def validate_dsn(dsn: str) -> bool:
    """Validate that remote database connection strings enforce TLS (sslmode=require)."""
    parsed = urlparse(dsn)
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1", "0.0.0.0"):
        return True

    params = parse_qs(parsed.query)
    sslmode = params.get("sslmode", [""])[0]
    if sslmode != "require":
        raise ConfigError(
            f"Remote database host '{host}' requires 'sslmode=require' in connection string."
        )
    return True


_SAFE_SCHEMA_RE = re.compile(r"^[a-z][a-z0-9_]*$")


class PostgresMemory:
    """Neon (serverless Postgres) persistent memory backend."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        schema: str = "calbi",
        pool_size: int = 5,
    ) -> None:
        self.dsn = dsn or os.getenv("DATABASE_URL")
        if not self.dsn:
            raise ConfigError("DATABASE_URL environment variable or dsn parameter is required.")

        validate_dsn(self.dsn)

        if not _SAFE_SCHEMA_RE.match(schema):
            raise ValueError(f"Invalid schema name '{schema}'. Must match ^[a-z][a-z0-9_]*$")
        self.schema = schema
        self.pool_size = pool_size

    def _get_connection(self) -> Any:
        """Establish connection with retries to handle Neon serverless cold starts."""
        if psycopg is None:
            raise ConfigError(
                "psycopg is required for PostgresMemory. "
                "Install with: pip install 'calbi[postgres]'"
            )
        assert self.dsn is not None
        attempts = 0
        max_attempts = 3
        while attempts < max_attempts:
            attempts += 1
            try:
                conn = psycopg.connect(self.dsn, autocommit=True)
                return conn
            except Exception as e:
                if attempts >= max_attempts:
                    raise ConfigError(
                        f"Failed to connect to Postgres after {max_attempts} attempts: {e}"
                    ) from e
                time.sleep(0.5 * (2 ** (attempts - 1)))
        raise ConfigError("Failed to connect to Postgres.")

    def migrate(self) -> None:
        """Apply idempotent database migrations in the designated schema."""
        schema_ident = sql.Identifier(self.schema)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # 1. Create schema
                cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema_ident))

                # 2. Schema version table
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.schema_version (
                            version INTEGER PRIMARY KEY,
                            applied_at TIMESTAMPTZ NOT NULL
                        );
                        """
                    ).format(schema_ident)
                )

                # 3. Playbooks table
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.playbooks (
                            record_id TEXT PRIMARY KEY,
                            task TEXT NOT NULL,
                            playbook_json JSONB NOT NULL,
                            status TEXT NOT NULL,
                            run_count INTEGER NOT NULL DEFAULT 1,
                            last_used TIMESTAMPTZ NOT NULL
                        );
                        """
                    ).format(schema_ident)
                )

                # 4. GIN Index for full-text search
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS playbooks_task_fts_idx
                        ON {}.playbooks USING gin(to_tsvector('english', task));
                        """
                    ).format(schema_ident)
                )

                # 5. Tool failures table
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE IF NOT EXISTS {}.tool_failures (
                            id SERIAL PRIMARY KEY,
                            task TEXT NOT NULL,
                            playbook_name TEXT NOT NULL,
                            step_id TEXT,
                            tool TEXT,
                            error TEXT NOT NULL,
                            created_at TIMESTAMPTZ NOT NULL
                        );
                        """
                    ).format(schema_ident)
                )

                # Record version 1
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.schema_version (version, applied_at)
                        VALUES (1, NOW())
                        ON CONFLICT (version) DO NOTHING;
                        """
                    ).format(schema_ident)
                )

    def save_run(
        self,
        task: str,
        playbook: Playbook,
        run: RunState,
        verdict: Verdict | None = None,
    ) -> str:
        """Save a completed playbook run."""
        now = datetime.now(timezone.utc)
        pb_dump = json.dumps(playbook.model_dump(mode="json"))
        schema_ident = sql.Identifier(self.schema)

        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT record_id, status, run_count
                        FROM {}.playbooks
                        WHERE task = %s AND playbook_json = %s::jsonb
                        """
                    ).format(schema_ident),
                    (task, pb_dump),
                )
                existing = cur.fetchone()

                if existing:
                    rec_id, status, count = existing
                    new_count = count + 1
                    new_status = status
                    if new_count >= 3 and status == "succeeded":
                        new_status = "validated"

                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {}.playbooks
                            SET run_count = %s, status = %s, last_used = %s
                            WHERE record_id = %s
                            """
                        ).format(schema_ident),
                        (new_count, new_status, now, rec_id),
                    )
                    return str(rec_id)

                rec_id = f"rec_{uuid.uuid4().hex[:12]}"
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.playbooks
                        (record_id, task, playbook_json, status, run_count, last_used)
                        VALUES (%s, %s, %s::jsonb, 'succeeded', 1, %s)
                        """
                    ).format(schema_ident),
                    (rec_id, task, pb_dump, now),
                )
                return rec_id

    def find_similar(self, task: str, limit: int = 3) -> list[PlaybookRecord]:
        """Find validated or reusable playbooks using Postgres tsvector search."""
        schema_ident = sql.Identifier(self.schema)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT record_id, task, playbook_json::text, status, run_count, last_used
                        FROM {}.playbooks
                        WHERE to_tsvector('english', task) @@ plainto_tsquery('english', %s)
                          AND status IN ('validated', 'reusable')
                        LIMIT %s
                        """
                    ).format(schema_ident),
                    (task, limit),
                )
                rows = cur.fetchall()

        records = []
        for r_id, r_task, pb_json, status, count, last_used in rows:
            pb_dict = json.loads(pb_json)
            records.append(
                PlaybookRecord(
                    record_id=r_id,
                    task=r_task,
                    playbook=load_playbook(pb_dict),
                    status=status,
                    run_count=count,
                    last_used=last_used,
                )
            )
        return records

    def record_failure(
        self,
        task: str,
        playbook_name: str,
        step_id: str | None,
        tool: str | None,
        error: str,
    ) -> None:
        """Record an execution failure or tool limitation."""
        now = datetime.now(timezone.utc)
        schema_ident = sql.Identifier(self.schema)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {}.tool_failures
                        (task, playbook_name, step_id, tool, error, created_at)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """
                    ).format(schema_ident),
                    (task, playbook_name, step_id, tool, error, now),
                )

    def tool_limitations(self, tool: str) -> list[str]:
        """Retrieve known failure modes and limitations for a tool."""
        schema_ident = sql.Identifier(self.schema)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT error FROM {}.tool_failures
                        WHERE tool = %s
                        ORDER BY id DESC LIMIT 10
                        """
                    ).format(schema_ident),
                    (tool,),
                )
                return [row[0] for row in cur.fetchall()]

    def promote(self, record_id: str, to: Literal["validated", "reusable"]) -> None:
        """Promote a playbook record's trust level."""
        schema_ident = sql.Identifier(self.schema)
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("UPDATE {}.playbooks SET status = %s WHERE record_id = %s").format(
                        schema_ident
                    ),
                    (to, record_id),
                )
