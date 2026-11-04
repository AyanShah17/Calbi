from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Literal

from calbi.ai.verifier import Verdict
from calbi.core.schema import Playbook, load_playbook
from calbi.core.state import RunState
from calbi.memory.base import PlaybookRecord


class SQLiteMemory:
    """SQLite-backed persistent memory store with FTS5 keyword retrieval."""

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.conn = sqlite3.connect(self.path, isolation_level=None)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY,
                applied_at TIMESTAMP
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS playbooks (
                record_id TEXT PRIMARY KEY,
                task TEXT NOT NULL,
                playbook_json TEXT NOT NULL,
                status TEXT NOT NULL,
                run_count INTEGER NOT NULL DEFAULT 1,
                last_used TIMESTAMP NOT NULL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS tool_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT NOT NULL,
                playbook_name TEXT NOT NULL,
                step_id TEXT,
                tool TEXT,
                error TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL
            );
            """
        )

        # Attempt to create FTS5 table
        self.has_fts = False
        try:
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS playbooks_fts USING fts5(
                    task,
                    content='playbooks',
                    content_rowid='rowid'
                );
                """
            )
            self.has_fts = True
        except sqlite3.OperationalError:
            self.has_fts = False

        # Record schema version 1
        cur.execute(
            "INSERT OR IGNORE INTO schema_version (version, applied_at) VALUES (1, ?)",
            (datetime.now(timezone.utc).isoformat(),),
        )

    def save_run(
        self,
        task: str,
        playbook: Playbook,
        run: RunState,
        verdict: Verdict | None = None,
    ) -> str:
        """Save a playbook execution run and update usage count."""
        now = datetime.now(timezone.utc).isoformat()
        pb_dump = json.dumps(playbook.model_dump(mode="json"))

        cur = self.conn.cursor()
        cur.execute(
            (
                "SELECT record_id, status, run_count FROM playbooks "
                "WHERE task = ? AND playbook_json = ?"
            ),
            (task, pb_dump),
        )
        existing = cur.fetchone()

        if existing:
            rec_id, status, count = existing
            new_count = count + 1
            new_status = status
            # Auto promote to validated after 3 passing runs
            if new_count >= 3 and status == "succeeded":
                new_status = "validated"

            cur.execute(
                "UPDATE playbooks SET run_count = ?, status = ?, last_used = ? WHERE record_id = ?",
                (new_count, new_status, now, rec_id),
            )
            return str(rec_id)

        rec_id = f"rec_{uuid.uuid4().hex[:12]}"
        cur.execute(
            """
            INSERT INTO playbooks (record_id, task, playbook_json, status, run_count, last_used)
            VALUES (?, ?, ?, 'succeeded', 1, ?)
            """,
            (rec_id, task, pb_dump, now),
        )
        if self.has_fts:
            cur.execute(
                "INSERT INTO playbooks_fts(rowid, task) VALUES (last_insert_rowid(), ?)", (task,)
            )
        return rec_id

    def find_similar(self, task: str, limit: int = 3) -> list[PlaybookRecord]:
        """Find validated or reusable playbooks matching task tokens."""
        cur = self.conn.cursor()
        # Clean query tokens
        tokens = [t.strip() for t in task.split() if len(t.strip()) > 2]
        if not tokens:
            return []

        rows = []
        if self.has_fts:
            fts_query = " OR ".join(tokens)
            try:
                cur.execute(
                    """
                    SELECT p.record_id, p.task, p.playbook_json, p.status, p.run_count, p.last_used
                    FROM playbooks p
                    JOIN playbooks_fts f ON p.rowid = f.rowid
                    WHERE playbooks_fts MATCH ? AND p.status IN ('validated', 'reusable')
                    LIMIT ?
                    """,
                    (fts_query, limit),
                )
                rows = cur.fetchall()
            except sqlite3.OperationalError:
                rows = []

        if not rows:
            # Fallback to LIKE
            like_clause = " OR ".join("task LIKE ?" for _ in tokens)
            params = [f"%{t}%" for t in tokens] + [limit]
            cur.execute(
                f"""
                SELECT record_id, task, playbook_json, status, run_count, last_used
                FROM playbooks
                WHERE ({like_clause}) AND status IN ('validated', 'reusable')
                LIMIT ?
                """,
                params,
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
                    last_used=datetime.fromisoformat(last_used),
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
        now = datetime.now(timezone.utc).isoformat()
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO tool_failures (task, playbook_name, step_id, tool, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (task, playbook_name, step_id, tool, error, now),
        )

    def tool_limitations(self, tool: str) -> list[str]:
        """Retrieve known failure modes and limitations for a tool."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT error FROM tool_failures WHERE tool = ? ORDER BY id DESC LIMIT 10",
            (tool,),
        )
        return [row[0] for row in cur.fetchall()]

    def promote(self, record_id: str, to: Literal["validated", "reusable"]) -> None:
        """Promote a playbook record's trust level."""
        cur = self.conn.cursor()
        cur.execute("UPDATE playbooks SET status = ? WHERE record_id = ?", (to, record_id))
