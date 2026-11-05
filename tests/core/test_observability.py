from __future__ import annotations

import io
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from calbi.core.events import Event, EventBus
from calbi.core.observability import JSONLEventSink, Redactor, SpanRecord, TraceCollector


class SampleConfig(BaseModel):
    app_name: str
    api_key: str
    db_password: str


def test_redactor_scrubs_sensitive_keys_case_insensitively() -> None:
    redactor = Redactor()
    payload: dict[str, Any] = {
        "user_name": "calbi_user",
        "api_key": "raw-key-123",
        "API_KEY": "raw-key-upper",
        "nested": {
            "password": "p@ssword!",
            "AUTH_TOKEN": "token_abc_xyz",
            "db_dsn": "postgres://user:pass@host/db",
            "safe_number": 42,
        },
        "items": [
            {"secret": "hidden_value", "label": "public_label"},
            "regular_string",
        ],
    }

    redacted = redactor.redact(payload)

    assert isinstance(redacted, dict)
    assert redacted["user_name"] == "calbi_user"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["API_KEY"] == "[REDACTED]"
    assert redacted["nested"]["password"] == "[REDACTED]"
    assert redacted["nested"]["AUTH_TOKEN"] == "[REDACTED]"
    assert redacted["nested"]["db_dsn"] == "[REDACTED]"
    assert redacted["nested"]["safe_number"] == 42
    assert redacted["items"][0]["secret"] == "[REDACTED]"
    assert redacted["items"][0]["label"] == "public_label"
    assert redacted["items"][1] == "regular_string"


def test_redactor_scrubs_inline_bearer_and_dsn() -> None:
    redactor = Redactor()

    # Bearer header scrubbing
    msg_bearer = "Authorization header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.e30.secret"
    scrubbed_bearer = redactor.redact(msg_bearer)
    assert "Bearer [REDACTED]" in scrubbed_bearer
    assert "eyJhbGci" not in scrubbed_bearer

    # Postgres DSN password scrubbing
    msg_dsn = "Failed connecting to postgresql://calbi_admin:super_secret_pw@db.internal:5432/calbi_db"
    scrubbed_dsn = redactor.redact(msg_dsn)
    assert (
        scrubbed_dsn
        == "Failed connecting to postgresql://calbi_admin:[REDACTED]@db.internal:5432/calbi_db"
    )
    assert "super_secret_pw" not in scrubbed_dsn

    # Alternative scheme postgres://
    msg_alt_dsn = "postgres://root:very_secret_pwd@127.0.0.1:5432/test"
    assert (
        redactor.redact(msg_alt_dsn)
        == "postgres://root:[REDACTED]@127.0.0.1:5432/test"
    )


def test_redactor_scrubs_api_key_prefixes() -> None:
    redactor = Redactor()

    # OpenAI / OpenRouter style sk- prefix
    msg_sk = "Using OpenAI key sk-proj-12345678abcdef for inference"
    scrubbed_sk = redactor.redact(msg_sk)
    assert scrubbed_sk == "Using OpenAI key [REDACTED] for inference"
    assert "sk-proj" not in scrubbed_sk

    # NVIDIA NIM style nvapi- prefix
    msg_nv = "Using NVIDIA key nvapi-abcde123456789 for NIM service"
    scrubbed_nv = redactor.redact(msg_nv)
    assert scrubbed_nv == "Using NVIDIA key [REDACTED] for NIM service"
    assert "nvapi-" not in scrubbed_nv


def test_redactor_extra_patterns() -> None:
    extra = re.compile(r"ACCOUNT_\d{6}")
    redactor = Redactor(extra_patterns=[extra])

    msg = "Customer ID is ACCOUNT_123456 in system"
    res = redactor.redact(msg)
    assert res == "Customer ID is [REDACTED] in system"


def test_redactor_truncates_payload_cap() -> None:
    redactor = Redactor(max_string_bytes=2048)

    # String under 2048 bytes remains untouched
    short_str = "a" * 2048
    assert redactor.redact(short_str) == short_str

    # String over 2048 bytes is capped to 2048 bytes + suffix
    long_str = "b" * 3000
    capped = redactor.redact(long_str)
    expected_prefix = "b" * 2048
    assert capped.startswith(expected_prefix)
    assert capped.endswith("... [truncated]")
    assert len(capped) == 2048 + len("... [truncated]")


def test_redactor_truncates_multibyte_utf8_safely() -> None:
    # 4-byte UTF-8 emoji repeated
    emoji = "🚀"  # 4 bytes in UTF-8
    redactor = Redactor(max_string_bytes=10)
    # 3 emojis = 12 bytes > 10 bytes cap
    capped = redactor.redact(emoji * 3)
    assert capped.endswith("... [truncated]")
    # 10 // 4 = 2 emojis fit in 10 bytes (8 bytes), 3rd emoji doesn't fit
    assert capped.startswith("🚀🚀")


def test_redactor_handles_pydantic_model() -> None:
    redactor = Redactor()
    model = SampleConfig(
        app_name="calbi_service",
        api_key="sk-proj-9876543210",
        db_password="password123",
    )

    redacted = redactor.redact(model)
    assert isinstance(redacted, dict)
    assert redacted["app_name"] == "calbi_service"
    assert redacted["api_key"] == "[REDACTED]"
    assert redacted["db_password"] == "[REDACTED]"


def test_redactor_non_destructive_event() -> None:
    redactor = Redactor()
    original_data = {
        "password": "secret_password",
        "details": {"api_key": "sk-1234567890"},
        "status": "active",
    }
    event = Event(
        ts=datetime.now(timezone.utc),
        run_id="run_100",
        type="step.started",
        step_id="step_a",
        data=dict(original_data),
    )

    redacted_event = redactor.redact_event(event)

    # Verify new event is created and redacted
    assert redacted_event is not event
    assert redacted_event.run_id == "run_100"
    assert redacted_event.step_id == "step_a"
    assert redacted_event.data["password"] == "[REDACTED]"
    assert redacted_event.data["details"]["api_key"] == "[REDACTED]"
    assert redacted_event.data["status"] == "active"

    # Original event data MUST NOT be modified
    assert event.data["password"] == "secret_password"
    assert event.data["details"]["api_key"] == "sk-1234567890"


def test_jsonl_event_sink_writes_to_stream() -> None:
    stream = io.StringIO()
    sink = JSONLEventSink(target=stream)
    now = datetime.now(timezone.utc)

    event = Event(
        ts=now,
        run_id="run_test_stream",
        type="run.started",
        data={"api_key": "sk-1234567890", "playbook": "test_pb"},
    )
    sink(event)

    content = stream.getvalue()
    lines = content.strip().splitlines()
    assert len(lines) == 1

    parsed = json.loads(lines[0])
    assert parsed["run_id"] == "run_test_stream"
    assert parsed["type"] == "run.started"
    assert parsed["data"]["api_key"] == "[REDACTED]"
    assert parsed["data"]["playbook"] == "test_pb"


def test_jsonl_event_sink_subscribed_to_event_bus(tmp_path: Path) -> None:
    output_file = tmp_path / "events.jsonl"
    sink = JSONLEventSink(target=output_file)

    bus = EventBus()
    bus.subscribe(sink)

    now = datetime.now(timezone.utc)
    ev1 = Event(
        ts=now,
        run_id="run_42",
        type="run.started",
        data={"playbook": "deploy", "token": "super_token"},
    )
    ev2 = Event(
        ts=now,
        run_id="run_42",
        type="step.started",
        step_id="step_build",
        data={"dsn": "postgres://user:pass@host/db", "attempt": 1},
    )
    ev3 = Event(
        ts=now,
        run_id="run_42",
        type="run.completed",
        data={"status": "succeeded"},
    )

    bus.emit(ev1)
    bus.emit(ev2)
    bus.emit(ev3)

    assert output_file.exists()
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3

    parsed1 = json.loads(lines[0])
    assert parsed1["data"]["token"] == "[REDACTED]"
    assert parsed1["data"]["playbook"] == "deploy"

    parsed2 = json.loads(lines[1])
    assert parsed2["step_id"] == "step_build"
    assert parsed2["data"]["dsn"] == "[REDACTED]"
    assert parsed2["data"]["attempt"] == 1

    parsed3 = json.loads(lines[2])
    assert parsed3["data"]["status"] == "succeeded"


def test_jsonl_event_sink_thread_safety(tmp_path: Path) -> None:
    output_file = tmp_path / "threaded_events.jsonl"
    sink = JSONLEventSink(target=output_file)

    num_threads = 8
    events_per_thread = 25

    def worker(worker_id: int) -> None:
        for i in range(events_per_thread):
            ev = Event(
                ts=datetime.now(timezone.utc),
                run_id=f"run_{worker_id}",
                type="step.started",
                step_id=f"step_{i}",
                data={"worker": worker_id, "index": i, "secret_key": "confidential"},
            )
            sink(ev)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == num_threads * events_per_thread

    for line in lines:
        data = json.loads(line)
        assert data["data"]["secret_key"] == "[REDACTED]"


def test_trace_collector_pairs_parent_child_spans() -> None:
    collector = TraceCollector()
    bus = EventBus()
    bus.subscribe(collector)

    t0 = datetime(2026, 11, 5, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 5, 12, 0, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 5, 12, 0, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 11, 5, 12, 0, 3, tzinfo=timezone.utc)

    # 1. run.started
    bus.emit(
        Event(
            ts=t0,
            run_id="run_alpha",
            type="run.started",
            data={"playbook": "etl_pipeline", "api_key": "sk-1234567890"},
        )
    )

    # 2. step.started
    bus.emit(
        Event(
            ts=t1,
            run_id="run_alpha",
            type="step.started",
            step_id="fetch_data",
            data={"attempt": 1, "url": "https://api.internal/data"},
        )
    )

    # 3. step.completed
    bus.emit(
        Event(
            ts=t2,
            run_id="run_alpha",
            type="step.completed",
            step_id="fetch_data",
            data={"output": {"rows": 100}},
        )
    )

    # 4. run.completed
    bus.emit(
        Event(
            ts=t3,
            run_id="run_alpha",
            type="run.completed",
            data={"status": "succeeded", "duration_s": 3.0},
        )
    )

    spans = collector.spans
    assert len(spans) == 2

    root_span = next(s for s in spans if s.name == "run:etl_pipeline")
    step_span = next(s for s in spans if s.name == "step:fetch_data")

    # Root span checks
    assert root_span.trace_id == "run_alpha"
    assert root_span.parent_span_id is None
    assert root_span.status == "ok"
    assert root_span.start_ts == t0
    assert root_span.end_ts == t3
    assert root_span.attributes["api_key"] == "[REDACTED]"
    assert root_span.attributes["status"] == "succeeded"

    # Step span checks
    assert step_span.trace_id == "run_alpha"
    assert step_span.parent_span_id == root_span.span_id
    assert step_span.status == "ok"
    assert step_span.start_ts == t1
    assert step_span.end_ts == t2
    assert step_span.attributes["url"] == "https://api.internal/data"
    assert step_span.attributes["output"] == {"rows": 100}


def test_trace_collector_step_and_run_error_status() -> None:
    collector = TraceCollector()
    t0 = datetime(2026, 11, 5, 13, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 5, 13, 0, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 5, 13, 0, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 11, 5, 13, 0, 3, tzinfo=timezone.utc)

    # run.started
    collector(
        Event(
            ts=t0,
            run_id="run_err",
            type="run.started",
            data={"playbook": "failing_flow"},
        )
    )

    # step.started
    collector(
        Event(
            ts=t1,
            run_id="run_err",
            type="step.started",
            step_id="flaky_step",
            data={"attempt": 1},
        )
    )

    # step.failed
    collector(
        Event(
            ts=t2,
            run_id="run_err",
            type="step.failed",
            step_id="flaky_step",
            data={"error": "Connection failed to postgresql://user:pass@host/db"},
        )
    )

    # run.failed
    collector(
        Event(
            ts=t3,
            run_id="run_err",
            type="run.failed",
            data={"status": "failed", "error": "Step execution aborted"},
        )
    )

    spans = collector.spans
    assert len(spans) == 2

    root_span = next(s for s in spans if s.name == "run:failing_flow")
    step_span = next(s for s in spans if s.name == "step:flaky_step")

    assert root_span.status == "error"
    assert root_span.end_ts == t3

    assert step_span.status == "error"
    assert step_span.end_ts == t2
    # Verify inline DSN was redacted in attributes
    assert "user:[REDACTED]@" in step_span.attributes["error"]


def test_trace_collector_multiple_runs_and_filtering() -> None:
    collector = TraceCollector()
    t = datetime.now(timezone.utc)

    collector(Event(ts=t, run_id="run_1", type="run.started", data={"playbook": "p1"}))
    collector(Event(ts=t, run_id="run_2", type="run.started", data={"playbook": "p2"}))

    assert len(collector.spans) == 2
    assert len(collector.get_spans("run_1")) == 1
    assert collector.get_spans("run_1")[0].trace_id == "run_1"
    assert len(collector.get_spans("run_2")) == 1
    assert collector.get_spans("run_2")[0].trace_id == "run_2"

    collector.clear()
    assert len(collector.spans) == 0


def test_span_record_pydantic_serialization() -> None:
    t0 = datetime(2026, 11, 5, 14, 0, 0, tzinfo=timezone.utc)
    span = SpanRecord(
        trace_id="trace_xyz",
        span_id="span_123",
        parent_span_id=None,
        name="run:test",
        start_ts=t0,
        status="ok",
        attributes={"key": "val"},
    )

    json_str = span.model_dump_json()
    assert "trace_xyz" in json_str
    assert "run:test" in json_str

    dumped = span.model_dump()
    assert dumped["status"] == "ok"
    assert dumped["attributes"] == {"key": "val"}
