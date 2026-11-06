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
    msg_dsn = (
        "Failed connecting to postgresql://calbi_admin:super_secret_pw@db.internal:5432/calbi_db"
    )
    scrubbed_dsn = redactor.redact(msg_dsn)
    assert (
        scrubbed_dsn
        == "Failed connecting to postgresql://calbi_admin:[REDACTED]@db.internal:5432/calbi_db"
    )
    assert "super_secret_pw" not in scrubbed_dsn

    # Alternative scheme postgres://
    msg_alt_dsn = "postgres://root:very_secret_pwd@127.0.0.1:5432/test"
    assert redactor.redact(msg_alt_dsn) == "postgres://root:[REDACTED]@127.0.0.1:5432/test"


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


def test_redactor_preserves_token_count_metrics() -> None:
    redactor = Redactor()
    payload: dict[str, Any] = {
        "max_tokens": 4096,
        "tokens_used": 1520,
        "token_count": 350,
        "prompt_tokens": 120,
        "completion_tokens": 230,
        "total_tokens": 350,
        "input_tokens": 100,
        "output_tokens": 250,
        "tokens": 500,
        "token_budget": 8000,
        "cached_tokens": 50,
        "reasoning_tokens": 64,
        "prompt_token_count": 120,
        "candidates_token_count": 230,
        "total_token_count": 350,
        "num_tokens": 42,
        "token_usage": {"prompt": 120, "completion": 230},
        # Secrets in the same payload MUST still be redacted
        "api_token": "secret_token_12345",
        "auth_token": "secret_auth_abc",
        "access_token": "secret_access_xyz",
        "session_token": "secret_session_qwe",
        "github_token": "ghp_1234567890abcdef",
        "api_key": "sk-1234567890",
    }

    redacted = redactor.redact(payload)
    assert isinstance(redacted, dict)

    # All token count/metric keys preserved with original values
    assert redacted["max_tokens"] == 4096
    assert redacted["tokens_used"] == 1520
    assert redacted["token_count"] == 350
    assert redacted["prompt_tokens"] == 120
    assert redacted["completion_tokens"] == 230
    assert redacted["total_tokens"] == 350
    assert redacted["input_tokens"] == 100
    assert redacted["output_tokens"] == 250
    assert redacted["tokens"] == 500
    assert redacted["token_budget"] == 8000
    assert redacted["cached_tokens"] == 50
    assert redacted["reasoning_tokens"] == 64
    assert redacted["prompt_token_count"] == 120
    assert redacted["candidates_token_count"] == 230
    assert redacted["total_token_count"] == 350
    assert redacted["num_tokens"] == 42
    assert redacted["token_usage"] == {"prompt": 120, "completion": 230}

    # All secret tokens strictly redacted
    assert redacted["api_token"] == "[REDACTED]"
    assert redacted["auth_token"] == "[REDACTED]"
    assert redacted["access_token"] == "[REDACTED]"
    assert redacted["session_token"] == "[REDACTED]"
    assert redacted["github_token"] == "[REDACTED]"
    assert redacted["api_key"] == "[REDACTED]"


def test_redactor_handles_camel_kebab_case_token_metrics() -> None:
    redactor = Redactor()
    payload = {
        "maxTokens": 2048,
        "max-tokens": 2048,
        "MAX_TOKENS": 2048,
        "tokensUsed": 100,
        "tokens-used": 100,
        "tokenCount": 50,
        "token-count": 50,
        "promptTokens": 30,
        "prompt-tokens": 30,
        "completionTokens": 70,
        "completion-tokens": 70,
        "totalTokens": 100,
        "total-tokens": 100,
        # Camel/kebab secret tokens MUST still be redacted
        "apiToken": "secret_api",
        "api-token": "secret_api",
        "authToken": "secret_auth",
        "auth-token": "secret_auth",
        "accessToken": "secret_access",
        "access-token": "secret_access",
        "sessionToken": "secret_session",
        "session-token": "secret_session",
        "githubToken": "secret_github",
        "github-token": "secret_github",
    }

    redacted = redactor.redact(payload)
    assert isinstance(redacted, dict)

    # Metrics preserved
    assert redacted["maxTokens"] == 2048
    assert redacted["max-tokens"] == 2048
    assert redacted["MAX_TOKENS"] == 2048
    assert redacted["tokensUsed"] == 100
    assert redacted["tokens-used"] == 100
    assert redacted["tokenCount"] == 50
    assert redacted["token-count"] == 50
    assert redacted["promptTokens"] == 30
    assert redacted["prompt-tokens"] == 30
    assert redacted["completionTokens"] == 70
    assert redacted["completion-tokens"] == 70
    assert redacted["totalTokens"] == 100
    assert redacted["total-tokens"] == 100

    # Secret tokens redacted
    assert redacted["apiToken"] == "[REDACTED]"
    assert redacted["api-token"] == "[REDACTED]"
    assert redacted["authToken"] == "[REDACTED]"
    assert redacted["auth-token"] == "[REDACTED]"
    assert redacted["accessToken"] == "[REDACTED]"
    assert redacted["access-token"] == "[REDACTED]"
    assert redacted["sessionToken"] == "[REDACTED]"
    assert redacted["session-token"] == "[REDACTED]"
    assert redacted["githubToken"] == "[REDACTED]"
    assert redacted["github-token"] == "[REDACTED]"


def test_redactor_strictly_redacts_plural_secret_tokens() -> None:
    redactor = Redactor()
    payload = {
        "api_tokens": ["secret1", "secret2"],
        "auth_tokens": ["token_a", "token_b"],
        "token": "raw_secret_value",
    }

    redacted = redactor.redact(payload)
    assert isinstance(redacted, dict)
    assert redacted["api_tokens"] == "[REDACTED]"
    assert redacted["auth_tokens"] == "[REDACTED]"
    assert redacted["token"] == "[REDACTED]"


def test_redactor_avoids_false_positive_substring_redaction() -> None:
    redactor = Redactor()
    payload = {
        "tokenization": "bpe",
        "tokenizer_name": "cl100k_base",
        "stakeholder": "admin_team",
        "monkey_patch": True,
    }

    redacted = redactor.redact(payload)
    assert isinstance(redacted, dict)
    assert redacted["tokenization"] == "bpe"
    assert redacted["tokenizer_name"] == "cl100k_base"
    assert redacted["stakeholder"] == "admin_team"
    assert redacted["monkey_patch"] is True


def test_redactor_custom_allowed_and_sensitive_keys() -> None:
    redactor = Redactor(
        sensitive_keys={"custom_pin", "api_key"},
        allowed_keys={"pin_count", "max_tokens"},
    )
    payload = {
        "custom_pin": "1234",
        "pin_count": 4,
        "max_tokens": 1000,
        "api_key": "sk-test",
    }
    redacted = redactor.redact(payload)
    assert isinstance(redacted, dict)
    assert redacted["custom_pin"] == "[REDACTED]"
    assert redacted["pin_count"] == 4
    assert redacted["max_tokens"] == 1000
    assert redacted["api_key"] == "[REDACTED]"


def test_trace_collector_retry_aware_step_retrying() -> None:
    collector = TraceCollector()
    bus = EventBus()
    bus.subscribe(collector)

    t0 = datetime(2026, 11, 6, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 6, 10, 0, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 6, 10, 0, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 11, 6, 10, 0, 3, tzinfo=timezone.utc)
    t4 = datetime(2026, 11, 6, 10, 0, 4, tzinfo=timezone.utc)
    t5 = datetime(2026, 11, 6, 10, 0, 5, tzinfo=timezone.utc)

    # 1. run.started
    bus.emit(
        Event(
            ts=t0,
            run_id="run_retry_1",
            type="run.started",
            data={"playbook": "flaky_pipeline"},
        )
    )

    # 2. step.started attempt 1
    bus.emit(
        Event(
            ts=t1,
            run_id="run_retry_1",
            type="step.started",
            step_id="fetch_data",
            data={"attempt": 1, "url": "https://api.upstream.com"},
        )
    )

    # 3. step.retrying attempt 1 fails
    bus.emit(
        Event(
            ts=t2,
            run_id="run_retry_1",
            type="step.retrying",
            step_id="fetch_data",
            data={
                "attempt": 1,
                "error": "Connection reset by peer with postgresql://u:p@db/d",
                "delay_s": 1.0,
            },
        )
    )

    # 4. step.started attempt 2
    bus.emit(
        Event(
            ts=t3,
            run_id="run_retry_1",
            type="step.started",
            step_id="fetch_data",
            data={"attempt": 2, "url": "https://api.upstream.com"},
        )
    )

    # 5. step.completed attempt 2 succeeds
    bus.emit(
        Event(
            ts=t4,
            run_id="run_retry_1",
            type="step.completed",
            step_id="fetch_data",
            data={"output": {"status": "success", "total_tokens": 120}},
        )
    )

    # 6. run.completed
    bus.emit(
        Event(
            ts=t5,
            run_id="run_retry_1",
            type="run.completed",
            data={"status": "succeeded"},
        )
    )

    spans = collector.spans
    # 1 root span + 2 step attempt spans = 3 spans total
    assert len(spans) == 3

    root_span = next(s for s in spans if s.name == "run:flaky_pipeline")
    step_spans = collector.get_step_spans("run_retry_1", "fetch_data")
    assert len(step_spans) == 2

    span_attempt_1 = step_spans[0]
    span_attempt_2 = step_spans[1]

    # Attempt 1 span assertions: distinctly tracked, closed on retry, status error
    assert span_attempt_1.span_id == "span_retry_1_fetch_data_1" or "1" in span_attempt_1.span_id
    assert span_attempt_1.status == "error"
    assert span_attempt_1.start_ts == t1
    assert span_attempt_1.end_ts == t2
    assert span_attempt_1.parent_span_id == root_span.span_id
    assert "postgresql://u:[REDACTED]@db/d" in span_attempt_1.attributes["error"]

    # Attempt 2 span assertions: distinctly tracked, closed on completion, status ok
    assert span_attempt_2.span_id == "span_retry_1_fetch_data_2" or "2" in span_attempt_2.span_id
    assert span_attempt_2.status == "ok"
    assert span_attempt_2.start_ts == t3
    assert span_attempt_2.end_ts == t4
    assert span_attempt_2.parent_span_id == root_span.span_id
    assert span_attempt_2.attributes["output"]["total_tokens"] == 120

    # Root span
    assert root_span.status == "ok"
    assert root_span.start_ts == t0
    assert root_span.end_ts == t5


def test_trace_collector_multiple_retries_and_terminal_failure() -> None:
    collector = TraceCollector()
    t0 = datetime(2026, 11, 6, 11, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 6, 11, 0, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 6, 11, 0, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 11, 6, 11, 0, 3, tzinfo=timezone.utc)
    t4 = datetime(2026, 11, 6, 11, 0, 4, tzinfo=timezone.utc)
    t5 = datetime(2026, 11, 6, 11, 0, 5, tzinfo=timezone.utc)
    t6 = datetime(2026, 11, 6, 11, 0, 6, tzinfo=timezone.utc)
    t7 = datetime(2026, 11, 6, 11, 0, 7, tzinfo=timezone.utc)

    # run.started
    collector(
        Event(ts=t0, run_id="run_fail_retries", type="run.started", data={"playbook": "crash_job"})
    )

    # Attempt 1: started -> retrying
    collector(
        Event(
            ts=t1,
            run_id="run_fail_retries",
            type="step.started",
            step_id="network_call",
            data={"attempt": 1},
        )
    )
    collector(
        Event(
            ts=t2,
            run_id="run_fail_retries",
            type="step.retrying",
            step_id="network_call",
            data={"attempt": 1, "error": "timeout 1"},
        )
    )

    # Attempt 2: started -> retrying
    collector(
        Event(
            ts=t3,
            run_id="run_fail_retries",
            type="step.started",
            step_id="network_call",
            data={"attempt": 2},
        )
    )
    collector(
        Event(
            ts=t4,
            run_id="run_fail_retries",
            type="step.retrying",
            step_id="network_call",
            data={"attempt": 2, "error": "timeout 2"},
        )
    )

    # Attempt 3: started -> failed
    collector(
        Event(
            ts=t5,
            run_id="run_fail_retries",
            type="step.started",
            step_id="network_call",
            data={"attempt": 3},
        )
    )
    collector(
        Event(
            ts=t6,
            run_id="run_fail_retries",
            type="step.failed",
            step_id="network_call",
            data={"attempts": 3, "error": "exhausted"},
        )
    )

    # run.failed
    collector(Event(ts=t7, run_id="run_fail_retries", type="run.failed", data={"status": "failed"}))

    spans = collector.get_spans("run_fail_retries")
    assert len(spans) == 4  # 1 root + 3 attempts

    step_spans = collector.get_step_spans("run_fail_retries", "network_call")
    assert len(step_spans) == 3

    # All 3 attempts recorded with status="error" and accurate durations
    assert step_spans[0].status == "error"
    assert step_spans[0].start_ts == t1
    assert step_spans[0].end_ts == t2
    assert step_spans[0].attributes["error"] == "timeout 1"

    assert step_spans[1].status == "error"
    assert step_spans[1].start_ts == t3
    assert step_spans[1].end_ts == t4
    assert step_spans[1].attributes["error"] == "timeout 2"

    assert step_spans[2].status == "error"
    assert step_spans[2].start_ts == t5
    assert step_spans[2].end_ts == t6
    assert step_spans[2].attributes["error"] == "exhausted"


def test_trace_collector_supersedes_in_progress_attempt() -> None:
    collector = TraceCollector()
    t0 = datetime(2026, 11, 6, 12, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 6, 12, 0, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 6, 12, 0, 2, tzinfo=timezone.utc)
    t3 = datetime(2026, 11, 6, 12, 0, 3, tzinfo=timezone.utc)

    collector(Event(ts=t0, run_id="run_sup", type="run.started", data={"playbook": "sup_pb"}))

    # Attempt 1 starts but never gets step.retrying before Attempt 2 starts
    collector(
        Event(ts=t1, run_id="run_sup", type="step.started", step_id="work", data={"attempt": 1})
    )
    # Attempt 2 starts: should close attempt 1
    collector(
        Event(ts=t2, run_id="run_sup", type="step.started", step_id="work", data={"attempt": 2})
    )
    # Attempt 2 completes
    collector(
        Event(ts=t3, run_id="run_sup", type="step.completed", step_id="work", data={"done": True})
    )

    step_spans = collector.get_step_spans("run_sup", "work")
    assert len(step_spans) == 2

    assert step_spans[0].status == "error"
    assert step_spans[0].end_ts == t2
    assert step_spans[0].attributes.get("superseded_by_retry") is True

    assert step_spans[1].status == "ok"
    assert step_spans[1].end_ts == t3


def test_trace_collector_compound_keys_and_clear() -> None:
    collector = TraceCollector()
    t = datetime.now(timezone.utc)

    collector(Event(ts=t, run_id="run_ck", type="run.started", data={"playbook": "p"}))
    collector(Event(ts=t, run_id="run_ck", type="step.started", step_id="s1", data={"attempt": 1}))
    collector(Event(ts=t, run_id="run_ck", type="step.retrying", step_id="s1", data={"attempt": 1}))
    collector(Event(ts=t, run_id="run_ck", type="step.started", step_id="s1", data={"attempt": 2}))
    collector(
        Event(ts=t, run_id="run_ck", type="step.completed", step_id="s1", data={"attempt": 2})
    )

    # Verify compound keys
    assert ("run_ck", "s1", 1) in collector._step_spans
    assert ("run_ck", "s1", 2) in collector._step_spans
    assert collector._step_spans[("run_ck", "s1", 1)] != collector._step_spans[("run_ck", "s1", 2)]

    collector.clear()
    assert len(collector.spans) == 0
    assert len(collector._step_spans) == 0
    assert len(collector._latest_step_spans) == 0
    assert len(collector._latest_step_attempts) == 0
