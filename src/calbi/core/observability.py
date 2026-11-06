from __future__ import annotations

import re
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, TextIO

from pydantic import BaseModel, Field

from calbi.core.events import Event

DEFAULT_ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "max_tokens",
        "tokens_used",
        "token_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "tokens",
        "token_budget",
        "tokens_budget",
        "cached_tokens",
        "reasoning_tokens",
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "tokens_remaining",
        "num_tokens",
        "token_usage",
        "tokens_total",
        "prompt_tokens_details",
        "completion_tokens_details",
    }
)


class Redactor:
    """Recursively redacts sensitive keys, scrubs credentials, and caps payload byte length."""

    def __init__(
        self,
        *,
        max_string_bytes: int = 2048,
        sensitive_keys: frozenset[str] | set[str] = frozenset(
            {
                "api_key",
                "password",
                "secret",
                "token",
                "authorization",
                "dsn",
                "database_url",
            }
        ),
        allowed_keys: frozenset[str] | set[str] = DEFAULT_ALLOWED_KEYS,
        extra_patterns: Sequence[re.Pattern[str]] = (),
    ) -> None:
        self.max_string_bytes = max_string_bytes
        self.sensitive_keys = {self._normalize_key(k) for k in sensitive_keys}
        self.allowed_keys = {self._normalize_key(k) for k in allowed_keys}
        self.extra_patterns = list(extra_patterns)

        # Standard credential scrubbing regular expressions
        self._bearer_pattern = re.compile(r"(Bearer\s+)[A-Za-z0-9._\-~+/=]+", re.IGNORECASE)
        self._dsn_pattern = re.compile(r"(postgres(?:ql)?://[^:/]*:)[^@]+(@)", re.IGNORECASE)
        self._api_key_sk_pattern = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
        self._api_key_nv_pattern = re.compile(r"nvapi-[A-Za-z0-9_\-]{8,}")

    @staticmethod
    def _normalize_key(key: Any) -> str:
        """Normalize key across camelCase, kebab-case, dots, and underscores."""
        s = str(key)
        s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
        s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s)
        s = re.sub(r"[-.\s:]+", "_", s)
        return s.lower().strip("_")

    def is_sensitive_key(self, key: Any) -> bool:
        """Determine if a dictionary key represents sensitive credentials."""
        norm = self._normalize_key(key)
        if not norm:
            return False

        # 1. Non-secret token count/usage allowlist check
        if norm in self.allowed_keys:
            return False

        # 2. Check against sensitive keys with delimiter-aware word boundary matching
        padded_norm = f"_{norm}_"
        for sens in self.sensitive_keys:
            if not sens:
                continue
            padded_sens = f"_{sens}_"
            if padded_sens in padded_norm:
                return True
            # Support plural tokens for secret tokens (e.g., auth_tokens, api_tokens)
            if sens == "token" and "_tokens_" in padded_norm:
                return True

        return False

    def redact(self, value: Any) -> Any:
        """Recursively redact values in dicts, lists, sets, tuples, and strings."""
        if isinstance(value, str):
            return self._redact_string(value)
        if isinstance(value, dict):
            redacted_dict: dict[str, Any] = {}
            for k, v in value.items():
                if self.is_sensitive_key(k):
                    redacted_dict[k] = "[REDACTED]"
                else:
                    redacted_dict[k] = self.redact(v)
            return redacted_dict
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.redact(v) for v in value)
        if isinstance(value, set):
            return {self.redact(v) for v in value}
        if isinstance(value, Event):
            return self.redact_event(value)
        if isinstance(value, BaseModel):
            return self.redact(value.model_dump(mode="python"))
        return value

    def _redact_string(self, text: str) -> str:
        """Apply regex credential scrubbing and byte length capping."""
        s = self._bearer_pattern.sub(r"\1[REDACTED]", text)
        s = self._dsn_pattern.sub(r"\1[REDACTED]\2", s)
        s = self._api_key_sk_pattern.sub("[REDACTED]", s)
        s = self._api_key_nv_pattern.sub("[REDACTED]", s)
        for pat in self.extra_patterns:
            s = pat.sub("[REDACTED]", s)

        # Byte length capping (UTF-8)
        encoded = s.encode("utf-8")
        if len(encoded) > self.max_string_bytes:
            truncated = encoded[: self.max_string_bytes].decode("utf-8", errors="ignore")
            s = truncated + "... [truncated]"
        return s

    def redact_event(self, event: Event) -> Event:
        """Return a new Event instance with redacted payload data (non-destructive)."""
        redacted_data = self.redact(event.data)
        safe_data: dict[str, Any] = (
            redacted_data if isinstance(redacted_data, dict) else {"value": redacted_data}
        )
        return Event(
            ts=event.ts,
            run_id=event.run_id,
            type=event.type,
            step_id=event.step_id,
            data=safe_data,
        )


class JSONLEventSink:
    """Thread-safe event subscriber that writes redacted JSONL lines to a file or stream."""

    def __init__(
        self,
        target: Path | str | TextIO,
        redactor: Redactor | None = None,
    ) -> None:
        self.target = target
        self.redactor = redactor or Redactor()
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        """Process event, apply redaction, and write JSON line."""
        redacted = self.redactor.redact_event(event)
        line = redacted.model_dump_json() + "\n"
        with self._lock:
            if isinstance(self.target, (str, Path)):
                p = Path(self.target)
                p.parent.mkdir(parents=True, exist_ok=True)
                with open(p, mode="a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
            else:
                self.target.write(line)
                if hasattr(self.target, "flush"):
                    self.target.flush()

    def close(self) -> None:
        """Close target stream if applicable."""
        with self._lock:
            if not isinstance(self.target, (str, Path)) and hasattr(self.target, "close"):
                self.target.close()


class SpanRecord(BaseModel):
    """OpenTelemetry-compatible span representation for workflow tracing."""

    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    name: str
    start_ts: datetime
    end_ts: datetime | None = None
    status: Literal["ok", "error", "in_progress"]
    attributes: dict[str, Any] = Field(default_factory=dict)


class TraceCollector:
    """Builds hierarchical trace -> run -> step spans from EventBus events."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        self.redactor = redactor or Redactor()
        self._spans: dict[str, SpanRecord] = {}
        self._run_spans: dict[str, str] = {}  # run_id -> span_id
        self._step_spans: dict[
            tuple[str, str, int], str
        ] = {}  # (run_id, step_id, attempt) -> span_id
        self._latest_step_spans: dict[
            tuple[str, str], str
        ] = {}  # (run_id, step_id) -> latest span_id
        self._latest_step_attempts: dict[
            tuple[str, str], int
        ] = {}  # (run_id, step_id) -> latest attempt
        self._lock = threading.Lock()

    def __call__(self, event: Event) -> None:
        """Process an event from EventBus and map to hierarchical spans."""
        with self._lock:
            raw_redacted = self.redactor.redact(event.data)
            redacted_data: dict[str, Any] = (
                raw_redacted if isinstance(raw_redacted, dict) else {"payload": raw_redacted}
            )

            if event.type == "run.started":
                root_span_id = f"span_{event.run_id}"
                self._run_spans[event.run_id] = root_span_id
                playbook_name = str(redacted_data.get("playbook", "unknown"))
                self._spans[root_span_id] = SpanRecord(
                    trace_id=event.run_id,
                    span_id=root_span_id,
                    parent_span_id=None,
                    name=f"run:{playbook_name}",
                    start_ts=event.ts,
                    status="in_progress",
                    attributes=redacted_data,
                )

            elif event.type == "step.started":
                parent_id = self._run_spans.get(event.run_id)
                step_name = event.step_id or "unknown"
                step_pair = (event.run_id, step_name)

                # If prior attempt was still in_progress, close it with status="error"
                prior_span_id = self._latest_step_spans.get(step_pair)
                if prior_span_id and prior_span_id in self._spans:
                    prior_span = self._spans[prior_span_id]
                    if prior_span.status == "in_progress":
                        prior_span.end_ts = event.ts
                        prior_span.status = "error"
                        prior_span.attributes.setdefault("superseded_by_retry", True)

                # Determine attempt number
                attempt_val = redacted_data.get("attempt")
                if attempt_val is not None:
                    try:
                        attempt = int(attempt_val)
                    except (ValueError, TypeError):
                        attempt = self._latest_step_attempts.get(step_pair, 0) + 1
                else:
                    attempt = self._latest_step_attempts.get(step_pair, 0) + 1

                compound_key = (event.run_id, step_name, attempt)
                child_span_id = f"span_{event.run_id}_{step_name}_{attempt}"
                self._step_spans[compound_key] = child_span_id
                self._latest_step_spans[step_pair] = child_span_id
                self._latest_step_attempts[step_pair] = attempt

                self._spans[child_span_id] = SpanRecord(
                    trace_id=event.run_id,
                    span_id=child_span_id,
                    parent_span_id=parent_id,
                    name=f"step:{step_name}",
                    start_ts=event.ts,
                    status="in_progress",
                    attributes=redacted_data,
                )

            elif event.type == "step.retrying":
                step_name = event.step_id or "unknown"
                step_pair = (event.run_id, step_name)
                attempt_val = redacted_data.get("attempt")
                try:
                    attempt = (
                        int(attempt_val)
                        if attempt_val is not None
                        else self._latest_step_attempts.get(step_pair, 1)
                    )
                except (ValueError, TypeError):
                    attempt = self._latest_step_attempts.get(step_pair, 1)

                compound_key = (event.run_id, step_name, attempt)
                span_id = self._step_spans.get(compound_key) or self._latest_step_spans.get(
                    step_pair
                )
                if span_id and span_id in self._spans:
                    span = self._spans[span_id]
                    span.end_ts = event.ts
                    span.status = "error"
                    span.attributes.update(redacted_data)
                else:
                    # Fallback span if step.retrying arrived without step.started
                    fallback_id = f"span_{event.run_id}_{step_name}_{attempt}"
                    parent_id = self._run_spans.get(event.run_id)
                    self._spans[fallback_id] = SpanRecord(
                        trace_id=event.run_id,
                        span_id=fallback_id,
                        parent_span_id=parent_id,
                        name=f"step:{step_name}",
                        start_ts=event.ts,
                        end_ts=event.ts,
                        status="error",
                        attributes=redacted_data,
                    )
                    self._step_spans[compound_key] = fallback_id
                    self._latest_step_spans[step_pair] = fallback_id
                    self._latest_step_attempts[step_pair] = attempt

            elif event.type == "step.completed":
                step_name = event.step_id or "unknown"
                step_pair = (event.run_id, step_name)
                attempt_val = redacted_data.get("attempt")
                span_id = None
                if attempt_val is not None:
                    try:
                        span_id = self._step_spans.get((event.run_id, step_name, int(attempt_val)))
                    except (ValueError, TypeError):
                        pass
                if not span_id:
                    span_id = self._latest_step_spans.get(step_pair)

                if span_id and span_id in self._spans:
                    span = self._spans[span_id]
                    span.end_ts = event.ts
                    span.status = "ok"
                    span.attributes.update(redacted_data)
                else:
                    # Completed without a matching step.started span
                    attempt = self._latest_step_attempts.get(step_pair, 1)
                    fallback_id = f"span_{event.run_id}_{step_name}_{attempt}_completed"
                    parent_id = self._run_spans.get(event.run_id)
                    self._spans[fallback_id] = SpanRecord(
                        trace_id=event.run_id,
                        span_id=fallback_id,
                        parent_span_id=parent_id,
                        name=f"step:{step_name}",
                        start_ts=event.ts,
                        end_ts=event.ts,
                        status="ok",
                        attributes=redacted_data,
                    )

            elif event.type == "step.failed":
                step_name = event.step_id or "unknown"
                step_pair = (event.run_id, step_name)
                attempt_val = redacted_data.get("attempt") or redacted_data.get("attempts")
                span_id = None
                if attempt_val is not None:
                    try:
                        span_id = self._step_spans.get((event.run_id, step_name, int(attempt_val)))
                    except (ValueError, TypeError):
                        pass
                if not span_id:
                    span_id = self._latest_step_spans.get(step_pair)

                if span_id and span_id in self._spans:
                    span = self._spans[span_id]
                    span.end_ts = event.ts
                    span.status = "error"
                    span.attributes.update(redacted_data)
                else:
                    # Failed before started (e.g. denied by policy)
                    attempt = self._latest_step_attempts.get(step_pair, 1)
                    fallback_id = f"span_{event.run_id}_{step_name}_{attempt}_failed"
                    parent_id = self._run_spans.get(event.run_id)
                    self._spans[fallback_id] = SpanRecord(
                        trace_id=event.run_id,
                        span_id=fallback_id,
                        parent_span_id=parent_id,
                        name=f"step:{step_name}",
                        start_ts=event.ts,
                        end_ts=event.ts,
                        status="error",
                        attributes=redacted_data,
                    )

            elif event.type == "run.completed":
                existing_run_span_id = self._run_spans.get(event.run_id)
                if existing_run_span_id and existing_run_span_id in self._spans:
                    span = self._spans[existing_run_span_id]
                    span.end_ts = event.ts
                    span.status = "ok"
                    span.attributes.update(redacted_data)

            elif event.type == "run.failed":
                existing_run_span_id = self._run_spans.get(event.run_id)
                if existing_run_span_id and existing_run_span_id in self._spans:
                    span = self._spans[existing_run_span_id]
                    span.end_ts = event.ts
                    span.status = "error"
                    span.attributes.update(redacted_data)
                else:
                    fallback_id = f"span_{event.run_id}"
                    self._spans[fallback_id] = SpanRecord(
                        trace_id=event.run_id,
                        span_id=fallback_id,
                        parent_span_id=None,
                        name="run:failed",
                        start_ts=event.ts,
                        end_ts=event.ts,
                        status="error",
                        attributes=redacted_data,
                    )

    @property
    def spans(self) -> list[SpanRecord]:
        """Return all recorded spans in order of recording."""
        with self._lock:
            return list(self._spans.values())

    def get_spans(self, run_id: str | None = None) -> list[SpanRecord]:
        """Filter spans optionally by trace_id (run_id)."""
        with self._lock:
            if run_id is None:
                return list(self._spans.values())
            return [s for s in self._spans.values() if s.trace_id == run_id]

    def get_step_spans(self, run_id: str, step_id: str) -> list[SpanRecord]:
        """Return all attempt spans for a given step in a run."""
        with self._lock:
            return [
                s
                for s in self._spans.values()
                if s.trace_id == run_id and s.name == f"step:{step_id}"
            ]

    def clear(self) -> None:
        """Clear all recorded spans."""
        with self._lock:
            self._spans.clear()
            self._run_spans.clear()
            self._step_spans.clear()
            self._latest_step_spans.clear()
            self._latest_step_attempts.clear()


__all__ = [
    "DEFAULT_ALLOWED_KEYS",
    "JSONLEventSink",
    "Redactor",
    "SpanRecord",
    "TraceCollector",
]
