# Calbi Observability & Redaction

Calbi provides production-grade observability, credential redaction, and OpenTelemetry-compatible tracing with zero external telemetry dependencies.

## Architecture & Guarantees

1. **Secret & Credential Scrubbing**: Sensitive keys and credentials are automatically redacted before logging, streaming, or storing in traces.
2. **Payload Size Capping**: Long strings are capped at 2048 bytes (UTF-8 byte length) to prevent memory explosion from large responses or raw HTML/dumps.
3. **Non-Destructive Redaction**: EventBus events and payloads remain unmodified in-memory for downstream execution logic; redaction operates on copies.
4. **Thread-Safe Sinks**: File and stream sinks employ synchronization locks for concurrent step and run execution.
5. **Zero External Dependency OTel Hierarchy**: `TraceCollector` reconstructs hierarchical trace, run, and step spans matching OpenTelemetry semantics.

---

## Redactor

`calbi.core.observability.Redactor` recursively scrubs dictionaries, lists, sets, tuples, Pydantic models, and raw strings.

### Sensitive Keys (Case-Insensitive Substrings)
By default, dictionary keys containing any of the following substrings have their values replaced with `"[REDACTED]"`:
- `api_key`
- `password`
- `secret`
- `token`
- `authorization`
- `dsn`
- `database_url`

### Inline Pattern Scrubbing
Raw strings are scrubbed for common credentials:
- **Bearer Tokens**: `Bearer <token>` &rarr; `Bearer [REDACTED]`
- **Postgres DSN Passwords**: `postgresql://user:pass@host/db` &rarr; `postgresql://user:[REDACTED]@host/db`
- **OpenAI / OpenRouter API Keys**: `sk-<8+ chars>` &rarr; `[REDACTED]`
- **NVIDIA NIM API Keys**: `nvapi-<8+ chars>` &rarr; `[REDACTED]`
- **Custom Patterns**: User-provided `Sequence[re.Pattern[str]]` substituted with `"[REDACTED]"`.

### Payload Cap
Strings longer than `max_string_bytes` (default `2048`) are truncated at byte boundaries and suffixed with `"... [truncated]"`. Multi-byte UTF-8 sequences are preserved safely without corruption.

### Example
```python
from calbi.core.observability import Redactor

redactor = Redactor(max_string_bytes=2048)
clean_data = redactor.redact({
    "user": "alice",
    "API_KEY": "sk-proj-1234567890",
    "error_log": "Connection failed for postgresql://admin:secret123@db:5432/app",
})
# clean_data["API_KEY"] == "[REDACTED]"
# clean_data["error_log"] == "Connection failed for postgresql://admin:[REDACTED]@db:5432/app"
```

---

## JSONLEventSink

`JSONLEventSink` attaches directly to `calbi.core.events.EventBus` and writes redacted JSONL lines to a file path or standard stream.

### Usage
```python
from pathlib import Path
from calbi.core.events import EventBus
from calbi.core.observability import JSONLEventSink

bus = EventBus()
sink = JSONLEventSink(target=Path("/var/log/calbi/events.jsonl"))
bus.subscribe(sink)

# Any events emitted by the runtime will be redacted and written to the JSONL file
```

---

## TraceCollector & SpanRecord

`TraceCollector` reconstructs execution spans from EventBus lifecycle events:
- `run.started` &rarr; creates root span `name="run:<playbook>"` with `status="in_progress"`
- `run.completed` / `run.failed` &rarr; updates root span with `end_ts` and `status="ok"` / `"error"`
- `step.started` &rarr; creates child span `name="step:<step_id>"` referencing parent root span
- `step.completed` / `step.failed` &rarr; updates child span with `end_ts` and `status="ok"` / `"error"`

### SpanRecord Fields
- `trace_id: str`: Corresponds to `run_id`
- `span_id: str`: Unique identifier for the span
- `parent_span_id: str | None`: Identifier of the parent span (or `None` for root spans)
- `name: str`: Operation name (e.g., `run:playbook_name`, `step:step_id`)
- `start_ts: datetime`: UTC timestamp when operation started
- `end_ts: datetime | None`: UTC timestamp when operation finished
- `status: Literal["ok", "error", "in_progress"]`: Current execution status
- `attributes: dict[str, Any]`: Scrubbed event attributes (inputs, outputs, errors, durations)

### Usage
```python
from calbi.core.events import EventBus
from calbi.core.observability import TraceCollector

bus = EventBus()
collector = TraceCollector()
bus.subscribe(collector)

# Retrieve all recorded spans
for span in collector.spans:
    print(f"{span.name} [{span.status}]: parent={span.parent_span_id} duration={span.end_ts - span.start_ts if span.end_ts else 'running'}")
```
