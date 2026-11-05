# Sandboxed Step Execution

`calbi.core.executor` provides pluggable execution abstractions for running playbook tool steps with strict isolation, credential scrubbing, and deterministic timeouts.

## Architectural Rationale

In an autonomous agent system, tools may interact with untrusted inputs, external networks, or potentially buggy third-party libraries. Calbi enforces a strict dual-system architecture:
> **The AI proposes. The runtime disposes.**

To prevent malicious tools, compromised dependencies, or prompt injections from exfiltrating orchestrator secrets or hanging the event loop, step execution is decoupled into the `StepExecutor` abstraction.

```
       +---------------------------------------------+
       |             Calbi Runtime Engine            |
       |  (orchestration, policy, approval, events)  |
       +---------------------------------------------+
                              |
                     StepExecutor Protocol
                              |
               +--------------+--------------+
               |                             |
      InProcessExecutor             SubprocessExecutor
      - Event loop (async)          - Isolated child PID
      - Worker thread (sync)        - Environment strictly scrubbed
      - JSON output check           - Secrets stripped (sk-*, passwords)
      - Cooperative timeout         - SIGKILL on timeout
```

---

## The `StepExecutor` Protocol

All executors implement the `@runtime_checkable` `StepExecutor` protocol:

```python
from typing import Any, Protocol, runtime_checkable
from calbi.core.registry import ToolSpec


@runtime_checkable
class StepExecutor(Protocol):
    async def execute(
        self,
        spec: ToolSpec,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Execute a tool specification with the provided inputs and optional timeout."""
        ...
```

---

## 1. `InProcessExecutor`

`InProcessExecutor` is the default lightweight executor.

### Behavior & Guarantees
- **Async Execution**: Asynchronous coroutine tools (`is_async=True`) execute directly on the active `asyncio` event loop.
- **Sync Offloading**: Synchronous callables execute in a separate worker thread via `asyncio.to_thread`, ensuring blocking operations do not stall the event loop.
- **Timeout Enforcement**: Enforces `timeout_s` via `asyncio.wait_for`. If execution exceeds `timeout_s`, `TimeoutError` is raised immediately.
- **Serialization Verification**: Verifies that tool return values are JSON-serializable (`json.dumps(result)`). If a tool returns an un-serializable object (e.g. socket, unpicklable class), `ExecutionIsolationError` is raised.

### Example

```python
from calbi.core.executor import InProcessExecutor
from calbi.core.registry import ToolSpec

executor = InProcessExecutor()
spec = ToolSpec(
    name="add",
    description="Add two numbers",
    fn=lambda a, b: a + b,  # lambdas permitted in-process
    is_async=False,
    input_schema={"type": "object"},
)

result = await executor.execute(spec, {"a": 10, "b": 32}, timeout_s=5.0)
assert result == 42
```

---

## 2. `SubprocessExecutor`

`SubprocessExecutor` executes tools inside an isolated child Python process (`sys.executable -c ...`).

### Security Model & Isolation Boundaries

1. **Address Space Isolation (Separate PID)**:
   The tool runs in a separate operating system process with its own memory space and interpreter state. A crash, memory leak, or segfault inside the tool does not affect the host runtime.

2. **Credential Scrubbing & Environment Sanitization**:
   The child process does **not** inherit the parent's `os.environ`. By default, only a minimal set of system environment variables is preserved:
   ```python
   allowed_env_keys = frozenset({"PATH", "LANG", "LC_ALL", "SYSTEMROOT"})
   ```
   Crucial credentials present in the host process environment—such as `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `DATABASE_URL`, or `SECRET_KEY`—are strictly stripped and cannot be accessed or exfiltrated by the child tool process.

3. **Importability Verification & Anti-Code Injection**:
   Only top-level, importable module functions can be executed in a subprocess.
   - Lambdas (`<lambda>`) are rejected with `ExecutionIsolationError`.
   - Local closures (`<locals>`) are rejected with `ExecutionIsolationError`.
   - Functions defined in `__main__` or un-resolvable references are rejected.

4. **Hard Process Termination on Timeout**:
   If a tool process hangs or exceeds `timeout_s`:
   - `proc.kill()` (SIGKILL) is dispatched immediately to the child process.
   - The runtime awaits `proc.wait()` to ensure the process is fully reaped without leaving zombie processes.
   - A `TimeoutError` is raised without hanging the parent process or test suite.

5. **Payload Bounds & IPC Integrity**:
   - Communication occurs over standard input/output (`stdin`/`stdout`) via structured JSON payloads.
   - Outputs exceeding `max_output_bytes` (default 1 MB) are rejected with `ExecutionIsolationError`.
   - Non-zero process exit codes and corrupted non-JSON outputs raise `ExecutionIsolationError`.

### Example

```python
import os
from calbi.core.executor import SubprocessExecutor
from calbi.core.registry import ToolSpec

# Host process holds sensitive credentials
os.environ["SECRET_KEY"] = "super-secret-token"


# Define top-level tool function
def get_safe_status() -> dict[str, str | None]:
    return {
        "status": "ready",
        "secret": os.environ.get("SECRET_KEY"),  # Returns None in child!
    }


executor = SubprocessExecutor(
    allowed_env_keys=frozenset({"PATH"}),
    max_output_bytes=1024 * 1024,
)

spec = ToolSpec(
    name="get_status",
    description="Check status",
    fn=get_safe_status,
    is_async=False,
    input_schema={"type": "object"},
)

result = await executor.execute(spec, {}, timeout_s=10.0)
assert result["status"] == "ready"
assert result["secret"] is None  # Credential was scrubbed
```

---

## Error Hierarchy

- `CalbiError`: Base exception for all Calbi errors.
  - `ExecutionIsolationError`: Raised when:
    - A tool output or input is not JSON-serializable.
    - A child process exits with a non-zero exit code.
    - A child process outputs non-JSON data.
    - A child process output exceeds `max_output_bytes`.
    - A non-importable function, lambda, or local closure is supplied to `SubprocessExecutor`.
- `TimeoutError`: Standard Python built-in exception raised when `timeout_s` is exceeded.
