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

5. **Streaming Payload Bounds & OOM Prevention**:
   - Child `stdout` is streamed incrementally with a hard cap of `max_output_bytes + 1` rather than buffering unbounded output into parent memory via `proc.communicate()`.
   - If output exceeds `max_output_bytes` (default 1 MB), the child process is terminated immediately (`proc.kill()`) and `ExecutionIsolationError` is raised, bounding parent memory RSS to $O(\text{max\_output\_bytes})$.

6. **Sentinel-Framed IPC Integrity**:
   - JSON results are transmitted within a dedicated frame bounded by sentinel lines (`__CALBI_OUTPUT_START__` and `__CALBI_OUTPUT_END__`).
   - Incidental user logs, debug `print()` statements, or third-party diagnostic output written to stdout before or after the sentinel payload do not corrupt JSON deserialization.

### Example

```python
import os
from calbi.core.executor import SubprocessExecutor
from calbi.core.registry import ToolSpec

# Host process holds sensitive credentials
os.environ["SECRET_KEY"] = "super-secret-token"


# Define top-level tool function
def get_safe_status() -> dict[str, str | None]:
    # Incidental debug logging will not corrupt payload parsing
    print("Debug: checking worker status...")
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

## 3. Threat Model & Security Boundaries (Non-Goals)

`SubprocessExecutor` provides process-level separation and credential scrubbing for autonomous agent workflows. Understanding both its guarantees and its non-goals is essential for secure production deployments.

### What SubprocessExecutor Protects Against

| Protection Area | Mechanism | Guarantees |
| --- | --- | --- |
| **Parent Memory & Interpreter Integrity** | Separate OS process (`sys.executable -c`) | Tool segfaults, memory leaks, unhandled exceptions, or interpreter crashes cannot crash or corrupt the parent Calbi runtime. |
| **Event Loop Blocking** | Subprocess concurrency with `asyncio` | CPU-intensive tool execution or blocking native calls do not stall the parent orchestrator event loop. |
| **Credential & Secret Leakage** | Explicit environment allowlist | Environment variables (e.g. `OPENROUTER_API_KEY`, `NVIDIA_API_KEY`, `DATABASE_URL`) are stripped by default; tools cannot inspect host secrets via `os.environ`. |
| **Runaway Execution & Hangs** | Deterministic timeout enforcement | When `timeout_s` is exceeded, child process is terminated immediately via SIGKILL (`proc.kill()`) and reaped via `proc.wait()`, preventing orphan/zombie processes. |
| **Memory Exhaustion DoS** | Incremental streaming byte cap (`max_output_bytes + 1`) | Prevents child processes from exhausting parent memory by streaming unbounded stdout (e.g., infinite loops or 50+ MB payloads). Child is killed as soon as the cap is reached. |
| **Incidental Output Corruption** | Sentinel framing (`__CALBI_OUTPUT_START__` / `__CALBI_OUTPUT_END__`) | Debug prints, logging libraries, and warnings printed to stdout do not corrupt JSON IPC payload deserialization. |
| **Arbitrary Code Injection via Callables** | Importability checks | Disallows `<lambda>`, `<locals>`, and dynamically defined closures, enforcing that only statically resolvable, imported functions can be executed. |

### What SubprocessExecutor Does NOT Protect Against (Non-Goals)

`SubprocessExecutor` is a **process isolation boundary, not a security hypervisor or container sandbox**. It does NOT provide kernel-level virtualization or complete hostility containment:

1. **Root & Kernel Privilege Escalation**:
   - The child process runs under the same OS user (UID/GID) as the host application.
   - If the host runs with administrative or root privileges, the child process has the same system permissions.

2. **Filesystem Read/Write Access**:
   - The child inherits standard filesystem access rights of the executing user.
   - Untrusted tools can read files accessible to the user (e.g., `~/.ssh`, configuration files, local databases) or overwrite workspace files unless filesystem permissions or chroots/mount namespaces are configured.

3. **Network Egress & Exfiltration**:
   - Subprocesses have unrestricted network egress by default.
   - A malicious or compromised dependency can open outbound TCP/UDP connections to exfiltrate data or interact with internal network endpoints (e.g., cloud metadata services like `169.254.169.254`).

4. **Hardware & Resource Starvation (Pre-Output)**:
   - While output byte counts and wall-clock time are capped, CPU core usage, fork bombs (`os.fork()`), and transient RAM consumption inside the child before termination are not restricted without OS-level cgroups.

5. **Side-Channels & Shared OS Resources**:
   - Process-level isolation does not protect against CPU cache timing attacks, shared IPC/POSIX semaphores, or shared `/tmp` directory inspection.

---

## 4. Recommended Production Sandboxing Architecture

For production deployments executing untrusted LLM-generated code or untrusted third-party tools, `StepExecutor` should be integrated into a defense-in-depth sandboxing architecture:

```
+-----------------------------------------------------------------------+
|                         Host / Orchestrator                           |
|  Calbi Runtime Engine (Policy, Verifier, EventBus, Secrets Kept Here) |
+-----------------------------------------------------------------------+
                                  |
                                  v
+-----------------------------------------------------------------------+
|                    Defense-in-Depth Sandbox Layer                     |
|                                                                       |
|  1. Containerization (Docker / Podman / Kubernetes):                  |
|     - Run as unprivileged non-root user (`UID 10001`)                 |
|     - Read-only root filesystem (`--read-only`)                       |
|     - Ephemeral tmpfs for scratch storage (`--tmpfs /tmp:rw,noexec`)  |
|     - Drop all Linux capabilities (`--cap-drop=ALL`)                  |
|                                                                       |
|  2. Network Sandboxing & Egress Control:                              |
|     - Isolated network namespace (`--net=none` for offline tools)     |
|     - Cilium / eBPF / iptables egress allowlists                      |
|     - Block cloud metadata IP (`169.254.169.254`)                     |
|                                                                       |
|  3. Resource Quotas (cgroups v2):                                     |
|     - CPU limits (`cpu.max`) & memory limits (`memory.max`)           |
|     - Process count limits (`pids.max`) to defeat fork bombs          |
|                                                                       |
|  4. MicroVM / System Call Interception (Optional Hardened Tier):      |
|     - gVisor (`runsc`) user-space kernel interception                 |
|     - Firecracker microVMs or nsjail sandboxing                       |
+-----------------------------------------------------------------------+
```

### Implementing Custom Production Executors

Because Calbi executors follow the simple `@runtime_checkable` `StepExecutor` protocol:

```python
class StepExecutor(Protocol):
    async def execute(
        self,
        spec: ToolSpec,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any: ...
```

Production systems can implement custom containerized or microVM executors (e.g. `DockerExecutor`, `NsjailExecutor`, `FirecrackerExecutor`) that conform to `StepExecutor` while enforcing hardware and network boundaries tailored to the deployment environment.

---

## 5. Error Hierarchy

- `CalbiError`: Base exception for all Calbi errors.
  - `ExecutionIsolationError`: Raised when:
    - A tool output or input is not JSON-serializable.
    - A child process exits with a non-zero exit code.
    - A child process outputs non-JSON data or missing sentinel framing.
    - A child process output exceeds `max_output_bytes`.
    - A non-importable function, lambda, or local closure is supplied to `SubprocessExecutor`.
- `TimeoutError`: Standard Python built-in exception raised when `timeout_s` is exceeded.
