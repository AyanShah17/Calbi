from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
from typing import Any, Protocol, runtime_checkable

from calbi.core.registry import ToolSpec
from calbi.core.schema import CalbiError


class ExecutionIsolationError(CalbiError):
    """Raised when process isolation, environment scrubbing, or subprocess execution fails."""


@runtime_checkable
class StepExecutor(Protocol):
    """Protocol defining the interface for step executors."""

    async def execute(
        self,
        spec: ToolSpec,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """Execute a tool specification with the provided inputs and optional timeout."""
        ...


class InProcessExecutor:
    """Default executor running async tools on event loop and sync tools via asyncio.to_thread."""

    async def execute(
        self,
        spec: ToolSpec,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        is_async = spec.is_async or inspect.iscoroutinefunction(spec.fn)
        if is_async:
            coro = spec.fn(**inputs)
        else:
            coro = asyncio.to_thread(spec.fn, **inputs)

        if timeout_s is not None:
            try:
                result = await asyncio.wait_for(coro, timeout=timeout_s)
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise TimeoutError(f"Tool '{spec.name}' exceeded timeout of {timeout_s}s") from exc
        else:
            result = await coro

        try:
            json.dumps(result)
        except (TypeError, OverflowError) as exc:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' output is not JSON-serializable: {exc}"
            ) from exc

        return result


_OUTPUT_START_SENTINEL = "__CALBI_OUTPUT_START__"
_OUTPUT_END_SENTINEL = "__CALBI_OUTPUT_END__"

_CHILD_RUNNER_CODE = """
import asyncio
import importlib
import inspect
import json
import os
import sys

_START_SENTINEL = "__CALBI_OUTPUT_START__"
_END_SENTINEL = "__CALBI_OUTPUT_END__"

def _run() -> None:
    try:
        raw_in = sys.stdin.read()
        if not raw_in:
            sys.stderr.write("No input payload received on stdin\\n")
            sys.exit(1)

        payload = json.loads(raw_in)
        mod_name = payload["module"]
        qualname = payload["qualname"]
        inputs = payload.get("inputs", {})
        max_bytes = payload.get("max_output_bytes")
        cwd = payload.get("cwd")
        extra_sys_paths = payload.get("sys_path", [])

        if cwd and cwd not in sys.path:
            sys.path.insert(0, cwd)
        for p in extra_sys_paths:
            if p not in sys.path:
                sys.path.append(p)

        mod = importlib.import_module(mod_name)
        fn = mod
        for part in qualname.split("."):
            fn = getattr(fn, part)

        if inspect.iscoroutinefunction(fn):
            result = asyncio.run(fn(**inputs))
        elif callable(fn):
            res = fn(**inputs)
            if inspect.iscoroutine(res):
                result = asyncio.run(res)
            else:
                result = res
        else:
            raise TypeError(f"Target '{mod_name}.{qualname}' is not callable")

        encoded = json.dumps(result)
        raw_out = encoded.encode("utf-8")
        if max_bytes is not None and len(raw_out) > max_bytes:
            sys.stderr.write(
                f"Subprocess output size {len(raw_out)} bytes exceeds max limit {max_bytes}\\n"
            )
            sys.exit(2)

        sys.stdout.write(f"\\n{_START_SENTINEL}\\n{encoded}\\n{_END_SENTINEL}\\n")
        sys.stdout.flush()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    _run()
"""


async def _feed_stdin(proc: asyncio.subprocess.Process, payload: bytes) -> None:
    if proc.stdin is None:
        return
    try:
        proc.stdin.write(payload)
        await proc.stdin.drain()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        try:
            proc.stdin.close()
            await proc.stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError, Exception):
            pass


async def _read_stdout_stream(
    proc: asyncio.subprocess.Process,
    max_output_bytes: int,
) -> tuple[bytes, bool]:
    """Stream stdout from proc up to hard cap max_output_bytes + 1.

    If output exceeds max_output_bytes, kill proc and return (received_bytes, True).
    """
    if proc.stdout is None:
        return b"", False

    chunks: list[bytes] = []
    total_bytes = 0
    exceeded = False
    hard_cap = max_output_bytes + 1

    while True:
        to_read = min(65536, hard_cap - total_bytes)
        if to_read <= 0:
            exceeded = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            break

        chunk = await proc.stdout.read(to_read)
        if not chunk:
            break

        chunks.append(chunk)
        total_bytes += len(chunk)

        if total_bytes > max_output_bytes:
            exceeded = True
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            break

    return b"".join(chunks), exceeded


async def _read_stderr_stream(
    proc: asyncio.subprocess.Process,
    max_bytes: int = 65536,
) -> bytes:
    """Read stderr from proc up to max_bytes, discarding extra to avoid unbounded memory."""
    if proc.stderr is None:
        return b""

    chunks: list[bytes] = []
    total = 0

    while True:
        chunk = await proc.stderr.read(65536)
        if not chunk:
            break
        if total < max_bytes:
            take = min(len(chunk), max_bytes - total)
            chunks.append(chunk[:take])
            total += take

    return b"".join(chunks)


class SubprocessExecutor:
    """Runs top-level importable tool functions in an isolated child Python process."""

    def __init__(
        self,
        *,
        allowed_env_keys: set[str] | frozenset[str] = frozenset(
            {"PATH", "LANG", "LC_ALL", "SYSTEMROOT"}
        ),
        max_output_bytes: int = 1_048_576,
    ) -> None:
        self.allowed_env_keys: frozenset[str] = frozenset(allowed_env_keys)
        self.max_output_bytes: int = max_output_bytes

    async def execute(
        self,
        spec: ToolSpec,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        fn = spec.fn
        name = getattr(fn, "__name__", "")
        mod_name = getattr(fn, "__module__", None)
        qual_name = getattr(fn, "__qualname__", None)

        if not mod_name or not qual_name:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' function has no __module__ or __qualname__ "
                "and cannot be isolated in a subprocess."
            )

        if name == "<lambda>" or "<locals>" in qual_name:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' ({qual_name}) is a lambda or local closure "
                "and cannot be isolated in a subprocess."
            )

        if mod_name == "__main__":
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' is defined in __main__ and cannot be isolated in a subprocess."
            )

        if inspect.ismethod(fn):
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' is a bound method and cannot be isolated in a subprocess."
            )

        try:
            mod = importlib.import_module(mod_name)
            target: Any = mod
            for part in qual_name.split("."):
                target = getattr(target, part)
        except Exception as exc:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' ({mod_name}.{qual_name}) cannot be imported: {exc}"
            ) from exc

        try:
            payload = json.dumps(
                {
                    "module": mod_name,
                    "qualname": qual_name,
                    "inputs": inputs,
                    "max_output_bytes": self.max_output_bytes,
                    "cwd": os.getcwd(),
                    "sys_path": [p for p in sys.path if p and os.path.exists(p)],
                }
            ).encode("utf-8")
        except (TypeError, OverflowError) as exc:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' inputs are not JSON-serializable: {exc}"
            ) from exc

        child_env = {k: v for k, v in os.environ.items() if k in self.allowed_env_keys}

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                _CHILD_RUNNER_CODE,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=child_env,
                cwd=os.getcwd(),
            )
        except Exception as exc:
            raise ExecutionIsolationError(
                f"Failed to spawn child process for tool '{spec.name}': {exc}"
            ) from exc

        stdin_task = asyncio.create_task(_feed_stdin(proc, payload))
        stdout_task = asyncio.create_task(_read_stdout_stream(proc, self.max_output_bytes))
        stderr_task = asyncio.create_task(_read_stderr_stream(proc, 65536))

        try:
            if timeout_s is not None:
                (stdout_bytes, exceeded), stderr_bytes = await asyncio.wait_for(
                    asyncio.gather(stdout_task, stderr_task),
                    timeout=timeout_s,
                )
            else:
                (stdout_bytes, exceeded), stderr_bytes = await asyncio.gather(
                    stdout_task, stderr_task
                )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            stdin_task.cancel()
            stdout_task.cancel()
            stderr_task.cancel()
            raise TimeoutError(
                f"Tool '{spec.name}' subprocess exceeded timeout of {timeout_s}s"
            ) from exc
        finally:
            if not stdin_task.done():
                stdin_task.cancel()
            await proc.wait()

        if exceeded or len(stdout_bytes) > self.max_output_bytes:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' output exceeded maximum allowed size of "
                f"{self.max_output_bytes} bytes (received at least {len(stdout_bytes)} bytes)."
            )

        if proc.returncode != 0:
            err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' subprocess exited with non-zero code "
                f"{proc.returncode}: {err_msg}"
            )

        try:
            decoded = stdout_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' subprocess produced invalid JSON output: {exc}"
            ) from exc

        start_idx = decoded.rfind(_OUTPUT_START_SENTINEL)
        if start_idx == -1:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' subprocess produced invalid JSON output: "
                "missing sentinel framing."
            )

        payload_start = start_idx + len(_OUTPUT_START_SENTINEL)
        end_idx = decoded.find(_OUTPUT_END_SENTINEL, payload_start)
        if end_idx == -1:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' subprocess produced invalid JSON output: "
                "missing sentinel framing."
            )

        json_str = decoded[payload_start:end_idx].strip()

        try:
            result = json.loads(json_str)
        except Exception as exc:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' subprocess produced invalid JSON output: {exc}"
            ) from exc

        return result
