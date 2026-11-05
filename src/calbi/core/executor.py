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
            result = await asyncio.wait_for(coro, timeout=timeout_s)
        else:
            result = await coro

        try:
            json.dumps(result)
        except (TypeError, OverflowError) as exc:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' output is not JSON-serializable: {exc}"
            ) from exc

        return result


_CHILD_RUNNER_CODE = """
import asyncio
import importlib
import inspect
import json
import os
import sys

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

        sys.stdout.write(encoded)
        sys.stdout.flush()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    _run()
"""


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

        try:
            if timeout_s is not None:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(input=payload),
                    timeout=timeout_s,
                )
            else:
                stdout_bytes, stderr_bytes = await proc.communicate(input=payload)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise TimeoutError(
                f"Tool '{spec.name}' subprocess exceeded timeout of {timeout_s}s"
            ) from exc

        if len(stdout_bytes) > self.max_output_bytes:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' output exceeded maximum allowed size of "
                f"{self.max_output_bytes} bytes (received {len(stdout_bytes)} bytes)."
            )

        if proc.returncode != 0:
            err_msg = stderr_bytes.decode("utf-8", errors="replace").strip()
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' subprocess exited with non-zero code "
                f"{proc.returncode}: {err_msg}"
            )

        try:
            decoded = stdout_bytes.decode("utf-8")
            result = json.loads(decoded)
        except Exception as exc:
            raise ExecutionIsolationError(
                f"Tool '{spec.name}' subprocess produced invalid JSON output: {exc}"
            ) from exc

        return result
