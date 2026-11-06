from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import Any

import pytest

from calbi.core.executor import (
    ExecutionIsolationError,
    InProcessExecutor,
    StepExecutor,
    SubprocessExecutor,
)
from calbi.core.registry import ToolSpec

# ---------------------------------------------------------------------------
# Top-level helper functions for SubprocessExecutor & InProcessExecutor tests
# (SubprocessExecutor requires top-level importable callables)
# ---------------------------------------------------------------------------


def sync_add(a: int, b: int) -> int:
    """Synchronous integer addition."""
    return a + b


async def async_multiply(a: int, b: int) -> int:
    """Asynchronous integer multiplication."""
    await asyncio.sleep(0.01)
    return a * b


def get_current_pid() -> int:
    """Return current process ID."""
    return os.getpid()


async def async_get_current_pid() -> int:
    """Asynchronously return current process ID."""
    await asyncio.sleep(0.01)
    return os.getpid()


def read_env_secrets() -> dict[str, Any]:
    """Inspect environment variables for secret keys and standard keys."""
    return {
        "SECRET_KEY": os.environ.get("SECRET_KEY"),
        "OPENROUTER_API_KEY": os.environ.get("OPENROUTER_API_KEY"),
        "NVIDIA_API_KEY": os.environ.get("NVIDIA_API_KEY"),
        "DATABASE_URL": os.environ.get("DATABASE_URL"),
        "has_path": "PATH" in os.environ,
    }


def read_custom_env(key: str) -> str | None:
    """Read a specific environment variable by key."""
    return os.environ.get(key)


def sync_slow_tool(duration: float = 0.4) -> None:
    """Sleep for a short duration; triggers timeout without blocking thread teardown."""
    time.sleep(duration)


def infinite_sync_sleep() -> None:
    """Sleep for 100 seconds (intended to be killed by subprocess proc.kill())."""
    time.sleep(100)


async def infinite_async_sleep() -> None:
    """Asynchronously sleep for 300 seconds (intended to be killed by timeout)."""
    await asyncio.sleep(300)


def exit_with_code(code: int = 42) -> None:
    """Explicitly exit with a non-zero code."""
    sys.exit(code)


def corrupt_stdout_tool() -> None:
    """Write invalid non-JSON data directly to stdout and exit immediately without sentinels."""
    sys.stdout.write("THIS_IS_DEFINITELY_NOT_JSON!!!\n")
    sys.stdout.flush()
    os._exit(0)


def tool_with_incidental_stdout_prints() -> dict[str, Any]:
    """Tool that produces incidental debug prints before returning."""
    print("DEBUG: entering tool function")
    sys.stdout.write("THIS_IS_INCIDENTAL_PRINT_OUTPUT!!!\n")
    print("DEBUG: finishing tool function")
    return {"success": True, "answer": 42}


def tool_with_atexit_noise() -> str:
    """Tool that registers an atexit handler to print noise AFTER the child runner's sentinel."""
    import atexit

    def _after_cleanup() -> None:
        sys.stdout.write("TRAILING NOISE AFTER SENTINEL\n")
        sys.stdout.flush()

    atexit.register(_after_cleanup)
    print("LEADING NOISE BEFORE SENTINEL")
    return "payload_with_noise_before_and_after"


def stream_50mb_tool() -> None:
    """Tool that emits ~50MB of data on stdout to test streaming byte cap."""
    chunk = "X" * 65536
    for _ in range(800):  # ~52 MB
        sys.stdout.write(chunk)
        sys.stdout.flush()


def infinite_stdout_tool() -> None:
    """Tool that infinitely prints to stdout until killed."""
    while True:
        sys.stdout.write("A" * 1024)
        sys.stdout.flush()


def emit_corrupt_sentinel_json_tool() -> None:
    """Tool that writes invalid JSON inside valid sentinels and exits immediately."""
    sys.stdout.write("\n__CALBI_OUTPUT_START__\n{bad_json_not_quoted:\n__CALBI_OUTPUT_END__\n")
    sys.stdout.flush()
    os._exit(0)


def emit_missing_end_sentinel_tool() -> None:
    """Tool that writes start sentinel without end sentinel and exits immediately."""
    sys.stdout.write('\n__CALBI_OUTPUT_START__\n{"foo": "bar"}\n')
    sys.stdout.flush()
    os._exit(0)


def failing_tool(msg: str = "child failure") -> None:
    """Raise an unhandled exception inside the tool."""
    raise RuntimeError(msg)


def unserializable_output_tool() -> Any:
    """Return an object that cannot be JSON serialized."""
    return {"unsupported_object": object()}


def large_output_tool(size: int = 5000) -> str:
    """Return a string of specified character size."""
    return "A" * size


# ---------------------------------------------------------------------------
# Test Suites
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_protocol_conformance() -> None:
    """Verify InProcessExecutor and SubprocessExecutor implement StepExecutor protocol."""
    in_proc = InProcessExecutor()
    sub_proc = SubprocessExecutor()

    assert isinstance(in_proc, StepExecutor)
    assert isinstance(sub_proc, StepExecutor)


@pytest.mark.asyncio
async def test_in_process_sync_tool() -> None:
    """Verify InProcessExecutor executes sync tools and returns result."""
    executor = InProcessExecutor()
    spec = ToolSpec(
        name="sync_add",
        description="add numbers",
        fn=sync_add,
        is_async=False,
        input_schema={"type": "object"},
    )

    result = await executor.execute(spec, {"a": 10, "b": 32})
    assert result == 42


@pytest.mark.asyncio
async def test_in_process_async_tool() -> None:
    """Verify InProcessExecutor executes async tools and returns result."""
    executor = InProcessExecutor()
    spec = ToolSpec(
        name="async_multiply",
        description="multiply numbers",
        fn=async_multiply,
        is_async=True,
        input_schema={"type": "object"},
    )

    result = await executor.execute(spec, {"a": 6, "b": 7})
    assert result == 42


@pytest.mark.asyncio
async def test_in_process_sync_timeout() -> None:
    """Verify InProcessExecutor enforces timeout on sync tools."""
    executor = InProcessExecutor()
    spec = ToolSpec(
        name="sync_slow_tool",
        description="sleeps",
        fn=sync_slow_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        await executor.execute(spec, {"duration": 0.4}, timeout_s=0.05)
    assert (time.perf_counter() - start) < 0.3


@pytest.mark.asyncio
async def test_in_process_async_timeout() -> None:
    """Verify InProcessExecutor enforces timeout on async tools."""
    executor = InProcessExecutor()
    spec = ToolSpec(
        name="infinite_async_sleep",
        description="sleeps async",
        fn=infinite_async_sleep,
        is_async=True,
        input_schema={"type": "object"},
    )

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        await executor.execute(spec, {}, timeout_s=0.1)
    assert (time.perf_counter() - start) < 1.0


@pytest.mark.asyncio
async def test_in_process_unserializable_output_raises_isolation_error() -> None:
    """Verify InProcessExecutor raises error when tool output is not JSON-serializable."""
    executor = InProcessExecutor()
    spec = ToolSpec(
        name="unserializable_output_tool",
        description="bad output",
        fn=unserializable_output_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {})
    assert "not JSON-serializable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_in_process_tool_exception_propagates() -> None:
    """Verify unhandled tool exceptions propagate directly from InProcessExecutor."""
    executor = InProcessExecutor()
    spec = ToolSpec(
        name="failing_tool",
        description="fails",
        fn=failing_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(RuntimeError, match="in_process failure"):
        await executor.execute(spec, {"msg": "in_process failure"})


@pytest.mark.asyncio
async def test_subprocess_runs_in_separate_pid() -> None:
    """Verify SubprocessExecutor executes top-level tool in a child process with a distinct PID."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="get_current_pid",
        description="returns pid",
        fn=get_current_pid,
        is_async=False,
        input_schema={"type": "object"},
    )

    parent_pid = os.getpid()
    child_pid = await executor.execute(spec, {})

    assert isinstance(child_pid, int)
    assert child_pid != parent_pid
    assert child_pid > 0


@pytest.mark.asyncio
async def test_subprocess_runs_async_in_separate_pid() -> None:
    """Verify SubprocessExecutor executes top-level async tool in a separate PID."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="async_get_current_pid",
        description="returns pid async",
        fn=async_get_current_pid,
        is_async=True,
        input_schema={"type": "object"},
    )

    parent_pid = os.getpid()
    child_pid = await executor.execute(spec, {})

    assert isinstance(child_pid, int)
    assert child_pid != parent_pid


@pytest.mark.asyncio
async def test_subprocess_strips_secret_env_vars() -> None:
    """Verify SubprocessExecutor scrubs secret environment variables from the child process."""
    os.environ["SECRET_KEY"] = "super_secret_value"
    os.environ["OPENROUTER_API_KEY"] = "sk-or-test-key-12345"
    os.environ["NVIDIA_API_KEY"] = "nvapi-secret-token"
    os.environ["DATABASE_URL"] = "postgresql://user:secret@localhost:5432/calbi"

    try:
        executor = SubprocessExecutor()
        spec = ToolSpec(
            name="read_env_secrets",
            description="reads env",
            fn=read_env_secrets,
            is_async=False,
            input_schema={"type": "object"},
        )

        env_in_child = await executor.execute(spec, {})

        assert env_in_child["SECRET_KEY"] is None
        assert env_in_child["OPENROUTER_API_KEY"] is None
        assert env_in_child["NVIDIA_API_KEY"] is None
        assert env_in_child["DATABASE_URL"] is None
        assert env_in_child["has_path"] is True
    finally:
        os.environ.pop("SECRET_KEY", None)
        os.environ.pop("OPENROUTER_API_KEY", None)
        os.environ.pop("NVIDIA_API_KEY", None)
        os.environ.pop("DATABASE_URL", None)


@pytest.mark.asyncio
async def test_subprocess_allows_configured_env_keys() -> None:
    """Verify SubprocessExecutor allows explicitly whitelisted environment keys."""
    os.environ["CUSTOM_ALLOWED_TOKEN"] = "allowed_val_999"
    try:
        executor = SubprocessExecutor(allowed_env_keys=frozenset({"PATH", "CUSTOM_ALLOWED_TOKEN"}))
        spec = ToolSpec(
            name="read_custom_env",
            description="read custom env",
            fn=read_custom_env,
            is_async=False,
            input_schema={"type": "object"},
        )

        result = await executor.execute(spec, {"key": "CUSTOM_ALLOWED_TOKEN"})
        assert result == "allowed_val_999"
    finally:
        os.environ.pop("CUSTOM_ALLOWED_TOKEN", None)


@pytest.mark.asyncio
async def test_subprocess_kills_infinite_sleep_on_timeout() -> None:
    """Verify SubprocessExecutor terminates hanging child process on timeout promptly."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="infinite_sync_sleep",
        description="sleeps forever",
        fn=infinite_sync_sleep,
        is_async=False,
        input_schema={"type": "object"},
    )

    start = time.perf_counter()
    with pytest.raises(TimeoutError) as exc_info:
        await executor.execute(spec, {}, timeout_s=0.2)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0
    assert "timeout of 0.2s" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_kills_async_infinite_sleep_on_timeout() -> None:
    """Verify SubprocessExecutor terminates hanging async child process promptly."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="infinite_async_sleep",
        description="sleeps forever async",
        fn=infinite_async_sleep,
        is_async=True,
        input_schema={"type": "object"},
    )

    start = time.perf_counter()
    with pytest.raises(TimeoutError):
        await executor.execute(spec, {}, timeout_s=0.2)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0


@pytest.mark.asyncio
async def test_subprocess_non_zero_exit_raises_isolation_error() -> None:
    """Verify child process exit code != 0 raises ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="exit_with_code",
        description="exits with code",
        fn=exit_with_code,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {"code": 17})
    assert "exited with non-zero code" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_tool_crash_raises_isolation_error() -> None:
    """Verify unhandled tool exception in child raises ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="failing_tool",
        description="crashes child",
        fn=failing_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {"msg": "catastrophic_crash"})
    assert "exited with non-zero code" in str(exc_info.value)
    assert "catastrophic_crash" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_corrupt_stdout_raises_isolation_error() -> None:
    """Verify non-JSON output on stdout raises ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="corrupt_stdout_tool",
        description="bad stdout",
        fn=corrupt_stdout_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {})
    assert "invalid JSON output" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_unserializable_output_raises_isolation_error() -> None:
    """Verify tool returning non-JSON-serializable data raises ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="unserializable_output_tool",
        description="unserializable output",
        fn=unserializable_output_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {})
    assert "exited with non-zero code" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_max_output_bytes_exceeded_raises_isolation_error() -> None:
    """Verify child output exceeding max_output_bytes raises ExecutionIsolationError."""
    executor = SubprocessExecutor(max_output_bytes=256)
    spec = ToolSpec(
        name="large_output_tool",
        description="large output",
        fn=large_output_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {"size": 2048})
    assert "exceeded" in str(exc_info.value) or "exited with non-zero code" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_rejects_lambda_tool() -> None:
    """Verify SubprocessExecutor rejects lambdas with ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="lambda_tool",
        description="lambda",
        fn=lambda x: x + 1,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {"x": 5})
    assert "lambda or local closure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_rejects_local_closure() -> None:
    """Verify SubprocessExecutor rejects locally scoped closure functions."""

    def local_function(a: int) -> int:
        return a * 2

    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="local_closure_tool",
        description="local closure",
        fn=local_function,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {"a": 4})
    assert "lambda or local closure" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_rejects_unserializable_inputs() -> None:
    """Verify non-JSON-serializable input arguments raise ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="sync_add",
        description="add numbers",
        fn=sync_add,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {"a": 1, "b": object()})
    assert "inputs are not JSON-serializable" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_incidental_stdout_prints_succeed() -> None:
    """Verify incidental print() output from user tool does not corrupt JSON parsing."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="tool_with_incidental_stdout_prints",
        description="prints noise and returns result",
        fn=tool_with_incidental_stdout_prints,
        is_async=False,
        input_schema={"type": "object"},
    )

    result = await executor.execute(spec, {})
    assert result == {"success": True, "answer": 42}


@pytest.mark.asyncio
async def test_subprocess_sentinel_framed_with_noise_before_and_after() -> None:
    """Verify sentinel framing works when noise is emitted before and after sentinel payload."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="tool_with_atexit_noise",
        description="noise before and after sentinel",
        fn=tool_with_atexit_noise,
        is_async=False,
        input_schema={"type": "object"},
    )

    result = await executor.execute(spec, {})
    assert result == "payload_with_noise_before_and_after"


@pytest.mark.asyncio
async def test_subprocess_streaming_output_cap_kills_child_bounded_memory() -> None:
    """Verify tool emitting 50MB is killed incrementally without buffering entire stream."""
    executor = SubprocessExecutor(max_output_bytes=64 * 1024)
    spec = ToolSpec(
        name="stream_50mb_tool",
        description="emits 50MB",
        fn=stream_50mb_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    start = time.perf_counter()
    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {})
    duration = time.perf_counter() - start

    assert "exceeded maximum allowed size" in str(exc_info.value)
    # Child was killed promptly without waiting for all 50MB to be generated/buffered
    assert duration < 5.0


@pytest.mark.asyncio
async def test_subprocess_streaming_output_cap_infinite_loop() -> None:
    """Verify tool infinitely printing to stdout is terminated by byte cap."""
    executor = SubprocessExecutor(max_output_bytes=1024)
    spec = ToolSpec(
        name="infinite_stdout_tool",
        description="infinite stdout printer",
        fn=infinite_stdout_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    start = time.perf_counter()
    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {}, timeout_s=5.0)
    duration = time.perf_counter() - start

    assert "exceeded maximum allowed size" in str(exc_info.value)
    assert duration < 2.0


@pytest.mark.asyncio
async def test_subprocess_byte_cap_exact_boundary() -> None:
    """Verify byte cap allows exact boundary output and rejects exceeding output."""
    executor_ok = SubprocessExecutor(max_output_bytes=5000)
    spec = ToolSpec(
        name="large_output_tool",
        description="large output",
        fn=large_output_tool,
        is_async=False,
        input_schema={"type": "object"},
    )
    res = await executor_ok.execute(spec, {"size": 2000})
    assert len(res) == 2000

    executor_capped = SubprocessExecutor(max_output_bytes=500)
    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor_capped.execute(spec, {"size": 2000})
    assert "exceed" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_subprocess_malformed_json_inside_sentinels_raises_isolation_error() -> None:
    """Verify malformed JSON inside valid sentinels raises ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="emit_corrupt_sentinel_json_tool",
        description="bad json in sentinel",
        fn=emit_corrupt_sentinel_json_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {})
    assert "invalid JSON output" in str(exc_info.value)


@pytest.mark.asyncio
async def test_subprocess_missing_start_or_end_sentinel_raises_isolation_error() -> None:
    """Verify missing end sentinel raises ExecutionIsolationError."""
    executor = SubprocessExecutor()
    spec = ToolSpec(
        name="emit_missing_end_sentinel_tool",
        description="missing end sentinel",
        fn=emit_missing_end_sentinel_tool,
        is_async=False,
        input_schema={"type": "object"},
    )

    with pytest.raises(ExecutionIsolationError) as exc_info:
        await executor.execute(spec, {})
    assert "missing sentinel framing" in str(exc_info.value)
