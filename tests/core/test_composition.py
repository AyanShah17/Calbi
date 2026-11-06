"""Tests for Calbi core composition: subplaybooks and concurrent map tools."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from calbi.core.composition import (
    _CALL_DEPTH,
    SubPlaybookError,
    register_map_tool,
    register_subplaybook,
)
from calbi.core.executor import InProcessExecutor, StepExecutor
from calbi.core.policy import ApprovalRequest, Autonomy, Policy
from calbi.core.registry import DuplicateToolError, ToolRegistry, ToolSpec, UnknownToolError
from calbi.core.runtime import Runtime
from calbi.core.schema import CalbiError, InputSpec, Playbook, Risk, Step
from calbi.core.state import RunStatus, StepStatus

# ---------------------------------------------------------------------------
# Subplaybook Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subplaybook_output_passing() -> None:
    """Subplaybook registered as tool in parent playbook passes output downstream."""
    reg = ToolRegistry()

    @reg.tool(name="multiply_step")
    def multiply(x: int, y: int) -> dict[str, int]:
        return {"product": x * y}

    @reg.tool(name="format_step")
    def format_result(val: int) -> dict[str, str]:
        return {"formatted": f"Result is {val}"}

    # Define subplaybook that multiplies two numbers and outputs product
    sub_pb = Playbook(
        name="calc_product",
        objective="Calculate product of two inputs",
        inputs={
            "a": InputSpec(type="integer", description="First factor"),
            "b": InputSpec(type="integer", description="Second factor"),
        },
        steps=[
            Step(
                id="mult",
                tool="multiply_step",
                inputs={"x": "{{ inputs.a }}", "y": "{{ inputs.b }}"},
            )
        ],
        outputs={"total": "{{ steps.mult.output.product }}"},
    )

    rt = Runtime(reg, Policy())
    spec = register_subplaybook(reg, sub_pb, rt, name="calc_tool")

    assert spec.name == "calc_tool"
    assert spec.is_async is True
    assert spec.risk == Risk.READ
    assert "calc_tool" in reg._tools

    # Define parent playbook invoking calc_tool and passing its output to format_step
    parent_pb = Playbook(
        name="parent_calc_pipeline",
        inputs={"num1": InputSpec(type="integer"), "num2": InputSpec(type="integer")},
        steps=[
            Step(
                id="sub",
                tool="calc_tool",
                inputs={"a": "{{ inputs.num1 }}", "b": "{{ inputs.num2 }}"},
            ),
            Step(
                id="formatter",
                tool="format_step",
                depends_on=["sub"],
                inputs={"val": "{{ steps.sub.output.total }}"},
            ),
        ],
        outputs={"final_output": "{{ steps.formatter.output.formatted }}"},
    )

    state = await rt.run(parent_pb, inputs={"num1": 6, "num2": 7})

    assert state.status == RunStatus.SUCCEEDED
    assert state.steps["sub"].status == StepStatus.SUCCEEDED
    assert state.steps["sub"].output == {"total": 42}
    assert state.steps["formatter"].status == StepStatus.SUCCEEDED
    assert state.steps["formatter"].output == {"formatted": "Result is 42"}
    assert state.outputs["final_output"] == "Result is 42"


@pytest.mark.asyncio
async def test_subplaybook_risk_and_approval_inheritance() -> None:
    """Subplaybook inherits the maximum risk and requires_approval from constituent steps."""
    reg = ToolRegistry()

    @reg.tool(name="read_data", risk=Risk.READ, idempotent=True)
    def read_data() -> dict[str, str]:
        return {"data": "info"}

    @reg.tool(name="write_audit", risk=Risk.WRITE, idempotent=True)
    def write_audit(msg: str) -> dict[str, bool]:
        return {"logged": True}

    @reg.tool(name="drop_table", risk=Risk.DESTRUCTIVE, requires_approval=True, idempotent=False)
    def drop_table(table: str) -> dict[str, str]:
        return {"dropped": table}

    # 1. Playbook with only READ steps
    pb_read = Playbook(
        name="read_only_pb",
        steps=[Step(id="s1", tool="read_data")],
        outputs={"res": "{{ steps.s1.output.data }}"},
    )
    rt = Runtime(reg, Policy())
    spec_read = register_subplaybook(reg, pb_read, rt)
    assert spec_read.risk == Risk.READ
    assert spec_read.requires_approval is False
    assert spec_read.idempotent is True

    # 2. Playbook with READ and WRITE steps
    pb_write = Playbook(
        name="read_write_pb",
        steps=[
            Step(id="s1", tool="read_data"),
            Step(id="s2", tool="write_audit", inputs={"msg": "test"}),
        ],
        outputs={"res": "{{ steps.s1.output.data }}"},
    )
    spec_write = register_subplaybook(reg, pb_write, rt)
    assert spec_write.risk == Risk.WRITE
    assert spec_write.requires_approval is False
    assert spec_write.idempotent is True

    # 3. Playbook containing DESTRUCTIVE step with requires_approval=True
    pb_destructive = Playbook(
        name="destructive_pb",
        steps=[
            Step(id="s1", tool="read_data"),
            Step(id="s2", tool="drop_table", inputs={"table": "temp"}),
        ],
        outputs={"res": "{{ steps.s1.output.data }}"},
    )
    spec_destructive = register_subplaybook(reg, pb_destructive, rt)
    assert spec_destructive.risk == Risk.DESTRUCTIVE
    assert spec_destructive.requires_approval is True
    assert spec_destructive.idempotent is False


@pytest.mark.asyncio
async def test_subplaybook_max_depth_recursion_guard() -> None:
    """Recursive subplaybook calling itself exceeds max_depth and raises SubPlaybookError."""
    reg = ToolRegistry()

    @reg.tool(name="recurse_step")
    async def recurse_step(count: int) -> dict[str, Any]:
        res = await reg.get("recursive_tool").fn(count=count + 1)
        return dict(res) if isinstance(res, dict) else {"count": count}

    recursive_pb = Playbook(
        name="recursive_pb",
        inputs={"count": InputSpec(type="integer")},
        steps=[
            Step(
                id="rec",
                tool="recurse_step",
                inputs={"count": "{{ inputs.count }}"},
            )
        ],
        outputs={"res": "{{ steps.rec.output.res }}"},
    )

    rt = Runtime(reg, Policy())
    register_subplaybook(reg, recursive_pb, rt, name="recursive_tool", max_depth=3)

    assert _CALL_DEPTH.get() == 0

    # Invoking the recursive subplaybook tool directly or through runtime should exceed max_depth
    with pytest.raises(SubPlaybookError) as exc_info:
        await reg.get("recursive_tool").fn(count=1)

    assert "exceeded maximum recursion depth of 3" in str(exc_info.value)
    # ContextVar call depth must be safely restored to 0
    assert _CALL_DEPTH.get() == 0


@pytest.mark.asyncio
async def test_subplaybook_mutual_recursion_guard() -> None:
    """Mutual recursion between two subplaybooks exceeds max_depth and raises SubPlaybookError."""
    reg = ToolRegistry()

    # Tool B invokes tool_a, and Playbook A calls tool_b
    @reg.tool(name="tool_b")
    async def tool_b() -> dict[str, Any]:
        res = await reg.get("tool_a").fn()
        return dict(res) if isinstance(res, dict) else {}

    pb_a = Playbook(
        name="pb_a",
        steps=[Step(id="call_b", tool="tool_b")],
        outputs={"res": "{{ steps.call_b.output.res }}"},
    )

    rt = Runtime(reg, Policy())
    register_subplaybook(reg, pb_a, rt, name="tool_a", max_depth=3)

    assert _CALL_DEPTH.get() == 0

    with pytest.raises(SubPlaybookError) as exc_info:
        await reg.get("tool_a").fn()

    assert "exceeded maximum recursion depth of 3" in str(exc_info.value)
    assert _CALL_DEPTH.get() == 0


@pytest.mark.asyncio
async def test_subplaybook_execution_failure_raises_subplaybook_error() -> None:
    """When subplaybook runtime execution fails, SubPlaybookError is raised with error details."""
    reg = ToolRegistry()

    @reg.tool(name="failing_tool")
    def fail() -> None:
        raise RuntimeError("Database connection timed out.")

    failing_pb = Playbook(
        name="failing_pb",
        steps=[Step(id="fail_step", tool="failing_tool")],
    )

    rt = Runtime(reg, Policy())
    spec = register_subplaybook(reg, failing_pb, rt)

    with pytest.raises(SubPlaybookError) as exc_info:
        await spec.fn()

    assert "failed with status" in str(exc_info.value)
    assert "Database connection timed out" in str(exc_info.value)
    assert _CALL_DEPTH.get() == 0


def test_subplaybook_duplicate_name_rejected() -> None:
    """Registering a subplaybook with an already existing tool name raises DuplicateToolError."""
    reg = ToolRegistry()

    @reg.tool(name="existing_tool")
    def existing() -> str:
        return "exists"

    pb = Playbook(
        name="existing_tool",
        steps=[Step(id="s1", tool="existing_tool")],
    )

    rt = Runtime(reg, Policy())
    with pytest.raises(DuplicateToolError):
        register_subplaybook(reg, pb, rt)


def test_subplaybook_schema_derivation() -> None:
    """Input and output JSON schemas are properly synthesized from playbook specs."""
    reg = ToolRegistry()

    @reg.tool(name="echo")
    def echo(val: str) -> dict[str, str]:
        return {"val": val}

    pb = Playbook(
        name="schema_test_pb",
        objective="Test schema synthesis",
        inputs={
            "req_str": InputSpec(type="string", required=True, description="Required text"),
            "opt_int": InputSpec(
                type="integer", required=False, default=10, description="Optional int"
            ),
        },
        steps=[Step(id="s1", tool="echo", inputs={"val": "{{ inputs.req_str }}"})],
        outputs={"out1": "{{ steps.s1.output.val }}", "out2": "fixed_val"},
    )

    rt = Runtime(reg, Policy())
    spec = register_subplaybook(reg, pb, rt)

    # Input schema checks
    assert spec.input_schema["type"] == "object"
    assert "req_str" in spec.input_schema["properties"]
    assert spec.input_schema["properties"]["req_str"]["type"] == "string"
    assert spec.input_schema["properties"]["req_str"]["description"] == "Required text"
    assert "req_str" in spec.input_schema["required"]

    assert "opt_int" in spec.input_schema["properties"]
    assert spec.input_schema["properties"]["opt_int"]["type"] == "integer"
    assert spec.input_schema["properties"]["opt_int"]["default"] == 10
    assert "opt_int" not in spec.input_schema["required"]

    # Output schema checks
    assert spec.output_schema is not None
    assert spec.output_schema["type"] == "object"
    assert "out1" in spec.output_schema["properties"]
    assert "out2" in spec.output_schema["properties"]


def test_subplaybook_fails_closed_on_unknown_tool() -> None:
    """Registering a subplaybook with an unknown tool fails closed with SubPlaybookError."""
    reg = ToolRegistry()
    pb = Playbook(
        name="unknown_tool_pb",
        steps=[Step(id="s1", tool="non_existent_tool")],
    )
    rt = Runtime(reg, Policy())
    with pytest.raises(SubPlaybookError) as exc_info:
        register_subplaybook(reg, pb, rt)
    assert "references unknown tool 'non_existent_tool'" in str(exc_info.value)


def test_subplaybook_fails_closed_on_unregistered_self_reference() -> None:
    """Registering a self-referential subplaybook where tool is not yet registered
    raises SubPlaybookError.
    """
    reg = ToolRegistry()
    pb = Playbook(
        name="self_ref_pb",
        steps=[Step(id="s1", tool="self_ref_pb")],
    )
    rt = Runtime(reg, Policy())
    with pytest.raises(SubPlaybookError) as exc_info:
        register_subplaybook(reg, pb, rt)
    assert "references unknown tool 'self_ref_pb'" in str(exc_info.value)


def test_subplaybook_fails_closed_prevents_risk_laundering() -> None:
    """A subplaybook cannot conceal an unknown destructive step behind a registered read step."""
    reg = ToolRegistry()

    @reg.tool(name="benign_read", risk=Risk.READ)
    def benign_read() -> str:
        return "safe"

    pb = Playbook(
        name="laundering_attempt_pb",
        steps=[
            Step(id="s1", tool="benign_read"),
            Step(id="s2", tool="unregistered_destructive_tool"),
        ],
    )
    rt = Runtime(reg, Policy())
    with pytest.raises(SubPlaybookError) as exc_info:
        register_subplaybook(reg, pb, rt)
    assert "references unknown tool 'unregistered_destructive_tool'" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Map Tool Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_register_map_tool_concurrent_execution() -> None:
    """register_map_tool processes items concurrently with order and concurrency bounds."""
    reg = ToolRegistry()

    active_executions = 0
    max_observed_concurrency = 0
    lock = asyncio.Lock()

    @reg.tool(name="fetch_item")
    async def fetch_item(item_id: int) -> dict[str, Any]:
        nonlocal active_executions, max_observed_concurrency
        async with lock:
            active_executions += 1
            if active_executions > max_observed_concurrency:
                max_observed_concurrency = active_executions

        await asyncio.sleep(0.02)

        async with lock:
            active_executions -= 1

        return {"id": item_id, "square": item_id * item_id}

    # Register map tool with max_concurrency=3
    map_spec = register_map_tool(
        reg,
        base_tool_name="fetch_item",
        name="map_fetch",
        items_param="item_ids",
        item_arg="item_id",
        max_items=20,
        max_concurrency=3,
    )

    assert map_spec.name == "map_fetch"
    assert "map_fetch" in reg._tools

    input_items = [1, 2, 3, 4, 5, 6, 7, 8]
    result = await map_spec.fn(item_ids=input_items)

    assert result["count"] == 8
    # Assert exact order is preserved
    expected_results = [{"id": i, "square": i * i} for i in input_items]
    assert result["results"] == expected_results

    # Assert concurrency was bounded by max_concurrency=3
    assert max_observed_concurrency <= 3
    assert max_observed_concurrency >= 2


@pytest.mark.asyncio
async def test_register_map_tool_max_items_enforced() -> None:
    """Mapping over items exceeding max_items raises CalbiError."""
    reg = ToolRegistry()

    @reg.tool(name="simple_echo")
    def simple_echo(val: str) -> str:
        return val

    map_spec = register_map_tool(
        reg,
        base_tool_name="simple_echo",
        item_arg="val",
        max_items=5,
    )

    # Valid invocation within max_items
    valid_res = await map_spec.fn(items=["a", "b", "c"])
    assert valid_res["count"] == 3
    assert valid_res["results"] == ["a", "b", "c"]

    # Exceeding max_items raises CalbiError
    with pytest.raises(CalbiError) as exc_info:
        await map_spec.fn(items=["a", "b", "c", "d", "e", "f"])

    assert "exceeds maximum allowed (5)" in str(exc_info.value)


@pytest.mark.asyncio
async def test_register_map_tool_inherits_risk_and_approval() -> None:
    """Map tool inherits risk, requires_approval, and idempotent metadata from base tool."""
    reg = ToolRegistry()

    @reg.tool(
        name="delete_account",
        risk=Risk.DESTRUCTIVE,
        requires_approval=True,
        idempotent=False,
        tags=["admin"],
        limitations="Cannot delete root",
    )
    def delete_account(acc_id: str) -> dict[str, str]:
        return {"deleted": acc_id}

    map_spec = register_map_tool(
        reg,
        base_tool_name="delete_account",
        item_arg="acc_id",
        items_param="acc_ids",
    )

    assert map_spec.risk == Risk.DESTRUCTIVE
    assert map_spec.requires_approval is True
    assert map_spec.idempotent is False
    assert "map" in map_spec.tags
    assert "batch" in map_spec.tags
    assert "admin" in map_spec.tags
    assert map_spec.limitations == "Cannot delete root"


@pytest.mark.asyncio
async def test_register_map_tool_sync_base_and_static_kwargs() -> None:
    """Synchronous base tool runs via asyncio.to_thread and respects additional static kwargs."""
    reg = ToolRegistry()

    @reg.tool(name="prefix_transform")
    def prefix_transform(text: str, prefix: str = "item_") -> str:
        return f"{prefix}{text.upper()}"

    map_spec = register_map_tool(
        reg,
        base_tool_name="prefix_transform",
        item_arg="text",
        items_param="texts",
    )

    # Call with static argument prefix="custom:"
    res = await map_spec.fn(texts=["alpha", "beta", "gamma"], prefix="custom:")

    assert res["count"] == 3
    assert res["results"] == ["custom:ALPHA", "custom:BETA", "custom:GAMMA"]


@pytest.mark.asyncio
async def test_register_map_tool_in_playbook_dag() -> None:
    """Map tool can be used as a step inside a Playbook DAG."""
    reg = ToolRegistry()

    @reg.tool(name="cube")
    def cube(n: int) -> int:
        return n * n * n

    register_map_tool(reg, base_tool_name="cube", name="map_cube", item_arg="n", items_param="nums")

    pb = Playbook(
        name="batch_pipeline",
        inputs={"input_numbers": InputSpec(type="array")},
        steps=[
            Step(
                id="batch_step",
                tool="map_cube",
                inputs={"nums": "{{ inputs.input_numbers }}"},
            )
        ],
        outputs={"cube_results": "{{ steps.batch_step.output.results }}"},
    )

    rt = Runtime(reg, Policy())
    state = await rt.run(pb, inputs={"input_numbers": [2, 3, 4]})

    assert state.status == RunStatus.SUCCEEDED
    assert state.steps["batch_step"].output == {"results": [8, 27, 64], "count": 3}
    assert state.outputs["cube_results"] == [8, 27, 64]


@pytest.mark.asyncio
async def test_map_tool_validation_errors() -> None:
    """Map tool checks missing items_param, invalid parameter type, and invalid init bounds."""
    reg = ToolRegistry()

    @reg.tool(name="dummy")
    def dummy(x: int) -> int:
        return x

    # Invalid max_items or max_concurrency
    with pytest.raises(ValueError):
        register_map_tool(reg, "dummy", item_arg="x", max_items=0)

    with pytest.raises(ValueError):
        register_map_tool(reg, "dummy", item_arg="x", max_concurrency=0)

    # Unknown base tool
    with pytest.raises(UnknownToolError):
        register_map_tool(reg, "non_existent_tool", item_arg="x")

    map_spec = register_map_tool(reg, "dummy", item_arg="x")

    # Duplicate name
    with pytest.raises(DuplicateToolError):
        register_map_tool(reg, "dummy", item_arg="x")

    # Missing items param
    with pytest.raises(CalbiError) as exc_missing:
        await map_spec.fn()
    assert "Missing required parameter 'items'" in str(exc_missing.value)

    # Non-list items param
    with pytest.raises(CalbiError) as exc_type:
        await map_spec.fn(items="not-a-list")
    assert "must be a list" in str(exc_type.value)

    # Empty list is supported
    empty_res = await map_spec.fn(items=[])
    assert empty_res == {"results": [], "count": 0}


class RecordingExecutor(StepExecutor):
    """Test helper StepExecutor that records calls and forwards to InProcessExecutor."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []

    async def execute(
        self,
        spec: ToolSpec,
        inputs: dict[str, Any],
        *,
        timeout_s: float | None = None,
    ) -> Any:
        self.calls.append((spec.name, dict(inputs), timeout_s))
        return await InProcessExecutor().execute(spec, inputs, timeout_s=timeout_s)


@pytest.mark.asyncio
async def test_register_map_tool_custom_executor() -> None:
    """register_map_tool uses the provided StepExecutor for each item."""
    reg = ToolRegistry()

    @reg.tool(name="double_num")
    def double_num(n: int) -> int:
        return n * 2

    rec_executor = RecordingExecutor()
    map_spec = register_map_tool(
        reg,
        base_tool_name="double_num",
        item_arg="n",
        executor=rec_executor,
        per_item_timeout_s=5.0,
    )

    res = await map_spec.fn(items=[10, 20, 30])
    assert res == {"results": [20, 40, 60], "count": 3}

    assert len(rec_executor.calls) == 3
    assert rec_executor.calls[0] == ("double_num", {"n": 10}, 5.0)
    assert rec_executor.calls[1] == ("double_num", {"n": 20}, 5.0)
    assert rec_executor.calls[2] == ("double_num", {"n": 30}, 5.0)


@pytest.mark.asyncio
async def test_register_map_tool_per_item_timeout() -> None:
    """register_map_tool enforces per_item_timeout_s on mapped item execution."""
    reg = ToolRegistry()

    @reg.tool(name="slow_tool")
    async def slow_tool(delay: float) -> str:
        await asyncio.sleep(delay)
        return f"waited {delay}s"

    map_spec = register_map_tool(
        reg,
        base_tool_name="slow_tool",
        item_arg="delay",
        per_item_timeout_s=0.03,
    )

    # Item finishes faster than per_item_timeout_s
    fast_res = await map_spec.fn(items=[0.005, 0.01])
    assert fast_res["count"] == 2

    # Item takes longer than per_item_timeout_s
    with pytest.raises(TimeoutError) as exc_info:
        await map_spec.fn(items=[0.1])
    assert "exceeded timeout" in str(exc_info.value)


def test_register_map_tool_invalid_timeout() -> None:
    """Non-positive per_item_timeout_s raises ValueError."""
    reg = ToolRegistry()

    @reg.tool(name="ping")
    def ping(msg: str) -> str:
        return msg

    with pytest.raises(ValueError) as exc1:
        register_map_tool(reg, "ping", item_arg="msg", per_item_timeout_s=0.0)
    assert "greater than 0" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        register_map_tool(reg, "ping", item_arg="msg", per_item_timeout_s=-2.5)
    assert "greater than 0" in str(exc2.value)


@pytest.mark.asyncio
async def test_register_map_tool_policy_refused_destructive() -> None:
    """A DESTRUCTIVE base tool mapped under Autonomy.AUTONOMOUS is refused without approver."""
    reg = ToolRegistry()

    @reg.tool(name="wipe_db", risk=Risk.DESTRUCTIVE)
    def wipe_db(db_name: str) -> str:
        return f"wiped {db_name}"

    autonomous_policy = Policy(autonomy=Autonomy.AUTONOMOUS)
    map_spec = register_map_tool(
        reg,
        base_tool_name="wipe_db",
        item_arg="db_name",
        policy=autonomous_policy,
    )

    with pytest.raises(CalbiError) as exc_info:
        await map_spec.fn(items=["db_dev", "db_staging"])
    assert "requires approval by policy, but no approver configured" in str(exc_info.value)


@pytest.mark.asyncio
async def test_register_map_tool_policy_denied() -> None:
    """Tool denied by policy is rejected when map tool executes."""
    reg = ToolRegistry()

    @reg.tool(name="blocked_tool")
    def blocked(x: int) -> int:
        return x

    policy = Policy(denied_tools={"blocked_tool"})
    map_spec = register_map_tool(
        reg,
        base_tool_name="blocked_tool",
        item_arg="x",
        policy=policy,
    )

    with pytest.raises(CalbiError) as exc_info:
        await map_spec.fn(items=[1, 2])
    assert "denied by policy" in str(exc_info.value)


@pytest.mark.asyncio
async def test_register_map_tool_policy_approver_flow() -> None:
    """Policy approver callback is invoked for items requiring approval."""
    reg = ToolRegistry()

    @reg.tool(name="deploy_service", risk=Risk.DEPLOY, requires_approval=True)
    def deploy_service(service: str) -> str:
        return f"deployed {service}"

    approved_items: list[str] = []

    async def approver_callback(req: ApprovalRequest) -> bool:
        srv = str(req.inputs.get("service"))
        if srv == "malicious_service":
            return False
        approved_items.append(srv)
        return True

    policy = Policy(autonomy=Autonomy.CONTROLLED, approver=approver_callback)
    map_spec = register_map_tool(
        reg,
        base_tool_name="deploy_service",
        item_arg="service",
        policy=policy,
    )

    # 1. Approval granted for all items
    ok_res = await map_spec.fn(items=["frontend", "backend"])
    assert ok_res["count"] == 2
    assert "frontend" in approved_items
    assert "backend" in approved_items

    # 2. Approval rejected for malicious_service
    with pytest.raises(CalbiError) as exc_info:
        await map_spec.fn(items=["malicious_service"])
    assert "Approval denied" in str(exc_info.value)
