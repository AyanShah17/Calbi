"""Playbook composition tools: subplaybooks and concurrent for-each mapping.

This module allows playbooks to be registered and reused as modular tools within
other playbooks, and provides batched concurrent mapping of existing tools over
item collections with bounded concurrency and strict bounds.
"""

from __future__ import annotations

import asyncio
import contextvars
import inspect
from typing import Any

from calbi.core.registry import DuplicateToolError, ToolRegistry, ToolSpec, UnknownToolError
from calbi.core.runtime import Runtime
from calbi.core.schema import CalbiError, Playbook, Risk
from calbi.core.state import RunStatus

# Task-local call depth tracking across subplaybook invocations
_CALL_DEPTH: contextvars.ContextVar[int] = contextvars.ContextVar(
    "calbi_subplaybook_depth", default=0
)


class SubPlaybookError(CalbiError):
    """Raised when sub-playbook registration or execution fails."""


def register_subplaybook(
    registry: ToolRegistry,
    playbook: Playbook,
    runtime: Runtime,
    *,
    name: str | None = None,
    description: str | None = None,
    max_depth: int = 3,
) -> ToolSpec:
    """Register a Playbook as a reusable tool in ToolRegistry.

    Args:
        registry: The ToolRegistry in which to register the subplaybook.
        playbook: The Playbook to expose as a tool.
        runtime: The Runtime engine used to execute the playbook DAG.
        name: Optional custom tool name (defaults to playbook.name).
        description: Optional custom description (defaults to playbook.objective).
        max_depth: Maximum allowed recursion depth to prevent infinite call loops.

    Returns:
        The registered ToolSpec for the subplaybook.

    Raises:
        DuplicateToolError: If a tool with the target name is already registered.
    """
    tool_name = name or playbook.name
    if tool_name in registry._tools:
        raise DuplicateToolError(f"Tool with name '{tool_name}' is already registered.")

    tool_desc = description or playbook.objective or f"Sub-playbook {playbook.name}"

    # Calculate effective risk: maximum Risk among all constituent steps
    effective_risk = Risk.READ
    requires_approval = False
    all_idempotent = True

    for step in playbook.steps:
        try:
            step_spec = registry.get(step.tool)
            if step_spec.risk > effective_risk:
                effective_risk = step_spec.risk
            if step_spec.requires_approval:
                requires_approval = True
            if not step_spec.idempotent:
                all_idempotent = False
        except UnknownToolError:
            # Step references a tool not yet in registry (e.g. self-reference or forward-reference)
            pass

    # Synthesize JSON input schema from playbook.inputs
    properties: dict[str, Any] = {}
    required: list[str] = []
    for param_name, input_spec in playbook.inputs.items():
        prop: dict[str, Any] = {}
        if input_spec.description:
            prop["description"] = input_spec.description
        if input_spec.type != "any":
            prop["type"] = input_spec.type
        if input_spec.default is not None:
            prop["default"] = input_spec.default
        properties[param_name] = prop
        if input_spec.required and input_spec.default is None:
            required.append(param_name)

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {k: {"title": k} for k in playbook.outputs},
    }

    async def subplaybook_fn(**inputs: Any) -> dict[str, Any]:
        current_depth = _CALL_DEPTH.get()
        if current_depth >= max_depth:
            raise SubPlaybookError(
                f"Sub-playbook '{tool_name}' exceeded maximum recursion depth of {max_depth}."
            )

        token = _CALL_DEPTH.set(current_depth + 1)
        try:
            run_state = await runtime.run(playbook, inputs=inputs)
            if run_state.status != RunStatus.SUCCEEDED:
                detail = run_state.error or f"status {run_state.status.value}"
                for s_st in run_state.steps.values():
                    if s_st.error:
                        detail = f"{detail} ({s_st.id}: {s_st.error})"
                        break
                raise SubPlaybookError(
                    f"Sub-playbook '{tool_name}' failed with status "
                    f"{run_state.status.value}: {detail}"
                )
            return dict(run_state.outputs)
        except SubPlaybookError:
            raise
        except Exception as exc:
            raise SubPlaybookError(
                f"Sub-playbook '{tool_name}' encountered execution error: {exc}"
            ) from exc
        finally:
            _CALL_DEPTH.reset(token)

    spec = ToolSpec(
        name=tool_name,
        description=tool_desc.strip(),
        fn=subplaybook_fn,
        is_async=True,
        input_schema=input_schema,
        output_schema=output_schema,
        risk=effective_risk,
        requires_approval=requires_approval,
        idempotent=all_idempotent,
        tags=["subplaybook"],
        limitations="",
    )

    registry._tools[tool_name] = spec
    return spec


def register_map_tool(
    registry: ToolRegistry,
    base_tool_name: str,
    *,
    name: str | None = None,
    items_param: str = "items",
    item_arg: str,
    max_items: int = 50,
    max_concurrency: int = 8,
) -> ToolSpec:
    """Register a batched for-each tool that maps base_tool over a list of items concurrently.

    Args:
        registry: The ToolRegistry containing base_tool_name.
        base_tool_name: Name of the registered tool to map over.
        name: Optional custom tool name (defaults to 'map_{base_tool_name}').
        items_param: Name of the list parameter accepted by the mapped tool.
        item_arg: Argument name on base_tool to which each element is bound.
        max_items: Maximum items allowed per invocation (raises CalbiError if exceeded).
        max_concurrency: Maximum number of concurrent base tool executions via Semaphore.

    Returns:
        The registered ToolSpec for the concurrent mapping tool.

    Raises:
        UnknownToolError: If base_tool_name is not registered.
        DuplicateToolError: If target tool name is already registered.
        ValueError: If max_items or max_concurrency is less than 1.
    """
    if max_items < 1:
        raise ValueError("max_items must be at least 1.")
    if max_concurrency < 1:
        raise ValueError("max_concurrency must be at least 1.")

    base_spec = registry.get(base_tool_name)
    tool_name = name or f"map_{base_tool_name}"
    if tool_name in registry._tools:
        raise DuplicateToolError(f"Tool with name '{tool_name}' is already registered.")

    tool_desc = (
        f"Batch concurrent map of '{base_tool_name}' over '{items_param}' (max {max_items} items)."
    )

    # Synthesize input schema: take base tool properties, remove item_arg, add items_param array
    properties = dict(base_spec.input_schema.get("properties", {}))
    properties.pop(item_arg, None)
    properties[items_param] = {
        "type": "array",
        "description": f"List of items to map over '{item_arg}' (max {max_items}).",
    }

    base_required = [r for r in base_spec.input_schema.get("required", []) if r != item_arg]
    required = [items_param] + base_required

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    output_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "results": {"type": "array", "description": "Mapped execution results in order."},
            "count": {"type": "integer", "description": "Number of items processed."},
        },
        "required": ["results", "count"],
    }

    sem = asyncio.Semaphore(max_concurrency)

    async def map_fn(**kwargs: Any) -> dict[str, Any]:
        if items_param not in kwargs:
            raise CalbiError(f"Missing required parameter '{items_param}'.")
        raw_items = kwargs.pop(items_param)
        if not isinstance(raw_items, (list, tuple)):
            raise CalbiError(
                f"Parameter '{items_param}' must be a list, got {type(raw_items).__name__}."
            )

        if len(raw_items) > max_items:
            raise CalbiError(
                f"Items count ({len(raw_items)}) exceeds maximum allowed ({max_items})."
            )

        async def _execute_single(item: Any) -> Any:
            call_kwargs = {**kwargs, item_arg: item}
            async with sem:
                if base_spec.is_async or inspect.iscoroutinefunction(base_spec.fn):
                    res = await base_spec.fn(**call_kwargs)
                else:
                    res = await asyncio.to_thread(base_spec.fn, **call_kwargs)
                if inspect.isawaitable(res):
                    res = await res
                return res

        results = await asyncio.gather(*(_execute_single(item) for item in raw_items))
        return {"results": list(results), "count": len(results)}

    spec = ToolSpec(
        name=tool_name,
        description=tool_desc,
        fn=map_fn,
        is_async=True,
        input_schema=input_schema,
        output_schema=output_schema,
        risk=base_spec.risk,
        requires_approval=base_spec.requires_approval,
        idempotent=base_spec.idempotent,
        tags=list(base_spec.tags) + ["map", "batch"],
        limitations=base_spec.limitations,
    )

    registry._tools[tool_name] = spec
    return spec
