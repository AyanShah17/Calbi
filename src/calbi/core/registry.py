from __future__ import annotations

import builtins
import inspect
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast, get_type_hints, overload

from pydantic import create_model

from calbi.core.schema import CalbiError, Risk

F = TypeVar("F", bound=Callable[..., Any])


class UnknownToolError(CalbiError):
    """Raised when a requested tool is not found in the registry."""


class DuplicateToolError(CalbiError):
    """Raised when attempting to register a tool with an already existing name."""


@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[..., Any]
    is_async: bool
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    risk: Risk = Risk.READ
    requires_approval: bool = False
    idempotent: bool = False
    tags: list[str] = field(default_factory=list)
    limitations: str = ""


def _extract_input_schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """Derive JSON-schema from function parameter annotations using pydantic."""
    sig = inspect.signature(fn)
    hints = get_type_hints(fn) if hasattr(fn, "__annotations__") else {}
    fields: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        param_type = hints.get(param_name, Any)
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (param_type, ...)
        else:
            fields[param_name] = (param_type, param.default)

    if not fields:
        return {"type": "object", "properties": {}, "required": []}

    try:
        model = create_model(f"{fn.__name__}Input", **fields)
        schema = model.model_json_schema()
        schema.pop("title", None)
        return cast(dict[str, Any], schema)
    except Exception:
        properties: dict[str, Any] = {}
        required: list[str] = []
        for p_name, param in sig.parameters.items():
            if p_name in ("self", "cls"):
                continue
            properties[p_name] = {"title": p_name}
            if param.default is inspect.Parameter.empty:
                required.append(p_name)
        return {"type": "object", "properties": properties, "required": required}


class ToolRegistry:
    """Registry for capabilities exposed to Calbi."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    @overload
    def register(
        self,
        fn: F,
        *,
        name: str | None = None,
        description: str | None = None,
        risk: Risk = Risk.READ,
        requires_approval: bool = False,
        idempotent: bool = False,
        tags: Sequence[str] = (),
        limitations: str = "",
    ) -> F: ...

    @overload
    def register(
        self,
        fn: None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        risk: Risk = Risk.READ,
        requires_approval: bool = False,
        idempotent: bool = False,
        tags: Sequence[str] = (),
        limitations: str = "",
    ) -> Callable[[F], F]: ...

    def register(
        self,
        fn: F | None = None,
        *,
        name: str | None = None,
        description: str | None = None,
        risk: Risk = Risk.READ,
        requires_approval: bool = False,
        idempotent: bool = False,
        tags: Sequence[str] = (),
        limitations: str = "",
    ) -> F | Callable[[F], F]:
        """Register a function as a tool. Can be used as a decorator or direct call."""

        def decorator(f: F) -> F:
            tool_name = name or f.__name__
            if tool_name in self._tools:
                raise DuplicateToolError(f"Tool with name '{tool_name}' is already registered.")

            tool_desc = description or inspect.getdoc(f) or ""
            is_async = inspect.iscoroutinefunction(f)
            input_schema = _extract_input_schema(f)

            spec = ToolSpec(
                name=tool_name,
                description=tool_desc.strip(),
                fn=f,
                is_async=is_async,
                input_schema=input_schema,
                risk=risk,
                requires_approval=requires_approval,
                idempotent=idempotent,
                tags=list(tags),
                limitations=limitations,
            )
            self._tools[tool_name] = spec
            return f

        if fn is not None:
            return decorator(fn)
        return decorator

    # Alias tool to register for convenience (@registry.tool(...))
    tool = register

    def register_tools(self, fns: Iterable[Callable[..., Any]]) -> None:
        """Register multiple callable tools."""
        for fn in fns:
            self.register(fn)

    def get(self, name: str) -> ToolSpec:
        """Get a ToolSpec by tool name, raising UnknownToolError if missing."""
        if name not in self._tools:
            raise UnknownToolError(f"Tool '{name}' is not registered.")
        return self._tools[name]

    def list(self) -> builtins.list[ToolSpec]:
        """Return all registered ToolSpecs."""
        return list(self._tools.values())

    def describe(self) -> builtins.list[dict[str, Any]]:
        """Return LLM-safe manifest of registered tools. Never includes callables."""
        manifest: builtins.list[dict[str, Any]] = []
        for spec in self._tools.values():
            manifest.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "input_schema": spec.input_schema,
                    "output_schema": spec.output_schema,
                    "risk": spec.risk.value,
                    "requires_approval": spec.requires_approval,
                    "idempotent": spec.idempotent,
                    "limitations": spec.limitations,
                }
            )
        return manifest
