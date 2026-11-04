from typing import Any

import pytest

from calbi.core.registry import DuplicateToolError, ToolRegistry, ToolSpec, UnknownToolError
from calbi.core.schema import CalbiError, Risk


def test_registry_errors_subclass_calbi_error() -> None:
    assert issubclass(DuplicateToolError, CalbiError)
    assert issubclass(UnknownToolError, CalbiError)


def test_register_decorator_and_call() -> None:
    reg = ToolRegistry()

    @reg.tool(name="math.add", description="Add two numbers", risk=Risk.READ)
    def add(a: int, b: int = 0) -> int:
        return a + b

    def multiply(x: float, y: float) -> float:
        """Multiply numbers."""
        return x * y

    reg.register(multiply, name="math.mul", risk=Risk.READ)

    spec_add = reg.get("math.add")
    assert isinstance(spec_add, ToolSpec)
    assert spec_add.name == "math.add"
    assert spec_add.is_async is False
    assert spec_add.input_schema["properties"]["a"]["type"] == "integer"

    spec_mul = reg.get("math.mul")
    assert spec_mul.name == "math.mul"
    assert spec_mul.description == "Multiply numbers."


def test_register_async_tool() -> None:
    reg = ToolRegistry()

    @reg.tool()
    async def fetch_data(url: str) -> dict[str, Any]:
        return {"data": url}

    spec = reg.get("fetch_data")
    assert spec.is_async is True
    assert spec.name == "fetch_data"


def test_register_tools_iterable() -> None:
    reg = ToolRegistry()

    def t1() -> None:
        pass

    def t2() -> None:
        pass

    reg.register_tools([t1, t2])
    assert len(reg.list()) == 2
    assert reg.get("t1").name == "t1"
    assert reg.get("t2").name == "t2"


def test_duplicate_tool_error() -> None:
    reg = ToolRegistry()

    def t1() -> None:
        pass

    reg.register(t1, name="same_name")
    with pytest.raises(DuplicateToolError):
        reg.register(t1, name="same_name")


def test_unknown_tool_error() -> None:
    reg = ToolRegistry()
    with pytest.raises(UnknownToolError):
        reg.get("non_existent")


def test_describe_is_llm_safe() -> None:
    reg = ToolRegistry()

    @reg.tool(
        name="safe_tool", description="A safe tool", risk=Risk.WRITE, limitations="No empty strings"
    )
    def do_work(text: str) -> str:
        return text.upper()

    manifest = reg.describe()
    assert isinstance(manifest, list)
    assert len(manifest) == 1
    item = manifest[0]
    assert item["name"] == "safe_tool"
    assert item["description"] == "A safe tool"
    assert item["risk"] == "write"
    assert item["limitations"] == "No empty strings"
    assert "fn" not in item

    # Ensure JSON serializable
    import json

    json_str = json.dumps(manifest)
    assert "safe_tool" in json_str
