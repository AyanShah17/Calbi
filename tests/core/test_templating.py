from typing import Any

import pytest

from calbi.core.schema import CalbiError
from calbi.core.templating import TemplateError, evaluate, referenced_steps, render


def test_template_error_subclass_calbi_error() -> None:
    assert issubclass(TemplateError, CalbiError)


def test_typed_passthrough() -> None:
    ctx = {"inputs": {"n": 42, "flag": True, "data": {"key": "val"}}}
    assert render("{{ inputs.n }}", ctx) == 42
    assert render("{{ inputs.flag }}", ctx) is True
    assert render("{{ inputs.data }}", ctx) == {"key": "val"}


def test_embedded_string_interpolation() -> None:
    ctx = {"inputs": {"dataset": "sales", "ext": "csv"}}
    assert render("data/{{ inputs.dataset }}.{{ inputs.ext }}", ctx) == "data/sales.csv"


def test_nested_dict_and_list_rendering() -> None:
    ctx = {
        "inputs": {"file": "in.txt", "count": 5},
        "steps": {"step1": {"output": {"done": True}}},
    }
    raw = {
        "target": "{{ inputs.file }}",
        "nested": [
            "{{ inputs.count }}",
            {"status": "{{ steps.step1.output.done }}"},
            "plain_string",
        ],
        "num": 123,
    }
    rendered = render(raw, ctx)
    assert rendered == {
        "target": "in.txt",
        "nested": [
            5,
            {"status": True},
            "plain_string",
        ],
        "num": 123,
    }


def test_evaluate_expressions_and_operators() -> None:
    ctx = {
        "steps": {
            "val": {"output": {"valid": True, "count": 10, "items": [1, 2, 3]}},
        },
        "inputs": {"threshold": 5},
    }
    assert evaluate("steps.val.output.valid", ctx) is True
    assert evaluate("steps.val.output.count > inputs.threshold", ctx) is True
    assert evaluate("steps.val.output.count == 10 and steps.val.output.valid", ctx) is True
    assert evaluate("not (steps.val.output.count < 5)", ctx) is True
    assert evaluate("len(steps.val.output.items) == 3", ctx) is True
    assert evaluate("1 in steps.val.output.items", ctx) is True
    assert evaluate("steps.val.output.count + 2 * 3", ctx) == 16


def test_missing_key_names_path() -> None:
    ctx: dict[str, Any] = {"inputs": {"data": {}}}
    with pytest.raises(TemplateError) as exc_info:
        evaluate("inputs.data.missing", ctx)
    assert "missing" in str(exc_info.value)


def test_referenced_steps() -> None:
    obj = {
        "a": "{{ steps.step_one.output.path }}",
        "b": "foo {{ steps.step_two.output.valid }} bar",
        "c": ["{{ steps.step_three.x + steps.step_one.y }}"],
        "d": "no step here",
    }
    refs = referenced_steps(obj)
    assert refs == {"step_one", "step_two", "step_three"}


HOSTILE_STRINGS = [
    "__import__('os').system('ls')",
    "().__class__.__base__.__subclasses__()",
    "open('/etc/passwd').read()",
    "exec('import sys')",
    "eval('2+2')",
    "(lambda: 1)()",
    "[x for x in ().__class__.__base__.__subclasses__()]",
    "{x: 1 for x in [1, 2]}",
    "globals()",
    "locals()",
    "breakpoint()",
    "compile('1', '', 'eval')",
    "getattr(inputs, '__class__')",
    "setattr(inputs, 'x', 1)",
    "delattr(inputs, 'x')",
    "__builtins__",
    "inputs.__class__",
    "inputs.__doc__",
    "inputs.__dict__",
    "__import__('subprocess').check_output('whoami')",
] + [f"__import__('math_{i}')" for i in range(30)]


@pytest.mark.parametrize("hostile", HOSTILE_STRINGS)
def test_hostile_strings_raise_template_error(hostile: str) -> None:
    ctx = {"inputs": {"x": 1}, "steps": {}}
    with pytest.raises(TemplateError):
        evaluate(hostile, ctx)
