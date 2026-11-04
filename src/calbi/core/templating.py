from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

from calbi.core.schema import CalbiError


class TemplateError(CalbiError):
    """Raised when template parsing or evaluation fails."""


_TEMPLATE_EXACT_RE = re.compile(r"^\s*\{\{\s*(.*?)\s*\}\}\s*$")
_TEMPLATE_FIND_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")


def _lookup_path(obj: Any, path: str) -> Any:
    """Helper to look up attribute or item from obj, raising TemplateError on missing."""
    if isinstance(obj, Mapping):
        if path in obj:
            return obj[path]
        raise TemplateError(f"Key '{path}' not found in mapping.")
    if hasattr(obj, path):
        return getattr(obj, path)
    raise TemplateError(f"Attribute or key '{path}' not found on {type(obj).__name__}.")


class _SafeEvaluator(ast.NodeVisitor):
    def __init__(self, ctx: dict[str, Any]) -> None:
        self.ctx = ctx

    def evaluate(self, node: ast.AST) -> Any:
        return self.visit(node)

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        return node.value

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("_"):
            raise TemplateError(f"Access to private/dunder variable '{node.id}' is forbidden.")
        if node.id in self.ctx:
            return self.ctx[node.id]
        if node.id in ("True", "False", "None"):
            return {"True": True, "False": False, "None": None}[node.id]
        raise TemplateError(f"Variable '{node.id}' not found in context.")

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("_"):
            raise TemplateError(f"Access to private/dunder attribute '{node.attr}' is forbidden.")
        target = self.visit(node.value)
        return _lookup_path(target, node.attr)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        target = self.visit(node.value)
        slice_val = self.visit(node.slice)
        try:
            return target[slice_val]
        except (KeyError, IndexError, TypeError) as e:
            raise TemplateError(f"Subscript access error for key/index '{slice_val}': {e}") from e

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not operand
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise TemplateError(f"Unary operator '{type(node.op).__name__}' is not allowed.")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise TemplateError(f"Binary operator '{type(node.op).__name__}' is not allowed.")

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        if isinstance(node.op, ast.And):
            for val_node in node.values:
                val = self.visit(val_node)
                if not val:
                    return val
            return val
        if isinstance(node.op, ast.Or):
            for val_node in node.values:
                val = self.visit(val_node)
                if val:
                    return val
            return val
        raise TemplateError(f"Boolean operator '{type(node.op).__name__}' is not allowed.")

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            right = self.visit(comparator)
            result: bool
            if isinstance(op, ast.Eq):
                result = left == right
            elif isinstance(op, ast.NotEq):
                result = left != right
            elif isinstance(op, ast.Lt):
                result = left < right
            elif isinstance(op, ast.LtE):
                result = left <= right
            elif isinstance(op, ast.Gt):
                result = left > right
            elif isinstance(op, ast.GtE):
                result = left >= right
            elif isinstance(op, ast.In):
                result = left in right
            elif isinstance(op, ast.NotIn):
                result = left not in right
            elif isinstance(op, ast.Is):
                result = left is right
            elif isinstance(op, ast.IsNot):
                result = left is not right
            else:
                raise TemplateError(f"Comparison operator '{type(op).__name__}' is not allowed.")
            if not result:
                return False
            left = right
        return True

    def visit_Call(self, node: ast.Call) -> Any:
        if isinstance(node.func, ast.Name) and node.func.id == "len":
            if len(node.args) != 1 or node.keywords:
                raise TemplateError("len() expects exactly one argument and no keyword arguments.")
            arg_val = self.visit(node.args[0])
            try:
                return len(arg_val)
            except Exception as e:
                raise TemplateError(f"Error executing len(): {e}") from e
        raise TemplateError(f"Function call to '{ast.dump(node.func)}' is not allowed.")

    def visit_List(self, node: ast.List) -> Any:
        return [self.visit(elt) for elt in node.elts]

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Dict(self, node: ast.Dict) -> Any:
        return {
            self.visit(k): self.visit(v)
            for k, v in zip(node.keys, node.values, strict=True)
            if k is not None
        }

    def generic_visit(self, node: ast.AST) -> Any:
        msg = f"Syntax construct '{type(node).__name__}' is not allowed in templates."
        raise TemplateError(msg)


def evaluate(expr: str, ctx: dict[str, Any]) -> Any:
    """Evaluate a single Python expression in the restricted safe evaluator."""
    if not expr or not expr.strip():
        return None
    try:
        parsed = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise TemplateError(f"Invalid expression syntax '{expr}': {e}") from e

    evaluator = _SafeEvaluator(ctx)
    return evaluator.evaluate(parsed)


def render(value: Any, ctx: dict[str, Any]) -> Any:
    """Recursively render templates in dicts, lists, and strings."""
    if isinstance(value, str):
        # Check if it is exactly one "{{ expr }}"
        match_exact = _TEMPLATE_EXACT_RE.match(value)
        if match_exact:
            expr = match_exact.group(1)
            return evaluate(expr, ctx)

        # Check if there are embedded templates
        if "{{" in value and "}}" in value:

            def repl(m: re.Match[str]) -> str:
                res = evaluate(m.group(1), ctx)
                return str(res)

            return _TEMPLATE_FIND_RE.sub(repl, value)

        return value

    if isinstance(value, Mapping):
        return {k: render(v, ctx) for k, v in value.items()}

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [render(item, ctx) for item in value]

    return value


class _StepReferenceCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.step_ids: set[str] = set()

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if isinstance(node.value, ast.Name) and node.value.id == "steps":
            self.step_ids.add(node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if isinstance(node.value, ast.Name) and node.value.id == "steps":
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                self.step_ids.add(node.slice.value)
        self.generic_visit(node)


def referenced_steps(value: Any) -> set[str]:
    """Extract all step IDs referenced via steps.<id> or steps['<id>']."""
    collector = _StepReferenceCollector()

    def scan_str(s: str) -> None:
        for m in _TEMPLATE_FIND_RE.finditer(s):
            expr = m.group(1).strip()
            try:
                tree = ast.parse(expr, mode="eval")
                collector.visit(tree)
            except SyntaxError:
                pass

    def walk(obj: Any) -> None:
        if isinstance(obj, str):
            scan_str(obj)
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                if isinstance(k, str):
                    scan_str(k)
                walk(v)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for item in obj:
                walk(item)

    walk(value)
    return collector.step_ids
