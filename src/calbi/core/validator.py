from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, Field

from calbi.core.policy import Autonomy, Policy
from calbi.core.registry import ToolRegistry, UnknownToolError
from calbi.core.schema import Playbook, Risk
from calbi.core.templating import referenced_steps

_TEMPLATE_FIND_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")


class Issue(BaseModel):
    """Validation issue detected during playbook analysis."""

    severity: Literal["error", "warning"]
    code: str
    message: str
    step_id: str | None = None


class ValidationReport(BaseModel):
    """Report summarizing playbook validation results."""

    ok: bool
    issues: list[Issue] = Field(default_factory=list)
    max_risk: Risk = Risk.READ
    approvals_needed: list[str] = Field(default_factory=list)
    step_count: int = 0


def _find_cycle(adj: dict[str, list[str]]) -> list[str] | None:
    """Find cycle path using DFS, returning list of node ids in cycle or None."""
    visited: dict[str, int] = {}  # 0 = unvisited, 1 = visiting, 2 = visited
    parent: dict[str, str | None] = {}

    for node in adj:
        visited[node] = 0
        parent[node] = None

    def dfs(u: str) -> list[str] | None:
        visited[u] = 1
        for v in adj.get(u, []):
            if v not in visited:
                continue
            if visited[v] == 1:
                cycle = [v, u]
                curr = parent[u]
                while curr is not None and curr != v:
                    cycle.append(curr)
                    curr = parent[curr]
                cycle.append(v)
                cycle.reverse()
                return cycle
            elif visited[v] == 0:
                parent[v] = u
                res = dfs(v)
                if res is not None:
                    return res
        visited[u] = 2
        return None

    for node in adj:
        if visited[node] == 0:
            res = dfs(node)
            if res is not None:
                return res
    return None


_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.Subscript,
    ast.UnaryOp,
    ast.UAdd,
    ast.USub,
    ast.Not,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Is,
    ast.IsNot,
    ast.Call,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Load,
)


class _TemplateASTValidator(ast.NodeVisitor):
    def __init__(self, declared_inputs: set[str]) -> None:
        self.declared_inputs = declared_inputs
        self.syntax_errors: list[str] = []
        self.unknown_inputs: list[str] = []

    def visit(self, node: ast.AST) -> Any:
        if not isinstance(node, _ALLOWED_AST_NODES):
            self.syntax_errors.append(f"Syntax construct '{type(node).__name__}' is forbidden.")
            return
        super().visit(node)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id.startswith("_"):
            self.syntax_errors.append(
                f"Access to private or dunder variable '{node.id}' is forbidden."
            )
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        if node.attr.startswith("_"):
            self.syntax_errors.append(
                f"Access to private or dunder attribute '{node.attr}' is forbidden."
            )
        if isinstance(node.value, ast.Name) and node.value.id == "inputs":
            if node.attr not in self.declared_inputs:
                self.unknown_inputs.append(node.attr)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        if isinstance(node.value, ast.Name) and node.value.id == "inputs":
            if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                if node.slice.value not in self.declared_inputs:
                    self.unknown_inputs.append(node.slice.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> Any:
        if not (isinstance(node.func, ast.Name) and node.func.id == "len"):
            self.syntax_errors.append(
                f"Function calls other than len() are forbidden: '{ast.dump(node.func)}'."
            )
        elif len(node.args) != 1 or node.keywords:
            self.syntax_errors.append("len() expects exactly one argument.")
        self.generic_visit(node)


def _check_template_syntax(
    expr: str,
    declared_inputs: set[str],
) -> tuple[list[str], list[str]]:
    """Validate expression syntax and check input references."""
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        return [str(e)], []

    validator = _TemplateASTValidator(declared_inputs)
    validator.visit(tree)
    return validator.syntax_errors, validator.unknown_inputs


def _scan_templates(
    val: Any,
    declared_inputs: set[str],
    step_id: str | None,
    issues: list[Issue],
) -> None:
    """Extract and validate all templates in val."""

    def check_str(s: str) -> None:
        for m in _TEMPLATE_FIND_RE.finditer(s):
            expr = m.group(1).strip()
            syntax_errs, unknown_inputs = _check_template_syntax(expr, declared_inputs)
            for err in syntax_errs:
                issues.append(
                    Issue(
                        severity="error",
                        code="bad_template",
                        message=f"Invalid template expression '{expr}': {err}",
                        step_id=step_id,
                    )
                )
            for unk in unknown_inputs:
                issues.append(
                    Issue(
                        severity="error",
                        code="unknown_input_ref",
                        message=f"Template references undeclared input '{unk}' in '{expr}'.",
                        step_id=step_id,
                    )
                )

    def walk(obj: Any) -> None:
        if isinstance(obj, str):
            check_str(obj)
        elif isinstance(obj, Mapping):
            for k, v in obj.items():
                if isinstance(k, str):
                    check_str(k)
                walk(v)
        elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
            for item in obj:
                walk(item)

    walk(val)


def validate(
    pb: Playbook,
    registry: ToolRegistry,
    policy: Policy,
    inputs: dict[str, Any] | None = None,
) -> ValidationReport:
    """Validate a playbook against registered tools, policy limits, and input specs."""
    issues: list[Issue] = []
    approvals_needed: list[str] = []
    max_risk = Risk.READ
    step_count = len(pb.steps)

    # 1. Step count check
    if step_count > policy.max_steps:
        msg = f"Playbook has {step_count} steps, exceeding limit of {policy.max_steps}."
        issues.append(Issue(severity="error", code="too_many_steps", message=msg))

    # 2. Input validation
    declared_inputs = set(pb.inputs.keys())
    if inputs is not None:
        for name, spec in pb.inputs.items():
            if spec.required and name not in inputs:
                issues.append(
                    Issue(
                        severity="error",
                        code="missing_required_input",
                        message=f"Required input '{name}' is missing.",
                    )
                )
            elif name in inputs:
                val = inputs[name]
                valid = True
                expected = spec.type
                if expected == "string" and not isinstance(val, str):
                    valid = False
                elif expected == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
                    valid = False
                elif expected == "number" and (
                    not isinstance(val, (int, float)) or isinstance(val, bool)
                ):
                    valid = False
                elif expected == "boolean" and not isinstance(val, bool):
                    valid = False
                elif expected == "object" and not isinstance(val, dict):
                    valid = False
                elif expected == "array" and not isinstance(val, list):
                    valid = False

                if not valid:
                    got_type = type(val).__name__
                    msg = f"Input '{name}' expected type '{expected}', got '{got_type}'."
                    issues.append(Issue(severity="error", code="bad_input_type", message=msg))

    # 3. Dependency graph and cycle check
    all_step_ids = {s.id for s in pb.steps}
    adj: dict[str, list[str]] = {s.id: [] for s in pb.steps}
    for step in pb.steps:
        for dep in step.depends_on:
            if dep not in all_step_ids:
                issues.append(
                    Issue(
                        severity="error",
                        code="unknown_dependency",
                        message=f"Step '{step.id}' depends on unknown step '{dep}'.",
                        step_id=step.id,
                    )
                )
            else:
                adj[step.id].append(dep)

    cycle = _find_cycle(adj)
    if cycle:
        issues.append(
            Issue(
                severity="error",
                code="cycle",
                message=f"Cycle detected in step dependencies: {' -> '.join(cycle)}",
                step_id=cycle[0],
            )
        )

    # Compute transitive ancestors for each step
    transitive_deps: dict[str, set[str]] = {s.id: set() for s in pb.steps}
    if not cycle:
        for step in pb.steps:
            queue = list(step.depends_on)
            visited = set()
            while queue:
                curr = queue.pop(0)
                if curr in visited or curr not in adj:
                    continue
                visited.add(curr)
                transitive_deps[step.id].add(curr)
                queue.extend(adj[curr])

    # 4. Step-level validation
    for step in pb.steps:
        _scan_templates(step.inputs, declared_inputs, step.id, issues)
        if step.condition:
            _scan_templates(step.condition, declared_inputs, step.id, issues)

        refs = referenced_steps(step.inputs)
        if step.condition:
            refs = refs.union(referenced_steps(step.condition))

        for ref_id in refs:
            if ref_id not in all_step_ids:
                issues.append(
                    Issue(
                        severity="error",
                        code="unknown_step_ref",
                        message=f"Step '{step.id}' references unknown step '{ref_id}'.",
                        step_id=step.id,
                    )
                )
            elif ref_id != step.id and ref_id not in transitive_deps[step.id]:
                issues.append(
                    Issue(
                        severity="error",
                        code="unknown_step_ref",
                        message=(
                            f"Step '{step.id}' references output of step '{ref_id}' "
                            f"without declaring it in depends_on."
                        ),
                        step_id=step.id,
                    )
                )

        try:
            tool_spec = registry.get(step.tool)
        except UnknownToolError:
            issues.append(
                Issue(
                    severity="error",
                    code="unknown_tool",
                    message=f"Tool '{step.tool}' is not registered.",
                    step_id=step.id,
                )
            )
            continue

        if tool_spec.risk > max_risk:
            max_risk = tool_spec.risk

        # At advisory level, validate against controlled permissions
        # (since advisory level creates without executing)
        check_autonomy = (
            Autonomy.CONTROLLED if policy.autonomy == Autonomy.ADVISORY else policy.autonomy
        )
        decision = policy.decide(tool_spec, check_autonomy)
        if decision == "deny":
            issues.append(
                Issue(
                    severity="error",
                    code="tool_denied",
                    message=f"Tool '{step.tool}' execution is denied by current policy.",
                    step_id=step.id,
                )
            )
        elif decision == "needs_approval":
            approvals_needed.append(step.id)
            issues.append(
                Issue(
                    severity="warning",
                    code="needs_approval",
                    message=f"Step '{step.id}' with tool '{step.tool}' will require approval.",
                    step_id=step.id,
                )
            )

        if not tool_spec.idempotent and step.retry.max_attempts > 1:
            issues.append(
                Issue(
                    severity="warning",
                    code="retry_non_idempotent",
                    message=(
                        f"Step '{step.id}' has retries (attempts={step.retry.max_attempts}) "
                        f"on non-idempotent tool '{step.tool}'."
                    ),
                    step_id=step.id,
                )
            )

        schema = tool_spec.input_schema
        required_args = schema.get("required", [])
        properties = schema.get("properties", {})

        for req in required_args:
            if req not in step.inputs:
                issues.append(
                    Issue(
                        severity="error",
                        code="tool_arg_missing",
                        message=f"Required argument '{req}' for tool '{step.tool}' is missing.",
                        step_id=step.id,
                    )
                )

        if properties:
            for arg_name in step.inputs:
                if arg_name not in properties:
                    issues.append(
                        Issue(
                            severity="error",
                            code="tool_arg_unknown",
                            message=f"Unknown argument '{arg_name}' for tool '{step.tool}'.",
                            step_id=step.id,
                        )
                    )

    # 5. Output templates validation
    for out_name, out_val in pb.outputs.items():
        _scan_templates(out_val, declared_inputs, None, issues)
        out_refs = referenced_steps(out_val)
        for ref_id in out_refs:
            if ref_id not in all_step_ids:
                issues.append(
                    Issue(
                        severity="error",
                        code="unreachable_output",
                        message=f"Output '{out_name}' references unknown step '{ref_id}'.",
                    )
                )

    has_errors = any(i.severity == "error" for i in issues)
    return ValidationReport(
        ok=not has_errors,
        issues=issues,
        max_risk=max_risk,
        approvals_needed=approvals_needed,
        step_count=step_count,
    )
