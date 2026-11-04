import ast
from pathlib import Path


def test_core_does_not_import_ai_or_memory() -> None:
    """Ensure architectural boundary: calbi.core must NOT import calbi.ai or calbi.memory."""
    core_dir = Path(__file__).resolve().parent.parent / "src" / "calbi" / "core"
    assert core_dir.exists() and core_dir.is_dir()

    forbidden_prefixes = ("calbi.ai", "calbi.memory")

    violations: list[str] = []
    for py_file in core_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if any(alias.name.startswith(p) for p in forbidden_prefixes):
                        violations.append(f"{py_file.name}:{node.lineno} imports '{alias.name}'")
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(node.module.startswith(p) for p in forbidden_prefixes):
                    violations.append(f"{py_file.name}:{node.lineno} imports from '{node.module}'")

    assert not violations, (
        "Architectural violation: calbi.core imports forbidden layers:\n" + "\n".join(violations)
    )
