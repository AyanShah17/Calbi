from calbi.core.policy import Autonomy, Policy
from calbi.core.registry import ToolRegistry
from calbi.core.schema import InputSpec, Playbook, RetryPolicy, Risk, Step
from calbi.core.validator import validate


def setup_test_registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(
        name="dataset.inspect", description="Inspect dataset", risk=Risk.READ, idempotent=True
    )
    def inspect_dataset(file: str) -> dict[str, str]:
        return {"status": "ok", "valid": "true"}

    @reg.tool(name="dataset.analyze", description="Analyze dataset", risk=Risk.READ)
    def analyze_dataset(file: str) -> dict[str, str]:
        return {"path": "/tmp/out.csv"}

    @reg.tool(name="system.delete", description="Delete data", risk=Risk.DESTRUCTIVE)
    def delete_data(target: str) -> bool:
        return True

    return reg


def test_valid_playbook_passes_validation() -> None:
    reg = setup_test_registry()
    policy = Policy(autonomy=Autonomy.CONTROLLED)

    pb = Playbook(
        name="test_pipeline",
        inputs={"dataset": InputSpec(type="string", required=True)},
        steps=[
            Step(
                id="inspect",
                tool="dataset.inspect",
                inputs={"file": "{{ inputs.dataset }}"},
            ),
            Step(
                id="analyze",
                tool="dataset.analyze",
                depends_on=["inspect"],
                inputs={"file": "{{ inputs.dataset }}"},
                condition="{{ steps.inspect.output.valid }}",
            ),
        ],
        outputs={"result": "{{ steps.analyze.output.path }}"},
    )

    report = validate(pb, reg, policy, inputs={"dataset": "sales.csv"})
    assert report.ok is True
    assert report.step_count == 2
    assert report.max_risk == Risk.READ
    assert report.approvals_needed == []
    assert len([i for i in report.issues if i.severity == "error"]) == 0


def test_unknown_tool_issue() -> None:
    reg = setup_test_registry()
    policy = Policy()

    pb = Playbook(
        name="test",
        steps=[Step(id="s1", tool="non_existent.tool")],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    codes = [i.code for i in report.issues]
    assert "unknown_tool" in codes


def test_tool_denied_issue() -> None:
    reg = setup_test_registry()
    policy = Policy(denied_tools={"dataset.inspect"})

    pb = Playbook(
        name="test",
        steps=[Step(id="s1", tool="dataset.inspect", inputs={"file": "test"})],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    codes = [i.code for i in report.issues]
    assert "tool_denied" in codes


def test_unknown_dependency_issue() -> None:
    reg = setup_test_registry()
    policy = Policy()

    pb = Playbook(
        name="test",
        steps=[Step(id="s1", tool="dataset.inspect", depends_on=["missing_step"])],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    codes = [i.code for i in report.issues]
    assert "unknown_dependency" in codes


def test_cycle_detection() -> None:
    reg = setup_test_registry()
    policy = Policy()

    pb = Playbook(
        name="test",
        steps=[
            Step(id="s1", tool="dataset.inspect", depends_on=["s2"]),
            Step(id="s2", tool="dataset.inspect", depends_on=["s1"]),
        ],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    issue = next(i for i in report.issues if i.code == "cycle")
    assert "s1" in issue.message
    assert "s2" in issue.message


def test_unknown_step_ref_without_transitive_dependency() -> None:
    reg = setup_test_registry()
    policy = Policy()

    pb = Playbook(
        name="test",
        steps=[
            Step(id="s1", tool="dataset.inspect"),
            Step(
                id="s2",
                tool="dataset.analyze",
                # s2 references s1 output but does NOT depend on s1
                inputs={"file": "{{ steps.s1.output.valid }}"},
            ),
        ],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    codes = [i.code for i in report.issues]
    assert "unknown_step_ref" in codes


def test_unknown_input_ref() -> None:
    reg = setup_test_registry()
    policy = Policy()

    pb = Playbook(
        name="test",
        inputs={"real_input": InputSpec(type="string")},
        steps=[
            Step(
                id="s1", tool="dataset.inspect", inputs={"file": "{{ inputs.unregistered_input }}"}
            ),
        ],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    codes = [i.code for i in report.issues]
    assert "unknown_input_ref" in codes


def test_missing_required_input_and_bad_type() -> None:
    reg = setup_test_registry()
    policy = Policy()

    pb = Playbook(
        name="test",
        inputs={
            "req": InputSpec(type="integer", required=True),
        },
        steps=[Step(id="s1", tool="dataset.inspect", inputs={"file": "x"})],
    )
    # Missing input
    r1 = validate(pb, reg, policy, inputs={})
    assert r1.ok is False
    assert "missing_required_input" in [i.code for i in r1.issues]

    # Bad type
    r2 = validate(pb, reg, policy, inputs={"req": "not-an-int"})
    assert r2.ok is False
    assert "bad_input_type" in [i.code for i in r2.issues]


def test_tool_arg_missing_and_unknown() -> None:
    reg = setup_test_registry()
    policy = Policy()

    # file is required for dataset.inspect
    pb1 = Playbook(
        name="test",
        steps=[Step(id="s1", tool="dataset.inspect", inputs={})],
    )
    r1 = validate(pb1, reg, policy)
    assert r1.ok is False
    assert "tool_arg_missing" in [i.code for i in r1.issues]

    # unexpected arg
    pb2 = Playbook(
        name="test",
        steps=[Step(id="s1", tool="dataset.inspect", inputs={"file": "f", "unexpected": 123})],
    )
    r2 = validate(pb2, reg, policy)
    assert r2.ok is False
    assert "tool_arg_unknown" in [i.code for i in r2.issues]


def test_too_many_steps() -> None:
    reg = setup_test_registry()
    policy = Policy(max_steps=2)

    pb = Playbook(
        name="test",
        steps=[
            Step(id="s1", tool="dataset.inspect", inputs={"file": "a"}),
            Step(id="s2", tool="dataset.inspect", inputs={"file": "b"}),
            Step(id="s3", tool="dataset.inspect", inputs={"file": "c"}),
        ],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    assert "too_many_steps" in [i.code for i in report.issues]


def test_bad_template_syntax() -> None:
    reg = setup_test_registry()
    policy = Policy()

    pb = Playbook(
        name="test",
        steps=[
            Step(id="s1", tool="dataset.inspect", inputs={"file": "{{ __import__('os') }}"}),
        ],
    )
    report = validate(pb, reg, policy)
    assert report.ok is False
    assert "bad_template" in [i.code for i in report.issues]


def test_needs_approval_warning_and_max_risk() -> None:
    reg = setup_test_registry()
    policy = Policy(autonomy=Autonomy.CONTROLLED)

    pb = Playbook(
        name="test",
        steps=[
            Step(id="s1", tool="system.delete", inputs={"target": "/tmp"}),
        ],
    )
    report = validate(pb, reg, policy)
    # Warning, so ok is still True unless errors exist
    assert report.ok is True
    assert report.max_risk == Risk.DESTRUCTIVE
    assert report.approvals_needed == ["s1"]
    assert any(i.code == "needs_approval" and i.severity == "warning" for i in report.issues)


def test_retry_non_idempotent_warning() -> None:
    reg = setup_test_registry()
    policy = Policy()

    # dataset.analyze is idempotent=False by default
    pb = Playbook(
        name="test",
        steps=[
            Step(
                id="s1",
                tool="dataset.analyze",
                inputs={"file": "x"},
                retry=RetryPolicy(max_attempts=3),
            ),
        ],
    )
    report = validate(pb, reg, policy)
    assert report.ok is True
    assert any(i.code == "retry_non_idempotent" and i.severity == "warning" for i in report.issues)
