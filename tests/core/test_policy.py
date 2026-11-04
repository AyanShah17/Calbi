from calbi.core.policy import Autonomy, Policy
from calbi.core.registry import ToolSpec
from calbi.core.schema import Risk


def make_spec(
    name: str = "tool.test",
    risk: Risk = Risk.READ,
    requires_approval: bool = False,
) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="test tool",
        fn=lambda: None,
        is_async=False,
        input_schema={},
        output_schema=None,
        risk=risk,
        requires_approval=requires_approval,
    )


def test_policy_denied_tools() -> None:
    pol = Policy(denied_tools={"dangerous.tool"})
    spec = make_spec(name="dangerous.tool", risk=Risk.READ)
    assert pol.decide(spec, Autonomy.AUTONOMOUS) == "deny"


def test_policy_allowed_tools_whitelist() -> None:
    pol = Policy(allowed_tools={"allowed.tool"})
    spec1 = make_spec(name="allowed.tool", risk=Risk.READ)
    spec2 = make_spec(name="other.tool", risk=Risk.READ)
    assert pol.decide(spec1, Autonomy.AUTONOMOUS) == "allow"
    assert pol.decide(spec2, Autonomy.AUTONOMOUS) == "deny"


def test_always_approve_risks_enforced_at_all_autonomy_levels() -> None:
    pol = Policy()
    for risk in [Risk.DESTRUCTIVE, Risk.EXTERNAL, Risk.FINANCIAL, Risk.DEPLOY]:
        spec = make_spec(risk=risk)
        for autonomy in [Autonomy.CONTROLLED, Autonomy.AUTONOMOUS, Autonomy.ADAPTIVE]:
            assert pol.decide(spec, autonomy) == "needs_approval"


def test_spec_requires_approval_enforced_at_all_autonomy_levels() -> None:
    pol = Policy()
    spec = make_spec(risk=Risk.READ, requires_approval=True)
    for autonomy in [Autonomy.CONTROLLED, Autonomy.AUTONOMOUS, Autonomy.ADAPTIVE]:
        assert pol.decide(spec, autonomy) == "needs_approval"


def test_autonomy_advisory_denies_all_execution() -> None:
    pol = Policy()
    spec = make_spec(risk=Risk.READ)
    assert pol.decide(spec, Autonomy.ADVISORY) == "deny"


def test_autonomy_supervised_requires_approval_for_all() -> None:
    pol = Policy()
    spec = make_spec(risk=Risk.READ)
    assert pol.decide(spec, Autonomy.SUPERVISED) == "needs_approval"


def test_risk_exceeding_max_risk_auto_needs_approval() -> None:
    pol = Policy(max_risk_auto=Risk.READ)
    spec_read = make_spec(risk=Risk.READ)
    spec_write = make_spec(risk=Risk.WRITE)
    assert pol.decide(spec_read, Autonomy.AUTONOMOUS) == "allow"
    assert pol.decide(spec_write, Autonomy.AUTONOMOUS) == "needs_approval"


def test_policy_budgets_defaults() -> None:
    pol = Policy()
    assert pol.max_steps == 50
    assert pol.max_replans == 3
    assert pol.max_total_attempts == 200
    assert pol.max_runtime_s == 600.0
