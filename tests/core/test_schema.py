from pathlib import Path

import pytest
from pydantic import ValidationError

from calbi.core.schema import (
    CalbiError,
    Playbook,
    RetryPolicy,
    Risk,
    SchemaError,
    Step,
    dump_playbook,
    load_playbook,
)


def test_risk_ordering() -> None:
    assert Risk.READ < Risk.WRITE < Risk.DESTRUCTIVE < Risk.EXTERNAL < Risk.FINANCIAL < Risk.DEPLOY
    assert Risk.READ <= Risk.READ
    assert Risk.DEPLOY > Risk.READ
    assert not (Risk.WRITE < Risk.READ)


def test_retry_policy_validation() -> None:
    rp = RetryPolicy(max_attempts=3, backoff="exponential", base_delay_s=1.0, max_delay_s=10.0)
    assert rp.max_attempts == 3
    assert rp.backoff == "exponential"

    with pytest.raises(ValidationError):
        RetryPolicy(max_attempts=0)


def test_step_id_regex() -> None:
    Step(id="valid_step_1", tool="test.tool")
    Step(id="a", tool="test.tool")

    with pytest.raises(ValidationError):
        Step(id="1invalid", tool="test.tool")

    with pytest.raises(ValidationError):
        Step(id="Invalid_Cap", tool="test.tool")

    with pytest.raises(ValidationError):
        Step(id="invalid-dash", tool="test.tool")

    with pytest.raises(ValidationError):
        Step(id="", tool="test.tool")


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        Step(id="valid", tool="test.tool", unauthorized_field=True)  # type: ignore[call-arg]

    with pytest.raises(ValidationError):
        Playbook(
            name="test",
            steps=[Step(id="s1", tool="t1")],
            arbitrary="forbidden",  # type: ignore[call-arg]
        )


def test_playbook_empty_steps_forbidden() -> None:
    with pytest.raises(ValidationError):
        Playbook(name="empty", steps=[])


def test_playbook_duplicate_step_ids_forbidden() -> None:
    with pytest.raises(ValidationError):
        Playbook(
            name="dups",
            steps=[
                Step(id="step_a", tool="t1"),
                Step(id="step_a", tool="t2"),
            ],
        )


def test_load_and_dump_playbook_roundtrip() -> None:
    yaml_text = """
name: analyze_dataset
version: 1
objective: Analyze an uploaded dataset.
inputs:
  dataset:
    type: string
    required: true
    description: Path to file
steps:
  - id: inspect
    tool: dataset.inspect
    inputs:
      file: "{{ inputs.dataset }}"
    retry:
      max_attempts: 3
      backoff: exponential
      base_delay_s: 0.5
      max_delay_s: 30.0
    timeout_s: 60.0
    on_failure: fail
  - id: analyze
    tool: dataset.analyze
    depends_on:
      - inspect
    condition: "{{ steps.inspect.output.valid }}"
    on_failure: replan
outputs:
  report: "{{ steps.analyze.output.path }}"
metadata:
  owner: data-team
"""
    pb = load_playbook(yaml_text)
    assert pb.name == "analyze_dataset"
    assert len(pb.steps) == 2
    assert pb.steps[0].id == "inspect"
    assert pb.steps[0].retry.max_attempts == 3
    assert pb.steps[1].depends_on == ["inspect"]
    assert pb.outputs["report"] == "{{ steps.analyze.output.path }}"

    dumped = dump_playbook(pb)
    pb2 = load_playbook(dumped)
    assert pb == pb2


def test_load_playbook_from_dict_and_path(tmp_path: Path) -> None:
    data = {
        "name": "sample",
        "steps": [{"id": "step1", "tool": "sample.tool"}],
    }
    pb1 = load_playbook(data)
    assert pb1.name == "sample"

    file_path = tmp_path / "playbook.yaml"
    file_path.write_text(dump_playbook(pb1), encoding="utf-8")

    pb2 = load_playbook(file_path)
    assert pb2.name == "sample"
    assert pb2 == pb1


def test_load_playbook_invalid_yaml_raises_schema_error() -> None:
    with pytest.raises(SchemaError, match="Invalid YAML"):
        load_playbook("not: valid: yaml: [")


def test_load_playbook_validation_error_raises_schema_error() -> None:
    with pytest.raises(SchemaError):
        load_playbook("name: missing_steps")


def test_calbi_error_hierarchy() -> None:
    assert issubclass(SchemaError, CalbiError)
    assert issubclass(CalbiError, Exception)
