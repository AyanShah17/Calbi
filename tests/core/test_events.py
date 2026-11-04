import json
from datetime import datetime, timezone

from calbi.core.events import Event, EventBus


def test_event_creation_and_fields() -> None:
    now = datetime.now(timezone.utc)
    ev = Event(
        ts=now,
        run_id="run_123",
        type="step.started",
        step_id="step_a",
        data={"attempt": 1},
    )
    assert ev.run_id == "run_123"
    assert ev.type == "step.started"
    assert ev.step_id == "step_a"
    assert ev.data["attempt"] == 1


def test_event_bus_subscribe_and_emit() -> None:
    bus = EventBus()
    received: list[Event] = []

    bus.subscribe(received.append)

    ev1 = Event(
        ts=datetime.now(timezone.utc),
        run_id="r1",
        type="run.started",
        step_id=None,
        data={},
    )
    bus.emit(ev1)

    assert len(bus.history) == 1
    assert len(received) == 1
    assert received[0] == ev1


def test_event_bus_to_jsonl() -> None:
    bus = EventBus()
    ev1 = Event(
        ts=datetime.now(timezone.utc),
        run_id="r1",
        type="run.started",
        step_id=None,
        data={"v": 1},
    )
    ev2 = Event(
        ts=datetime.now(timezone.utc),
        run_id="r1",
        type="run.completed",
        step_id=None,
        data={"v": 2},
    )
    bus.emit(ev1)
    bus.emit(ev2)

    jsonl = bus.to_jsonl()
    lines = [json.loads(line) for line in jsonl.strip().split("\n")]
    assert len(lines) == 2
    assert lines[0]["type"] == "run.started"
    assert lines[1]["type"] == "run.completed"
