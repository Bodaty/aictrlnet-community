"""SSE frames encode any tool result value instead of ending the stream.

Community "list my workflows" ended in `error` (1 Oct): a tool result carried
WorkflowDefinition rows and `json.dumps` raised at the SSE frame.
"""
import enum
import json
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel

from core.sse import sse_frame
from models.community_complete import WorkflowDefinition


class _Colour(enum.Enum):
    RED = "red"


class _Model(BaseModel):
    when: datetime


@dataclass
class _Point:
    x: int


def _data(frame):
    event, data, blank, end = frame.split("\n")
    assert event == "event: tool_complete" and blank == "" and end == ""
    return json.loads(data[len("data: "):])


def test_orm_rows_and_odd_values_encode():
    row = WorkflowDefinition(id="wf-1", name="Invoice intake")
    frame = sse_frame("tool_complete", {
        "result": {"workflows": [row]},
        "cost": Decimal("1.25"),
        "id": UUID("12345678-1234-5678-1234-567812345678"),
        "colour": _Colour.RED,
        "model": _Model(when=datetime(2026, 10, 8, 9, 0)),
        "point": _Point(3),
        "tags": {"a"},
    })
    data = _data(frame)
    assert data["result"]["workflows"] == [{"type": "WorkflowDefinition", "id": "wf-1", "name": "Invoice intake"}]
    assert data["cost"] == "1.25" and data["colour"] == "red" and data["tags"] == ["a"]
    assert data["model"] == {"when": "2026-10-08T09:00:00"} and data["point"] == {"x": 3}
    assert data["id"] == "12345678-1234-5678-1234-567812345678"


def test_orm_rows_never_carry_other_columns():
    from models.user import User

    user = User(id="u-1", email="a@b.c", username="a", hashed_password="$2b$secret")
    data = _data(sse_frame("tool_complete", {"user": user}))
    assert data == {"user": {"type": "User", "id": "u-1"}}


def test_unknown_objects_fall_back_to_text():
    class Opaque:
        def __str__(self):
            return "opaque"

    assert _data(sse_frame("tool_complete", {"v": Opaque()})) == {"v": "opaque"}
