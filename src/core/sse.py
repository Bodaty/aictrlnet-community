"""Server-sent event frames whose data is always valid JSON.

A conversation event carries tool results, and a tool may return ORM rows,
Decimals, enums or Pydantic models. Plain `json.dumps` raised on the first such
value and ended the stream mid-turn (Community "list my workflows", 1 Oct).
"""

import dataclasses
import enum
import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any
from uuid import UUID

_ORM_SUMMARY_FIELDS = ("id", "name", "title", "status")


def json_default(obj: Any) -> Any:
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
    if isinstance(obj, (UUID, Decimal)):
        return str(obj)
    if isinstance(obj, enum.Enum):
        return obj.value
    if isinstance(obj, (set, frozenset, tuple)):
        return list(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    from sqlalchemy import inspect as sa_inspect

    state = sa_inspect(obj, raiseerr=False)
    if state is not None and hasattr(state, "mapper"):
        # An identity summary only: a row can carry secrets (password hashes,
        # tokens) that must never reach the browser; a tool that wants more
        # fields returns them explicitly. Loaded values only — a lazy load
        # here would run I/O inside json.dumps.
        loaded = state.dict
        summary = {"type": type(obj).__name__}
        summary.update({k: loaded[k] for k in _ORM_SUMMARY_FIELDS if k in loaded})
        return summary
    return str(obj)


def sse_frame(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=json_default)}\n\n"
