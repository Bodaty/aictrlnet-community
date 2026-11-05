"""One recorded version of a workflow's definition (its steps), per tenant."""

import contextlib
import json
import uuid
from contextvars import ContextVar
from datetime import datetime
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import JSON, DateTime, Integer, String, UniqueConstraint, func, insert, inspect, select
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TenantMixin


class WorkflowDefinitionVersion(Base, TenantMixin):
    __tablename__ = "workflow_definition_versions"
    __table_args__ = (
        UniqueConstraint("workflow_id", "version", name="uq_workflow_definition_version"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    workflow_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    definition: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    template_name: Mapped[Optional[str]] = mapped_column(String(256))
    template_version: Mapped[Optional[str]] = mapped_column(String(32))
    reason: Mapped[str] = mapped_column(String(16), nullable=False)
    created_by: Mapped[Optional[str]] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


_context: ContextVar[Optional[Dict[str, Optional[str]]]] = ContextVar(
    "definition_version_context", default=None
)


@contextlib.contextmanager
def version_context(reason: str, actor: Optional[str] = None) -> Iterator[None]:
    token = _context.set({"reason": reason, "actor": actor})
    try:
        yield
    finally:
        _context.reset(token)


def _fields_of(definition: Any) -> Dict[str, Any]:
    if isinstance(definition, str):
        try:
            definition = json.loads(definition)
        except ValueError:
            return {}
    return definition if isinstance(definition, dict) else {}


def _clip(value: Any, length: int) -> Optional[str]:
    return None if value is None else str(value)[:length]


def _record(connection, target, default_reason: str) -> None:
    ctx = _context.get() or {}
    definition = target.definition or {}
    fields = _fields_of(definition)
    version = target.version or 1
    table = WorkflowDefinitionVersion.__table__
    latest = connection.execute(
        select(func.max(table.c.version)).where(table.c.workflow_id == target.id)
    ).scalar()
    if latest is not None and version <= latest:
        version = latest + 1
    connection.execute(
        insert(table).values(
            id=WorkflowDefinitionVersion.id.default.arg(None),
            tenant_id=target.tenant_id,
            workflow_id=target.id,
            version=version,
            definition=definition,
            template_name=_clip(fields.get("name"), 256),
            template_version=_clip(fields.get("version"), 32),
            reason=ctx.get("reason") or default_reason,
            created_by=ctx.get("actor"),
        )
    )


def after_insert(mapper, connection, target) -> None:
    _record(connection, target, "created")


def after_update(mapper, connection, target) -> None:
    history = inspect(target).attrs.definition.history
    if not history.has_changes():
        return
    if history.deleted and history.added and history.deleted[0] == history.added[0]:
        return
    _record(connection, target, "edited")
