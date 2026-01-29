"""Base model for SQLAlchemy ORM."""

from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, func, text
from datetime import datetime
from typing import Optional
import uuid

# Note: We define DEFAULT_TENANT_ID here to avoid circular imports.
# The authoritative value is in core.tenant_context - this MUST match.
# (core/__init__.py imports models.base, and importing core.tenant_context
# from here would create a circular dependency)
DEFAULT_TENANT_ID = "default-tenant"


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all database models with async support."""
    pass


class TimestampMixin:
    """Mixin for timestamp fields."""
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now()
    )


class UUIDMixin:
    """Mixin for UUID primary key."""
    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4())
    )


class TenantMixin:
    """Mixin for multi-tenant support.

    Note: tenant_id is required (nullable=False) but has a default
    from DEFAULT_TENANT_ID for self-hosted single-tenant deployments.
    The server_default ensures database-level consistency.
    """
    tenant_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
        default=DEFAULT_TENANT_ID,
        server_default=text(f"'{DEFAULT_TENANT_ID}'")
    )


__all__ = ["Base", "TimestampMixin", "UUIDMixin", "TenantMixin"]