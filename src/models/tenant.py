"""Unified Tenant model for all editions.

This provides the infrastructure for multi-tenant SaaS hosting.
Community uses basic fields, Enterprise uses extended fields.
The model follows the accretive pattern - all fields are here,
editions use what they need.
"""

from typing import Optional, Dict, Any
from sqlalchemy import String, Boolean, JSON, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, UUIDMixin, TimestampMixin


class Tenant(Base, UUIDMixin, TimestampMixin):
    """Unified tenant model for multi-tenancy infrastructure.

    Used by:
    - Community: Basic tenant isolation for SaaS
    - Business: Same as Community
    - Enterprise: Full multi-tenant features with quotas, policies, federation

    Fields marked [Enterprise] are primarily used by Enterprise Edition.

    Note: Enterprise-specific relationships (users, policies, quotas, usage)
    are defined in the Enterprise models (TenantUser, CrossTenantPolicy, etc.)
    via back_populates pointing to this model.
    """
    __tablename__ = "tenants"

    # Basic identification (all editions)
    name: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        unique=True,
        comment="URL-safe tenant identifier"
    )
    display_name: Mapped[str] = mapped_column(
        String(256),
        nullable=False,
        comment="Human-readable tenant name"
    )

    # [Enterprise] Extended identification
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="[Enterprise] Detailed tenant description"
    )
    domain: Mapped[Optional[str]] = mapped_column(
        String(256),
        unique=True,
        nullable=True,
        comment="[Enterprise] Custom domain for tenant"
    )

    # Status - using String instead of Enum to avoid migration conflicts
    status: Mapped[str] = mapped_column(
        String(20),
        default="active",
        nullable=False,
        comment="active, inactive, suspended, trial"
    )

    # Flexible settings storage (all editions)
    settings: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        default=dict,
        comment="Tenant-specific configuration"
    )

    # [Enterprise] Extended configuration
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        default=dict,
        comment="[Enterprise] Advanced tenant configuration (tier, billing, etc.)"
    )
    tenant_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        default=dict,
        comment="[Enterprise] Additional metadata (contacts, billing info, etc.)"
    )

    # For self-hosted single-tenant mode
    is_default: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True for the auto-created default tenant in self-hosted mode"
    )

    def __repr__(self) -> str:
        return f"<Tenant {self.name} ({self.status})>"


# Status constants
class TenantStatus:
    """Tenant status values."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    TRIAL = "trial"  # [Enterprise] Trial period status


__all__ = ["Tenant", "TenantStatus"]
