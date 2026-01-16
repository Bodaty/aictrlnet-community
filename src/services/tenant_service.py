"""Basic tenant service for Community Edition.

This provides basic tenant operations for the multi-tenant SaaS infrastructure.
Enterprise Edition extends this with:
- Sub-tenant management
- Cross-tenant access
- Resource quotas
- Federation
"""

from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
import logging

from models.tenant import Tenant, TenantStatus
from core.tenant_context import DEFAULT_TENANT_ID

logger = logging.getLogger(__name__)


class TenantService:
    """Basic tenant operations for Community Edition.

    Provides:
    - Get tenant by ID or name
    - Get/create default tenant (for self-hosted mode)
    - List tenants (for admin)

    Enterprise Edition extends this with:
    - Create/update/delete tenants
    - Sub-tenant hierarchy
    - Cross-tenant access
    - Resource quotas
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID.

        Args:
            tenant_id: The tenant ID to look up

        Returns:
            Tenant if found, None otherwise
        """
        result = await self.db.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        )
        return result.scalar_one_or_none()

    async def get_tenant_by_name(self, name: str) -> Optional[Tenant]:
        """Get tenant by name.

        Args:
            name: The tenant name (URL-safe identifier)

        Returns:
            Tenant if found, None otherwise
        """
        result = await self.db.execute(
            select(Tenant).where(Tenant.name == name)
        )
        return result.scalar_one_or_none()

    async def get_default_tenant(self) -> Optional[Tenant]:
        """Get the default tenant (for self-hosted mode).

        Returns:
            The default tenant if exists, None otherwise
        """
        result = await self.db.execute(
            select(Tenant).where(Tenant.is_default == True)
        )
        return result.scalar_one_or_none()

    async def get_or_create_default_tenant(self) -> Tenant:
        """Get or create the default tenant.

        This ensures the default tenant exists for self-hosted deployments.
        Called during application startup.

        Returns:
            The default tenant (existing or newly created)
        """
        tenant = await self.get_default_tenant()
        if tenant:
            return tenant

        # Also check by ID in case is_default flag wasn't set
        tenant = await self.get_tenant(DEFAULT_TENANT_ID)
        if tenant:
            # Set is_default flag if missing
            if not tenant.is_default:
                tenant.is_default = True
                await self.db.commit()
                await self.db.refresh(tenant)
            return tenant

        # Create default tenant
        tenant = Tenant(
            id=DEFAULT_TENANT_ID,
            name="default",
            display_name="Default Tenant",
            status=TenantStatus.ACTIVE,
            is_default=True,
            settings={}
        )

        try:
            self.db.add(tenant)
            await self.db.commit()
            await self.db.refresh(tenant)
            logger.info("Created default tenant for self-hosted mode")
        except IntegrityError:
            # Race condition - another process created it
            await self.db.rollback()
            tenant = await self.get_tenant(DEFAULT_TENANT_ID)
            if not tenant:
                raise RuntimeError("Failed to create or retrieve default tenant")

        return tenant

    async def list_tenants(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[str] = None
    ) -> List[Tenant]:
        """List all tenants.

        Args:
            skip: Number of records to skip (pagination)
            limit: Maximum records to return
            status: Optional status filter

        Returns:
            List of tenants
        """
        query = select(Tenant)

        if status:
            query = query.where(Tenant.status == status)

        query = query.order_by(Tenant.name).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return list(result.scalars().all())

    async def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists.

        Args:
            tenant_id: The tenant ID to check

        Returns:
            True if tenant exists, False otherwise
        """
        tenant = await self.get_tenant(tenant_id)
        return tenant is not None

    async def is_tenant_active(self, tenant_id: str) -> bool:
        """Check if a tenant exists and is active.

        Args:
            tenant_id: The tenant ID to check

        Returns:
            True if tenant exists and is active, False otherwise
        """
        tenant = await self.get_tenant(tenant_id)
        # Compare case-insensitively since Enterprise uses ACTIVE, Community uses active
        return tenant is not None and tenant.status.upper() == TenantStatus.ACTIVE.upper()


__all__ = ["TenantService"]
