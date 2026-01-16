"""Tenant context management for request-scoped tenant isolation.

This module provides the infrastructure for multi-tenant SaaS:
- Request-scoped tenant context using contextvars
- Thread-safe tenant ID storage
- Helper functions for getting/setting current tenant

Usage:
    from core.tenant_context import get_current_tenant_id, set_current_tenant_id

    # In middleware (automatic)
    set_current_tenant_id(tenant_id_from_jwt)

    # In services (read context)
    tenant_id = get_current_tenant_id()

    # Manual context (for background tasks)
    with TenantContext(tenant_id):
        # All operations use this tenant
        pass
"""

from contextvars import ContextVar
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Default tenant ID for self-hosted single-tenant mode
DEFAULT_TENANT_ID = "default-tenant"

# Request-scoped tenant context using contextvars (async-safe)
_current_tenant_id: ContextVar[Optional[str]] = ContextVar(
    'current_tenant_id',
    default=None
)


def get_current_tenant_id() -> str:
    """Get the current tenant ID from request context.

    Returns:
        The current tenant_id, or DEFAULT_TENANT_ID if not set.
        This ensures self-hosted mode always works with "default-tenant".
    """
    tenant_id = _current_tenant_id.get()
    return tenant_id if tenant_id else DEFAULT_TENANT_ID


def set_current_tenant_id(tenant_id: str) -> None:
    """Set the current tenant ID in request context.

    Args:
        tenant_id: The tenant ID to set for the current request.
    """
    _current_tenant_id.set(tenant_id)
    logger.debug(f"Tenant context set to: {tenant_id}")


def clear_tenant_context() -> None:
    """Clear the tenant context.

    Called at the end of request processing to reset context.
    """
    _current_tenant_id.set(None)


def get_tenant_id_or_none() -> Optional[str]:
    """Get the current tenant ID, returning None if not explicitly set.

    Unlike get_current_tenant_id(), this does NOT fall back to default.
    Useful for checking if tenant context was explicitly set.
    """
    return _current_tenant_id.get()


class TenantContext:
    """Context manager for tenant scoping.

    Useful for background tasks or when you need to temporarily
    switch tenant context:

        async def process_for_tenant(tenant_id: str):
            with TenantContext(tenant_id):
                # All operations here use the specified tenant
                await some_tenant_specific_operation()

        # After the with block, previous context is restored
    """

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self._token = None
        self._previous_tenant_id = None

    def __enter__(self):
        self._previous_tenant_id = _current_tenant_id.get()
        self._token = _current_tenant_id.set(self.tenant_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._token is not None:
            _current_tenant_id.reset(self._token)
        return False

    async def __aenter__(self):
        return self.__enter__()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return self.__exit__(exc_type, exc_val, exc_tb)


__all__ = [
    "DEFAULT_TENANT_ID",
    "get_current_tenant_id",
    "set_current_tenant_id",
    "clear_tenant_context",
    "get_tenant_id_or_none",
    "TenantContext",
]
