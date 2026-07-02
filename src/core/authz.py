"""Fail-closed tenant/ownership authorization helpers.

Central place for the "does this principal own this row" check so every by-id
endpoint enforces tenancy the same way and can't silently fall open. Used by the
workflow / IAM / MCP / conversation endpoints to close cross-tenant IDORs.

Design choices:
- **Fail closed.** If the caller's tenant can't be derived, or the resource has
  no tenant, or they differ → deny. Never default to a shared/permissive tenant.
- **404, not 403, on tenant mismatch.** Returning 403 would confirm the resource
  exists in another tenant (existence oracle); 404 matches the "not found" a
  properly-scoped query would have produced.
- **Superuser bypass.** Platform superusers may act across tenants (admin tooling
  relies on it); everyone else is confined to their own tenant.
"""

from typing import Any, Optional

from fastapi import HTTPException, status

from core.user_utils import get_safe_attr
from core.tenant_context import get_current_tenant_id


def is_superuser(current_user: Any) -> bool:
    return bool(get_safe_attr(current_user, "is_superuser", False))


def resolve_caller_tenant(current_user: Any) -> Optional[str]:
    """The caller's tenant, from the principal or the request context.

    Returns None when no tenant can be established — callers must treat None as
    "deny" for non-superusers rather than substituting a default.
    """
    tenant = get_safe_attr(current_user, "tenant_id", None)
    if tenant:
        return str(tenant)
    ctx = get_current_tenant_id()
    return str(ctx) if ctx else None


def assert_tenant_access(
    resource: Any,
    current_user: Any,
    *,
    resource_name: str = "Resource",
) -> Any:
    """Raise 404 unless the caller may access this resource's tenant.

    `resource` must expose a `tenant_id` attribute. Superusers bypass. Returns
    the resource for convenient chaining.
    """
    if is_superuser(current_user):
        return resource

    caller_tenant = resolve_caller_tenant(current_user)
    resource_tenant = get_safe_attr(resource, "tenant_id", None)

    if (
        caller_tenant is None
        or resource_tenant is None
        or str(resource_tenant) != str(caller_tenant)
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{resource_name} not found",
        )
    return resource


def require_tenant(current_user: Any) -> str:
    """Return the caller's tenant or raise 403 — for create/list scoping where
    a tenant is mandatory and there is no resource to compare against."""
    if is_superuser(current_user):
        # Superuser still needs *a* tenant for created rows; fall back to context.
        return resolve_caller_tenant(current_user) or get_current_tenant_id()
    caller_tenant = resolve_caller_tenant(current_user)
    if not caller_tenant:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="No tenant context for this request",
        )
    return caller_tenant
