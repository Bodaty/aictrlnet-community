"""Who an IAM caller is, for the per-tenant, per-owner IAM rules (G45).

IAM agents belong to a tenant (row-level security keeps every tenant to its
own) and, unless they are the tenant's system agents, to an owner:

- member        : sees their own agents and the tenant's system agents (no
                  owner); sends, heartbeats and updates only as their own.
- tenant admin  : sees and acts as every agent in the tenant.
- superuser     : the same, in whichever tenant they act in.

Community has no tenant-administrator role. A higher edition contributes one
with ``register_tenant_admin_check`` (Enterprise: ``tenant_users.role ==
'admin'``). The shared default tenant holds every self-signup and has no single
owner, so nobody but a superuser administers it.
"""

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, List, Optional

from sqlalchemy import select

from core.tenant_context import DEFAULT_TENANT_ID
from core.user_utils import get_safe_attr, get_safe_user_id

# Where existing agents with no knowable owner were parked by the G45
# migration. No user, signup or session holds this tenant, so RLS hides the
# rows from everyone but the admin bypass.
IAM_UNOWNED_TENANT = "iam-unowned"

# Ids the auth layer substitutes when it cannot resolve a real user; they are
# no identity, never an owner (same set as core/mcp_access.py).
_SENTINEL_IDS = {"unknown", "anonymous", "none", ""}

TenantAdminCheck = Callable[[Any, str, str], Awaitable[bool]]
_TENANT_ADMIN_CHECKS: List[TenantAdminCheck] = []


def register_tenant_admin_check(check: TenantAdminCheck) -> None:
    """Contribute a ``check(db, user_id, tenant_id) -> bool`` for tenant administrators."""
    if check not in _TENANT_ADMIN_CHECKS:
        _TENANT_ADMIN_CHECKS.append(check)


@dataclass(frozen=True)
class IAMCaller:
    """The principal the IAM service scopes agents and their messages to."""

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    is_superuser: bool = False
    is_tenant_admin: bool = False

    @property
    def sees_all(self) -> bool:
        return self.is_superuser or self.is_tenant_admin


def _clean_id(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    value = str(raw)
    return None if value.strip().lower() in _SENTINEL_IDS else value


async def _is_tenant_admin(db, user_id: Optional[str], tenant_id: Optional[str]) -> bool:
    if not user_id or not tenant_id or tenant_id == DEFAULT_TENANT_ID:
        return False
    for check in _TENANT_ADMIN_CHECKS:
        if await check(db, user_id, tenant_id):
            return True
    return False


async def _caller(db, user_id: Optional[str], tenant_id: Optional[str], is_superuser: bool) -> IAMCaller:
    return IAMCaller(
        user_id=user_id,
        tenant_id=tenant_id,
        is_superuser=is_superuser,
        is_tenant_admin=not is_superuser and await _is_tenant_admin(db, user_id, tenant_id),
    )


async def iam_caller(db, current_user: Any) -> IAMCaller:
    """The caller behind a request: a User object or the dict Business/Enterprise pass.

    The tenant is the one the session acts in (what RLS checks), not the user's
    recorded tenant: an OAuth client token can act in another tenant, and an
    administrator of their home tenant is not one there.
    """
    from core.database import session_tenant

    user_id = _clean_id(get_safe_user_id(current_user) or get_safe_attr(current_user, "sub"))
    tenant = session_tenant(db) if db is not None else get_safe_attr(current_user, "tenant_id")
    # is_admin is the dict shape's alias; mirrors core/mcp_access.mcp_caller.
    is_super = bool(
        get_safe_attr(current_user, "is_superuser", False)
        or get_safe_attr(current_user, "is_admin", False)
    )
    return await _caller(db, user_id if user_id else None, str(tenant) if tenant else None,
                         is_super and user_id is not None)


async def iam_caller_for_user(db, user_id: Any, tenant_id: Optional[str]) -> IAMCaller:
    """The caller for a run started by ``user_id`` (nodes and background work)."""
    from models.user import User

    uid = _clean_id(user_id)
    is_super = False
    if uid:
        is_super = bool(await db.scalar(select(User.is_superuser).where(User.id == uid)))
    return await _caller(db, uid, str(tenant_id) if tenant_id else None, is_super)
