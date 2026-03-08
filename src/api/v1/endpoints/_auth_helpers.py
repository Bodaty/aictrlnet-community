"""Shared authorization helpers for Community endpoints.

Provides safe user attribute access that handles both dict and ORM User objects.
"""

from core.user_utils import get_safe_user_id, get_safe_attr  # noqa: F401 re-export


def get_safe_tenant_id(current_user, default: str = "default-tenant") -> str:
    if isinstance(current_user, dict):
        return current_user.get("tenant_id", default) or default
    return getattr(current_user, 'tenant_id', default) or default
