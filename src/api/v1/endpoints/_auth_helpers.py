"""Shared authorization helpers for Community endpoints.

Provides safe user attribute access that handles both dict and ORM User objects.
"""

from typing import Optional


def get_safe_user_id(current_user) -> Optional[str]:
    if isinstance(current_user, dict):
        return current_user.get("id")
    return getattr(current_user, 'id', None)


def get_safe_tenant_id(current_user, default: str = "default-tenant") -> str:
    if isinstance(current_user, dict):
        return current_user.get("tenant_id", default) or default
    return getattr(current_user, 'tenant_id', default) or default


def get_safe_attr(current_user, attr: str, default=None):
    if isinstance(current_user, dict):
        return current_user.get(attr, default)
    return getattr(current_user, attr, default)
