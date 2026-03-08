"""Shared authorization helpers for Community endpoints.

Provides safe user attribute access that handles both dict and ORM User objects.
"""

from typing import Optional

from core.user_utils import get_safe_user_id, get_safe_attr  # noqa: F401 re-export


def get_safe_tenant_id(current_user) -> Optional[str]:
    if isinstance(current_user, dict):
        return current_user.get("tenant_id")
    return getattr(current_user, 'tenant_id', None)


def is_superuser(current_user) -> bool:
    if isinstance(current_user, dict):
        return current_user.get("is_superuser", False) or current_user.get("is_admin", False)
    return getattr(current_user, 'is_superuser', False) or getattr(current_user, 'is_admin', False)
