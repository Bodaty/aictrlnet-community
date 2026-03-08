"""Pure helper functions for safe user attribute access.

These handle both dict and ORM User objects. Located in core/ so that
core/dependencies.py and core/security.py can import without triggering
the api.v1 package init chain.
"""
from typing import Optional


def get_safe_user_id(current_user) -> Optional[str]:
    if isinstance(current_user, dict):
        return current_user.get("id")
    return getattr(current_user, 'id', None)


def get_safe_attr(current_user, attr: str, default=None):
    if isinstance(current_user, dict):
        return current_user.get(attr, default)
    return getattr(current_user, attr, default)
