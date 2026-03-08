"""Common dependencies for FastAPI endpoints."""

from typing import Optional, Dict, Any, Union, List
from fastapi import Depends, HTTPException, status

from .config import get_settings
from .security import get_current_active_user, get_current_user
from .tenant_context import get_current_tenant_id
from .user_utils import get_safe_attr, get_safe_user_id


async def get_current_user_safe(
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user as a dictionary, handling both User objects and dicts.

    This ensures consistent handling across all endpoints
    regardless of whether we get a User model or dict.
    """
    uid = str(get_safe_user_id(current_user) or 'unknown')
    return {
        'id': uid,
        'sub': get_safe_attr(current_user, 'sub', uid),
        'email': get_safe_attr(current_user, 'email', 'unknown@example.com'),
        'name': get_safe_attr(current_user, 'name', None) or get_safe_attr(current_user, 'username', 'Test User'),
        'username': get_safe_attr(current_user, 'username', 'unknown'),
        'is_active': get_safe_attr(current_user, 'is_active', True),
        'is_superuser': get_safe_attr(current_user, 'is_superuser', False),
        'is_admin': get_safe_attr(current_user, 'is_superuser', False),
        'tenant_id': get_safe_attr(current_user, 'tenant_id', None) or get_current_tenant_id(),
        'edition': get_safe_attr(current_user, 'edition', 'community'),
        'roles': get_safe_attr(current_user, 'roles', ['user']),
        'permissions': get_safe_attr(current_user, 'permissions', [])
    }


async def require_superuser(current_user: Dict[str, Any] = Depends(get_current_user_safe)) -> None:
    """
    Dependency to require superuser/admin privileges.
    
    Raises HTTPException if user is not a superuser.
    """
    if not current_user.get('is_superuser', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Admin access required."
        )


def get_edition() -> str:
    """Get current edition."""
    settings = get_settings()
    return settings.EDITION.lower()


class EditionChecker:
    """Check if user has access to edition features."""
    
    def __init__(self, required_editions: Union[str, List[str]]):
        if isinstance(required_editions, str):
            self.required_editions = [required_editions.lower()]
        else:
            self.required_editions = [ed.lower() for ed in required_editions]
        self.edition_hierarchy = {
            "community": 0,
            "business": 1,
            "enterprise": 2,
        }
    
    async def __call__(
        self,
        current_user: Any = Depends(get_current_active_user),
        edition: str = Depends(get_edition),
    ) -> Any:
        """Check edition access."""
        # Handle both dict and User object
        user_edition = (get_safe_attr(current_user, 'edition', 'community') or 'community').lower()
        current_level = self.edition_hierarchy.get(edition, 0)
        user_level = self.edition_hierarchy.get(user_edition, 0)
        
        # Check if user has access to any of the required editions
        has_access = False
        for required_edition in self.required_editions:
            required_level = self.edition_hierarchy.get(required_edition, 0)
            if current_level >= required_level and user_level >= required_level:
                has_access = True
                break
        
        if not has_access:
            editions_str = " or ".join(self.required_editions)
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This feature requires {editions_str} edition",
            )
        
        return current_user


def require_edition(edition: Union[str, List[str]]):
    """Require specific edition(s) for endpoint."""
    return EditionChecker(edition)


class FeatureLimiter:
    """Check feature limits based on edition."""
    
    def __init__(self, feature: str):
        self.feature = feature
    
    async def __call__(
        self,
        current_user: Any = Depends(get_current_active_user),
        edition: str = Depends(get_edition),
    ) -> Dict[str, Any]:
        """Check feature limit."""
        settings = get_settings()
        edition_features = settings.get_edition_features()
        limit = edition_features.get(self.feature, 0)
        
        return {
            "user": current_user,
            "feature": self.feature,
            "limit": limit,
            "unlimited": limit == -1,
        }


def check_feature_limit(feature: str):
    """Check if user has reached feature limit."""
    return FeatureLimiter(feature)


# Convenience dependencies for common edition requirements
require_business_or_enterprise = require_edition("business")
require_enterprise = require_edition("enterprise")


class PaginationParams:
    """Common pagination parameters."""
    
    def __init__(
        self,
        skip: int = 0,
        limit: int = 100,
        sort_by: Optional[str] = None,
        sort_order: str = "asc",
    ):
        self.skip = skip
        self.limit = min(limit, 1000)  # Max 1000 items
        self.sort_by = sort_by
        self.sort_order = sort_order


class FilterParams:
    """Common filter parameters."""
    
    def __init__(
        self,
        search: Optional[str] = None,
        category: Optional[str] = None,
        status: Optional[str] = None,
        created_after: Optional[str] = None,
        created_before: Optional[str] = None,
    ):
        self.search = search
        self.category = category
        self.status = status
        self.created_after = created_after
        self.created_before = created_before