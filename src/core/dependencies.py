"""Common dependencies for FastAPI endpoints."""

from typing import Optional, Dict, Any, Union, List
from fastapi import Depends, HTTPException, status

from .config import get_settings
from .security import get_current_active_user, get_current_user
from .tenant_context import get_current_tenant_id


async def get_current_user_safe(
    current_user = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Get current user as a dictionary, handling both User objects and dicts.
    
    This ensures consistent handling across all endpoints
    regardless of whether we get a User model or dict.
    """
    # Use getattr to safely access attributes regardless of type
    if isinstance(current_user, dict):
        # Already a dict, just ensure all expected keys exist
        return {
            'id': current_user.get('id', 'unknown'),
            'sub': current_user.get('sub', current_user.get('id', 'unknown')),
            'email': current_user.get('email', 'unknown@example.com'),
            'name': current_user.get('name', 'Test User'),
            'username': current_user.get('username', 'unknown'),
            'is_active': current_user.get('is_active', True),
            'is_superuser': current_user.get('is_superuser', False),
            'is_admin': current_user.get('is_superuser', False),  # Alias
            'tenant_id': current_user.get('tenant_id') or get_current_tenant_id(),
            'edition': current_user.get('edition', 'community'),
            'roles': current_user.get('roles', ['user']),
            'permissions': current_user.get('permissions', [])
        }
    else:
        # It's a User object, extract attributes
        return {
            'id': str(getattr(current_user, 'id', 'unknown')),
            'sub': str(getattr(current_user, 'id', 'unknown')),
            'email': getattr(current_user, 'email', 'unknown@example.com'),
            'name': getattr(current_user, 'name', getattr(current_user, 'username', 'Test User')),
            'username': getattr(current_user, 'username', 'unknown'),
            'is_active': getattr(current_user, 'is_active', True),
            'is_superuser': getattr(current_user, 'is_superuser', False),
            'is_admin': getattr(current_user, 'is_superuser', False),  # Alias
            'tenant_id': getattr(current_user, 'tenant_id', None) or get_current_tenant_id(),
            'edition': getattr(current_user, 'edition', 'community'),
            'roles': getattr(current_user, 'roles', ['user']),
            'permissions': getattr(current_user, 'permissions', [])
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
        if hasattr(current_user, 'edition'):
            user_edition = current_user.edition.lower() if current_user.edition else "community"
        else:
            user_edition = current_user.get("edition", "community").lower()
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