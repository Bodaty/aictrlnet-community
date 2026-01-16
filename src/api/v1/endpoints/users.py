"""User-related endpoints."""

from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, or_
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.security import get_current_active_user, get_password_hash
from core.dependencies import get_current_user_safe, require_superuser
from models.user import User
from schemas.user import UserCreate, UserUpdate, UserResponse

# Import sub-routers
from .api_keys import router as api_keys_router
from .webhooks import router as webhooks_router

router = APIRouter()

# Include sub-routers
router.include_router(api_keys_router)
router.include_router(webhooks_router)


@router.get("/me")
async def get_current_user(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user information."""
    # Handle both User object and dict
    if hasattr(current_user, 'id'):
        # User object
        return {
            "id": str(current_user.id),
            "email": current_user.email,
            "name": current_user.full_name or current_user.username or "Test User",
            "edition": current_user.edition,
            "roles": ["admin"] if current_user.is_superuser else ["user"],
            "permissions": [],
            "metadata": {
                "last_login": str(current_user.last_login_at) if hasattr(current_user, 'last_login_at') and current_user.last_login_at else None,
                "created_at": str(current_user.created_at) if hasattr(current_user, 'created_at') else None,
                "preferences": current_user.preferences if hasattr(current_user, 'preferences') and current_user.preferences else {
                    "language": "en",
                    "timezone": "UTC",
                    "theme": "light"
                }
            }
        }
    else:
        # Dict (legacy)
        return {
            "id": current_user.get("sub", "anonymous"),
            "email": current_user.get("email", "user@example.com"),
            "name": current_user.get("name", "Test User"),
            "edition": current_user.get("edition", "community"),
            "roles": current_user.get("roles", ["user"]),
            "permissions": current_user.get("permissions", []),
            "metadata": {
                "last_login": current_user.get("last_login"),
                "created_at": current_user.get("created_at"),
                "preferences": {
                    "language": "en",
                    "timezone": "UTC",
                    "theme": "light"
                }
            }
        }


@router.get("/me/preferences")
async def get_current_user_preferences(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user's preferences."""
    # Handle both User object and dict
    if hasattr(current_user, 'preferences'):
        # User object - return actual preferences from database
        preferences = current_user.preferences or {}
        
        # Merge with defaults to ensure all expected fields exist
        default_preferences = {
            "notifications": {
                "email": True,
                "in_app": True,
                "browser": False,
                "approval_requests": True,
                "workflow_completion": True,
                "system_alerts": True
            },
            "ui_preferences": {
                "theme": "light",
                "compact_mode": False,
                "show_welcome": True,
                "default_view": "workflows"
            }
        }
        
        # Deep merge preferences with defaults
        merged_preferences = default_preferences.copy()
        if preferences:
            if "notifications" in preferences:
                merged_preferences["notifications"].update(preferences["notifications"])
            if "ui_preferences" in preferences:
                merged_preferences["ui_preferences"].update(preferences["ui_preferences"])
            # Add any other top-level preferences
            for key, value in preferences.items():
                if key not in ["notifications", "ui_preferences"]:
                    merged_preferences[key] = value
        
        return {"preferences": merged_preferences}
    else:
        # Dict (legacy) - return default preferences
        return {
            "preferences": {
                "notifications": {
                    "email": True,
                    "in_app": True,
                    "browser": False,
                    "approval_requests": True,
                    "workflow_completion": True,
                    "system_alerts": True
                },
                "ui_preferences": {
                    "theme": "light",
                    "compact_mode": False,
                    "show_welcome": True,
                    "default_view": "workflows"
                }
            }
        }


@router.put("/me/preferences")
async def update_current_user_preferences(
    preferences_data: Dict[str, Any],
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update current user's preferences."""
    # Handle both User object and dict
    if hasattr(current_user, 'id'):
        # User object - update preferences in database
        # Get fresh user object to ensure we have write access
        result = await db.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Extract preferences from request body
        # The frontend sends { preferences: {...} }
        new_preferences = preferences_data.get("preferences", preferences_data)
        
        # Merge with existing preferences
        existing_preferences = user.preferences or {}
        
        # Create a deep copy to avoid mutation issues
        import copy
        updated_preferences = copy.deepcopy(existing_preferences)
        
        # Deep merge for nested objects
        if "notifications" in new_preferences:
            if "notifications" not in updated_preferences:
                updated_preferences["notifications"] = {}
            updated_preferences["notifications"].update(new_preferences["notifications"])
        
        if "ui_preferences" in new_preferences:
            if "ui_preferences" not in updated_preferences:
                updated_preferences["ui_preferences"] = {}
            updated_preferences["ui_preferences"].update(new_preferences["ui_preferences"])
        
        # Add any other top-level preferences
        for key, value in new_preferences.items():
            if key not in ["notifications", "ui_preferences"]:
                updated_preferences[key] = value
        
        # Update user preferences with the new object
        user.preferences = updated_preferences
        
        # Ensure the update is marked as dirty for SQLAlchemy
        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(user, "preferences")
        
        await db.commit()
        await db.refresh(user)
        
        return {"preferences": user.preferences}
    else:
        # Dict (legacy) - just return the updated preferences
        new_preferences = preferences_data.get("preferences", preferences_data)
        return {"preferences": new_preferences}


@router.get("/app-settings")
async def get_app_settings(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get user's application settings."""
    # Default settings
    default_settings = {
        "aiModel": "llama3.1-local",  # Default AI model (legacy - 8B for backward compatibility)
        "preferredFastModel": "llama3.2:1b",  # Fast tier model (~1-2s per call)
        "preferredBalancedModel": "llama3.2:3b",  # Balanced tier model (~3-5s per call)
        "preferredQualityModel": "llama3.1:8b-instruct-q4_K_M",  # Quality tier model (~20-25s per call)
        "theme": "light",
        "notifications": {
            "email": True,
            "inApp": True,
            "workflow": True
        },
        "workspace": {
            "defaultView": "grid",
            "autoSave": True,
            "saveInterval": 30
        },
        "nlp": {
            "enabled": True,
            "streaming": True,
            "suggestions": True
        }
    }

    # Handle both User object and dict
    if hasattr(current_user, 'id'):
        user_id = str(current_user.id)

        # Try to get saved settings from database
        result = await db.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if user and user.preferences and 'app_settings' in user.preferences:
            # Merge saved settings with defaults
            saved_settings = user.preferences.get('app_settings', {})
            settings = {**default_settings, **saved_settings}
            return {
                "userId": user_id,
                **settings
            }
    else:
        user_id = current_user.get("sub", "anonymous")

    return {
        "userId": user_id,
        **default_settings
    }


@router.put("/app-settings")
async def update_app_settings(
    settings: Dict[str, Any],
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user's application settings."""
    # Handle both User object and dict
    if hasattr(current_user, 'id'):
        # User object - update preferences in database
        result = await db.execute(
            select(User).where(User.id == current_user.id)
        )
        user = result.scalar_one_or_none()

        if user:
            # Store app settings in user preferences
            if not user.preferences:
                user.preferences = {}

            # Store app settings under 'app_settings' key
            user.preferences['app_settings'] = settings

            # ALSO store aiModel and tier models at top level for LLM service to find
            if 'aiModel' in settings:
                user.preferences['aiModel'] = settings['aiModel']
            if 'preferredFastModel' in settings:
                user.preferences['preferredFastModel'] = settings['preferredFastModel']
            if 'preferredBalancedModel' in settings:
                user.preferences['preferredBalancedModel'] = settings['preferredBalancedModel']
            if 'preferredQualityModel' in settings:
                user.preferences['preferredQualityModel'] = settings['preferredQualityModel']

            # Ensure the update is marked as dirty for SQLAlchemy
            from sqlalchemy.orm.attributes import flag_modified
            flag_modified(user, "preferences")

            await db.commit()
            await db.refresh(user)

            return {
                **settings,
                "userId": str(user.id),
                "updated": True
            }

    # Fallback for dict or if user not found
    user_id = current_user.get("sub", "anonymous") if isinstance(current_user, dict) else str(current_user.id)
    return {
        **settings,
        "userId": user_id,
        "updated": True
    }

@router.patch("/app-settings")
async def update_app_settings_patch(
    settings: Dict[str, Any],
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user's application settings (PATCH method for backwards compatibility)."""
    return await update_app_settings(settings, current_user, db)


@router.get("/profile")
async def get_user_profile(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db),
):
    """Get user's profile information."""
    return {
        "id": current_user.get("sub", "anonymous"),
        "email": current_user.get("email", "user@example.com"),
        "name": current_user.get("name", "Test User"),
        "edition": current_user.get("edition", "community"),
        "roles": current_user.get("roles", ["user"]),
        "permissions": current_user.get("permissions", [])
    }


@router.get("/preferences")
async def get_user_preferences(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db),
):
    """Get user's preferences."""
    return {
        "language": "en",
        "timezone": "UTC",
        "dateFormat": "MM/DD/YYYY",
        "timeFormat": "12h",
        "firstDayOfWeek": "sunday"
    }


@router.patch("/preferences")
async def update_user_preferences(
    preferences: Dict[str, Any],
    current_user: dict = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Update user's preferences."""
    return {
        **preferences,
        "updated": True
    }


# User Management endpoints (admin only)
@router.get("", response_model=Dict[str, Any])
async def list_users(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    search: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    _: None = Depends(require_superuser)
):
    """
    List all users with pagination and search.
    
    Requires admin privileges.
    """
    # Build query
    query = select(User)
    
    # Add search filter if provided
    if search:
        query = query.where(
            or_(
                User.email.ilike(f"%{search}%"),
                User.username.ilike(f"%{search}%"),
                User.full_name.ilike(f"%{search}%")
            )
        )
    
    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total_result = await db.execute(count_query)
    total = total_result.scalar()
    
    # Apply pagination
    query = query.limit(limit).offset(offset).order_by(User.created_at.desc())
    
    # Execute query
    result = await db.execute(query)
    users = result.scalars().all()
    
    # Convert to response format
    users_data = []
    for user in users:
        users_data.append({
            "id": str(user.id),
            "email": user.email,
            "name": user.full_name or user.username or user.email.split("@")[0],
            "username": user.username,
            "edition": user.edition,
            "roles": ["admin"] if user.is_superuser else ["user"],
            "is_active": user.is_active,
            "is_superuser": user.is_superuser,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "last_login": user.last_login_at.isoformat() if hasattr(user, 'last_login_at') and user.last_login_at else None
        })
    
    return {
        "users": users_data,
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/{user_id}", response_model=Dict[str, Any])
async def get_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    _: None = Depends(require_superuser)
):
    """
    Get a specific user by ID.
    
    Requires admin privileges.
    """
    # Query user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.full_name or user.username or user.email.split("@")[0],
        "username": user.username,
        "edition": user.edition,
        "roles": ["admin"] if user.is_superuser else ["user"],
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login_at.isoformat() if hasattr(user, 'last_login_at') and user.last_login_at else None,
        "metadata": {
            "preferences": {
                "language": "en",
                "timezone": "UTC",
                "theme": "light"
            }
        }
    }


@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    _: None = Depends(require_superuser)
):
    """
    Create a new user.
    
    Requires admin privileges.
    """
    # Check if user already exists
    existing_user = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if existing_user.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email already exists"
        )
    
    # Create new user
    user = User(
        email=user_data.email,
        username=user_data.username or user_data.email.split("@")[0],
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        is_active=user_data.is_active if hasattr(user_data, 'is_active') else True,
        is_superuser=user_data.is_superuser if hasattr(user_data, 'is_superuser') else False,
        edition=user_data.edition if hasattr(user_data, 'edition') else 'community'
        # tenant_id is NULL for Community/Business (single-tenant), will be set by Enterprise
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.full_name or user.username,
        "username": user.username,
        "edition": user.edition,
        "roles": ["admin"] if user.is_superuser else ["user"],
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "created_at": user.created_at.isoformat() if user.created_at else None
    }


@router.put("/{user_id}", response_model=Dict[str, Any])
async def update_user(
    user_id: str,
    user_data: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    _: None = Depends(require_superuser)
):
    """
    Update a user.
    
    Requires admin privileges.
    """
    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update fields
    update_data = user_data.dict(exclude_unset=True)
    
    if "password" in update_data:
        update_data["hashed_password"] = get_password_hash(update_data.pop("password"))
    
    for field, value in update_data.items():
        if hasattr(user, field):
            setattr(user, field, value)
    
    await db.commit()
    await db.refresh(user)
    
    return {
        "id": str(user.id),
        "email": user.email,
        "name": user.full_name or user.username or user.email.split("@")[0],
        "username": user.username,
        "edition": user.edition,
        "roles": ["admin"] if user.is_superuser else ["user"],
        "is_active": user.is_active,
        "is_superuser": user.is_superuser,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login_at.isoformat() if hasattr(user, 'last_login_at') and user.last_login_at else None
    }


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    _: None = Depends(require_superuser)
):
    """
    Delete a user.
    
    Requires admin privileges.
    """
    # Get user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Don't allow deleting your own account
    current_user_id = current_user.get("id") or current_user.get("sub")
    if str(user.id) == str(current_user_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete your own account"
        )
    
    await db.delete(user)
    await db.commit()
    
    return None