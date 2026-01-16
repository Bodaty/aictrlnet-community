"""API endpoints for managing API keys."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from schemas import (
    APIKeyCreate, APIKeyResponse, APIKeyCreateResponse,
    APIKeyUpdate, APIKeyListResponse, APIKeyRevoke,
    SuccessResponse
)
from services.api_key_service import APIKeyService

router = APIRouter()


@router.get("/api-keys", response_model=APIKeyListResponse)
async def list_api_keys(
    include_inactive: bool = Query(False, description="Include revoked keys"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    List all API keys for the current user.
    
    Returns active keys by default. Set include_inactive=true to see all keys.
    """
    service = APIKeyService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    return await service.list_user_api_keys(
        user_id=user_id,
        include_inactive=include_inactive
    )


@router.post("/api-keys", response_model=APIKeyCreateResponse)
async def create_api_key(
    data: APIKeyCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Create a new API key.
    
    **Important**: The full API key is only shown once in the response.
    Save it securely - it cannot be retrieved again!
    
    Available scopes:
    - read:all, write:all - Full read/write access
    - read:tasks - Read tasks
    - write:tasks - Create/update tasks
    - read:workflows - Read workflows
    - write:workflows - Create/update workflows
    - admin - Full administrative access
    """
    service = APIKeyService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    return await service.create_api_key(
        user_id=user_id,
        data=data
    )


@router.get("/api-keys/{key_id}", response_model=APIKeyResponse)
async def get_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get details of a specific API key."""
    service = APIKeyService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    api_key = await service.get_api_key(
        user_id=user_id,
        key_id=key_id
    )
    
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return api_key


@router.put("/api-keys/{key_id}", response_model=APIKeyResponse)
async def update_api_key(
    key_id: str,
    data: APIKeyUpdate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Update API key metadata.
    
    Note: The key value itself cannot be changed.
    """
    service = APIKeyService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    api_key = await service.update_api_key(
        user_id=user_id,
        key_id=key_id,
        data=data
    )
    
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return api_key


@router.post("/api-keys/{key_id}/revoke", response_model=SuccessResponse)
async def revoke_api_key(
    key_id: str,
    data: APIKeyRevoke,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Revoke an API key.
    
    Revoked keys cannot be used but are kept for audit purposes.
    """
    service = APIKeyService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    success = await service.revoke_api_key(
        user_id=user_id,
        key_id=key_id,
        reason=data.reason
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return SuccessResponse(
        success=True,
        message="API key revoked successfully"
    )


@router.post("/api-keys/{key_id}/regenerate", response_model=APIKeyCreateResponse)
async def regenerate_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Regenerate an API key.
    
    This revokes the old key and creates a new one with the same settings.
    **Important**: The new key is only shown once!
    """
    service = APIKeyService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    
    # Get old key
    old_key = await service.get_api_key(user_id=user_id, key_id=key_id)
    if not old_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    # Revoke old key
    await service.revoke_api_key(
        user_id=user_id,
        key_id=key_id,
        reason="Regenerated"
    )
    
    # Create new key with same settings
    new_key_data = APIKeyCreate(
        name=f"{old_key.name} (Regenerated)",
        description=old_key.description,
        scopes=old_key.scopes,
        allowed_ips=old_key.allowed_ips,
        expires_in_days=None  # Reset expiration
    )
    
    return await service.create_api_key(
        user_id=user_id,
        data=new_key_data
    )


@router.delete("/api-keys/{key_id}", response_model=SuccessResponse)
async def delete_api_key(
    key_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Delete an API key permanently.
    
    This action cannot be undone. Consider revoking instead.
    """
    service = APIKeyService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    
    # Verify ownership
    api_key = await service.get_api_key(user_id=user_id, key_id=key_id)
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    await service.delete_api_key(key_id)
    
    return SuccessResponse(
        success=True,
        message="API key deleted successfully"
    )