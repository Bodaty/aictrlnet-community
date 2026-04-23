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
from api.v1.endpoints._auth_helpers import get_safe_user_id
from mcp_server.scopes import ALL_SCOPES, describe_scope, scope_action, scope_group

router = APIRouter()


@router.get("/api-keys/available-scopes")
async def list_available_scopes():
    """Return every valid API-key scope with its UI metadata.

    Public endpoint — no auth required. Consumed by the HitLai API-key
    picker + the OAuth consent screen so both render the same 71-scope
    taxonomy without hardcoding.
    """
    scopes = [
        {
            "scope": s,
            "description": describe_scope(s),
            "group": scope_group(s),
            "action": scope_action(s),
        }
        for s in sorted(ALL_SCOPES)
    ]
    return {"scopes": scopes, "total": len(scopes)}


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
    user_id = get_safe_user_id(current_user)
    return await service.list_user_api_keys(
        user_id=user_id,
        include_inactive=include_inactive
    )


@router.post("/api-keys", response_model=APIKeyCreateResponse)
async def create_api_key(
    data: APIKeyCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user),
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
    user_id = get_safe_user_id(current_user)
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
    user_id = get_safe_user_id(current_user)
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
    user_id = get_safe_user_id(current_user)
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
    user_id = get_safe_user_id(current_user)
    success = await service.revoke_api_key(
        user_id=user_id,
        key_id=key_id,
        reason=data.reason
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="API key not found")
    
    return SuccessResponse(
        success=True,
        data={"key_id": key_id, "revoked": True},
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
    user_id = get_safe_user_id(current_user)
    
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
    
    # Filter any legacy scopes that were valid at creation time but dropped
    # at request-time in Wave 2 Phase B. Regenerating a legacy-scoped key
    # otherwise fails the stricter validator. See also docs/MCP_V11_CLAIM_AUDIT.md.
    from mcp_server.scopes import ALL_SCOPES
    filtered_scopes = [s for s in (old_key.scopes or []) if s in ALL_SCOPES]

    # Create new key with same settings
    new_key_data = APIKeyCreate(
        name=f"{old_key.name} (Regenerated)",
        description=old_key.description,
        scopes=filtered_scopes,
        allowed_ips=old_key.allowed_ips,
        rate_limit_per_tool=getattr(old_key, "rate_limit_per_tool", None) or None,
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
    user_id = get_safe_user_id(current_user)
    
    # Verify ownership
    api_key = await service.get_api_key(user_id=user_id, key_id=key_id)
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    await service.delete_api_key(key_id)
    
    return SuccessResponse(
        success=True,
        data={"key_id": key_id, "deleted": True},
        message="API key deleted successfully"
    )