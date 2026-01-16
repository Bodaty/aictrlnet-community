"""MFA API endpoints."""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.mfa_service import MFAService
from schemas.mfa import (
    MFAInitRequest,
    MFACompleteRequest,
    MFAVerifyRequest,
    MFADisableRequest,
    MFAResetRequest,
    MFAStatusResponse,
    MFAInitResponse,
    MFACompleteResponse,
    MFAVerifyResponse,
    MFADisableResponse,
    MFARecoveryCodesResponse,
)

router = APIRouter(prefix="/users", tags=["mfa"])


async def is_admin(user, db: AsyncSession) -> bool:
    """Check if user is admin."""
    # Simple check - in real implementation would check roles/permissions
    # Handle both dict and User object
    if hasattr(user, 'is_superuser'):
        return user.is_superuser
    return user.get("is_superuser", False) if isinstance(user, dict) else False




@router.get("/{user_id}/mfa", response_model=MFAStatusResponse)
async def get_mfa_status(
    user_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get MFA status for a user."""

    # Check permissions
    if user_id != "me" and user_id != current_user.id:
        # Admin check for Business/Enterprise
        if not await is_admin(current_user, db):
            raise HTTPException(403, "Not authorized")
    
    if user_id == "me":
        user_id = current_user.id
    
    service = MFAService(db)
    return await service.get_mfa_status(user_id)


@router.get("/me/mfa", response_model=MFAStatusResponse)
async def get_my_mfa_status(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get MFA status for current user."""
    service = MFAService(db)
    return await service.get_mfa_status(current_user.id)


@router.post("/{user_id}/mfa/init", response_model=MFAInitResponse)
async def init_mfa(
    user_id: str,
    request: MFAInitRequest,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Initialize MFA enrollment."""
    
    # Same permission check
    if user_id != "me" and user_id != current_user.id:
        if not await is_admin(current_user, db):
            raise HTTPException(403, "Not authorized")
    
    if user_id == "me":
        user_id = current_user.id
    
    service = MFAService(db)
    return await service.init_mfa_enrollment(user_id, request.device_name)


@router.post("/{user_id}/mfa/complete", response_model=MFACompleteResponse)
async def complete_mfa(
    user_id: str,
    request: MFACompleteRequest,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Complete MFA enrollment."""
    
    if user_id != "me" and user_id != current_user.id:
        if not await is_admin(current_user, db):
            raise HTTPException(403, "Not authorized")
    
    if user_id == "me":
        user_id = current_user.id
    
    service = MFAService(db)
    return await service.complete_mfa_enrollment(
        user_id,
        request.verification_code,
        request.device_name
    )


@router.delete("/{user_id}/mfa", response_model=MFADisableResponse)
async def disable_mfa(
    user_id: str,
    request: MFADisableRequest,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Disable MFA."""
    
    admin_override = False
    
    if user_id != "me" and user_id != current_user.id:
        if await is_admin(current_user, db):
            admin_override = True
        else:
            raise HTTPException(403, "Not authorized")
    
    if user_id == "me":
        user_id = current_user.id
    
    service = MFAService(db)
    return await service.disable_mfa(
        user_id,
        request.password if not admin_override else None,
        admin_override,
        current_user.id if admin_override else None
    )


@router.post("/{user_id}/mfa/reset")
async def reset_mfa(
    user_id: str,
    request: MFAResetRequest,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Reset MFA (admin only)."""
    
    # Admin only
    if not await is_admin(current_user, db):
        raise HTTPException(403, "Admin access required")
    
    service = MFAService(db)
    return await service.disable_mfa(
        user_id,
        None,
        admin_override=True,
        performed_by=current_user.id
    )


@router.post("/{user_id}/mfa/verify")
async def verify_mfa(
    user_id: str,
    request: MFAVerifyRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify MFA code (for testing/admin purposes)."""
    
    # This endpoint is primarily for testing
    # Real verification happens during login
    service = MFAService(db)
    
    # Extract user_id from session token
    # This is a simplified version - real implementation would validate session
    result = await service.verify_mfa_code(user_id, request.code)
    
    if not result["valid"]:
        raise HTTPException(401, "Invalid MFA code")
    
    return {"message": "MFA code verified successfully"}


@router.get("/{user_id}/mfa/recovery-codes", response_model=MFARecoveryCodesResponse)
async def get_recovery_codes(
    user_id: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current recovery codes (shows count only for security)."""
    
    # Check permissions
    if user_id != "me" and user_id != current_user.id:
        if not await is_admin(current_user, db):
            raise HTTPException(403, "Not authorized")
    
    if user_id == "me":
        user_id = current_user.id
    
    # For security, we don't show actual codes, just count
    service = MFAService(db)
    user = await service.get_user(user_id)
    
    if not user.mfa_enabled:
        raise HTTPException(400, "MFA not enabled")
    
    # Return placeholder response
    return {
        "recovery_codes": ["********"] * 8,  # Don't show real codes
        "generated_at": user.mfa_enrolled_at
    }


@router.post("/{user_id}/mfa/recovery-codes", response_model=MFARecoveryCodesResponse)
async def regenerate_recovery_codes(
    user_id: str,
    password: str,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Regenerate recovery codes."""
    
    # Only user themselves can regenerate codes
    if user_id != "me" and user_id != current_user.id:
        raise HTTPException(403, "Can only regenerate your own recovery codes")
    
    if user_id == "me":
        user_id = current_user.id
    
    service = MFAService(db)
    return await service.regenerate_backup_codes(user_id, password)