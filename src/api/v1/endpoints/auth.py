"""Authentication endpoints."""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr
import uuid
import secrets

from core.database import get_db
from core.security import (
    verify_password,
    get_password_hash,
    create_access_token,
    create_refresh_token,
    get_current_active_user,
)
from core.config import get_settings
from core.cache import get_cache
from core.tenant_context import DEFAULT_TENANT_ID, get_current_tenant_id
from models import User
from services.mfa_service import MFAService
from schemas.mfa import LoginRequest, LoginResponse, MFAVerifyRequest, MFAVerifyResponse


router = APIRouter()


class UserCreate(BaseModel):
    """User registration schema."""
    email: EmailStr
    password: str
    username: Optional[str] = None
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    username: Optional[str]
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    edition: str
    created_at: datetime
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Token response schema."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"


class TokenData(BaseModel):
    """Token data schema."""
    username: Optional[str] = None


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db),
) -> User:
    """Register a new user."""
    # Check if user already exists
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    if result.scalar_one_or_none():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    
    # Check username if provided
    if user_data.username:
        result = await db.execute(
            select(User).where(User.username == user_data.username)
        )
        if result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Create new user
    settings = get_settings()
    verification_token = create_access_token(
        data={"sub": str(uuid.uuid4()), "email": user_data.email, "type": "email_verification"},
        expires_delta=timedelta(hours=24)
    )
    
    user = User(
        id=str(uuid.uuid4()),
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        edition=settings.EDITION,
        tenant_id=DEFAULT_TENANT_ID,
        is_active=True,
        is_superuser=False,
        email_verified=False,
        email_verification_token=verification_token,
    )
    
    db.add(user)
    await db.commit()
    await db.refresh(user)
    
    return user


@router.post("/login", response_model=LoginResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    """Login and get access token - with MFA support."""
    # Find user by email or username
    result = await db.execute(
        select(User).where(
            (User.email == form_data.username) | 
            (User.username == form_data.username)
        )
    )
    user = result.scalar_one_or_none()
    
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Check if MFA is enabled
    if user.mfa_enabled:
        # Create temporary session
        session_token = secrets.token_urlsafe(32)
        session_data = {
            "user_id": str(user.id),
            "email": user.email,
            "expires_at": (datetime.utcnow() + timedelta(minutes=5)).isoformat()
        }
        
        cache = await get_cache()
        await cache.set(
            f"mfa_session:{session_token}",
            session_data,
            expire=300  # 5 minutes
        )
        
        return LoginResponse(
            mfa_required=True,
            session_token=session_token,
            expires_in=300
        )
    
    # Update last login
    user.last_login_at = datetime.utcnow()
    await db.commit()
    
    # No MFA, generate tokens directly
    settings = get_settings()
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": str(user.id),
            "email": user.email,
            "is_superuser": user.is_superuser,
            "mfa_verified": False,
            "tenant_id": user.tenant_id or get_current_tenant_id(),
        },
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={
        "sub": str(user.id),
        "tenant_id": user.tenant_id or get_current_tenant_id(),
    })
    
    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        mfa_required=False
    )


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Get current user."""
    return current_user


@router.post("/logout")
async def logout(
    current_user: User = Depends(get_current_active_user),
) -> dict:
    """Logout user (client should discard token)."""
    return {"message": "Successfully logged out"}


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    token: str
    new_password: str


@router.post("/password-reset/request")
async def request_password_reset(
    reset_request: PasswordResetRequest,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Request password reset token."""
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == reset_request.email)
    )
    user = result.scalar_one_or_none()
    
    # Always return success to prevent email enumeration
    if user:
        # Generate reset token (valid for 1 hour)
        reset_token = create_access_token(
            data={"sub": user.id, "type": "password_reset"},
            expires_delta=timedelta(hours=1)
        )
        
        # In production, send email with reset link
        # For now, we'll just log it
        print(f"Password reset token for {user.email}: {reset_token}")
        
        # Store token in database for validation
        user.password_reset_token = reset_token
        user.password_reset_expires = datetime.utcnow() + timedelta(hours=1)
        await db.commit()
    
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    reset_data: PasswordResetConfirm,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Confirm password reset with token."""
    from jose import jwt, JWTError
    
    try:
        # Decode token
        settings = get_settings()
        payload = jwt.decode(reset_data.token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        token_type = payload.get("type")
        
        if not user_id or token_type != "password_reset":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid reset token"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Find user and verify token
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or user.password_reset_token != reset_data.token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid reset token"
        )
    
    # Check if token is expired
    if user.password_reset_expires and user.password_reset_expires < datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Reset token has expired"
        )
    
    # Update password
    user.hashed_password = get_password_hash(reset_data.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    await db.commit()
    
    return {"message": "Password successfully reset"}


class ChangePasswordRequest(BaseModel):
    """Change password request schema."""
    current_password: str
    new_password: str


@router.post("/password/change")
async def change_password(
    password_data: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Change password for authenticated user."""
    # Verify current password
    if not verify_password(password_data.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    # Update password
    current_user.hashed_password = get_password_hash(password_data.new_password)
    await db.commit()
    
    return {"message": "Password successfully changed"}


@router.post("/verify-email/{token}")
async def verify_email(
    token: str,
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Verify email with token."""
    from jose import jwt, JWTError
    
    try:
        # Decode token
        settings = get_settings()
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email = payload.get("email")
        token_type = payload.get("type")
        
        if not email or token_type != "email_verification":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid verification token"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Find user by email and token
    result = await db.execute(
        select(User).where(
            (User.email == email) & 
            (User.email_verification_token == token)
        )
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid verification token"
        )
    
    # Mark email as verified
    user.email_verified = True
    user.email_verification_token = None
    await db.commit()
    
    return {"message": "Email successfully verified"}


@router.post("/resend-verification")
async def resend_verification_email(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Resend verification email."""
    if current_user.email_verified:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already verified"
        )
    
    # Generate new verification token
    verification_token = create_access_token(
        data={"sub": current_user.id, "email": current_user.email, "type": "email_verification"},
        expires_delta=timedelta(hours=24)
    )
    
    current_user.email_verification_token = verification_token
    await db.commit()
    
    # In production, send email with verification link
    print(f"Email verification token for {current_user.email}: {verification_token}")
    
    return {"message": "Verification email sent"}


class RefreshTokenRequest(BaseModel):
    """Refresh token request schema."""
    refresh_token: str


@router.post("/token/refresh", response_model=Token)
async def refresh_access_token(
    token_data: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
) -> Token:
    """Refresh access token using refresh token."""
    from jose import jwt, JWTError
    
    try:
        # Decode refresh token
        settings = get_settings()
        payload = jwt.decode(token_data.refresh_token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id = payload.get("sub")
        token_type = payload.get("type")
        
        if not user_id or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token"
        )
    
    # Find user
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive"
        )
    
    # Create new access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id}, expires_delta=access_token_expires
    )
    
    # Optionally create new refresh token (rotating refresh tokens)
    new_refresh_token = create_refresh_token(data={"sub": user.id})
    
    return Token(access_token=access_token, refresh_token=new_refresh_token)


@router.post("/mfa/verify", response_model=MFAVerifyResponse)
async def verify_mfa_login(
    request: MFAVerifyRequest,
    db: AsyncSession = Depends(get_db)
):
    """Verify MFA code during login."""
    
    # Get temporary session from login
    cache = await get_cache()
    session_data = await cache.get(f"mfa_session:{request.session_token}")
    if not session_data:
        raise HTTPException(401, "Invalid or expired session")
    
    service = MFAService(db)
    result = await service.verify_mfa_code(
        session_data["user_id"],
        request.code
    )
    
    if not result["valid"]:
        raise HTTPException(401, "Invalid MFA code")
    
    # Generate final JWT token
    settings = get_settings()
    user = await service.get_user(session_data["user_id"])
    access_token = create_access_token(
        data={
            "sub": session_data["user_id"],
            "email": user.email,
            "is_superuser": user.is_superuser,
            "mfa_verified": True
        },
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # Clear temporary session
    await cache.delete(f"mfa_session:{request.session_token}")

    # Update last login (user already fetched above)
    user.last_login_at = datetime.utcnow()
    await db.commit()
    
    return MFAVerifyResponse(
        access_token=access_token,
        token_type="bearer",
        backup_code_used=result.get("backup_code_used", False),
        remaining_backup_codes=result.get("remaining_codes")
    )