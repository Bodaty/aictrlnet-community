"""Authentication endpoints."""

from datetime import datetime, timedelta
from typing import Optional, Dict, Tuple
from collections import defaultdict
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel, EmailStr, Field
import uuid
import secrets
import time

# Simple in-memory login rate limiter
# Tracks failed attempts per IP and per username
_login_attempts: Dict[str, Tuple[int, float]] = {}  # key -> (count, first_attempt_time)
_RATE_LIMIT_WINDOW = 300  # 5 minutes
_RATE_LIMIT_MAX_ATTEMPTS = 10  # max failures before lockout


def _check_login_rate_limit(key: str):
    """Check if login attempts for this key are rate limited."""
    now = time.time()
    if key in _login_attempts:
        count, first_time = _login_attempts[key]
        if now - first_time > _RATE_LIMIT_WINDOW:
            # Window expired, reset
            del _login_attempts[key]
        elif count >= _RATE_LIMIT_MAX_ATTEMPTS:
            remaining = int(_RATE_LIMIT_WINDOW - (now - first_time))
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Too many login attempts. Try again in {remaining} seconds.",
                headers={"Retry-After": str(remaining)}
            )


def _record_failed_login(key: str):
    """Record a failed login attempt."""
    now = time.time()
    if key in _login_attempts:
        count, first_time = _login_attempts[key]
        if now - first_time > _RATE_LIMIT_WINDOW:
            _login_attempts[key] = (1, now)
        else:
            _login_attempts[key] = (count + 1, first_time)
    else:
        _login_attempts[key] = (1, now)


def _clear_login_attempts(key: str):
    """Clear failed attempts on successful login."""
    _login_attempts.pop(key, None)

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
import logging

from models import User
from models.subscription import Subscription, SubscriptionPlan, SubscriptionStatus, BillingPeriod
from services.mfa_service import MFAService
from schemas.mfa import LoginRequest, LoginResponse, MFAVerifyRequest, MFAVerifyResponse
from schemas.user import UserCreate

logger = logging.getLogger(__name__)


router = APIRouter()


class UserResponse(BaseModel):
    """User response schema."""
    id: str
    email: str
    username: Optional[str]
    full_name: Optional[str]
    is_active: bool
    is_superuser: bool
    edition: str
    trial_active: Optional[bool] = None
    trial_days_remaining: Optional[int] = None
    trial_end: Optional[str] = None
    subscription_mismatch: Optional[bool] = None
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
    
    now = datetime.utcnow()
    trial_end = now + timedelta(days=settings.TRIAL_DAYS)
    user_id = str(uuid.uuid4())

    user = User(
        id=user_id,
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=get_password_hash(user_data.password),
        edition="business",  # All new users start with 14-day Business trial
        tenant_id=DEFAULT_TENANT_ID,
        is_active=True,
        is_superuser=False,
        email_verified=False,
        email_verification_token=verification_token,
    )

    db.add(user)

    # Auto-create 14-day Business trial subscription (no credit card required)
    try:
        plan_result = await db.execute(
            select(SubscriptionPlan).where(SubscriptionPlan.name == "business_starter")
        )
        plan = plan_result.scalar_one_or_none()
        if plan:
            trial_subscription = Subscription(
                id=str(uuid.uuid4()),
                user_id=user_id,
                tenant_id=DEFAULT_TENANT_ID,
                plan_id=plan.id,
                status=SubscriptionStatus.TRIALING,
                billing_period=BillingPeriod.MONTHLY,
                started_at=now,
                current_period_start=now,
                current_period_end=trial_end,
                trial_end=trial_end,
            )
            db.add(trial_subscription)
        else:
            logger.warning("Could not create trial subscription — business_starter plan not seeded")
    except Exception:
        logger.warning("Could not create trial subscription — plan lookup failed")

    await db.commit()
    await db.refresh(user)

    return user


@router.post("/login", response_model=LoginResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> LoginResponse:
    """Login and get access token - with MFA support."""
    # Rate limit check — by username and by IP
    client_ip = request.client.host if request.client else "unknown"
    _check_login_rate_limit(f"ip:{client_ip}")
    _check_login_rate_limit(f"user:{form_data.username}")

    # Find user by email or username
    result = await db.execute(
        select(User).where(
            (User.email == form_data.username) |
            (User.username == form_data.username)
        )
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(form_data.password, user.hashed_password):
        _record_failed_login(f"ip:{client_ip}")
        _record_failed_login(f"user:{form_data.username}")
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

    # Check tenant MFA policy — require enrollment if tenant mandates MFA
    if user.tenant_id and not user.mfa_enabled:
        try:
            from models.tenant import Tenant
            tenant = await db.get(Tenant, user.tenant_id)
            if tenant and getattr(tenant, 'mfa_required', False):
                # Check grace period
                mfa_grace_days = getattr(tenant, 'mfa_grace_period_days', 7)
                user_created = getattr(user, 'created_at', None)
                grace_expired = True
                if user_created and mfa_grace_days > 0:
                    from datetime import timezone
                    created = user_created if user_created.tzinfo else user_created.replace(tzinfo=timezone.utc)
                    grace_expired = datetime.now(timezone.utc) > created + timedelta(days=mfa_grace_days)
                if grace_expired:
                    session_token = secrets.token_urlsafe(32)
                    session_data = {
                        "user_id": str(user.id),
                        "email": user.email,
                        "expires_at": (datetime.utcnow() + timedelta(minutes=10)).isoformat()
                    }
                    cache = await get_cache()
                    await cache.set(f"mfa_enrollment:{session_token}", session_data, expire=600)
                    return LoginResponse(
                        mfa_required=True,
                        mfa_enrollment_required=True,
                        session_token=session_token,
                        expires_in=600
                    )
        except ImportError:
            logger.debug("Tenant model not available for MFA policy check")
        except Exception as e:
            logger.error(f"Error checking tenant MFA policy: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error verifying security policy"
            )

    # Check if MFA is enabled
    if user.mfa_enabled:
        # Create temporary session
        session_token = secrets.token_urlsafe(32)
        session_data = {
            "user_id": str(user.id),
            "email": user.email,
            "expires_at": (datetime.utcnow() + timedelta(minutes=10)).isoformat()
        }

        cache = await get_cache()
        await cache.set(
            f"mfa_session:{session_token}",
            session_data,
            expire=600  # 10 minutes — enough time to open authenticator app
        )

        return LoginResponse(
            mfa_required=True,
            session_token=session_token,
            expires_in=600
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
            "token_version": user.token_version or 0,
        },
        expires_delta=access_token_expires
    )
    refresh_token = create_refresh_token(data={
        "sub": str(user.id),
        "tenant_id": user.tenant_id or get_current_tenant_id(),
        "token_version": user.token_version or 0,
    })
    
    # Clear rate limit on successful login
    _clear_login_attempts(f"ip:{client_ip}")
    _clear_login_attempts(f"user:{form_data.username}")

    return LoginResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        mfa_required=False
    )


@router.get("/me", response_model=UserResponse)
async def read_users_me(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db),
):
    """Get current user with trial status."""
    # Check for trial subscription
    result = await db.execute(
        select(Subscription).where(
            Subscription.user_id == current_user.id,
            Subscription.status.in_([SubscriptionStatus.TRIALING, SubscriptionStatus.EXPIRED])
        )
    )
    trial_sub = result.scalar_one_or_none()

    trial_active = False
    trial_days = None
    trial_end_str = None

    if trial_sub and trial_sub.trial_end:
        now = datetime.utcnow()
        trial_active = trial_sub.status == SubscriptionStatus.TRIALING and trial_sub.trial_end > now
        trial_days = max(0, (trial_sub.trial_end - now).days) if trial_active else 0
        trial_end_str = trial_sub.trial_end.isoformat()

    # Check for edition/subscription mismatch
    subscription_mismatch = None
    if current_user.edition in ("business", "enterprise"):
        # Check if there's an active subscription backing this edition
        active_sub_result = await db.execute(
            select(Subscription).where(
                Subscription.user_id == current_user.id,
                Subscription.status.in_([SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIALING])
            )
        )
        active_sub = active_sub_result.scalar_one_or_none()
        if not active_sub:
            subscription_mismatch = True

    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        is_active=current_user.is_active,
        is_superuser=current_user.is_superuser,
        edition=current_user.edition,
        trial_active=trial_active,
        trial_days_remaining=trial_days,
        trial_end=trial_end_str,
        subscription_mismatch=subscription_mismatch,
        created_at=current_user.created_at,
    )


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
        logger.debug(f"Password reset token generated for {user.email}")
        
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
    
    # Update password and invalidate all existing tokens
    user.hashed_password = get_password_hash(reset_data.new_password)
    user.token_version = (user.token_version or 0) + 1
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
    
    # Update password and invalidate existing tokens
    current_user.hashed_password = get_password_hash(password_data.new_password)
    current_user.token_version = (current_user.token_version or 0) + 1
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
    logger.debug(f"Email verification token generated for {current_user.email}")
    
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

    # Validate token_version — reject old tokens after rotation
    token_version = payload.get("token_version", 0)
    if token_version != (user.token_version or 0):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has been revoked"
        )

    # Increment token_version to invalidate the old refresh token
    user.token_version = (user.token_version or 0) + 1
    await db.commit()

    # Create new access token
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={
            "sub": user.id,
            "tenant_id": user.tenant_id or payload.get("tenant_id"),
            "token_version": user.token_version,
        },
        expires_delta=access_token_expires
    )

    # Create new refresh token with updated version
    new_refresh_token = create_refresh_token(data={
        "sub": user.id,
        "tenant_id": user.tenant_id or payload.get("tenant_id"),
        "token_version": user.token_version,
    })

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
        raise HTTPException(401, "MFA session expired. Please log in again to get a new verification code.")
    
    service = MFAService(db)
    result = await service.verify_mfa_code(
        session_data["user_id"],
        request.code
    )
    
    if not result["valid"]:
        raise HTTPException(401, "Invalid MFA code. Check that the code is current and try again.")
    
    # Generate final JWT token
    settings = get_settings()
    user = await service.get_user(session_data["user_id"])
    access_token = create_access_token(
        data={
            "sub": session_data["user_id"],
            "email": user.email,
            "is_superuser": user.is_superuser,
            "mfa_verified": True,
            "tenant_id": user.tenant_id or get_current_tenant_id(),
            "token_version": user.token_version or 0,
        },
        expires_delta=timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    refresh_token = create_refresh_token(data={
        "sub": session_data["user_id"],
        "tenant_id": user.tenant_id or get_current_tenant_id(),
        "token_version": user.token_version or 0,
    })

    # Clear temporary session
    await cache.delete(f"mfa_session:{request.session_token}")

    # Update last login (user already fetched above)
    user.last_login_at = datetime.utcnow()
    await db.commit()

    return MFAVerifyResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        backup_code_used=result.get("backup_code_used", False),
        remaining_backup_codes=result.get("remaining_codes")
    )