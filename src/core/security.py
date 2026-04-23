"""Security utilities for authentication and authorization."""

from datetime import datetime, timedelta
from typing import Optional, Union, TYPE_CHECKING
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .config import get_settings
from .database import get_db
from .user_utils import get_safe_attr

# Avoid circular import
if TYPE_CHECKING:
    from models.user import User


# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme - use optional to allow dev token in header
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token with tenant_id claim.

    Args:
        data: Token payload. Should include 'sub' (user_id) and optionally 'tenant_id'.
              If tenant_id is not provided, defaults to DEFAULT_TENANT_ID from tenant_context.
        expires_delta: Optional expiration time delta.

    Returns:
        Encoded JWT token string with tenant_id claim.
    """
    from core.tenant_context import DEFAULT_TENANT_ID

    settings = get_settings()
    to_encode = data.copy()

    # Ensure tenant_id is always present in token
    if "tenant_id" not in to_encode:
        to_encode["tenant_id"] = DEFAULT_TENANT_ID

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token with longer expiration and tenant_id claim."""
    from core.tenant_context import DEFAULT_TENANT_ID

    settings = get_settings()
    to_encode = data.copy()

    # Ensure tenant_id is always present in refresh token
    if "tenant_id" not in to_encode:
        to_encode["tenant_id"] = DEFAULT_TENANT_ID

    expire = datetime.utcnow() + timedelta(days=30)  # 30 days refresh token
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt


async def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> "User":
    """Get current user from JWT token or API key."""
    # Import here to avoid circular import
    from models.user import User

    settings = get_settings()

    # DEV_ONLY_START
    # Development token for fast local development - only enabled in development environment
    # Check for dev token first, but ONLY in development environment
    if token == "dev-token-for-testing" and settings.ENVIRONMENT == "development":
        # Check if dev user exists in database
        result = await db.execute(
            select(User).where(User.email == "dev@aictrlnet.com")
        )
        dev_user = result.scalar_one_or_none()

        if not dev_user:
            # Create dev user if it doesn't exist
            # Use a valid UUID for the dev user to prevent UUID conversion errors
            from core.tenant_context import DEFAULT_TENANT_ID
            dev_user = User(
                id="00000000-0000-0000-0000-000000000001",  # Valid UUID for dev user
                email="dev@aictrlnet.com",
                username="devuser",
                hashed_password=get_password_hash("dev-password"),
                full_name="Development User",
                tenant_id=DEFAULT_TENANT_ID,
                is_active=True,
                is_superuser=True,
                edition="enterprise",
            )
            db.add(dev_user)
            await db.commit()
            await db.refresh(dev_user)

        return dev_user
    # DEV_ONLY_END

    if not token:
        # Fall back to X-API-Key header before rejecting
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            user = await _try_api_key_auth(api_key_header, request, db)
            if user:
                return user

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        # JWT decode failed — try OAuth2 access token (Business+ feature).
        # Pass request so _try_oauth2 can set tenant ContextVar + state.
        oauth2_user = await _try_oauth2_token_auth(token, db, request=request)
        if oauth2_user:
            return oauth2_user
        raise credentials_exception
    
    # Fetch user from database
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    # Validate token_version — reject tokens issued before rotation/password change
    token_version = payload.get("token_version")
    if token_version is not None and token_version != (user.token_version or 0):
        raise credentials_exception

    return user


async def get_current_active_user(current_user: "User" = Depends(get_current_user)) -> "User":
    """Get current active user.
    
    This function handles both User objects and special development cases
    to prevent UUID conversion errors in development.
    """
    if not get_safe_attr(current_user, 'is_active', True):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


class RoleChecker:
    """Dependency to check user roles."""

    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles

    async def __call__(self, user=Depends(get_current_user)):
        """Check if user has required role."""
        user_roles = get_safe_attr(user, "roles", []) or []
        if isinstance(user_roles, str):
            user_roles = [user_roles]
        if not any(role in self.allowed_roles for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )
        return user


# No need for separate dev function - it's integrated above


async def verify_token(token: str, db=None) -> Optional[dict]:
    """Verify a token and return user info without raising exceptions.

    Args:
        token: JWT token to verify
        db: Optional AsyncSession for database lookup. If provided, verifies
            user exists and is active in database.
    """
    from core.tenant_context import DEFAULT_TENANT_ID

    settings = get_settings()

    # DEV_ONLY_START
    # Development token for fast local development - only enabled in development environment
    # Check for dev token first, but ONLY in development environment
    if token == "dev-token-for-testing" and settings.ENVIRONMENT == "development":
        return {
            "id": "00000000-0000-0000-0000-000000000001",
            "username": "dev_user",
            "email": "dev@aictrlnet.com",
            "is_active": True,
            "edition": "enterprise",
            "role": "admin",
            "roles": ["user", "admin"],
            "tenant_id": DEFAULT_TENANT_ID,
        }
    # DEV_ONLY_END

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            return None

        # If database session provided, verify user exists and is active
        if db:
            from sqlalchemy import select
            from models.user import User

            result = await db.execute(select(User).where(User.id == user_id))
            user = result.scalar_one_or_none()

            if not user:
                return None  # User not found in database

            if not user.is_active:
                return None  # User is deactivated

            # Return real user data from database
            return {
                "id": str(user.id),
                "username": user.username,
                "email": user.email,
                "is_active": user.is_active,
                "edition": settings.EDITION,
                "role": getattr(user, 'role', 'user'),
                "roles": getattr(user, 'roles', ['user']),
                "tenant_id": getattr(user, 'tenant_id', DEFAULT_TENANT_ID),
            }

        # Fallback: return data from JWT payload (no DB verification)
        return {
            "id": user_id,
            "username": payload.get("username", "user"),
            "email": payload.get("email", "user@example.com"),
            "is_active": True,
            "edition": settings.EDITION,
            "role": "user",
            "roles": ["user"],
            "tenant_id": payload.get("tenant_id", DEFAULT_TENANT_ID),
        }
    except JWTError:
        return None


async def _try_api_key_auth(
    api_key_value: str, request: Request, db: AsyncSession
) -> Optional["User"]:
    """Attempt authentication via API key. Returns User or None.

    Wave 7 A7/A11: also re-syncs the tenant ContextVar to the
    authenticated user's tenant_id. The TenantMiddleware couldn't do
    this for API-key-only requests (no JWT to decode). Setting the
    ContextVar here ensures the plan gate and downstream handlers see
    the correct tenant, regardless of what ``X-Tenant-ID`` did (or
    didn't) set.
    """
    try:
        from core.tenant_context import set_current_tenant_id
        from services.api_key_service import APIKeyService

        svc = APIKeyService(db)
        ip_address = request.client.host if request.client else None
        user, key = await svc.verify_api_key(api_key_value, ip_address=ip_address)
        if user and key:
            request.state.api_key = key
            # Re-sync tenant ContextVar from the authenticated user.
            # The middleware's default-fallback or a prior (ignored)
            # X-Tenant-ID header attempt is superseded here.
            user_tenant = getattr(user, "tenant_id", None)
            if user_tenant:
                set_current_tenant_id(str(user_tenant))
                request.state.tenant_id = str(user_tenant)
            return user
    except Exception:
        pass
    return None


async def _try_oauth2_token_auth(
    token: str, db: AsyncSession, request: Optional[Request] = None
) -> Optional["User"]:
    """Attempt authentication via OAuth2 access token (Business+ feature).

    OAuth2 tokens are issued by POST /api/v1/oauth2/token
    (client_credentials grant). They're opaque tokens stored in
    ``oauth2_access_tokens``, not JWTs.

    Wave 7 A7: resolves the token's parent ``OAuth2Client.tenant_id``
    and sets the tenant ContextVar — the middleware couldn't do this
    because OAuth2 tokens aren't JWTs (and the middleware's sync-path
    helper returns None for this case by design).
    """
    try:
        import sys
        if "/workspace/editions/business/src" not in sys.path:
            sys.path.insert(0, "/workspace/editions/business/src")

        from aictrlnet_business.models.oauth2 import OAuth2Client  # type: ignore
        from aictrlnet_business.services.oauth2_service_async import (  # type: ignore
            OAuth2ServiceAsync,
        )
        from core.tenant_context import set_current_tenant_id
        from models.user import User

        svc = OAuth2ServiceAsync(db)
        access_token = await svc.verify_access_token(token)
        if access_token and not access_token.revoked:
            # Resolve tenant from the OAuth2 client (authoritative)
            client_row = (
                await db.execute(
                    select(OAuth2Client).where(
                        OAuth2Client.client_id == access_token.client_id
                    )
                )
            ).scalar_one_or_none()

            result = await db.execute(
                select(User).where(User.id == access_token.user_id)
            )
            user = result.scalar_one_or_none()
            if user and user.is_active:
                # Prefer the OAuth2 client's tenant (this is what the
                # subscription is scoped to). Fall back to user.tenant_id
                # if client lookup failed.
                resolved_tenant = None
                if client_row and client_row.tenant_id:
                    resolved_tenant = str(client_row.tenant_id)
                elif getattr(user, "tenant_id", None):
                    resolved_tenant = str(user.tenant_id)
                if resolved_tenant:
                    set_current_tenant_id(resolved_tenant)
                    if request is not None:
                        request.state.tenant_id = resolved_tenant
                return user
    except ImportError:
        pass
    except Exception:
        pass
    return None