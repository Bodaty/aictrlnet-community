"""Security utilities for authentication and authorization."""

from datetime import datetime, timedelta
from typing import Optional, Union, TYPE_CHECKING
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from .config import get_settings
from .database import get_db

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
    token: Optional[str] = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> "User":
    """Get current user from JWT token."""
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
        raise credentials_exception
    
    # Fetch user from database
    result = await db.execute(
        select(User).where(User.id == user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
    return user


async def get_current_active_user(current_user: "User" = Depends(get_current_user)) -> "User":
    """Get current active user.
    
    This function handles both User objects and special development cases
    to prevent UUID conversion errors in development.
    """
    # Handle the dev token case where current_user might have special attributes
    if hasattr(current_user, 'is_active'):
        if not current_user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
    # If no is_active attribute (shouldn't happen), assume active
    return current_user


class RoleChecker:
    """Dependency to check user roles."""
    
    def __init__(self, allowed_roles: list[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, user: dict = Depends(get_current_active_user)) -> dict:
        """Check if user has required role."""
        user_roles = user.get("roles", [])
        if not any(role in self.allowed_roles for role in user_roles):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions",
            )
        return user


# No need for separate dev function - it's integrated above


def verify_token(token: str) -> Optional[dict]:
    """Verify a token and return user info without raising exceptions."""
    from core.tenant_context import DEFAULT_TENANT_ID

    settings = get_settings()

    # DEV_ONLY_START
    # Development token for fast local development - only enabled in development environment
    # Check for dev token first, but ONLY in development environment
    if token == "dev-token-for-testing" and settings.ENVIRONMENT == "development":
        return {
            "id": "dev-user-123",
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

        # For development, return mock user
        # In production, fetch from database
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