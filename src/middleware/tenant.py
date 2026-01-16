"""Tenant middleware for extracting tenant context from requests.

This middleware:
1. Extracts tenant_id from JWT token (if present)
2. Falls back to X-Tenant-ID header (for service-to-service calls)
3. Sets the tenant context for the request
4. Adds X-Tenant-ID header to response for debugging
"""

from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from jose import jwt, JWTError
import logging

from core.config import get_settings
from core.tenant_context import (
    set_current_tenant_id,
    clear_tenant_context,
    DEFAULT_TENANT_ID,
)

logger = logging.getLogger(__name__)


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware to extract tenant_id from JWT and set request context.

    Priority for tenant_id extraction:
    1. JWT token claim (most secure)
    2. X-Tenant-ID header (for service-to-service)
    3. Default tenant (self-hosted mode)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        tenant_id = DEFAULT_TENANT_ID

        try:
            # Try to extract from JWT token first
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                extracted = self._extract_tenant_from_token(token)
                if extracted:
                    tenant_id = extracted

            # Allow X-Tenant-ID header override (for service-to-service calls)
            # Only if the request is from internal services
            header_tenant = request.headers.get("X-Tenant-ID")
            if header_tenant:
                # In production, you might want to validate this is from internal network
                tenant_id = header_tenant
                logger.debug(f"Using X-Tenant-ID header: {tenant_id}")

        except Exception as e:
            logger.debug(f"Could not extract tenant_id, using default: {e}")

        # Set context for this request
        set_current_tenant_id(tenant_id)

        # Also store in request.state for easy access in endpoints
        request.state.tenant_id = tenant_id

        try:
            response = await call_next(request)

            # Add tenant_id to response headers for debugging/tracing
            response.headers["X-Tenant-ID"] = tenant_id

            return response
        finally:
            # Always clear context after request
            clear_tenant_context()

    def _extract_tenant_from_token(self, token: str) -> str:
        """Extract tenant_id from JWT token.

        Args:
            token: The JWT token string

        Returns:
            The tenant_id claim from the token, or DEFAULT_TENANT_ID
        """
        settings = get_settings()

        # Handle dev token in development mode
        if token == "dev-token-for-testing":
            if settings.ENVIRONMENT == "development":
                return DEFAULT_TENANT_ID
            else:
                logger.warning("Dev token used in non-development environment")
                return DEFAULT_TENANT_ID

        try:
            # Decode without verification to get claims
            # Full verification happens in get_current_user dependency
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
                options={"verify_exp": False}  # Don't fail on expired here
            )
            return payload.get("tenant_id", DEFAULT_TENANT_ID)
        except JWTError as e:
            logger.debug(f"Could not decode JWT for tenant: {e}")
            return DEFAULT_TENANT_ID
        except Exception as e:
            logger.debug(f"Unexpected error extracting tenant from token: {e}")
            return DEFAULT_TENANT_ID


__all__ = ["TenantMiddleware"]
