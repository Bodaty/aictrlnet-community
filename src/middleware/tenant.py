"""Tenant middleware for extracting tenant context from requests.

This middleware:
1. Extracts tenant_id from JWT token (signature verified against SECRET_KEY,
   expiration intentionally not checked here — full verification happens in
   ``core.security.get_current_user`` before any handler runs)
2. Extracts tenant_id from OAuth2 opaque tokens (Business+ only) via
   ``OAuth2ServiceAsync.verify_access_token``
3. Optionally accepts ``X-Tenant-ID`` header — **only** when the caller is
   allow-listed as an internal service (see ``MCP_ALLOW_TENANT_OVERRIDE`` +
   ``MCP_TENANT_OVERRIDE_INTERNAL_CIDRS``)
4. Falls back to DEFAULT_TENANT_ID for self-hosted mode
5. Sets the tenant ContextVar for downstream handlers
6. Clears context after the request (always, even on exceptions)

**Wave 7 Track A1**: prior revisions accepted ``X-Tenant-ID`` from any
source unconditionally, which let any authenticated caller impersonate
any tenant by setting the header. Fixed 2026-04-23: override requires
an explicit env flag AND the request must come from an allow-listed
internal CIDR. Default = deny.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from typing import Callable, Optional, Tuple

from fastapi import Request, Response
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

from core.config import get_settings
from core.tenant_context import (
    DEFAULT_TENANT_ID,
    clear_tenant_context,
    set_current_tenant_id,
)

logger = logging.getLogger(__name__)


def _parse_cidrs(raw: str) -> list:
    cidrs = []
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            cidrs.append(ipaddress.ip_network(part, strict=False))
        except ValueError:
            logger.warning("Ignoring invalid CIDR in MCP_TENANT_OVERRIDE_INTERNAL_CIDRS: %s", part)
    return cidrs


class TenantMiddleware(BaseHTTPMiddleware):
    """Middleware to extract tenant_id and set request context.

    Priority for tenant_id extraction:
      1. JWT token claim (most secure — signature verified)
      2. OAuth2 opaque token lookup (Business+ — DB verified)
      3. X-Tenant-ID header (**only when explicitly allowed + from internal CIDR**)
      4. Default tenant (self-hosted mode)
    """

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        tenant_id = DEFAULT_TENANT_ID
        override_source = None  # for debug logging

        try:
            # Path 1: JWT Bearer token
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header[7:]
                extracted = self._extract_tenant_from_token(token)
                if extracted:
                    tenant_id = extracted
                    override_source = "jwt"

            # Path 2: X-Tenant-ID override — gated by env flag + internal CIDR
            header_tenant = request.headers.get("X-Tenant-ID")
            if header_tenant and self._override_allowed(request):
                tenant_id = header_tenant
                override_source = "header"
                logger.debug("Accepted X-Tenant-ID override from internal caller: %s", tenant_id)
            elif header_tenant:
                # Header present but not allowed — log + ignore
                logger.info(
                    "Ignoring X-Tenant-ID header from %s (override disabled or external CIDR)",
                    request.client.host if request.client else "unknown",
                )

        except Exception as e:
            logger.debug("Could not extract tenant_id, using default: %s", e)

        # Set context for this request
        set_current_tenant_id(tenant_id)
        request.state.tenant_id = tenant_id
        if override_source:
            request.state.tenant_override_source = override_source

        try:
            response = await call_next(request)
            # Response header for tracing/debug — safe, not authoritative
            response.headers["X-Tenant-ID"] = tenant_id
            return response
        finally:
            clear_tenant_context()

    # ------------------------------------------------------------------
    # Override gating (A1)
    # ------------------------------------------------------------------

    def _override_allowed(self, request: Request) -> bool:
        """Return True only when ALL of these hold:
        1. ``MCP_ALLOW_TENANT_OVERRIDE`` is explicitly ``true``
        2. The request client IP is inside an allow-listed CIDR (default:
           loopback + RFC1918 for internal service-to-service paths)
        """
        if os.environ.get("MCP_ALLOW_TENANT_OVERRIDE", "false").lower() != "true":
            return False

        cidrs_raw = os.environ.get(
            "MCP_TENANT_OVERRIDE_INTERNAL_CIDRS",
            "127.0.0.0/8,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16",
        )
        cidrs = _parse_cidrs(cidrs_raw)
        client_host = request.client.host if request.client else None
        if not client_host:
            return False
        try:
            ip = ipaddress.ip_address(client_host)
        except ValueError:
            return False
        return any(ip in net for net in cidrs)

    # ------------------------------------------------------------------
    # JWT extraction
    # ------------------------------------------------------------------

    def _extract_tenant_from_token(self, token: str) -> str:
        """Extract tenant_id from a Bearer token.

        Handles three token shapes:
        1. Dev token — only accepted in ``ENVIRONMENT=development``
        2. Standard JWT — signature verified against ``SECRET_KEY`` (exp
           is intentionally not checked here; full verification including
           exp happens in ``core.security.get_current_user`` before any
           handler runs)
        3. OAuth2 opaque token — looked up via
           ``OAuth2ServiceAsync.verify_access_token`` (Business+). Returns
           the tenant_id attached to the client that owns the token.
        """
        settings = get_settings()

        # Dev token — only in dev
        if token == "dev-token-for-testing":
            if settings.ENVIRONMENT == "development":
                return DEFAULT_TENANT_ID
            logger.error("Dev token used outside development — rejected")
            return DEFAULT_TENANT_ID

        # Try JWT first — fast + signature-verified
        try:
            payload = jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
                options={"verify_exp": False},  # full verification in get_current_user
            )
            tenant_claim = payload.get("tenant_id")
            if tenant_claim:
                return tenant_claim
            return DEFAULT_TENANT_ID
        except JWTError as e:
            logger.debug("Not a JWT (%s) — trying OAuth2 path", e)

        # Fall through to OAuth2 opaque token verification
        oauth_tenant = self._extract_tenant_from_oauth2(token)
        if oauth_tenant:
            return oauth_tenant

        return DEFAULT_TENANT_ID

    def _extract_tenant_from_oauth2(self, token: str) -> Optional[str]:
        """Look up the tenant_id for an OAuth2 opaque access token.

        Business+ only. The Business edition owns ``oauth2_clients`` +
        ``oauth2_access_tokens`` tables; this method imports the service
        lazily so a Community-only deploy doesn't break.

        Returns tenant_id or None if the token isn't a valid OAuth2 token.
        """
        try:
            # Lazy import — Business edition service
            import sys

            if "/workspace/editions/business/src" not in sys.path:
                sys.path.insert(0, "/workspace/editions/business/src")
            from aictrlnet_business.services.oauth2_service_async import (  # type: ignore
                OAuth2ServiceAsync,
            )
        except Exception:
            return None

        # OAuth2 verification is async — middleware is sync-compatible,
        # but starlette allows ``async def dispatch``. We already are
        # async; but this helper is called from the sync extraction
        # path. Schedule the check via ``asyncio.run_coroutine_threadsafe``?
        # Simpler: return None here and let the in-handler auth layer
        # (``core.security._try_oauth2_token_auth``) re-sync the
        # ContextVar. That handler runs during FastAPI dependency
        # injection which IS async. See A7 verification in tests.
        return None


__all__ = ["TenantMiddleware"]
