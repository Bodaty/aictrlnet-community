"""Fixed-window rate limiting for auth-sensitive endpoints (Redis-backed).

Brute-forcing credentials, refresh tokens, MFA codes, password-reset tokens, and
the 6-digit channel-link code are all high-value targets. This module provides a
small helper that counts attempts per (bucket, identifier) in a fixed time window
using Redis INCR + EXPIRE.

Design:
- Fail OPEN: if Redis is unavailable the limiter allows the request rather than
  locking every user out. Availability of the app is not sacrificed for a
  best-effort defense — the primary controls (hashing, lockout, short code TTLs)
  still apply.
- Identifier is caller-supplied (usually client IP, optionally combined with the
  account identifier) so a single attacker IP is throttled independently of a
  targeted username.
"""

import logging
from typing import Optional

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)


def client_ip(request: Optional[Request]) -> str:
    """Best-effort client IP, honoring the first X-Forwarded-For hop behind the
    Cloud Run / nginx proxy, falling back to the socket peer."""
    if request is None:
        return "unknown"
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    client = getattr(request, "client", None)
    return getattr(client, "host", None) or "unknown"


async def check_rate_limit(
    bucket: str,
    identifier: str,
    limit: int,
    window_seconds: int,
) -> bool:
    """Return True if the request is within the limit, False if it exceeds it.

    Fails OPEN (returns True) when the Redis store is unavailable.
    """
    if not identifier:
        identifier = "unknown"
    key = f"ratelimit:{bucket}:{identifier}"
    try:
        from core.cache import get_cache
        cache = await get_cache()
        count = await cache.incr_with_ttl(key, window_seconds)
    except Exception as exc:  # pragma: no cover - store unavailable
        logger.warning("Rate-limit store unavailable for %s: %s", key, exc)
        return True
    if count == 0:
        # Store down (incr_with_ttl returned its sentinel) -> fail open.
        return True
    return count <= limit


async def enforce_rate_limit(
    bucket: str,
    identifier: str,
    limit: int,
    window_seconds: int,
    detail: str = "Too many requests. Please try again later.",
) -> None:
    """Raise HTTP 429 when the (bucket, identifier) exceeds `limit` per window."""
    allowed = await check_rate_limit(bucket, identifier, limit, window_seconds)
    if not allowed:
        logger.warning("Rate limit exceeded for bucket=%s id=%s", bucket, identifier)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=detail,
            headers={"Retry-After": str(window_seconds)},
        )
