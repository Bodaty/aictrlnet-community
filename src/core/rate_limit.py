"""Fixed-window rate limiting for auth-sensitive endpoints (Redis-backed).

Brute-forcing credentials, refresh tokens, MFA codes, password-reset tokens, and
the 6-digit channel-link code are all high-value targets. This module provides a
small helper that counts attempts per (bucket, identifier) in a fixed time window
using Redis INCR + EXPIRE.

Design:
- Redis down: the limiter falls back to a per-process fixed window. It never
  locks every user out (counts stay per identifier), and attempts are still
  throttled — at worst `limit` x the number of worker processes. Deployments
  without Redis (GCP) would otherwise have no throttle at all.
- Identifier is caller-supplied (usually client IP, optionally combined with the
  account identifier) so a single attacker IP is throttled independently of a
  targeted username.
"""

import logging
import time
from typing import Dict, Optional, Tuple

from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# key -> (count, window_end) on the monotonic clock, used while Redis is down.
_memory_windows: Dict[str, Tuple[int, float]] = {}
_MEMORY_PRUNE_AT = 10_000


def _memory_incr(key: str, window_seconds: int) -> int:
    now = time.monotonic()
    if len(_memory_windows) >= _MEMORY_PRUNE_AT:
        for stale in [k for k, (_, end) in _memory_windows.items() if end <= now]:
            del _memory_windows[stale]
    count, end = _memory_windows.get(key, (0, now + window_seconds))
    if now >= end:
        count, end = 0, now + window_seconds
    _memory_windows[key] = (count + 1, end)
    return count + 1


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

    Falls back to a per-process window when the Redis store is unavailable.
    """
    if not identifier:
        identifier = "unknown"
    key = f"ratelimit:{bucket}:{identifier}"
    try:
        from core.cache import get_cache
        cache = await get_cache()
        count = await cache.incr_with_ttl(key, window_seconds)
    except Exception as exc:
        logger.warning("Rate-limit store unavailable for %s: %s", key, exc)
        count = 0
    if count == 0:
        # Store down (incr_with_ttl returned its sentinel or raised).
        count = _memory_incr(key, window_seconds)
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
