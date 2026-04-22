"""Per-tool, per-principal rate buckets for MCP tool calls.

Backed by Redis (``INCR`` + ``EXPIRE``) so bucket state survives process
restarts and scales across workers. When Redis is unavailable, falls
back to an in-memory counter (logged once per process — never blocks
the request).

Principal = one of:
- ``apikey:<id>`` for API-key auth
- ``oauth:<client_id>`` for OAuth2 client-credentials auth
- ``user:<id>`` for JWT auth (single-user rate as a safety net)

Bucket limits come from ``rate_limit_per_tool`` JSON column on
``api_keys`` / ``oauth2_clients``. Missing config falls back to the
legacy per-minute / per-day columns if present, otherwise the hard-coded
``DEFAULT_LIMITS`` below.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

# Applied only when the principal has no per-tool config AND no legacy
# per-minute / per-day config. Generous enough that well-behaved clients
# never hit these; low enough to cap runaway loops.
DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "per_minute": {"*": 120},    # 2 per second per tool default
    "per_day":    {"*": 10_000},
}

WINDOWS = {"per_minute": 60, "per_day": 86_400}


class RateError(Exception):
    def __init__(
        self,
        tool_name: str,
        window: str,
        limit: int,
        retry_after: int,
    ):
        self.tool_name = tool_name
        self.window = window
        self.limit = limit
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit for '{tool_name}' exceeded ({limit} {window}); "
            f"retry in {retry_after}s"
        )

    def to_payload(self) -> dict:
        return {
            "error": "rate_limit_exceeded",
            "tool": self.tool_name,
            "window": self.window,
            "limit": self.limit,
            "retry_after_seconds": self.retry_after,
        }


# ---------------------------------------------------------------------------
# Redis backend
# ---------------------------------------------------------------------------

_redis_client = None
_redis_checked = False
_redis_lock = threading.Lock()


def _get_redis():
    """Lazy redis client. Returns None if redis unavailable (logs once)."""
    global _redis_client, _redis_checked
    if _redis_checked:
        return _redis_client
    with _redis_lock:
        if _redis_checked:
            return _redis_client
        _redis_checked = True
        try:
            import redis.asyncio as redis_async  # type: ignore

            url = os.environ.get("REDIS_URL", "redis://redis:6379/0")
            _redis_client = redis_async.from_url(url, decode_responses=True)
        except Exception as e:
            logger.warning(
                "Redis unavailable for MCP rate buckets (%s); falling "
                "back to in-memory counter",
                e,
            )
            _redis_client = None
    return _redis_client


# ---------------------------------------------------------------------------
# In-memory fallback
# ---------------------------------------------------------------------------

_memory_state: dict[str, tuple[int, float]] = {}
_memory_lock = threading.Lock()


def _memory_incr(key: str, ttl_seconds: int) -> int:
    now = time.time()
    with _memory_lock:
        count, expires_at = _memory_state.get(key, (0, now + ttl_seconds))
        if now >= expires_at:
            count, expires_at = 0, now + ttl_seconds
        count += 1
        _memory_state[key] = (count, expires_at)
        return count


def _principal_from(api_key, oauth_client_id: Optional[str], user_id: Optional[str]) -> str:
    if api_key is not None:
        return f"apikey:{getattr(api_key, 'id', 'unknown')}"
    if oauth_client_id:
        return f"oauth:{oauth_client_id}"
    return f"user:{user_id or 'anonymous'}"


def _limits_for(principal: str, api_key, tool_name: str) -> dict[str, int]:
    """Resolve per-tool per-window limits for this principal."""
    out: dict[str, int] = {}
    per_tool = None
    if api_key is not None:
        per_tool = getattr(api_key, "rate_limit_per_tool", None) or {}
        if isinstance(per_tool, dict):
            entry = per_tool.get(tool_name) or per_tool.get("*") or {}
            for w in WINDOWS:
                if isinstance(entry, dict) and entry.get(w):
                    out[w] = int(entry[w])
    # Legacy per-minute / per-day columns
    if api_key is not None:
        legacy_min = getattr(api_key, "rate_limit_per_minute", None)
        legacy_day = getattr(api_key, "rate_limit_per_day", None)
        if "per_minute" not in out and legacy_min:
            out["per_minute"] = int(legacy_min)
        if "per_day" not in out and legacy_day:
            out["per_day"] = int(legacy_day)
    # Defaults
    for w, limit_map in DEFAULT_LIMITS.items():
        if w not in out:
            out[w] = limit_map.get(tool_name, limit_map.get("*", 0))
    return out


# ---------------------------------------------------------------------------
# Public check
# ---------------------------------------------------------------------------

async def check_rate(
    tool_name: str,
    api_key=None,
    oauth_client_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> None:
    """Raise ``RateError`` if the caller has exceeded any configured
    window for this tool.
    """
    principal = _principal_from(api_key, oauth_client_id, user_id)
    limits = _limits_for(principal, api_key, tool_name)
    client = _get_redis()

    for window, seconds in WINDOWS.items():
        limit = limits.get(window, 0)
        if limit <= 0:
            continue
        key = f"aictrlnet:ratelimit:{principal}:{tool_name}:{window}"
        count = await _incr(client, key, seconds)
        if count > limit:
            # Best-effort TTL lookup for retry_after
            retry_after = await _ttl(client, key) if client else seconds
            raise RateError(
                tool_name=tool_name,
                window=window,
                limit=limit,
                retry_after=max(1, retry_after),
            )


async def _incr(client, key: str, ttl: int) -> int:
    if client is None:
        return _memory_incr(key, ttl)
    try:
        pipe = client.pipeline()
        pipe.incr(key)
        pipe.expire(key, ttl, nx=True)  # only set TTL on first increment
        result = await pipe.execute()
        return int(result[0])
    except Exception as e:
        logger.warning("Redis INCR failed (%s); falling back to memory", e)
        return _memory_incr(key, ttl)


async def _ttl(client, key: str) -> int:
    try:
        ttl = await client.ttl(key)
        return int(ttl) if ttl > 0 else 60
    except Exception:
        return 60


__all__ = ["RateError", "check_rate"]
