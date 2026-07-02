"""Refresh-token revocation denylist (Redis-backed, graceful).

Lets an individual refresh token be revoked by its `jti` (e.g. on logout) and
checked at refresh time. Backed by the shared Redis cache; if Redis is
unavailable the helpers degrade safely (revoke is a no-op, is_revoked -> False)
so authentication keeps working — logout also bumps token_version as the
reliable revoke-all, so revocation is never solely dependent on Redis.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PREFIX = "revoked_refresh:"
# Cap TTL at the refresh-token lifetime (30 days) — no point keeping a denylist
# entry longer than the token could possibly be valid.
_MAX_TTL_SECONDS = 30 * 24 * 60 * 60


async def revoke_refresh_jti(jti: Optional[str], ttl_seconds: int = _MAX_TTL_SECONDS) -> bool:
    """Add a refresh token's jti to the denylist. Returns True if stored."""
    if not jti:
        return False
    try:
        from core.cache import get_cache
        cache = await get_cache()
        return await cache.set(f"{_PREFIX}{jti}", "1", expire=min(ttl_seconds, _MAX_TTL_SECONDS))
    except Exception as exc:  # pragma: no cover - Redis unavailable
        logger.warning("Could not record refresh-token revocation (jti=%s): %s", jti, exc)
        return False


async def is_refresh_revoked(jti: Optional[str]) -> bool:
    """Whether a refresh token's jti has been revoked. Fails open (False) if the
    denylist store is unavailable — logout's token_version bump is the backstop."""
    if not jti:
        return False
    try:
        from core.cache import get_cache
        cache = await get_cache()
        return await cache.exists(f"{_PREFIX}{jti}")
    except Exception as exc:  # pragma: no cover - Redis unavailable
        logger.warning("Could not check refresh-token revocation (jti=%s): %s", jti, exc)
        return False
