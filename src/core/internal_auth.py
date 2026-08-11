"""Shared-secret auth for calls to the internal microservices.

`ml-service` (:8003) and `ai-agent-framework-service` (:8004) have no user auth
of their own. Their ports are deliberately not published (see docker-compose.yml),
and this token is the second layer: every in-cluster caller presents it, and the
services reject requests without it.

Both sides are opt-in on the same env var, so an unset `INTERNAL_SERVICE_TOKEN`
leaves today's behaviour untouched:

  unset  -> services accept everything, callers send no header
  set    -> services require the header, callers send it

Read at call time rather than import time so a token rotated into the
environment takes effect on the next request instead of the next restart.

Every request to those two services must go through `internal_service_headers()`.
`tests/integration/regressions/test_internal_service_auth_headers.py` fails the
build if a call site forgets — without it a missed site 401s only in production,
once the token is switched on.
"""

import os
from typing import Dict, Optional

INTERNAL_TOKEN_HEADER = "X-Internal-Token"


def internal_service_token() -> str:
    """Return the configured shared token, or "" when auth is not enabled."""
    return os.environ.get("INTERNAL_SERVICE_TOKEN", "")


def internal_service_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Headers for a request to ml-service / ai-agent-framework-service.

    Returns `extra` unchanged when no token is configured, so callers can pass
    this unconditionally.
    """
    headers = dict(extra or {})
    token = internal_service_token()
    if token:
        headers[INTERNAL_TOKEN_HEADER] = token
    return headers
