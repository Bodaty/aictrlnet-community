"""Auto-discovering endpoint smoke tests for Community Edition.

Every registered API route is tested with a single request.
Any 5xx response is a test failure — it means the endpoint layer is broken.
"""

import os
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("EDITION", "community")
os.environ.setdefault("AICTRLNET_EDITION", "community")

import pytest
from httpx import AsyncClient, ASGITransport

from smoke_common.discovery import discover_routes
from smoke_common.runner import apply_overrides, smoke_one

# Discover routes at import time (safe — app creation doesn't trigger lifespan)
from core.app import create_app

_app = create_app()
apply_overrides(_app, "community")
_specs = discover_routes(_app)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _specs, ids=[s.test_id for s in _specs])
async def test_no_500(spec):
    """Assert endpoint does not return 5xx."""
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=10.0) as client:
        status, passed, detail = await smoke_one(client, spec)

    assert passed, f"{spec.method} {spec.path} -> {status}\n{detail}"
