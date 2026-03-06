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

from smoke_common.discovery import (
    discover_routes,
    discover_streaming_routes,
    discover_file_upload_routes,
    discover_501_routes,
    discover_external_service_routes,
)
from smoke_common.runner import (
    apply_overrides,
    smoke_one,
    smoke_one_streaming,
    smoke_one_file_upload,
    apply_external_service_patches,
)

# Discover routes at import time (safe — app creation doesn't trigger lifespan)
from core.app import create_app

_app = create_app()
apply_overrides(_app, "community")

# Apply external service patches before discovering those routes
_ext_patchers = apply_external_service_patches("community")

_specs = discover_routes(_app)
_streaming_specs = discover_streaming_routes(_app)
_upload_specs = discover_file_upload_routes(_app)
_501_specs = discover_501_routes(_app)
_external_specs = discover_external_service_routes(_app)


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _specs, ids=[s.test_id for s in _specs])
async def test_no_500(spec):
    """Assert endpoint does not return 5xx."""
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=10.0) as client:
        status, passed, detail = await smoke_one(client, spec)

    assert passed, f"{spec.method} {spec.path} -> {status}\n{detail}"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _streaming_specs, ids=[s.test_id for s in _streaming_specs])
async def test_streaming_no_500(spec):
    """Assert streaming endpoint does not return 5xx (headers only)."""
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=10.0) as client:
        status, passed, detail = await smoke_one_streaming(client, spec)

    assert passed, f"{spec.method} {spec.path} -> {status}\n{detail}"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _upload_specs, ids=[s.test_id for s in _upload_specs])
async def test_file_upload_no_500(spec):
    """Assert file-upload endpoint does not return 5xx."""
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=10.0) as client:
        status, passed, detail = await smoke_one_file_upload(client, spec)

    assert passed, f"{spec.method} {spec.path} -> {status}\n{detail}"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _501_specs, ids=[s.test_id for s in _501_specs])
async def test_expected_501(spec):
    """Assert 501-stub endpoint returns exactly 501."""
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=10.0) as client:
        status, passed, detail = await smoke_one(client, spec)

    assert status == 501, f"{spec.method} {spec.path} -> expected 501 got {status}\n{detail}"


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _external_specs, ids=[s.test_id for s in _external_specs])
async def test_external_service_no_500(spec):
    """Assert external-service endpoint does not return 5xx (services mocked)."""
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=10.0) as client:
        status, passed, detail = await smoke_one(client, spec)

    assert passed, f"{spec.method} {spec.path} -> {status}\n{detail}"
