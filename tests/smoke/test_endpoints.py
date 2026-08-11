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
    quarantined_for,
    quarantine_shrink_candidates,
    error_log_quarantined_for,
    error_log_shrink_candidates,
    discover_routes,
    discover_streaming_routes,
    discover_file_upload_routes,
    discover_501_routes,
    discover_external_service_routes,
)
from smoke_common.runner import (
    apply_overrides,
    smoke_one,
    smoke_one_with_body,
    smoke_one_with_body_capturing_errors,
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


# Endpoints that take a JSON body get a second, stronger probe: a
# schema-derived body so the handler body actually executes. The plain `{}`
# probe 422s at validation first, which is how a broken handler stayed green
# for months.
_body_specs = [s for s in _specs if s.method in ("POST", "PUT", "PATCH")]
_quarantined = quarantined_for("community")
_openapi = _app.openapi()


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _body_specs, ids=[s.test_id for s in _body_specs])
async def test_no_500_with_synthesized_body(spec):
    """Assert the handler does not 5xx when given a minimally valid body."""
    if spec.test_id in _quarantined:
        pytest.skip(f"quarantined: {spec.test_id} (see SMOKE_BODY_QUARANTINE_BY_EDITION)")

    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=15.0) as client:
        status, passed, detail = await smoke_one_with_body(client, spec, _openapi)

    assert passed, (
        f"{spec.test_id} -> {status} with a schema-valid body.\n"
        f"The empty-body probe passes because validation rejects it first; "
        f"this one reaches the handler.\n{detail}"
    )


@pytest.mark.asyncio
async def test_quarantine_only_shrinks():
    """A quarantined endpoint that now passes must be removed from the set.

    Without this the quarantine rots: entries stay after the underlying bug or
    mock gap is fixed, silently shrinking coverage again. Any name listed here
    that now survives a schema-derived body is reported so it can be deleted.
    """
    reportable = quarantine_shrink_candidates("community")
    quarantined = [s for s in _body_specs if s.test_id in reportable]
    if not quarantined:
        pytest.skip("no quarantined endpoints for this edition")

    now_passing = []
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=15.0) as client:
        for spec in quarantined:
            status, passed, _ = await smoke_one_with_body(client, spec, _openapi)
            if passed:
                now_passing.append(f"{spec.test_id} (-> {status})")

    assert not now_passing, (
        "These endpoints pass now and must be removed from "
        "SMOKE_BODY_QUARANTINE_BY_EDITION in tests/smoke_common/discovery.py:\n  "
        + "\n  ".join(sorted(now_passing))
    )


_error_log_quarantined = error_log_quarantined_for("community")


@pytest.mark.asyncio
@pytest.mark.parametrize("spec", _body_specs, ids=[s.test_id for s in _body_specs])
async def test_no_swallowed_error_with_body(spec):
    """A 2xx response must not be hiding an ERROR the handler logged.

    "Not 5xx" passes for a handler that catches its own exception, logs it, and
    returns a 200 error envelope — so the feature fails on every call while the
    suite stays green. Four MCP handlers did exactly that for months.
    """
    if spec.test_id in _quarantined or spec.test_id in _error_log_quarantined:
        pytest.skip(f"quarantined: {spec.test_id}")

    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=15.0) as client:
        status, _, _, errors = await smoke_one_with_body_capturing_errors(
            client, spec, _openapi
        )

    if not (200 <= status < 300):
        pytest.skip(f"{spec.test_id} -> {status}; swallowed-error check only applies to 2xx")

    assert not errors, (
        f"{spec.test_id} returned {status} but logged {len(errors)} ERROR record(s).\n"
        f"The handler swallowed an exception and answered as if it succeeded:\n  "
        + "\n  ".join(errors[:5])
    )


@pytest.mark.asyncio
async def test_error_log_quarantine_only_shrinks():
    """A swallowed-error entry that now runs clean must be removed.

    Same contract as test_quarantine_only_shrinks: without this the set rots,
    silently re-opening the gap it was meant to track.
    """
    if not _error_log_quarantined:
        pytest.skip("no swallowed-error quarantine for this edition")

    reportable = error_log_shrink_candidates("community")
    quarantined = [s for s in _body_specs if s.test_id in reportable]
    now_clean = []
    transport = ASGITransport(app=_app)
    async with AsyncClient(transport=transport, base_url="http://smoke", timeout=15.0) as client:
        for spec in quarantined:
            status, _, _, errors = await smoke_one_with_body_capturing_errors(
                client, spec, _openapi
            )
            if 200 <= status < 300 and not errors:
                now_clean.append(f"{spec.test_id} (-> {status})")

    assert not now_clean, (
        "These endpoints no longer log errors behind a 2xx and must be removed "
        "from SMOKE_ERROR_LOG_QUARANTINE_BY_EDITION in "
        "tests/smoke_common/discovery.py:\n  " + "\n  ".join(sorted(now_clean))
    )
