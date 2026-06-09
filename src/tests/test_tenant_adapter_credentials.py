"""Regression: tenant-scoped adapter credential resolution (GEO Phase B2).

covers: geo-phase-b2 tenant-credentials no-cross-tenant-leak

get_adapter_credentials_for_tenant must resolve:
  1. the executing org's own key,
  2. else the shared/free-tier row (tenant_id IS NULL),
  3. else None (adapter falls back to its env key),
and must NEVER return another tenant's key. This test seeds rows directly,
exercises each path, and asserts no cross-tenant leak.
"""
import uuid

import pytest
from sqlalchemy import delete, select

from core.database import get_session_maker
from core.crypto import encrypt_data
from models.adapter_config import UserAdapterConfig
from models.user import User
from nodes.template_utils import get_adapter_credentials_for_tenant

ENGINE = f"b2-test-engine-{uuid.uuid4().hex[:8]}"


async def _a_user_id():
    """A real users.id (user_id FKs to it). Tests run against the seeded dev DB."""
    async with get_session_maker()() as s:
        uid = (await s.execute(select(User.id).limit(1))).scalar_one_or_none()
    assert uid, "no users seeded — cannot run tenant-credential test"
    return uid


async def _seed(tenant_id, api_key, name):
    user_id = await _a_user_id()
    async with get_session_maker()() as s:
        s.add(UserAdapterConfig(
            user_id=user_id,
            tenant_id=tenant_id,
            adapter_type=ENGINE,
            name=name,
            credentials=encrypt_data({"api_key": api_key}),
            enabled=True,
        ))
        await s.commit()


async def _cleanup():
    async with get_session_maker()() as s:
        await s.execute(delete(UserAdapterConfig).where(UserAdapterConfig.adapter_type == ENGINE))
        await s.commit()


@pytest.fixture
async def seeded():
    await _cleanup()
    await _seed("org-a", "A-key", "orgA")
    await _seed(None, "shared-key", "shared")
    yield
    await _cleanup()


@pytest.mark.asyncio
async def test_org_gets_its_own_key(seeded):
    creds = await get_adapter_credentials_for_tenant(ENGINE, "org-a")
    assert creds == {"api_key": "A-key"}


@pytest.mark.asyncio
async def test_org_without_key_falls_back_to_shared(seeded):
    # org-b has no row of its own -> shared/free-tier (NULL) row, NOT org-a's.
    creds = await get_adapter_credentials_for_tenant(ENGINE, "org-b")
    assert creds == {"api_key": "shared-key"}


@pytest.mark.asyncio
async def test_default_tenant_uses_shared(seeded):
    creds = await get_adapter_credentials_for_tenant(ENGINE, "default-tenant")
    assert creds == {"api_key": "shared-key"}


@pytest.mark.asyncio
async def test_no_cross_tenant_leak_when_no_shared_row():
    # Only org-a has a key; org-b must get NOTHING (env fallback), never A's key.
    await _cleanup()
    await _seed("org-a", "A-only", "orgA")
    try:
        assert await get_adapter_credentials_for_tenant(ENGINE, "org-b") is None
        assert await get_adapter_credentials_for_tenant(ENGINE, "org-a") == {"api_key": "A-only"}
    finally:
        await _cleanup()


@pytest.mark.asyncio
async def test_unknown_engine_returns_none(seeded):
    assert await get_adapter_credentials_for_tenant("no-such-engine-xyz", "org-a") is None
