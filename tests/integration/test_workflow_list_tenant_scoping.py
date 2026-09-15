"""Regression: GET /api/v1/workflows/ must not return other tenants' workflows.

`list_workflows` ran `select(WorkflowDefinition)` with no tenant filter — it
injected `current_user` for authentication and then never used it for scoping.
Any authenticated user saw every workflow in the database.

Confirmed live on Beast 2026-08-19: a `default-tenant` user was returned all 40
workflows, while the database held 22 in `default-tenant` and 18 in `bodaty`.
The exposure mattered more after the Glass Company consolidation put all of the
real operating data into `bodaty`.

Not a design choice: `POST /{workflow_id}/schedules` and
`DELETE /schedules/{schedule_id}` in the same file both call
`assert_tenant_access`, and the list endpoint simply never did.

Caught by our own automation — no Trello card.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from smoke_common.db_probe import require_database  # <repo>/tests is on sys.path via ../conftest.py's egress-guard block

_REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(_REPO_ROOT / "editions" / "community" / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.community_complete import WorkflowDefinition  # noqa: E402
from api.v1.endpoints.workflows import list_workflows  # noqa: E402

_DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@postgres:5432/aictrlnet",
)
_TENANT_A = "test-tenant-scope-a"
_TENANT_B = "test-tenant-scope-b"


@pytest_asyncio.fixture
async def db():
    require_database(_DB_URL)
    engine = create_async_engine(_DB_URL, echo=False, poolclass=NullPool)
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        try:
            yield session
        finally:
            await session.close()
    await engine.dispose()


@pytest_asyncio.fixture(autouse=True)
async def seeded(db: AsyncSession):
    async def _purge():
        await db.execute(
            delete(WorkflowDefinition).where(
                WorkflowDefinition.tenant_id.in_([_TENANT_A, _TENANT_B])
            )
        )
        await db.commit()

    await _purge()
    ids = {}
    for tenant, label in ((_TENANT_A, "A"), (_TENANT_B, "B")):
        wid = str(uuid.uuid4())
        ids[label] = wid
        db.add(WorkflowDefinition(
            id=wid, name=f"scope-probe-{label}-{wid[:8]}",
            definition={"nodes": [], "edges": []}, tenant_id=tenant, active=True,
        ))
    await db.commit()
    yield ids
    await _purge()


async def _list_as(db, user):
    return await list_workflows(
        response=Response(), skip=0, limit=5000, category=None,
        is_template=None, search=None, db=db, current_user=user,
    )


@pytest.mark.asyncio
async def test_a_tenant_user_does_not_see_another_tenants_workflow(db, seeded):
    got = {str(w.id) for w in await _list_as(
        db, {"id": "u-a", "tenant_id": _TENANT_A, "is_superuser": False})}
    assert seeded["A"] in got, "caller must still see their own tenant's workflow"
    assert seeded["B"] not in got, "cross-tenant workflow leaked into the list"


@pytest.mark.asyncio
async def test_the_other_direction_too(db, seeded):
    got = {str(w.id) for w in await _list_as(
        db, {"id": "u-b", "tenant_id": _TENANT_B, "is_superuser": False})}
    assert seeded["B"] in got
    assert seeded["A"] not in got


@pytest.mark.asyncio
async def test_superuser_still_sees_across_tenants(db, seeded):
    """Matches assert_tenant_access, which bypasses for superusers — the
    operator tooling in this session depends on it."""
    got = {str(w.id) for w in await _list_as(
        db, {"id": "root", "tenant_id": _TENANT_A, "is_superuser": True})}
    assert seeded["A"] in got and seeded["B"] in got


@pytest.mark.asyncio
async def test_a_user_with_no_tenant_sees_nothing_rather_than_everything(db, seeded):
    """Fail closed. A missing tenant claim must not degrade into 'show all'."""
    got = {str(w.id) for w in await _list_as(
        db, {"id": "u-none", "tenant_id": None, "is_superuser": False})}
    assert seeded["A"] not in got and seeded["B"] not in got
