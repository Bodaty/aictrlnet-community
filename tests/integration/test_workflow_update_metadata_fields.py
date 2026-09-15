"""Regression: PUT /workflows/{id} silently discarded status and category.

`update_workflow` ran `setattr(workflow, field, value)` for every field in the
payload. `workflow_definitions` has no `status` or `category` column — both
live inside the JSON `workflow_metadata` column, which is where
`WorkflowService.create_workflow` writes them and where
`WorkflowResponse.extract_metadata_fields` reads them back.

So the write path set two transient Python attributes, committed cleanly,
returned 200, and stored nothing. A caller archiving a workflow was told it
worked and it did not.

Found by scripts/audit/schema_model_drift.py, which pairs each blanket-setattr
loop with its schema and model and compares fields against columns. It was the
single real finding across 32 such sites.

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
from schemas.workflow import WorkflowUpdate  # noqa: E402
from api.v1.endpoints.workflows import update_workflow  # noqa: E402

_DB_URL = os.environ.get(
    "DATABASE_URL", "postgresql+asyncpg://postgres:postgres@postgres:5432/aictrlnet",
)
_TENANT = "test-tenant-wf-metadata"
_ADMIN = {"id": "admin", "tenant_id": _TENANT, "is_superuser": True}


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


@pytest_asyncio.fixture
async def workflow_id(db: AsyncSession):
    async def _purge():
        await db.execute(delete(WorkflowDefinition).where(WorkflowDefinition.tenant_id == _TENANT))
        await db.commit()

    await _purge()
    wid = str(uuid.uuid4())
    db.add(WorkflowDefinition(
        id=wid, name=f"metadata-probe-{wid[:8]}", definition={"nodes": [], "edges": []},
        tenant_id=_TENANT, active=True,
        # mirrors what WorkflowService.create_workflow writes
        workflow_metadata={"status": "draft", "category": "ops", "template_id": "tpl-123"},
    ))
    await db.commit()
    yield wid
    await _purge()


async def _stored_metadata(db, wid):
    db.expire_all()
    row = (await db.execute(
        select(WorkflowDefinition).where(WorkflowDefinition.id == wid)
    )).scalar_one()
    return row.workflow_metadata or {}


@pytest.mark.asyncio
async def test_status_update_persists(db, workflow_id):
    await update_workflow(
        workflow_id=workflow_id, workflow_update=WorkflowUpdate(status="archived"),
        db=db, current_user=_ADMIN,
    )
    assert (await _stored_metadata(db, workflow_id))["status"] == "archived"


@pytest.mark.asyncio
async def test_category_update_persists(db, workflow_id):
    await update_workflow(
        workflow_id=workflow_id, workflow_update=WorkflowUpdate(category="finance"),
        db=db, current_user=_ADMIN,
    )
    assert (await _stored_metadata(db, workflow_id))["category"] == "finance"


@pytest.mark.asyncio
async def test_updating_one_metadata_field_preserves_the_others(db, workflow_id):
    """The obvious wrong fix replaces workflow_metadata wholesale."""
    await update_workflow(
        workflow_id=workflow_id, workflow_update=WorkflowUpdate(status="active"),
        db=db, current_user=_ADMIN,
    )
    meta = await _stored_metadata(db, workflow_id)
    assert meta["status"] == "active"
    assert meta["category"] == "ops", "unrelated metadata key was clobbered"
    assert meta["template_id"] == "tpl-123", "unrelated metadata key was clobbered"


@pytest.mark.asyncio
async def test_the_response_reflects_what_was_stored(db, workflow_id):
    resp = await update_workflow(
        workflow_id=workflow_id, workflow_update=WorkflowUpdate(status="archived", category="legal"),
        db=db, current_user=_ADMIN,
    )
    assert resp.status == "archived"
    assert resp.category == "legal"


@pytest.mark.asyncio
async def test_real_columns_still_update(db, workflow_id):
    """The fix must not break fields that do have columns."""
    await update_workflow(
        workflow_id=workflow_id, workflow_update=WorkflowUpdate(name="renamed-probe"),
        db=db, current_user=_ADMIN,
    )
    db.expire_all()
    row = (await db.execute(
        select(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id)
    )).scalar_one()
    assert row.name == "renamed-probe"
