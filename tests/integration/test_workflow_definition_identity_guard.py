"""The generic workflow update guards what a definition says it is.

- A workflow marked ``phi`` keeps the flag, its ``metadata.templateId`` and its
  name: other parts of the platform recognise its runs by name and template.
- Only an administrator marks a workflow ``phi`` or gives it a templateId,
  the same rule as for the autonomy dial.
- Only an administrator gives a workflow a name an edition reserved.
- The generic trigger route never inserts a server-managed trigger type.

The Community PUT handler and WorkflowService, called as the application calls
them, against the real database.

Caught by our own review — no Trello card.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import HTTPException
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool
from smoke_common.db_probe import require_database

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
_TENANT = "test-tenant-wf-identity"
_ADMIN = {"id": "admin", "tenant_id": _TENANT, "is_superuser": True}
_STAFF = {"id": "staff", "tenant_id": _TENANT, "is_superuser": False}


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
async def ids(db: AsyncSession):
    async def _purge():
        await db.execute(delete(WorkflowDefinition).where(WorkflowDefinition.tenant_id == _TENANT))
        await db.commit()

    await _purge()
    phi, plain = str(uuid.uuid4()), str(uuid.uuid4())
    db.add_all([
        WorkflowDefinition(id=phi, name="phi-probe", tenant_id=_TENANT, active=True,
                           definition={"phi": True, "metadata": {"templateId": "probe"},
                                       "nodes": [], "edges": []}),
        WorkflowDefinition(id=plain, name="plain-probe", tenant_id=_TENANT, active=True,
                           definition={"nodes": [], "edges": []}),
    ])
    await db.commit()
    yield {"phi": phi, "plain": plain}
    await _purge()


async def _row(db, wid):
    db.expire_all()
    return (await db.execute(select(WorkflowDefinition).where(WorkflowDefinition.id == wid))).scalar_one()


async def _put(db, wid, user, **fields):
    return await update_workflow(workflow_id=wid, workflow_update=WorkflowUpdate(**fields),
                                 db=db, current_user=user)


@pytest.mark.asyncio
@pytest.mark.parametrize("user", [_ADMIN, _STAFF])
async def test_a_phi_workflow_cannot_be_renamed(db, ids, user):
    with pytest.raises(HTTPException) as err:
        await _put(db, ids["phi"], user, name="renamed")
    assert err.value.status_code == 422
    await db.rollback()
    assert (await _row(db, ids["phi"])).name == "phi-probe"


@pytest.mark.asyncio
async def test_a_full_save_with_the_same_name_is_fine(db, ids):
    await _put(db, ids["phi"], _STAFF, name="phi-probe", description="kept")
    assert (await _row(db, ids["phi"])).description == "kept"


@pytest.mark.asyncio
@pytest.mark.parametrize("definition", [
    {"phi": True, "nodes": [], "edges": []},
    {"metadata": {"templateId": "probe"}, "nodes": [], "edges": []},
])
async def test_only_an_admin_marks_a_workflow_phi_or_gives_it_a_template(db, ids, definition):
    with pytest.raises(HTTPException) as err:
        await _put(db, ids["plain"], _STAFF, definition=definition)
    assert err.value.status_code == 403
    await db.rollback()
    assert (await _row(db, ids["plain"])).definition == {"nodes": [], "edges": []}

    await _put(db, ids["plain"], _ADMIN, definition=definition)
    stored = (await _row(db, ids["plain"])).definition
    for key, value in definition.items():
        assert stored[key] == value


@pytest.mark.asyncio
async def test_staff_can_still_edit_a_phi_workflows_steps(db, ids):
    await _put(db, ids["phi"], _STAFF, definition={"nodes": [{"id": "a", "type": "start"}],
                                                   "edges": [], "phi": True})
    stored = (await _row(db, ids["phi"])).definition
    assert stored["phi"] is True and stored["metadata"]["templateId"] == "probe"
    assert stored["nodes"][0]["id"] == "a"


@pytest.mark.asyncio
async def test_the_workflow_service_applies_the_same_guards(db, ids):
    from core.tenant_context import TenantContext
    from services.workflow import (
        DefinitionChangeForbidden,
        PhiDefinitionChangeForbidden,
        WorkflowService,
    )

    with TenantContext(_TENANT):
        service = WorkflowService(db)
        with pytest.raises(PhiDefinitionChangeForbidden):
            await service.update_workflow(ids["phi"], WorkflowUpdate(name="renamed"),
                                          actor_is_admin=True)
        await db.rollback()
        with pytest.raises(DefinitionChangeForbidden):
            await service.update_workflow(
                ids["plain"], WorkflowUpdate(definition={"phi": True, "nodes": [], "edges": []}))
        await db.rollback()
        await service.update_workflow(
            ids["plain"], WorkflowUpdate(definition={"phi": True, "nodes": [], "edges": []}),
            actor_is_admin=True)
    assert (await _row(db, ids["plain"])).definition["phi"] is True


# ---------------------------------------------------------------------------
# R78 on the Community paths: create, duplicate, the assistant's rename
# ---------------------------------------------------------------------------

async def _count(db):
    db.expire_all()
    return len((await db.execute(select(WorkflowDefinition.id).where(
        WorkflowDefinition.tenant_id == _TENANT))).all())


@pytest.mark.asyncio
async def test_only_an_admin_creates_a_phi_workflow(db, ids):
    from api.v1.endpoints.workflows import create_workflow
    from core.tenant_context import TenantContext
    from schemas.workflow import WorkflowCreate

    before = await _count(db)
    with TenantContext(_TENANT):
        with pytest.raises(HTTPException) as err:
            await create_workflow(WorkflowCreate(
                name="made", definition={"phi": True, "nodes": [], "edges": []}),
                db=db, current_user=_STAFF)
        assert err.value.status_code == 403
        assert await _count(db) == before
        await create_workflow(WorkflowCreate(
            name="made", definition={"phi": True, "nodes": [], "edges": []}),
            db=db, current_user=_ADMIN)
    assert await _count(db) == before + 1


@pytest.mark.asyncio
async def test_only_an_admin_duplicates_a_phi_workflow(db, ids):
    from api.v1.endpoints.workflows import duplicate_workflow

    before = await _count(db)
    with pytest.raises(HTTPException) as err:
        await duplicate_workflow(ids["phi"], db=db, current_user=_STAFF)
    assert err.value.status_code == 403
    assert await _count(db) == before
    await duplicate_workflow(ids["phi"], db=db, current_user=_ADMIN)
    await duplicate_workflow(ids["plain"], db=db, current_user=_STAFF)
    assert await _count(db) == before + 2


@pytest.mark.asyncio
async def test_the_assistant_cannot_rename_a_phi_workflow(db, ids):
    from core.tenant_context import TenantContext
    from services.tool_dispatcher import ToolDispatcher

    with TenantContext(_TENANT):
        dispatcher = ToolDispatcher(db)
        dispatcher._lazy_load_services()
        result = await dispatcher._update_workflow(
            {"workflow_id": ids["phi"], "name": "renamed",
             "description": "Reads the front desk's incoming faxes and files them for staff to check"},
            "admin", {})
    assert not result.success and result.data["status_code"] == 422
    assert (await _row(db, ids["phi"])).name == "phi-probe"


RESERVED = "reserved-probe-name"


@pytest.mark.asyncio
async def test_only_an_admin_gives_a_workflow_a_reserved_name(db, ids):
    from api.v1.endpoints.workflows import create_workflow
    from core.tenant_context import TenantContext
    from schemas.workflow import WorkflowCreate
    from services.workflow_identity import register_reserved_workflow_names

    register_reserved_workflow_names([RESERVED])
    before = await _count(db)
    with TenantContext(_TENANT):
        for call in (
            lambda: create_workflow(WorkflowCreate(name=RESERVED, definition={"nodes": [], "edges": []}),
                                    db=db, current_user=_STAFF),
            lambda: update_workflow(ids["plain"], WorkflowUpdate(name=RESERVED),
                                    db=db, current_user=_STAFF),
            lambda: update_workflow(ids["plain"], WorkflowUpdate(name=f"  {RESERVED.upper()} "),
                                    db=db, current_user=_STAFF),
        ):
            with pytest.raises(HTTPException) as err:
                await call()
            assert err.value.status_code == 422
        assert await _count(db) == before
        assert (await _row(db, ids["plain"])).name == "plain-probe"
        await update_workflow(ids["plain"], WorkflowUpdate(name=RESERVED), db=db, current_user=_ADMIN)
    assert (await _row(db, ids["plain"])).name == RESERVED


@pytest.mark.asyncio
async def test_a_server_managed_trigger_type_is_never_inserted_as_given(db, ids):
    """Community has no creator for the Documents inbox trigger: staff get 403,
    an administrator 422, and no row is written either way."""
    from api.v1.endpoints.workflows import create_workflow_trigger
    from models.workflow_execution import WorkflowTrigger
    from schemas.workflow_execution import WorkflowTriggerCreate

    data = WorkflowTriggerCreate(trigger_type="document_inbox", config={"subfolder": "x"})
    try:
        for user, status in ((_STAFF, 403), (_ADMIN, 422)):
            with pytest.raises(HTTPException) as err:
                await create_workflow_trigger(ids["plain"], data, db=db, current_user=user)
            assert err.value.status_code == status
        db.expire_all()
        assert (await db.execute(select(WorkflowTrigger.id).where(
            WorkflowTrigger.workflow_id == ids["plain"]))).all() == []
    finally:
        await db.execute(delete(WorkflowTrigger).where(WorkflowTrigger.workflow_id == ids["plain"]))
        await db.commit()
