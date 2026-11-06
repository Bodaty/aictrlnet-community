"""The IAM API serves every signed-in user, scoped to their agents (G45).

History (5 Nov 2026): IAM agents, messages and sessions had no tenant or owner,
so any signed-in user could list every agent, change one, send as it and read
the message history; the whole router was made superuser-only. Agents now carry
a tenant (row-level security) and an owner. A member sees their own agents and
the tenant's system agents and acts only as their own; anything else answers
404, exactly like an id that does not exist.

Found in the fix-queue security review - no Trello card.
"""

import uuid

import httpx
import pytest
import pytest_asyncio
from fastapi import FastAPI
from sqlalchemy import text

from api.v1.endpoints import iam
from core.database import get_db
from core.enforcement_dependency import enforce_agent_limit
from core.security import get_current_active_user

ALICE = "00000000-0000-4000-8000-0000000a11ce"
BOB = "00000000-0000-4000-8000-000000000b0b"


def _user(user_id):
    return {"id": user_id, "email": f"{user_id}@example.invalid", "is_active": True,
            "is_superuser": False, "tenant_id": "default-tenant"}


def _client(user_id, db):
    app = FastAPI()
    app.include_router(iam.router, prefix="/api/v1/iam")
    app.dependency_overrides[get_current_active_user] = lambda: _user(user_id)
    app.dependency_overrides[enforce_agent_limit] = lambda: None

    async def _db():
        yield db
    app.dependency_overrides[get_db] = _db
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test")


@pytest_asyncio.fixture
async def alices_agent(db):
    name = f"alice-api-{uuid.uuid4().hex[:8]}"
    async with _client(ALICE, db) as client:
        response = await client.post("/api/v1/iam/agents", json={"name": name, "agent_type": "workflow"})
    assert response.status_code == 200, response.text
    yield response.json()
    ids = "SELECT id FROM iam_agents WHERE name = :n"
    await db.execute(text(f"DELETE FROM iam_event_logs WHERE agent_id IN ({ids})"), {"n": name})
    await db.execute(text(f"DELETE FROM iam_messages WHERE sender_id IN ({ids})"), {"n": name})
    await db.execute(text("DELETE FROM iam_agents WHERE name = :n"), {"n": name})
    await db.commit()


@pytest.mark.asyncio
async def test_create_sets_the_callers_tenant_and_owner(alices_agent):
    assert alices_agent["owner_user_id"] == ALICE
    assert alices_agent["tenant_id"] == "default-tenant"


@pytest.mark.asyncio
async def test_an_ordinary_user_is_no_longer_refused(db, alices_agent):
    async with _client(ALICE, db) as client:
        listed = await client.get("/api/v1/iam/agents", params={"limit": 1000})
        read = await client.get(f"/api/v1/iam/agents/{alices_agent['id']}")
    assert listed.status_code == 200 and read.status_code == 200
    assert alices_agent["id"] in [a["id"] for a in listed.json()]


@pytest.mark.asyncio
async def test_an_ordinary_user_gets_404_for_another_members_agent(db, alices_agent):
    agent = f"/api/v1/iam/agents/{alices_agent['id']}"
    async with _client(BOB, db) as client:
        listed = await client.get("/api/v1/iam/agents", params={"limit": 1000})
        read = await client.get(agent)
        changed = await client.put(agent, json={"description": "mine now"})
        beat = await client.post(f"{agent}/heartbeat")
    assert alices_agent["id"] not in [a["id"] for a in listed.json()]
    assert (read.status_code, changed.status_code, beat.status_code) == (404, 404, 404)


@pytest.mark.asyncio
async def test_sending_as_another_members_agent_is_404(db, alices_agent):
    async with _client(BOB, db) as client:
        sent = await client.post("/api/v1/iam/messages", params={"sender_id": alices_agent["id"]},
                                 json={"message_type": "notification", "content": {"g45": True}})
        session = await client.post("/api/v1/iam/sessions", params={"initiator_id": alices_agent["id"]},
                                    json={"session_type": "g45"})
    assert (sent.status_code, session.status_code) == (404, 404)
