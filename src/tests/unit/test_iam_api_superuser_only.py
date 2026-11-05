"""The IAM API answers superusers only, until IAM agents carry a tenant (G45).

History (5 Nov 2026): every `/api/v1/iam` route checked only that the caller
was signed in. IAM agents, messages and sessions have no tenant or owner, so any
user of any tenant could list every agent, change one, heartbeat it, send
messages as it and read the message history. Until G45 scopes them per tenant,
the whole router requires a superuser and refuses everyone else with 403 —
the same answer whether or not the agent or message exists.

Found in the fix-queue security review - no Trello card.
"""

import re
import uuid

import httpx
import pytest
from fastapi import FastAPI

from api.v1.endpoints import iam
from core.database import get_db
from core.dependencies import get_current_user_safe
from core.security import get_current_active_user


def _user(is_superuser):
    return {"id": str(uuid.uuid4()), "email": "iam@example.invalid", "is_active": True,
            "is_superuser": is_superuser, "tenant_id": "default-tenant"}


def _app(user, db=None):
    app = FastAPI()
    app.include_router(iam.router, prefix="/api/v1/iam")
    app.dependency_overrides[get_current_user_safe] = lambda: user
    app.dependency_overrides[get_current_active_user] = lambda: user

    async def _db():
        yield db
    app.dependency_overrides[get_db] = _db
    return app


# A fixed id: parametrized ids must be identical in every xdist worker.
_SOME_ID = "00000000-0000-4000-8000-000000000045"


def _every_route():
    for route in iam.router.routes:
        path = re.sub(r"\{[^}]+\}", _SOME_ID, route.path)
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}):
            yield method, "/api/v1/iam" + path


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", list(_every_route()))
async def test_an_ordinary_user_is_refused_on_every_route(method, path):
    transport = httpx.ASGITransport(app=_app(_user(False)))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.request(method, path, json={})
    assert response.status_code == 403, (method, path, response.status_code)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/api/v1/iam/agents", "/api/v1/iam/monitoring/metrics"])
async def test_a_superuser_still_reads_iam(db, path):
    transport = httpx.ASGITransport(app=_app(_user(True), db))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(path)
    assert response.status_code == 200, response.text
