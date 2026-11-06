"""The pollers a service starts on its own obey BACKGROUND_WORKERS_ENABLED.

On a stack running the test gate the switch is off, so the scheduler, sweeps,
dispatchers and purges never start and nothing but the tests touches the test
tenants. Everywhere else it is on (default) and they start as before.
"""

import asyncio

import pytest

from core import background_workers
from core.background_workers import start_worker, started_workers


@pytest.fixture(autouse=True)
def _clean_registry():
    saved = list(background_workers._started)
    background_workers._started.clear()
    yield
    background_workers._started[:] = saved


@pytest.fixture
def workers_off(monkeypatch):
    monkeypatch.setenv("BACKGROUND_WORKERS_ENABLED", "false")


def test_workers_are_on_unless_switched_off(monkeypatch):
    monkeypatch.delenv("BACKGROUND_WORKERS_ENABLED", raising=False)
    assert background_workers.background_workers_enabled() is True


@pytest.mark.asyncio
async def test_a_worker_starts_and_is_recorded_when_on(monkeypatch):
    monkeypatch.delenv("BACKGROUND_WORKERS_ENABLED", raising=False)
    calls = []

    async def start():
        calls.append("async")

    assert await start_worker("sweep", start) is True
    assert await start_worker("purge", lambda: calls.append("sync")) is True
    assert calls == ["async", "sync"]
    assert started_workers() == ["sweep", "purge"]


@pytest.mark.asyncio
async def test_a_worker_never_starts_when_off(workers_off):
    calls = []
    assert await start_worker("sweep", lambda: calls.append("started")) is False
    assert calls == []
    assert started_workers() == []


@pytest.mark.asyncio
async def test_the_conversation_job_purge_is_not_scheduled_when_off(workers_off, monkeypatch):
    from services import conversation_jobs

    monkeypatch.setattr(conversation_jobs, "_last_purge", float("-inf"))
    before = set(conversation_jobs._tasks)
    conversation_jobs.schedule_purge()
    await asyncio.sleep(0)
    assert set(conversation_jobs._tasks) == before
    assert "conversation_job_purge" not in started_workers()


def _client(user):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from api.v1.endpoints import health
    from core.dependencies import get_current_user_safe

    app = FastAPI()
    app.include_router(health.router)
    app.dependency_overrides[get_current_user_safe] = lambda: user
    return TestClient(app)


def test_the_started_workers_are_reported_to_a_superuser_only(workers_off):
    background_workers.record_started("scheduler_service")
    member = _client({"id": "u1", "is_superuser": False}).get("/health/workers")
    assert member.status_code == 403
    body = _client({"id": "u2", "is_superuser": True}).get("/health/workers").json()
    assert body["enabled"] is False and body["started"] == ["scheduler_service"]
