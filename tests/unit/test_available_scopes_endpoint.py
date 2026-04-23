"""Unit tests for GET /api-keys/available-scopes."""

from fastapi.testclient import TestClient

from api.v1.endpoints.api_keys import list_available_scopes, router
from mcp_server.scopes import ALL_SCOPES, READ, WRITE


def test_returns_every_scope():
    """The handler function returns every ALL_SCOPES entry with metadata."""
    import asyncio

    result = asyncio.run(list_available_scopes())
    assert result["total"] == len(ALL_SCOPES)
    scopes = result["scopes"]
    assert {s["scope"] for s in scopes} == ALL_SCOPES


def test_every_scope_has_required_fields():
    import asyncio

    result = asyncio.run(list_available_scopes())
    for entry in result["scopes"]:
        assert set(entry.keys()) >= {"scope", "description", "group", "action"}
        assert isinstance(entry["description"], str) and entry["description"]
        assert entry["group"] in {
            "workflows", "adapters", "agents", "approvals", "knowledge",
            "governance", "org", "analytics", "admin", "other",
        }
        assert entry["action"] in {"read", "write", "other"}


def test_read_write_action_mapping():
    import asyncio

    result = asyncio.run(list_available_scopes())
    by_scope = {e["scope"]: e["action"] for e in result["scopes"]}
    for r in READ:
        assert by_scope[r] == "read"
    for w in WRITE:
        assert by_scope[w] == "write"


def test_scopes_sorted_deterministically():
    import asyncio

    result = asyncio.run(list_available_scopes())
    names = [e["scope"] for e in result["scopes"]]
    assert names == sorted(names)


def test_endpoint_reachable_via_fastapi():
    """End-to-end via a minimal FastAPI app with the router mounted."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    client = TestClient(app)
    resp = client.get("/api/v1/api-keys/available-scopes")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == len(ALL_SCOPES)
    assert "read:workflows" in {s["scope"] for s in data["scopes"]}
