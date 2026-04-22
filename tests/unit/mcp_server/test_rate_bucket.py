"""Unit tests for mcp_server.rate_bucket (in-memory fallback path).

The Redis path is covered by integration tests with a real Redis in
the stack; this file exercises the fallback that runs when Redis is
unavailable so every deploy has a working rate bucket regardless of
infrastructure.
"""

import asyncio
import pytest
from unittest.mock import patch

from mcp_server.rate_bucket import (
    DEFAULT_LIMITS,
    RateError,
    check_rate,
)


class _FakeApiKey:
    def __init__(self, id_="k1", per_tool=None, per_minute=None, per_day=None):
        self.id = id_
        self.rate_limit_per_tool = per_tool
        self.rate_limit_per_minute = per_minute
        self.rate_limit_per_day = per_day
        self.scopes = []


@pytest.fixture(autouse=True)
def _force_memory_backend(monkeypatch):
    """All tests here run against the in-memory bucket."""
    # Clear Redis globals so _get_redis picks the fallback path.
    from mcp_server import rate_bucket

    monkeypatch.setattr(rate_bucket, "_redis_client", None, raising=False)
    monkeypatch.setattr(rate_bucket, "_redis_checked", True, raising=False)
    # Clear in-memory state between tests.
    monkeypatch.setattr(rate_bucket, "_memory_state", {}, raising=False)


@pytest.mark.asyncio
async def test_under_default_limit_is_allowed():
    key = _FakeApiKey()
    for _ in range(3):
        await check_rate("get_workflow", api_key=key)


@pytest.mark.asyncio
async def test_over_per_minute_limit_raises():
    key = _FakeApiKey(per_tool={"create_workflow": {"per_minute": 2}})
    await check_rate("create_workflow", api_key=key)
    await check_rate("create_workflow", api_key=key)
    with pytest.raises(RateError) as exc:
        await check_rate("create_workflow", api_key=key)
    assert exc.value.window == "per_minute"
    assert exc.value.limit == 2


@pytest.mark.asyncio
async def test_separate_tools_have_separate_buckets():
    key = _FakeApiKey(
        per_tool={"tool_a": {"per_minute": 1}, "tool_b": {"per_minute": 1}}
    )
    await check_rate("tool_a", api_key=key)
    # tool_b still has room even though tool_a is exhausted
    await check_rate("tool_b", api_key=key)


@pytest.mark.asyncio
async def test_legacy_per_minute_column_used_when_no_per_tool():
    key = _FakeApiKey(per_minute=1)
    await check_rate("tool_x", api_key=key)
    with pytest.raises(RateError):
        await check_rate("tool_x", api_key=key)


@pytest.mark.asyncio
async def test_anonymous_user_also_rate_limited():
    # No api_key — user_id principal used
    from mcp_server import rate_bucket

    # Force low default by patching
    with patch.dict(
        rate_bucket.DEFAULT_LIMITS["per_minute"], {"*": 1}, clear=False
    ):
        await check_rate("anon_tool", user_id="u1")
        with pytest.raises(RateError):
            await check_rate("anon_tool", user_id="u1")


def test_rate_error_payload_shape():
    err = RateError(
        tool_name="x", window="per_minute", limit=10, retry_after=30
    )
    payload = err.to_payload()
    assert payload["error"] == "rate_limit_exceeded"
    assert payload["tool"] == "x"
    assert payload["window"] == "per_minute"
    assert payload["limit"] == 10
    assert payload["retry_after_seconds"] == 30


def test_default_limits_shape():
    assert "per_minute" in DEFAULT_LIMITS
    assert "per_day" in DEFAULT_LIMITS
    assert "*" in DEFAULT_LIMITS["per_minute"]
