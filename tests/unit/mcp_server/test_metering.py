"""Unit tests for mcp_server.metering.

Atomic DB behavior is exercised against a stub AsyncSession — the
production path uses Postgres ``UPDATE ... WHERE ... RETURNING`` which
cannot be exercised without a real DB. Those end-to-end concurrency
tests live in the integration suite.
"""

import asyncio
import pytest

from mcp_server.metering import (
    DEFAULT_LIMITS,
    QuotaError,
    RefundableError,
    TOOL_METERING,
    TOOL_TIMEOUT,
    ToolTimeoutError,
    get_default_limit,
    resolve_quantity,
    with_metering,
)


# ---------------------------------------------------------------------------
# Stub DB session
# ---------------------------------------------------------------------------

class _Row:
    def __init__(self, *values):
        self._values = values

    def __iter__(self):
        return iter(self._values)

    def __getitem__(self, i):
        return self._values[i]

    def first(self):
        return self


class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _StubSession:
    """Fake AsyncSession that tracks counter state in-process.

    Emulates the charge / refund / lookup SQL interactions enough for
    the orchestrator tests.
    """

    def __init__(self, counters=None, limits=None, idempotency=None):
        self.counters: dict = counters or {}
        self.limits: dict = limits or {}
        self.idempotency: dict = idempotency or {}
        self.executes = 0

    async def execute(self, stmt, params=None):
        self.executes += 1
        sql = str(stmt)
        params = params or {}

        if "INSERT INTO mcp_meters" in sql:
            tenant = params["tenant_id"]
            meter = params["meter"]
            qty = int(params["qty"])
            limit = int(params["default_limit"])
            current = int(self.counters.get((tenant, meter), 0))
            if limit > 0 and current + qty > limit:
                return _Result(None)
            self.counters[(tenant, meter)] = current + qty
            return _Result(_Row(current + qty, None))

        if "SELECT counter, period_end" in sql:
            tenant = params["tenant_id"]
            meter = params["meter"]
            return _Result(_Row(int(self.counters.get((tenant, meter), 0)), None))

        if "UPDATE mcp_meters" in sql:
            tenant = params["tenant_id"]
            meter = params["meter"]
            qty = int(params["qty"])
            cur = int(self.counters.get((tenant, meter), 0))
            self.counters[(tenant, meter)] = max(0, cur - qty)
            return _Result(None)

        if "SELECT response_json" in sql:
            tenant = params["tenant_id"]
            tool = params["tool_name"]
            key = params["key"]
            val = self.idempotency.get((tenant, tool, key))
            if not val:
                return _Result(None)
            return _Result(_Row(val["response"], val["hash"], None))

        if "INSERT INTO mcp_idempotency_keys" in sql:
            self.idempotency[
                (params["tenant_id"], params["tool_name"], params["key"])
            ] = {
                "response": params["response"],
                "hash": params["args_hash"],
            }
            return _Result(None)

        return _Result(None)

    async def commit(self):
        pass

    async def rollback(self):
        pass


# ---------------------------------------------------------------------------
# Registry sanity
# ---------------------------------------------------------------------------

def test_default_limits_has_all_tiers():
    for tier in ("community", "business", "enterprise"):
        assert tier in DEFAULT_LIMITS
        assert DEFAULT_LIMITS[tier]["llm_calls"] > 0


def test_default_limit_community_llm_matches_spec():
    # v11 spec says 5000 LLM calls for Community trial
    assert get_default_limit("community", "llm_calls") == 5_000


def test_default_limit_unknown_returns_zero():
    assert get_default_limit("community", "not_a_meter") == 0


def test_tool_metering_registry_contains_expected():
    assert "create_workflow" in TOOL_METERING
    assert TOOL_METERING["create_workflow"] == ("llm_calls", 1)
    assert "browser_execute" in TOOL_METERING
    # browser_execute uses callable for qty
    meter_name, qty_expr = TOOL_METERING["browser_execute"]
    assert meter_name == "browser_actions"
    assert callable(qty_expr)


def test_tool_timeout_registry_has_long_running_tools():
    assert TOOL_TIMEOUT.get("generate_adapter", 0) > 0
    assert TOOL_TIMEOUT.get("browser_execute", 0) > 0


# ---------------------------------------------------------------------------
# resolve_quantity
# ---------------------------------------------------------------------------

def test_resolve_quantity_literal():
    assert resolve_quantity(3, {}) == 3


def test_resolve_quantity_callable():
    qty = resolve_quantity(lambda args: len(args.get("actions", [])), {"actions": [1, 2, 3]})
    assert qty == 3


def test_resolve_quantity_callable_failure_returns_one():
    def boom(args):
        raise RuntimeError()
    assert resolve_quantity(boom, {}) == 1


# ---------------------------------------------------------------------------
# with_metering
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_with_metering_unmetered_tool_passes_through():
    db = _StubSession()
    called = []

    async def handler():
        called.append(True)
        return {"ok": True}

    out = await with_metering(
        tool_name="get_workflow",
        args={},
        tenant_id="t1",
        edition="community",
        db=db,
        handler=handler,
    )
    assert out == {"ok": True}
    assert called == [True]


@pytest.mark.asyncio
async def test_with_metering_metered_tool_charges_counter():
    db = _StubSession()
    async def handler():
        return {"ok": True}

    await with_metering(
        tool_name="create_workflow",
        args={},
        tenant_id="t1",
        edition="community",
        db=db,
        handler=handler,
    )
    assert db.counters[("t1", "llm_calls")] == 1


@pytest.mark.asyncio
async def test_with_metering_raises_quota_when_limit_exceeded():
    db = _StubSession()
    # Pre-fill counter near limit
    db.counters[("t1", "llm_calls")] = 4999  # community limit = 5000

    async def handler():
        return {"ok": True}

    # First charge (qty=1): 4999 + 1 = 5000 -> allowed
    await with_metering(
        tool_name="create_workflow",
        args={},
        tenant_id="t1",
        edition="community",
        db=db,
        handler=handler,
    )

    # Second charge: 5000 + 1 = 5001 > 5000 -> QuotaError
    with pytest.raises(QuotaError) as exc:
        await with_metering(
            tool_name="create_workflow",
            args={},
            tenant_id="t1",
            edition="community",
            db=db,
            handler=handler,
        )
    assert exc.value.meter == "llm_calls"
    assert exc.value.limit == 5_000


@pytest.mark.asyncio
async def test_with_metering_refund_on_refundable_error():
    db = _StubSession()

    async def handler():
        raise RefundableError("pre-LLM validation failed")

    with pytest.raises(RefundableError):
        await with_metering(
            tool_name="create_workflow",
            args={},
            tenant_id="t1",
            edition="community",
            db=db,
            handler=handler,
        )
    # Refunded -> counter back to 0
    assert db.counters.get(("t1", "llm_calls"), 0) == 0


@pytest.mark.asyncio
async def test_with_metering_does_not_refund_on_arbitrary_exception():
    db = _StubSession()

    async def handler():
        raise RuntimeError("side effects already happened")

    with pytest.raises(RuntimeError):
        await with_metering(
            tool_name="create_workflow",
            args={},
            tenant_id="t1",
            edition="community",
            db=db,
            handler=handler,
        )
    # Not refunded
    assert db.counters[("t1", "llm_calls")] == 1


@pytest.mark.asyncio
async def test_with_metering_timeout_refunds():
    db = _StubSession()

    async def handler():
        await asyncio.sleep(5)
        return {"ok": True}

    # Override the timeout for this specific call by monkey-patching the
    # registry — the orchestrator reads it at call time.
    from mcp_server import metering as m
    m.TOOL_TIMEOUT["create_workflow"] = 0.05

    try:
        with pytest.raises(ToolTimeoutError):
            await with_metering(
                tool_name="create_workflow",
                args={},
                tenant_id="t1",
                edition="community",
                db=db,
                handler=handler,
            )
        # Timeout refunds (no side effects guaranteed)
        assert db.counters.get(("t1", "llm_calls"), 0) == 0
    finally:
        # Restore default
        m.TOOL_TIMEOUT["create_workflow"] = 60.0


@pytest.mark.asyncio
async def test_with_metering_idempotency_replay_returns_cached():
    db = _StubSession()

    call_count = 0

    async def handler():
        nonlocal call_count
        call_count += 1
        return {"n": call_count}

    args1 = {"description": "x", "idempotency_key": "k-1"}
    first = await with_metering(
        tool_name="create_workflow",
        args=args1,
        tenant_id="t1",
        edition="community",
        db=db,
        handler=handler,
    )
    # Second invocation with same key + args should replay without running handler
    second = await with_metering(
        tool_name="create_workflow",
        args=args1,
        tenant_id="t1",
        edition="community",
        db=db,
        handler=handler,
    )
    assert first == second
    assert call_count == 1
    # Counter incremented only once
    assert db.counters[("t1", "llm_calls")] == 1


@pytest.mark.asyncio
async def test_with_metering_idempotency_key_reuse_different_args_raises():
    db = _StubSession()

    async def handler():
        return {"ok": True}

    await with_metering(
        tool_name="create_workflow",
        args={"description": "first", "idempotency_key": "k-2"},
        tenant_id="t1",
        edition="community",
        db=db,
        handler=handler,
    )
    with pytest.raises(ValueError):
        await with_metering(
            tool_name="create_workflow",
            args={"description": "DIFFERENT", "idempotency_key": "k-2"},
            tenant_id="t1",
            edition="community",
            db=db,
            handler=handler,
        )


def test_quota_error_payload_shape():
    err = QuotaError(meter="llm_calls", limit=5000, used=5000)
    payload = err.to_payload()
    assert payload["error"] == "quota_exceeded"
    assert payload["meter"] == "llm_calls"
    assert payload["limit"] == 5000
    assert payload["used"] == 5000
    assert payload["upgrade_url"]
