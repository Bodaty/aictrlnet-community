"""Atomic MCP metering + idempotency layer.

Wraps tool handler execution with:

1. **Idempotency lookup** — if the client supplied an ``idempotency_key``
   and we have a stored response for the same ``(tenant, tool, key)``
   tuple, return it directly (network-retry safe).
2. **Atomic quota decrement** — a single Postgres
   ``UPDATE ... WHERE counter + :qty <= limit RETURNING ...`` statement
   that rules out TOCTOU races between concurrent MCP calls.
3. **Timeout-wrapped execution** — ``asyncio.wait_for`` per tool.
4. **Refund on RefundableError** — only refund for exceptions we can
   prove did NOT produce side effects; never refund on arbitrary
   exceptions that may have partially completed.
5. **Idempotency store** — cache the response for future replays.

Counters live in a dedicated ``mcp_meters`` table (created by the
ZZZZ_mcp_idempotency_table migration) so MCP metering does not collide
with the existing per-edition ``basic_usage_metrics`` / Business
``usage_tracking`` tables. Those stay untouched as belt-and-suspenders
for non-MCP callers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

# Tool name -> (meter name, quantity expression)
# Quantity may be a literal int or a callable (args: dict) -> int.
TOOL_METERING: dict[str, tuple[str, Any]] = {
    # Original + Wave 1
    "create_workflow": ("llm_calls", 1),
    "send_message": ("llm_calls", 1),
    "assess_quality": ("llm_calls", 1),
    "nl_to_workflow": ("llm_calls", 1),
    "analyze_intent": ("llm_calls", 1),
    "research_api": ("llm_calls", 1),
    "generate_adapter": ("llm_calls", 2),
    "self_extend": ("llm_calls", 3),
    "browser_execute": (
        "browser_actions",
        lambda args: max(1, len(args.get("actions") or [])),
    ),
    # Wave 4
    "automate_company": ("llm_calls", 5),
    "promote_pattern_to_template": ("llm_calls", 1),
    "verify_quality": ("llm_calls", 1),
    "assess_data_quality": ("llm_calls", 1),
    "execute_agent": ("llm_calls", 1),
}

# Tool name -> timeout seconds. Default 30 for unlisted tools.
TOOL_TIMEOUT: dict[str, float] = {
    "create_workflow": 60.0,
    "execute_workflow": 30.0,  # just starts the execution; polled via get_execution_status
    "generate_adapter": 10.0,  # async-return; launches bg job
    "research_api": 60.0,
    "self_extend": 60.0,
    "browser_execute": 120.0,  # multi-action flows
    "automate_company": 10.0,  # async-return; launches bg job
    "nl_to_workflow": 60.0,
}

DEFAULT_TIMEOUT = 30.0
IDEMPOTENCY_TTL_HOURS = 24


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class RefundableError(Exception):
    """Raised inside a handler when an exception is guaranteed to have
    produced NO side effects — the meter charge should be refunded.

    Only raise this when you can prove no LLM call, no external API
    mutation, and no file write happened before the exception. Any
    ambiguity: don't subclass this.
    """


class QuotaError(Exception):
    """Raised when the caller exceeded their metered quota.

    Carries structured data for the transport layer to surface as an
    upgrade URL.
    """

    def __init__(
        self,
        meter: str,
        limit: int,
        used: int,
        reset_at: Optional[datetime] = None,
        upgrade_url: Optional[str] = None,
    ):
        self.meter = meter
        self.limit = limit
        self.used = used
        self.reset_at = reset_at
        self.upgrade_url = upgrade_url or f"/pricing?reason=quota&meter={meter}"
        super().__init__(
            f"Quota exceeded for meter '{meter}' (used {used}/{limit})"
        )

    def to_payload(self) -> dict:
        return {
            "error": "quota_exceeded",
            "meter": self.meter,
            "limit": self.limit,
            "used": self.used,
            "reset_at": self.reset_at.isoformat() if self.reset_at else None,
            "upgrade_url": self.upgrade_url,
        }


class ToolTimeoutError(Exception):
    """Raised when a tool handler exceeds TOOL_TIMEOUT."""


# ---------------------------------------------------------------------------
# Default per-edition limits (fallback when mcp_meters row has no limit)
# Matches values in subscription_plans.limits JSON + basic_usage_metrics
# defaults. Community tier mirrors basic_usage_metrics.api_calls_month.
# ---------------------------------------------------------------------------

DEFAULT_LIMITS: dict[str, dict[str, int]] = {
    "community": {"llm_calls": 5_000, "browser_actions": 1_000},
    "business":  {"llm_calls": 100_000, "browser_actions": 50_000},
    "enterprise": {"llm_calls": 10_000_000, "browser_actions": 1_000_000},
}


def get_default_limit(edition: str, meter: str) -> int:
    return DEFAULT_LIMITS.get(edition, DEFAULT_LIMITS["community"]).get(meter, 0)


# ---------------------------------------------------------------------------
# Quantity resolver
# ---------------------------------------------------------------------------

def resolve_quantity(qty_expr: Any, args: dict) -> int:
    if callable(qty_expr):
        try:
            return int(qty_expr(args))
        except Exception:
            return 1
    try:
        return int(qty_expr)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Atomic charge / refund
# ---------------------------------------------------------------------------

async def charge_atomic(
    tenant_id: str,
    meter: str,
    qty: int,
    limit: int,
    db: AsyncSession,
) -> tuple[bool, int, Optional[datetime]]:
    """Atomically increment the meter if headroom exists.

    Implementation: single SQL ``UPDATE ... WHERE new_value <= limit
    RETURNING counter, period_end``. Relies on row-level locking in
    Postgres for concurrency safety — no TOCTOU window.

    Returns ``(allowed, used_after, period_end)``.
    """
    # The mcp_meters table is created by the scope+idempotency migration;
    # schema: (tenant_id, meter, counter, limit_override, period_start,
    # period_end). UNIQUE(tenant_id, meter, period_start).
    # If the row doesn't exist yet, we insert with counter=qty and
    # limit_override=NULL (uses DEFAULT_LIMITS at read time).
    sql = text(
        """
        INSERT INTO mcp_meters (
            tenant_id, meter, counter, limit_override,
            period_start, period_end, updated_at
        )
        VALUES (
            :tenant_id, :meter, :qty, NULL,
            date_trunc('month', now()),
            date_trunc('month', now()) + interval '1 month',
            now()
        )
        ON CONFLICT (tenant_id, meter, period_start) DO UPDATE
        SET counter = mcp_meters.counter + :qty,
            updated_at = now()
        WHERE mcp_meters.counter + :qty <= COALESCE(mcp_meters.limit_override, :default_limit)
           OR :default_limit = 0
        RETURNING counter, period_end;
        """
    )
    try:
        result = await db.execute(
            sql,
            {
                "tenant_id": tenant_id,
                "meter": meter,
                "qty": qty,
                "default_limit": limit if limit > 0 else 0,
            },
        )
        row = result.first()
        await db.commit()
    except Exception as e:
        # Metering infra unavailable — never block the MCP call.
        # Log loudly; observability layer will surface the degraded state.
        logger.warning("charge_atomic failed (%s): %s; allowing request", meter, e)
        try:
            await db.rollback()
        except Exception:
            pass
        return True, 0, None

    if row is None:
        # The ON CONFLICT WHERE clause failed (over limit) and no row
        # was returned. Query the current value for the error payload.
        used, period_end = await _current_counter(db, tenant_id, meter)
        return False, used, period_end

    counter, period_end = row
    return True, int(counter), period_end


async def _current_counter(
    db: AsyncSession, tenant_id: str, meter: str
) -> tuple[int, Optional[datetime]]:
    sql = text(
        """
        SELECT counter, period_end
        FROM mcp_meters
        WHERE tenant_id = :tenant_id
          AND meter = :meter
          AND period_start = date_trunc('month', now())
        LIMIT 1
        """
    )
    try:
        row = (await db.execute(sql, {"tenant_id": tenant_id, "meter": meter})).first()
        if row:
            return int(row[0]), row[1]
    except Exception:
        pass
    return 0, None


async def refund(
    tenant_id: str, meter: str, qty: int, db: AsyncSession
) -> None:
    """Decrement the counter on RefundableError."""
    sql = text(
        """
        UPDATE mcp_meters
        SET counter = GREATEST(0, counter - :qty), updated_at = now()
        WHERE tenant_id = :tenant_id
          AND meter = :meter
          AND period_start = date_trunc('month', now())
        """
    )
    try:
        await db.execute(
            sql, {"tenant_id": tenant_id, "meter": meter, "qty": qty}
        )
        await db.commit()
    except Exception as e:
        logger.warning("refund failed (%s, %s): %s", tenant_id, meter, e)
        try:
            await db.rollback()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

def _args_hash(args: dict) -> str:
    return hashlib.sha256(
        json.dumps(args, sort_keys=True, default=str).encode()
    ).hexdigest()


async def idempotency_lookup(
    db: AsyncSession,
    tenant_id: str,
    tool_name: str,
    key: str,
    args: dict,
) -> Optional[dict]:
    """Return cached response if the same key has been seen before.

    ``args_hash`` mismatch is treated as a conflict — we do NOT replay
    the stored response because the caller actually sent different
    arguments. Raises ValueError in that case so the handler can return
    a 409-style error.
    """
    sql = text(
        """
        SELECT response_json, args_hash, created_at
        FROM mcp_idempotency_keys
        WHERE tenant_id = :tenant_id
          AND tool_name = :tool_name
          AND idempotency_key = :key
          AND created_at > now() - interval ':ttl hours'
        LIMIT 1
        """.replace(":ttl", str(IDEMPOTENCY_TTL_HOURS))
    )
    try:
        row = (
            await db.execute(
                sql, {"tenant_id": tenant_id, "tool_name": tool_name, "key": key}
            )
        ).first()
    except Exception as e:
        logger.warning("idempotency_lookup failed: %s", e)
        return None

    if not row:
        return None

    stored_response, stored_hash, _ = row
    if stored_hash != _args_hash(args):
        raise ValueError(
            "idempotency key reused with different arguments"
        )
    try:
        return json.loads(stored_response) if isinstance(stored_response, str) else stored_response
    except Exception:
        return None


async def idempotency_store(
    db: AsyncSession,
    tenant_id: str,
    tool_name: str,
    key: str,
    args: dict,
    response: dict,
) -> None:
    sql = text(
        """
        INSERT INTO mcp_idempotency_keys (
            tenant_id, tool_name, idempotency_key, args_hash,
            response_json, created_at
        ) VALUES (
            :tenant_id, :tool_name, :key, :args_hash, :response, now()
        )
        ON CONFLICT (tenant_id, tool_name, idempotency_key) DO NOTHING
        """
    )
    try:
        await db.execute(
            sql,
            {
                "tenant_id": tenant_id,
                "tool_name": tool_name,
                "key": key,
                "args_hash": _args_hash(args),
                "response": json.dumps(response, default=str),
            },
        )
        await db.commit()
    except Exception as e:
        logger.warning("idempotency_store failed: %s", e)
        try:
            await db.rollback()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Orchestrator — wraps handler execution
# ---------------------------------------------------------------------------

HandlerFn = Callable[[], Awaitable[dict]]


async def with_metering(
    tool_name: str,
    args: dict,
    tenant_id: str,
    edition: str,
    db: AsyncSession,
    handler: HandlerFn,
) -> dict:
    """Run ``handler`` under the MCP metering pipeline.

    Pipeline:
      1. Idempotency lookup (if key supplied).
      2. Resolve quantity for this tool/meter; charge atomically.
      3. Run handler under timeout.
      4. On RefundableError: refund + re-raise.
      5. Cache response for idempotency replay.
    """
    meter_entry = TOOL_METERING.get(tool_name)
    idem_key = args.get("idempotency_key") if isinstance(args, dict) else None
    timeout = TOOL_TIMEOUT.get(tool_name, DEFAULT_TIMEOUT)

    # 1. Idempotency replay
    if idem_key:
        cached = await idempotency_lookup(db, tenant_id, tool_name, idem_key, args)
        if cached is not None:
            return cached

    # 2. Meter charge
    qty = 0
    meter = None
    if meter_entry is not None:
        meter, qty_expr = meter_entry
        qty = resolve_quantity(qty_expr, args)
        if qty > 0 and tenant_id:
            limit = get_default_limit(edition, meter)
            allowed, used, period_end = await charge_atomic(
                tenant_id=tenant_id, meter=meter, qty=qty, limit=limit, db=db
            )
            if not allowed:
                raise QuotaError(
                    meter=meter, limit=limit, used=used, reset_at=period_end
                )

    # 3. Execute with timeout
    try:
        result = await asyncio.wait_for(handler(), timeout=timeout)
    except asyncio.TimeoutError as e:
        if meter and qty > 0:
            await refund(tenant_id, meter, qty, db)
        raise ToolTimeoutError(
            f"Tool '{tool_name}' exceeded {timeout}s timeout"
        ) from e
    except RefundableError:
        if meter and qty > 0:
            await refund(tenant_id, meter, qty, db)
        raise
    # Any other exception: side effects may have happened; do not refund.

    # 4. Idempotency store
    if idem_key and isinstance(result, dict):
        await idempotency_store(db, tenant_id, tool_name, idem_key, args, result)

    return result


__all__ = [
    "DEFAULT_LIMITS",
    "DEFAULT_TIMEOUT",
    "IDEMPOTENCY_TTL_HOURS",
    "QuotaError",
    "RefundableError",
    "TOOL_METERING",
    "TOOL_TIMEOUT",
    "ToolTimeoutError",
    "charge_atomic",
    "get_default_limit",
    "idempotency_lookup",
    "idempotency_store",
    "refund",
    "resolve_quantity",
    "with_metering",
]
