"""Metrics + structured logging for MCP pipeline gates.

Emits one metric per gate decision + one per tool invocation. Backed by
``prometheus_client`` if installed; degrades to a no-op counter
otherwise (same API). Every emit also produces a structured log line
so observability works even without Prometheus in place.

The counters are process-local; aggregation across workers happens at
the Prometheus scrape / log-shipper layer.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger("mcp.observability")

# ---------------------------------------------------------------------------
# Prometheus counters with no-op fallback
# ---------------------------------------------------------------------------

try:
    from prometheus_client import Counter, Histogram  # type: ignore

    TOOL_INVOCATIONS = Counter(
        "mcp_tool_invocations_total",
        "MCP tool call count",
        ["tool", "status", "plan", "edition_filter"],
    )
    PLAN_DENIED = Counter(
        "mcp_plan_gate_denied_total",
        "Plan-gate denials",
        ["tool", "current_plan", "required_plan"],
    )
    SCOPE_DENIED = Counter(
        "mcp_scope_denied_total",
        "Scope check denials",
        ["tool", "auth_type"],
    )
    RATE_DENIED = Counter(
        "mcp_rate_limited_total",
        "Rate-limit denials",
        ["tool", "window"],
    )
    QUOTA_DENIED = Counter(
        "mcp_quota_exceeded_total",
        "Quota denials",
        ["tool", "meter"],
    )
    COMPLIANCE_DENIED = Counter(
        "mcp_compliance_denied_total",
        "Compliance-gate denials",
        ["tool"],
    )
    TOOL_DURATION = Histogram(
        "mcp_tool_duration_seconds",
        "MCP tool handler duration",
        ["tool", "status"],
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
    )
    _PROM_AVAILABLE = True
except Exception:  # prometheus_client not installed
    _PROM_AVAILABLE = False

    class _NoopMetric:
        def labels(self, *_args, **_kwargs):  # noqa: D401
            return self

        def inc(self, _amount: float = 1.0) -> None:
            return

        def observe(self, _value: float) -> None:
            return

    TOOL_INVOCATIONS = _NoopMetric()  # type: ignore
    PLAN_DENIED = _NoopMetric()  # type: ignore
    SCOPE_DENIED = _NoopMetric()  # type: ignore
    RATE_DENIED = _NoopMetric()  # type: ignore
    QUOTA_DENIED = _NoopMetric()  # type: ignore
    COMPLIANCE_DENIED = _NoopMetric()  # type: ignore
    TOOL_DURATION = _NoopMetric()  # type: ignore


# ---------------------------------------------------------------------------
# Emit helpers — each also emits a structured log line
# ---------------------------------------------------------------------------

def record_invocation(
    tool: str,
    status: str,
    plan: str = "unknown",
    edition_filter: str = "none",
    duration_seconds: Optional[float] = None,
    tenant_id: Optional[str] = None,
) -> None:
    TOOL_INVOCATIONS.labels(
        tool=tool, status=status, plan=plan, edition_filter=edition_filter
    ).inc()
    if duration_seconds is not None:
        TOOL_DURATION.labels(tool=tool, status=status).observe(duration_seconds)
    logger.info(
        "mcp_tool_invocation",
        extra={
            "tool": tool,
            "status": status,
            "plan": plan,
            "edition_filter": edition_filter,
            "duration_seconds": duration_seconds,
            "tenant_id": tenant_id,
        },
    )


def record_plan_denied(
    tool: str, current_plan: str, required_plan: str, tenant_id: Optional[str] = None
) -> None:
    PLAN_DENIED.labels(
        tool=tool, current_plan=current_plan, required_plan=required_plan
    ).inc()
    logger.info(
        "mcp_plan_gate_denied",
        extra={
            "tool": tool,
            "current_plan": current_plan,
            "required_plan": required_plan,
            "tenant_id": tenant_id,
        },
    )


def record_scope_denied(
    tool: str, auth_type: str, missing: list[str], tenant_id: Optional[str] = None
) -> None:
    SCOPE_DENIED.labels(tool=tool, auth_type=auth_type).inc()
    logger.info(
        "mcp_scope_denied",
        extra={
            "tool": tool,
            "auth_type": auth_type,
            "missing_scopes": missing,
            "tenant_id": tenant_id,
        },
    )


def record_rate_denied(
    tool: str, window: str, limit: int, tenant_id: Optional[str] = None
) -> None:
    RATE_DENIED.labels(tool=tool, window=window).inc()
    logger.info(
        "mcp_rate_limited",
        extra={
            "tool": tool,
            "window": window,
            "limit": limit,
            "tenant_id": tenant_id,
        },
    )


def record_quota_denied(
    tool: str,
    meter: str,
    limit: int,
    used: int,
    tenant_id: Optional[str] = None,
) -> None:
    QUOTA_DENIED.labels(tool=tool, meter=meter).inc()
    logger.info(
        "mcp_quota_exceeded",
        extra={
            "tool": tool,
            "meter": meter,
            "limit": limit,
            "used": used,
            "tenant_id": tenant_id,
        },
    )


def record_compliance_denied(
    tool: str, reason: str, tenant_id: Optional[str] = None
) -> None:
    COMPLIANCE_DENIED.labels(tool=tool).inc()
    logger.info(
        "mcp_compliance_denied",
        extra={"tool": tool, "reason": reason, "tenant_id": tenant_id},
    )


# ---------------------------------------------------------------------------
# Timing context manager
# ---------------------------------------------------------------------------

@contextmanager
def tool_span(tool: str):
    """Usage:

        with tool_span("create_workflow") as span:
            ...
            span["status"] = "success"

    Returns a mutable dict so the caller can override status before the
    span closes. Auto-emits a duration metric on exit.
    """
    start = time.monotonic()
    span: dict = {"status": "unknown"}
    try:
        yield span
    finally:
        TOOL_DURATION.labels(
            tool=tool, status=span.get("status", "unknown")
        ).observe(time.monotonic() - start)


__all__ = [
    "record_compliance_denied",
    "record_invocation",
    "record_plan_denied",
    "record_quota_denied",
    "record_rate_denied",
    "record_scope_denied",
    "tool_span",
]
