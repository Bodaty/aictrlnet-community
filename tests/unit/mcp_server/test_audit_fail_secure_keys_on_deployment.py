"""M3 — MCP audit fail-secure keys on the DEPLOYMENT edition, not the
tenant's plan.

``_audit_if_enterprise``'s ``fail_secure`` now reads ``AICTRLNET_EDITION``
straight from the environment (the edition this process was deployed as),
never ``plan_tier`` (the caller's tenant plan) — see
mcp_server/tool_executor.py. Compliance ENFORCEMENT
(``_enforce_compliance_if_enterprise``) is untouched and still keys on
plan_tier; these tests only exercise the audit gate.

Runs in dev-community-1, where ``aictrlnet_enterprise`` is not importable,
so each test injects a stub ``aictrlnet_enterprise.services.mcp_compliance``
module (mirrored from
tests/integration/mcp_server/test_plan_gate_subversion.py) and stubs
``self_server.audit_session`` / ``self_server.resolve_self_server_id`` with
a no-op session, so a "could not connect" failure can never be confused
with the stub's scripted raise.
"""

from __future__ import annotations

import contextlib
import logging
import sys
import types

import pytest

from mcp_server import observability, self_server
from mcp_server.tool_executor import ComplianceError, _audit_if_enterprise


@contextlib.asynccontextmanager
async def _stub_audit_session():
    yield None


async def _stub_resolve_self_server_id(db):
    return self_server.SELF_SERVER_ID


def _install_boom_audit(monkeypatch):
    """Stub aictrlnet_enterprise.services.mcp_compliance.MCPComplianceManager
    so it records every call, then always raises."""
    calls = []

    class _BoomAudit:
        async def audit_mcp_operation(self, **kw):
            calls.append(kw)
            raise RuntimeError("audit backend unreachable")

    mod = types.ModuleType("aictrlnet_enterprise.services.mcp_compliance")
    mod.MCPComplianceManager = _BoomAudit
    monkeypatch.setitem(sys.modules, "aictrlnet_enterprise.services.mcp_compliance", mod)
    monkeypatch.setattr(self_server, "audit_session", _stub_audit_session)
    monkeypatch.setattr(self_server, "resolve_self_server_id", _stub_resolve_self_server_id)
    return calls


def _spy_record_audit_write_failed(monkeypatch):
    spy_calls = []
    monkeypatch.setattr(
        observability,
        "record_audit_write_failed",
        lambda **kw: spy_calls.append(kw),
    )
    return spy_calls


@pytest.mark.asyncio
async def test_enterprise_deployment_fails_secure_for_community_plan_tenant(monkeypatch):
    """AICTRLNET_EDITION=enterprise, plan_tier=community -> ComplianceError.

    The whole point of M3: a community-plan tenant on an Enterprise
    deployment must NOT be able to silently drop its audit trail.
    """
    calls = _install_boom_audit(monkeypatch)
    spy_calls = _spy_record_audit_write_failed(monkeypatch)
    monkeypatch.setenv("AICTRLNET_EDITION", "enterprise")
    monkeypatch.setenv("MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE", "true")

    with pytest.raises(ComplianceError, match="audit backend unreachable"):
        await _audit_if_enterprise(
            tool_name="query_analytics",
            request_data={},
            response_data=None,
            user_id="u1",
            tenant_id="community-tenant",
            duration_ms=10.0,
            status="success",
            db=None,
            plan_tier="community",
        )

    assert calls, "audit_mcp_operation was never called — test exercised nothing"
    assert len(spy_calls) == 1
    assert spy_calls[0]["fail_secure"] is True


@pytest.mark.asyncio
async def test_business_deployment_warns_even_for_enterprise_plan_tenant(monkeypatch, caplog):
    """AICTRLNET_EDITION=business, plan_tier=enterprise -> returns, WARNING logged.

    A non-Enterprise DEPLOYMENT never fails secure, even if the tenant's
    own plan happens to be enterprise-tier.
    """
    calls = _install_boom_audit(monkeypatch)
    spy_calls = _spy_record_audit_write_failed(monkeypatch)
    monkeypatch.setenv("AICTRLNET_EDITION", "business")
    monkeypatch.setenv("MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE", "true")

    with caplog.at_level(logging.WARNING, logger="mcp_server.tool_executor"):
        await _audit_if_enterprise(
            tool_name="evaluate_policy",
            request_data={},
            response_data=None,
            user_id="u1",
            tenant_id="ent-tenant",
            duration_ms=10.0,
            status="success",
            db=None,
            plan_tier="enterprise",
        )

    assert calls, "audit_mcp_operation was never called — test exercised nothing"
    assert len(spy_calls) == 1
    assert spy_calls[0]["fail_secure"] is False
    assert any(
        "Enterprise audit logging failed" in record.message for record in caplog.records
    )


@pytest.mark.asyncio
async def test_rollback_lever_disables_fail_secure_on_enterprise_deployment(monkeypatch):
    """AICTRLNET_EDITION=enterprise, MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE=false
    -> returns. The operator rollback lever still works on the new gate."""
    calls = _install_boom_audit(monkeypatch)
    spy_calls = _spy_record_audit_write_failed(monkeypatch)
    monkeypatch.setenv("AICTRLNET_EDITION", "enterprise")
    monkeypatch.setenv("MCP_COMPLIANCE_REQUIRED_FOR_ENTERPRISE", "false")

    await _audit_if_enterprise(
        tool_name="evaluate_policy",
        request_data={},
        response_data=None,
        user_id="u1",
        tenant_id="community-tenant",
        duration_ms=10.0,
        status="success",
        db=None,
        plan_tier="community",
    )

    assert calls, "audit_mcp_operation was never called — test exercised nothing"
    assert len(spy_calls) == 1
    assert spy_calls[0]["fail_secure"] is False
