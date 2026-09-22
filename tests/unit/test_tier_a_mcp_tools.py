"""Tier A-6: call every MCP tool once through execute_tool() - the production
pipeline (plan gate, scopes, metering, handler) - with schema-derived minimal
arguments, and classify the outcome. The only failing class is DEAD: success-
shaped output while doing nothing.

  REAL             a dict with substantive keys and no error/placeholder
  SURFACED         raised ToolExecutionError with a specific message (the tool refused - real)
  DECLARED_PENDING feature_pending AND the tool's description says so (contract)
  NEEDS_LIVE       SmokeEgressBlocked - the tool needs a third party; recorded unverified-live
  GATED            plan/scope/quota/rate gate refused the dev user - fixture problem, not a tool problem
  DEAD             feature_pending undeclared; or `error` under success; or None/empty; or Unknown tool

Ledger semantics as in the other Tier A gates (exact-set on DEAD tool names).
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))  # find tier_a_* helpers without shadowing src packages

import asyncio
import pytest
import pytest_asyncio
from sqlalchemy import text
from mcp_server.tools import get_tools_for_edition
from mcp_server.tool_executor import execute_tool, ToolExecutionError, ScopeError
from mcp_server.plan_gate import PlanError, PlanService
from mcp_server.metering import QuotaError
from mcp_server.rate_bucket import RateError
from tier_a_egress import SmokeEgressBlocked, synthesize_value

TOOLS = {t["name"]: t for t in get_tools_for_edition()}
DECLARED_PENDING = {n for n, t in TOOLS.items() if "feature_pending" in (t.get("description") or "")}
PER_TOOL_TIMEOUT = 20

# On a fresh DB (no Business demo seed), default-tenant's plan resolves
# from subscription rows that only that seed creates -- absent it, every
# copy of this sweep would resolve "community" and see 106/142 tools
# GATED before real classification is possible. Pin the tier this
# KNOWN_DEAD ledger was calibrated against instead. This is also the one
# line that ends this file's byte-identity with the Business copy (no
# test enforces that identity).
PINNED_PLAN_TIER = "community"


class _PinnedPlanService(PlanService):
    def __init__(self, db, tier: str):
        super().__init__(db)
        self._tier = tier

    async def _resolve(self, tenant_id):
        return self._tier

# A-30: tools that return feature_pending without declaring it (and F-6: the credential four).
# Shared across editions; only names present in this edition's catalogue are compared.
KNOWN_DEAD = {
    "allocate_resource": "A-30",
    "analyze_cost_trends": "A-30",
    "configure_update_notifications": "A-30",
    "create_credential": "A-30",
    "create_template": "A-30",
    "delete_credential": "A-30",
    "delete_template": "A-30",
    "discover_federated_capabilities": "A-30",
    "generate_signed_file_url": "A-30",
    "get_cost_analytics": "A-30",
    "get_credential": "A-30",
    "get_delegation_chain": "A-30",
    "get_execution_framework_trace": "A-30",
    "get_notification_preferences": "A-30",
    "get_org_discovery_logs": "A-30",
    "get_org_discovery_status": "A-30",
    "get_platform_cost_estimate": "A-30",
    "list_credentials": "A-30",
    "list_federated_peers": "A-30",
    "list_resource_pools": "A-30",
    "list_runtime_webhooks": "A-30",
    "optimize_workflow_cost": "A-30",
    "register_federated_peer": "A-30",
    "rescan_adapter_registry": "A-30",
    "rollback_workflow": "A-30",
    "set_channel_notification_rules": "A-30",
    "set_framework_priority": "A-30",
    "set_notification_frequency": "A-30",
    "share_resource_with_peer": "A-30",
    "update_notification_preferences": "A-30",
    "update_template": "A-30",
}


def _chain_has(exc, cls):
    while exc is not None:
        if isinstance(exc, cls):
            return True
        exc = exc.__cause__ or exc.__context__
    return False



@pytest_asyncio.fixture(autouse=True)
async def _clean_mcp_server_rows(db):
    """Delete the mcp_servers rows this sweep registers.

    register_mcp_server COMMITs on the request session, so the module's
    `finally: await db.rollback()` cannot undo it and every gate run used to
    leave another row behind.
    """
    before = {r[0] for r in (await db.execute(text("SELECT id FROM mcp_servers"))).all()}
    yield
    try:
        await db.rollback()
        rows = (await db.execute(text("SELECT id FROM mcp_servers"))).all()
        added = [r[0] for r in rows if r[0] not in before]
        if added:
            await db.execute(
                text("DELETE FROM mcp_server_capabilities WHERE server_id = ANY(:ids)"),
                {"ids": added},
            )
            await db.execute(text("DELETE FROM mcp_servers WHERE id = ANY(:ids)"), {"ids": added})
            await db.commit()
    except Exception:  # noqa: BLE001 - cleanup must not mask a real failure
        await db.rollback()

@pytest_asyncio.fixture
async def dev_user_id(db):
    await db.execute(text("select set_config('app.current_tenant_id', 'default-tenant', false)"))
    row = (await db.execute(text("SELECT id FROM users WHERE email='dev@aictrlnet.com'"))).first()
    assert row, "dev@aictrlnet.com missing - make fresh"
    return str(row[0])


async def _classify(name, db, dev_user_id, plan_service):
    args = synthesize_value(TOOLS[name]["inputSchema"], components={})
    try:
        result = await asyncio.wait_for(
            execute_tool(
                name, args, db, dev_user_id, tenant_id="default-tenant",
                plan_service=plan_service,
            ),
            timeout=PER_TOOL_TIMEOUT,
        )
    except ToolExecutionError as e:
        msg = str(e)
        return ("DEAD", msg) if msg.startswith("Unknown tool") else ("SURFACED", msg[:120])
    except (PlanError, ScopeError, QuotaError, RateError) as e:
        return "GATED", f"{type(e).__name__}: {str(e)[:80]}"
    except asyncio.TimeoutError:
        return "TIMEOUT", f">{PER_TOOL_TIMEOUT}s"
    except Exception as e:  # noqa: BLE001
        if _chain_has(e, SmokeEgressBlocked):
            return "NEEDS_LIVE", type(e).__name__
        return "RAISED", f"{type(e).__name__}: {str(e)[:120]}"
    finally:
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
    if not result or not isinstance(result, dict):
        return "DEAD", f"returned {result!r}"
    if result.get("status") == "feature_pending":
        return ("DECLARED_PENDING" if name in DECLARED_PENDING else "DEAD"), "feature_pending"
    if result.get("error"):
        return "DEAD", f"error under success: {str(result.get('error'))[:100]}"
    return "REAL", ",".join(sorted(result)[:5])


@pytest.mark.asyncio
async def test_every_mcp_tool_is_not_dead_or_is_in_the_ledger(db, dev_user_id):
    plan_service = _PinnedPlanService(db, PINNED_PLAN_TIER)
    classes, detail = {}, {}
    for name in sorted(TOOLS):
        k, d = await _classify(name, db, dev_user_id, plan_service)
        classes[name] = k; detail[name] = d
    from collections import Counter
    counts = Counter(classes.values())
    dead = {n: detail[n] for n, k in classes.items() if k == "DEAD"}
    raised = {n: detail[n] for n, k in classes.items() if k in ("RAISED", "TIMEOUT")}
    print(f"\\ntier-a mcp tools ({len(TOOLS)}): {dict(counts)}")
    print(f"tier-a mcp DEAD: {dead}")
    print(f"tier-a mcp RAISED/TIMEOUT: {raised}")
    print(f"tier-a mcp GATED: {[n for n, k in classes.items() if k == 'GATED']}")
    new = {n: d for n, d in dead.items() if n not in KNOWN_DEAD}
    fixed = {n: KNOWN_DEAD[n] for n in KNOWN_DEAD if n in TOOLS and n not in dead}
    assert not new, f"NEW dead MCP tools (file them): {new}"
    assert not fixed, f"FIXED - remove from KNOWN_DEAD and close: {fixed}"


def test_tool_count_matches_source():
    assert len(TOOLS) in (51, 158, 194), len(TOOLS)
