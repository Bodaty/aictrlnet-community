"""Plan-tier gate for MCP tool calls.

Maps each MCP tool to the minimum plan tier (``community`` / ``business``
/ ``enterprise``) required to invoke it. Checked BEFORE scope validation
so unauthorized-by-plan callers get a structured upgrade response, not a
permission-denied response.

``PlanService`` resolves the caller's effective plan tier from the
``subscriptions`` + ``subscription_plans`` tables. Sub-tiers (e.g.
``business_starter``, ``business_pro``) all resolve to ``business`` for
MCP access; sub-tier monetization stays in Stripe.

Cache: per-JSON-RPC-request. ``PlanService`` memoizes the lookup keyed
by ``tenant_id`` so a single HTTP request (which may carry a batch of
JSON-RPC calls) does not repeat the DB hit. The cache is invalidated
explicitly when a plan-mutating tool runs.
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Plan tier ranking
# ---------------------------------------------------------------------------

_TIER_RANK = {"community": 0, "business": 1, "enterprise": 2}
_VALID_TIERS = set(_TIER_RANK)


def _rank(tier: str) -> int:
    return _TIER_RANK.get(tier, 0)


def normalize_edition(edition: Optional[str]) -> str:
    """Collapse sub-tiers (business_starter, business_pro, ...) to the
    canonical tier string used for gating: community | business | enterprise.
    """
    if not edition:
        return "community"
    e = edition.lower().strip()
    if e in _VALID_TIERS:
        return e
    if e.startswith("business"):
        return "business"
    if e.startswith("enterprise"):
        return "enterprise"
    return "community"


# ---------------------------------------------------------------------------
# Tool -> minimum plan registry
# Populated for the original 11 tools and all Wave 1-6 additions.
# Tools not listed default to "community".
# ---------------------------------------------------------------------------

TOOL_MIN_PLAN: dict[str, str] = {
    # Original 11 tools
    "create_workflow": "community",
    "list_workflows": "community",
    "get_workflow": "community",
    "execute_workflow": "community",
    "get_execution_status": "community",
    "list_templates": "community",
    "assess_quality": "community",
    "send_message": "community",
    "evaluate_policy": "business",
    "list_policies": "business",
    "check_compliance": "enterprise",
    # Wave 1 — Three-Layer Reach + Control Spectrum + approvals
    "list_adapters": "community",
    "get_adapter": "community",
    "list_my_adapter_configs": "community",
    "test_adapter_config": "community",
    "nl_to_workflow": "community",
    "analyze_intent": "community",
    "get_workflow_autonomy": "community",
    "preview_autonomy": "community",
    "set_workflow_autonomy": "business",
    "research_api": "business",
    "generate_adapter": "business",
    "self_extend": "business",
    "list_generated_adapters": "business",
    "get_generated_adapter_status": "business",
    "get_generated_adapter_source": "business",
    "approve_adapter": "business",
    "reject_adapter": "business",
    "activate_adapter": "business",
    "browser_execute": "business",
    "list_pending_approvals": "business",
    "get_approval": "business",
    "approve_request": "business",
    "reject_request": "business",
    # Wave 2 — API-key economy + governance visibility
    "list_api_keys": "community",
    "get_api_key_usage": "community",
    "get_subscription": "community",
    "get_upgrade_options": "community",
    "list_ai_policies": "business",
    "get_ai_audit_logs": "business",
    "list_violations": "business",
    "create_policy": "business",
    # Wave 3 — Metering audit
    "get_trial_status": "community",
    "get_usage_report": "community",
    # Wave 4 — Horizontal + Living Platform + Agents
    "create_task": "community",
    "list_tasks": "community",
    "get_task": "community",
    "update_task": "community",
    "complete_task": "community",
    "list_conversations": "community",
    "get_conversation": "community",
    "list_linked_channels": "community",
    "request_channel_link_code": "community",
    "unlink_channel": "community",
    "send_channel_message": "business",
    "list_notifications": "business",
    "mark_notification_read": "business",
    "query_knowledge": "community",
    "suggest_next_actions": "community",
    "get_capabilities_summary": "community",
    "get_memory": "community",
    "set_memory": "community",
    "delete_memory": "community",
    "upload_file": "community",
    "list_staged_files": "community",
    "get_staged_file": "community",
    "search_templates": "community",
    "instantiate_template": "community",
    "list_agents": "business",
    "get_agent_capabilities": "business",
    "set_agent_autonomy": "business",
    "execute_agent": "business",
    "list_llm_models": "business",
    "get_llm_recommendation": "business",
    "list_pattern_candidates": "business",
    "promote_pattern_to_template": "business",
    "org_discovery_scan": "business",
    "get_org_landscape": "business",
    "get_org_recommendations": "business",
    "automate_company": "business",
    "get_company_automation_status": "business",
    "verify_quality": "business",
    "assess_data_quality": "community",
    "list_quality_dimensions": "community",
    # Wave 5 — Institute
    "list_institute_modules": "community",
    "enroll_in_module": "community",
    "get_certification_status": "community",
    # Wave 6 — Enterprise admin
    "query_analytics": "enterprise",
    "get_dashboard_metrics": "enterprise",
    "get_metric_trends": "enterprise",
    "get_audit_logs": "enterprise",
    "get_audit_summary": "enterprise",
    "run_compliance_check": "enterprise",
    "list_compliance_standards": "enterprise",
    "list_organizations": "enterprise",
    "list_tenants": "enterprise",
    "get_enterprise_risk_assessment": "enterprise",
    "federated_knowledge_query": "enterprise",
    "get_cross_tenant_insights": "enterprise",
    "list_fleet_agents": "enterprise",
    "get_fleet_autonomy_summary": "enterprise",
    "get_license_status": "enterprise",
    "list_license_entitlements": "enterprise",
}

# Tools that mutate the plan / subscription — executing one of these
# busts the per-request plan cache so subsequent calls in the same
# batch see fresh tier.
PLAN_MUTATING_TOOLS: set[str] = {"get_upgrade_options"}  # read-only; nothing mutates yet via MCP


# ---------------------------------------------------------------------------
# PlanError
# ---------------------------------------------------------------------------

class PlanError(Exception):
    """Raised when the caller's plan tier is below the tool's minimum.

    Carries structured payload so the MCP transport can surface an
    upgrade URL to the client.
    """

    def __init__(
        self,
        tool_name: str,
        required: str,
        current: str,
        upgrade_url: Optional[str] = None,
    ):
        self.tool_name = tool_name
        self.required = required
        self.current = current
        self.upgrade_url = upgrade_url or f"/pricing?from={current}&tool={tool_name}"
        super().__init__(
            f"Tool '{tool_name}' requires {required} plan (current: {current})"
        )

    def to_payload(self) -> dict:
        return {
            "error": "plan_upgrade_required",
            "tool": self.tool_name,
            "required_plan": self.required,
            "current_plan": self.current,
            "upgrade_url": self.upgrade_url,
        }


# ---------------------------------------------------------------------------
# PlanService
# ---------------------------------------------------------------------------

class PlanService:
    """Resolves the effective plan tier for a tenant.

    Caches per-instance (one ``PlanService`` per JSON-RPC HTTP request).
    ``bust_cache`` is called by the pipeline after a plan-mutating tool.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._cache: dict[str, str] = {}

    async def get_effective_edition(self, tenant_id: Optional[str]) -> str:
        """Return canonical plan tier: community | business | enterprise."""
        key = tenant_id or "__default__"
        if key in self._cache:
            return self._cache[key]

        tier = await self._resolve(tenant_id)
        self._cache[key] = tier
        return tier

    def bust_cache(self, tenant_id: Optional[str] = None) -> None:
        if tenant_id is None:
            self._cache.clear()
        else:
            self._cache.pop(tenant_id, None)
            self._cache.pop("__default__", None)

    async def _resolve(self, tenant_id: Optional[str]) -> str:
        # Import locally to avoid import-order issues with the models
        # layer (and to keep this module importable in contexts where
        # the DB schema isn't loaded yet — e.g. unit tests).
        try:
            from models.subscription import (  # type: ignore
                Subscription,
                SubscriptionPlan,
                SubscriptionStatus,
            )
        except ImportError:
            logger.debug("subscription models unavailable; defaulting to community")
            return "community"

        if not tenant_id:
            return "community"

        try:
            # Find an active/trialing/past_due subscription for the tenant.
            stmt = (
                select(SubscriptionPlan.edition)
                .join(Subscription, Subscription.plan_id == SubscriptionPlan.id)
                .where(
                    Subscription.tenant_id == tenant_id,
                    Subscription.status.in_([
                        SubscriptionStatus.ACTIVE,
                        SubscriptionStatus.TRIALING,
                        SubscriptionStatus.PAST_DUE,
                    ]),
                )
                .limit(1)
            )
            row = (await self.db.execute(stmt)).first()
            if row and row[0]:
                return normalize_edition(row[0])
        except Exception as e:  # defensive — never block MCP on plan-lookup failure
            logger.warning("plan lookup failed for tenant=%s: %s", tenant_id, e)

        return "community"


# ---------------------------------------------------------------------------
# Public gate
# ---------------------------------------------------------------------------

async def enforce_plan(
    tool_name: str,
    tenant_id: Optional[str],
    plan_service: PlanService,
) -> str:
    """Raise PlanError if the tenant's plan does not cover ``tool_name``.

    Returns the resolved plan tier on success (useful for downstream
    logging / metering).
    """
    required = TOOL_MIN_PLAN.get(tool_name, "community")
    current = await plan_service.get_effective_edition(tenant_id)
    if _rank(current) < _rank(required):
        raise PlanError(tool_name=tool_name, required=required, current=current)
    return current


__all__ = [
    "PLAN_MUTATING_TOOLS",
    "PlanError",
    "PlanService",
    "TOOL_MIN_PLAN",
    "enforce_plan",
    "normalize_edition",
]
