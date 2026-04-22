"""Unit tests for mcp_server.plan_gate."""

import pytest

from mcp_server.plan_gate import (
    PLAN_MUTATING_TOOLS,
    PlanError,
    PlanService,
    TOOL_MIN_PLAN,
    enforce_plan,
    normalize_edition,
)


class _FakePlanService:
    """Stand-in for PlanService that returns a canned tier.

    Matches the public interface used by ``enforce_plan``.
    """

    def __init__(self, tier: str):
        self._tier = tier
        self.lookups = 0

    async def get_effective_edition(self, tenant_id):
        self.lookups += 1
        return self._tier

    def bust_cache(self, tenant_id=None):
        pass


def test_normalize_edition_known_values():
    assert normalize_edition("community") == "community"
    assert normalize_edition("business") == "business"
    assert normalize_edition("enterprise") == "enterprise"


def test_normalize_edition_subtiers_collapse():
    assert normalize_edition("business_starter") == "business"
    assert normalize_edition("business_pro") == "business"
    assert normalize_edition("business_scale") == "business"
    assert normalize_edition("enterprise_plus") == "enterprise"


def test_normalize_edition_fallback():
    assert normalize_edition(None) == "community"
    assert normalize_edition("") == "community"
    assert normalize_edition("unknown_tier") == "community"


@pytest.mark.asyncio
async def test_enforce_plan_allows_equal_tier():
    svc = _FakePlanService("community")
    tier = await enforce_plan("create_workflow", "t1", svc)
    assert tier == "community"


@pytest.mark.asyncio
async def test_enforce_plan_allows_higher_tier():
    svc = _FakePlanService("enterprise")
    tier = await enforce_plan("list_workflows", "t1", svc)
    assert tier == "enterprise"


@pytest.mark.asyncio
async def test_enforce_plan_denies_lower_tier():
    svc = _FakePlanService("community")
    with pytest.raises(PlanError) as exc:
        await enforce_plan("evaluate_policy", "t1", svc)
    assert exc.value.required == "business"
    assert exc.value.current == "community"
    assert "upgrade" in exc.value.upgrade_url.lower() or "pricing" in exc.value.upgrade_url.lower()


@pytest.mark.asyncio
async def test_enforce_plan_denies_community_on_enterprise_tool():
    svc = _FakePlanService("community")
    with pytest.raises(PlanError):
        await enforce_plan("query_analytics", "t1", svc)


@pytest.mark.asyncio
async def test_enforce_plan_denies_business_on_enterprise_tool():
    svc = _FakePlanService("business")
    with pytest.raises(PlanError):
        await enforce_plan("check_compliance", "t1", svc)


@pytest.mark.asyncio
async def test_enforce_plan_business_allowed_on_business_tool():
    svc = _FakePlanService("business")
    tier = await enforce_plan("evaluate_policy", "t1", svc)
    assert tier == "business"


def test_plan_error_payload_shape():
    err = PlanError(tool_name="x", required="business", current="community")
    payload = err.to_payload()
    assert payload["error"] == "plan_upgrade_required"
    assert payload["tool"] == "x"
    assert payload["required_plan"] == "business"
    assert payload["current_plan"] == "community"
    assert payload["upgrade_url"]


def test_tool_min_plan_registry_is_tiered():
    # Spot checks covering every tier so the registry doesn't silently regress.
    assert TOOL_MIN_PLAN["create_workflow"] == "community"
    assert TOOL_MIN_PLAN["research_api"] == "business"
    assert TOOL_MIN_PLAN["generate_adapter"] == "business"
    assert TOOL_MIN_PLAN["check_compliance"] == "enterprise"
    assert TOOL_MIN_PLAN["query_analytics"] == "enterprise"


@pytest.mark.asyncio
async def test_plan_service_per_request_cache():
    """Two lookups for the same tenant in the same PlanService instance
    should hit the DB once."""

    class _CountingPlanService(PlanService):
        def __init__(self):
            super().__init__(db=None)  # type: ignore[arg-type]
            self._resolve_calls = 0

        async def _resolve(self, tenant_id):
            self._resolve_calls += 1
            return "business"

    svc = _CountingPlanService()
    t1 = await svc.get_effective_edition("tenant-a")
    t2 = await svc.get_effective_edition("tenant-a")
    assert t1 == t2 == "business"
    assert svc._resolve_calls == 1


@pytest.mark.asyncio
async def test_plan_service_bust_cache():
    class _CountingPlanService(PlanService):
        def __init__(self):
            super().__init__(db=None)  # type: ignore[arg-type]
            self._resolve_calls = 0

        async def _resolve(self, tenant_id):
            self._resolve_calls += 1
            return "business" if self._resolve_calls > 1 else "community"

    svc = _CountingPlanService()
    assert await svc.get_effective_edition("t1") == "community"
    svc.bust_cache("t1")
    assert await svc.get_effective_edition("t1") == "business"
    assert svc._resolve_calls == 2


def test_plan_mutating_tools_is_set_of_strings():
    assert isinstance(PLAN_MUTATING_TOOLS, set)
    for t in PLAN_MUTATING_TOOLS:
        assert isinstance(t, str)
        assert t in TOOL_MIN_PLAN
