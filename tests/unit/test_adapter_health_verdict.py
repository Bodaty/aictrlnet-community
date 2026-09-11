"""A-5: BaseAdapter.health_check must surface the adapter's own verdict.
Today the outer "status" is the lifecycle state (ready) regardless of what
_perform_health_check returned (base_adapter.py:296-322), and
AdapterConfigService.test_config trusts the outer key (adapter_config.py:234-245),
so invalid credentials test as SUCCESS. Declared open via A5_IS_OPEN.
"""
import pytest
from adapters.base_adapter import BaseAdapter
from adapters.models import AdapterConfig, AdapterCategory, AdapterRequest, AdapterResponse

A5_IS_OPEN = True


class _Probe(BaseAdapter):
    async def initialize(self): pass
    async def shutdown(self): pass
    def get_capabilities(self): return []
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(request_id="x", capability="x", status="success")
    async def _perform_health_check(self):
        return {"status": "unhealthy", "error": "invalid_auth"}


@pytest.mark.asyncio
async def test_health_check_reports_the_adapters_verdict_not_its_lifecycle_state():
    adapter = _Probe(AdapterConfig(name="probe", category=AdapterCategory.UTILITY))
    await adapter.start()  # lifecycle -> ready
    health = await adapter.health_check()
    verdict_surfaced = health.get("status") == "unhealthy" and health.get("error") == "invalid_auth"
    if A5_IS_OPEN:
        assert not verdict_surfaced, "A-5 declared open but the verdict now surfaces - flip A5_IS_OPEN and close it"
        assert health.get("status") == "ready", health  # documents the defect precisely
    else:
        assert verdict_surfaced, health
