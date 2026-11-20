"""Starting an adapter that is already registered reuses its registration.

Every workflow step that calls an adapter builds a new instance and starts it
(BaseNode.call_adapter, AIProcessNode). The first start registers the adapter
with the control plane; each later start used to be refused ("Component <name>
already registered"), logged at ERROR twice, and recovered by matching that
text in the callers.
"""
import logging
import uuid

import pytest

from adapters.base_adapter import BaseAdapter
from adapters.models import AdapterCategory, AdapterConfig, AdapterRequest, AdapterResponse, AdapterStatus
from control_plane.models import ComponentRegistrationRequest, ComponentType
from control_plane.registry import component_registry
from control_plane.services import control_plane_service


class _Probe(BaseAdapter):
    async def initialize(self): pass
    async def shutdown(self): pass
    def get_capabilities(self): return []
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        return AdapterResponse(request_id="x", capability="x", status="success")


def _named(name):
    return [c for c in component_registry.components.values() if c.name == name]


@pytest.mark.asyncio
async def test_starting_the_same_adapter_twice_reuses_its_registration(caplog):
    name = f"probe-{uuid.uuid4().hex[:8]}"
    first = _Probe(AdapterConfig(name=name, category=AdapterCategory.UTILITY))
    second = _Probe(AdapterConfig(name=name, category=AdapterCategory.UTILITY))

    caplog.set_level(logging.INFO)
    await first.start()
    await second.start()

    assert second.status == AdapterStatus.READY
    assert second._component_id == first._component_id
    assert len(_named(name)) == 1
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR] == []


@pytest.mark.asyncio
async def test_an_adapter_does_not_take_over_a_name_another_registrant_holds():
    name = f"probe-{uuid.uuid4().hex[:8]}"
    await control_plane_service.register_component(
        ComponentRegistrationRequest(name=name, type=ComponentType.ADAPTER, version="1.0.0"),
        user_id="someone-else",
    )
    adapter = _Probe(AdapterConfig(name=name, category=AdapterCategory.UTILITY))

    with pytest.raises(ValueError, match="already registered"):
        await adapter.start()
    assert [c.registered_by for c in _named(name)] == ["someone-else"]


@pytest.mark.asyncio
async def test_registering_a_component_twice_through_the_api_is_still_refused():
    name = f"comp-{uuid.uuid4().hex[:8]}"
    request = ComponentRegistrationRequest(name=name, type=ComponentType.ADAPTER, version="1.0.0")
    await control_plane_service.register_component(request, user_id="u1")

    with pytest.raises(ValueError, match="already registered"):
        await control_plane_service.register_component(request, user_id="u1")
