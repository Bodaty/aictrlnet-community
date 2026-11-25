"""An AI step can return its whole output under one key (``output_key``).

A JSON answer's top-level fields are copied onto the step's output, and a run's
data accumulates step over step (``{**input, **output}``), so without the key a
model's answer could overwrite anything an earlier step put in the run. Under
``output_key`` the output, parsed fields included, is one object under that key;
without it nothing changes.
"""
import json

import pytest

from adapters.base_adapter import BaseAdapter
from adapters.models import AdapterCapability, AdapterCategory, AdapterResponse
from adapters.registry import adapter_registry
from nodes.implementations.ai_process_node import AIProcessNode
from nodes.models import NodeConfig, NodeType

ANSWER = {"subject": "Hi", "body": "Hello", "contact": {"email": "evil@x.invalid"}, "work_item_id": "w-evil"}


class _JsonAdapter(BaseAdapter):
    async def initialize(self):
        return None

    async def shutdown(self):
        return None

    def get_capabilities(self):
        return [AdapterCapability(name="generate", description="generate")]

    async def execute(self, request):
        return AdapterResponse(request_id=request.id, capability=request.capability, status="success",
                               data={"text": json.dumps(ANSWER), "model": "m"}, duration_ms=0.0)


@pytest.fixture
def node_with(monkeypatch):
    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    adapter_registry.register_adapter_class("json_output_key", _JsonAdapter, AdapterCategory.AI)

    def make(**extra):
        return AIProcessNode(NodeConfig(id="draft", name="draft", type=NodeType.TASK, parameters={
            "ai_task": "generate", "adapter_id": "json_output_key", "model": "m", "prompt": "write", **extra}))
    return make


async def test_under_output_key_the_answers_fields_stay_inside_it(node_with):
    out = await node_with(output_key="draft").execute({"work_item_id": "w1"}, {})
    assert list(out) == ["draft"]
    assert out["draft"]["parsed_output"] == ANSWER
    assert out["draft"]["generated_text"] == json.dumps(ANSWER)
    accumulated = {"work_item_id": "w1", "contact": {"email": "ann@example.com"}, **out}
    assert accumulated["work_item_id"] == "w1" and accumulated["contact"] == {"email": "ann@example.com"}


async def test_without_output_key_the_output_is_unchanged(node_with):
    out = await node_with().execute({}, {})
    assert out["parsed_output"] == ANSWER
    assert out["work_item_id"] == "w-evil" and out["subject"] == "Hi"     # copied to the top level, as before


async def test_a_blank_output_key_is_ignored(node_with):
    out = await node_with(output_key="  ").execute({}, {})
    assert "parsed_output" in out and "  " not in out
