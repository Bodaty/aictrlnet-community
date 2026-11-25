"""An adapter step can return its whole output under one key (``output_key``).

A run's data accumulates step over step (``{**input, **output}``), so a failure a
step kept with ``fail_on_error: false`` (``_failed``, ``error``) stays visible to
every later step, including one reading another adapter's answer. Under
``output_key`` each call's answer, success or failure, is its own object, and a
failure carries the adapter's ``error_code`` (``HTTP_503``, ``resource_missing``).
"""
import pytest

from nodes.base_node import AdapterCallError
from nodes.implementations.adapter_node import AdapterNode
from nodes.models import NodeConfig, NodeType


def _node(**params):
    base = {"adapter_id": "stripe", "capability": "list_events", "fail_on_error": False}
    base.update(params)
    return AdapterNode(NodeConfig(name="fetch", type=NodeType.ADAPTER, parameters=base))


def _answer(node, monkeypatch, outcome):
    async def fake_call_adapter(**kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(node, "call_adapter", fake_call_adapter)


async def test_a_success_is_nested_under_the_key(monkeypatch):
    node = _node(output_key="stripe_fetch")
    _answer(node, monkeypatch, {"events": [], "newest_id": "evt_1"})
    out = await node.execute({}, {})
    assert out == {"stripe_fetch": {"events": [], "newest_id": "evt_1", "_adapter_called": "stripe",
                                    "_capability_used": "list_events"}}


async def test_a_kept_failure_is_nested_with_its_error_code(monkeypatch):
    node = _node(output_key="stripe_fetch")
    _answer(node, monkeypatch, AdapterCallError("Adapter error: No such event", "resource_missing"))
    out = await node.execute({}, {})
    assert set(out) == {"stripe_fetch"}
    failed = out["stripe_fetch"]
    assert failed["_failed"] is True and failed["error_code"] == "resource_missing"
    assert "No such event" in failed["error"]


async def test_a_kept_failure_without_the_key_carries_the_error_code_at_the_top(monkeypatch):
    node = _node()
    _answer(node, monkeypatch, AdapterCallError("Adapter error: down", "HTTP_503"))
    out = await node.execute({}, {})
    assert out["_failed"] is True and out["error_code"] == "HTTP_503"


async def test_a_dry_run_is_nested_too():
    node = _node(output_key="stripe_fetch")
    out = await node.execute({}, {"is_dry_run": True})
    assert out["stripe_fetch"]["_dry_run"] is True


async def test_without_the_key_the_output_is_unchanged(monkeypatch):
    node = _node()
    _answer(node, monkeypatch, {"events": []})
    assert await node.execute({}, {}) == {"events": [], "_adapter_called": "stripe",
                                          "_capability_used": "list_events"}


async def test_fail_on_error_still_raises_with_the_key(monkeypatch):
    node = _node(output_key="stripe_fetch", fail_on_error=True)
    _answer(node, monkeypatch, AdapterCallError("Adapter error: down", "HTTP_503"))
    with pytest.raises(AdapterCallError):
        await node.execute({}, {})
