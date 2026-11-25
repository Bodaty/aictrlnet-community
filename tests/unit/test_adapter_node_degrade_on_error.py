"""An adapter step can mark its run degraded when it keeps a failure (``degrade_on_error``).

With ``fail_on_error: false`` a failed call is kept in the step's output and the run
goes on. ``degrade_on_error: "<reason>"`` also puts a top-level ``degraded`` marker
in the output, beside any ``output_key``, so the run ends ``completed`` with that
marker on its ``output_data.degraded`` instead of hiding the failure.
"""
import pytest

from nodes.base_node import AdapterCallError
from nodes.implementations.adapter_node import AdapterNode
from nodes.models import NodeConfig, NodeType
from services.workflow_execution import collect_degraded


def _node(**params):
    base = {"adapter_id": "gmail", "capability": "send_email", "fail_on_error": False}
    base.update(params)
    return AdapterNode(NodeConfig(name="send", type=NodeType.ADAPTER, parameters=base))


def _answer(node, monkeypatch, outcome):
    async def fake_call_adapter(**kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(node, "call_adapter", fake_call_adapter)


async def test_a_kept_failure_marks_the_step_degraded_beside_the_key(monkeypatch):
    node = _node(output_key="send", degrade_on_error="send_failed")
    _answer(node, monkeypatch, AdapterCallError("Adapter error: token revoked", "HTTP_401"))
    out = await node.execute({}, {})
    assert out["send"]["_failed"] is True
    assert out["degraded"] == {"reason": "send_failed", "detail": {
        "adapter": "gmail", "capability": "send_email", "error_code": "HTTP_401"}}
    assert collect_degraded({"send": out}) == [{"node_id": "send", "reason": "send_failed", "detail": {
        "adapter": "gmail", "capability": "send_email", "error_code": "HTTP_401"}}]


async def test_a_kept_failure_without_the_key_is_degraded_too(monkeypatch):
    node = _node(degrade_on_error="send_failed")
    _answer(node, monkeypatch, AdapterCallError("Adapter error: down", "HTTP_503"))
    out = await node.execute({}, {})
    assert out["_failed"] is True and out["degraded"]["reason"] == "send_failed"


async def test_a_success_is_not_degraded(monkeypatch):
    node = _node(output_key="send", degrade_on_error="send_failed")
    _answer(node, monkeypatch, {"message_id": "m1"})
    out = await node.execute({}, {})
    assert "degraded" not in out and out["send"]["message_id"] == "m1"


async def test_a_dry_run_is_not_degraded():
    node = _node(output_key="send", degrade_on_error="send_failed")
    out = await node.execute({}, {"is_dry_run": True})
    assert "degraded" not in out and out["send"]["_dry_run"] is True


async def test_without_the_parameter_a_kept_failure_is_unchanged(monkeypatch):
    node = _node(output_key="send")
    _answer(node, monkeypatch, AdapterCallError("Adapter error: down", "HTTP_503"))
    out = await node.execute({}, {})
    assert set(out) == {"send"}


async def test_fail_on_error_still_raises(monkeypatch):
    node = _node(output_key="send", degrade_on_error="send_failed", fail_on_error=True)
    _answer(node, monkeypatch, AdapterCallError("Adapter error: down", "HTTP_503"))
    with pytest.raises(AdapterCallError):
        await node.execute({}, {})
