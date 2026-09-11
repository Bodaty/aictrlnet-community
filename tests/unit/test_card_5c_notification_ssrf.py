"""Regression: security #5c - NotificationNode webhook channel posted to a
recipient-supplied URL with no SSRF validation. Fails on pre-fix code.
"""
import pytest
from nodes.registry import get_node_registry
from nodes.models import NodeConfig, NodeType, NodeInstance


def _node(**params):
    params.setdefault("custom_node_type", "notification")
    cfg = NodeConfig(id="n", name="n", type=NodeType.TASK, parameters=params)
    return get_node_registry().create_node(cfg)


@pytest.mark.asyncio
async def test_webhook_channel_refuses_internal_target():
    node = _node(channel="webhook")
    out = await node._send_webhook(
        [{"url": "http://169.254.169.254/latest/meta-data/"}],
        {"subject": "s", "body": "b"},
    )
    results = out.get("results", out if isinstance(out, list) else [])
    assert results, out
    r = results[0]
    assert r.get("success") is False
    assert r.get("error"), "internal target must be refused with an error, not silently posted"
