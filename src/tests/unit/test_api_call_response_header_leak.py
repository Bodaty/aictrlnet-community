"""Regression: an apiCall node's RESPONSE headers must not become the next
apiCall's REQUEST headers via the executor's {**input, **output} accumulation.

Live failure (presence-audit workflow, Aug 4 2026): GitHub's
`content-length: 2355` response header leaked into 9 subsequent surface GETs,
all failing with httpx "Too little data for declared Content-Length".
Caught by our own automation — no Trello card.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from nodes.implementations.api_call_node import APICallNode
from nodes.models import NodeConfig, NodeType


def _node(node_id, url):
    return APICallNode(NodeConfig(
        id=node_id,
        name=f"apiCall-{node_id}",
        type=NodeType.TASK,
        parameters={"url": url, "method": "GET"},
    ))


def _mock_response():
    response = MagicMock()
    response.status_code = 200
    response.headers = httpx.Headers({
        "content-type": "application/json",
        "content-length": "2355",
    })
    response.is_redirect = False
    response.json.return_value = {"ok": True}
    response.raise_for_status.return_value = None
    return response


@pytest.mark.asyncio
async def test_response_headers_do_not_leak_into_next_request():
    client = MagicMock()
    client.request = AsyncMock(return_value=_mock_response())
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=client)
    client_cm.__aexit__ = AsyncMock(return_value=False)

    with patch("core.ssrf.validate_outbound_url"), \
         patch("nodes.implementations.api_call_node.httpx.AsyncClient", return_value=client_cm):
        first = await _node("a", "https://api.github.com/repos/x/y").execute({}, {})

        assert "headers" not in first
        assert first["response_headers"]["content-length"] == "2355"

        # Chain exactly like the executor does: {**input_data, **output_data}.
        accumulated = {**{}, **first}
        await _node("b", "https://example.com/surface").execute(accumulated, {})

    request_headers = client.request.call_args_list[-1].kwargs["headers"]
    leaked = {
        k for k in request_headers
        if k.lower() in {"content-length", "transfer-encoding", "connection"}
    }
    assert not leaked, f"response headers leaked into outgoing request: {leaked}"
