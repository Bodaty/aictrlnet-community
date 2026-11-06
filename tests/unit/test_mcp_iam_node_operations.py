"""The `mcp` and `iam` workflow nodes call service methods that exist.

Both nodes used to call 14 and 10 methods their services never had
(`MCPService.get_all_tools`, `IAMService.get_all_agents`, ...), so every
operation failed with AttributeError. Here every service is an autospec of the
real class: a call to a method that is missing, or with the wrong signature,
fails the test. Each operation is run once; operations with nothing real behind
them are asserted to fail with a clear error, not to return a result.
"""
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from nodes.implementations.iam_node import IAMNode, IAMOperationUnavailable
from nodes.implementations.mcp_node import MCPNode, MCPOperationUnavailable
from nodes.models import NodeConfig, NodeType
from schemas.iam import IAMAgentResponse, IAMAgentMetrics, IAMMessageResponse
from services.iam import IAMService
from services.mcp_service import MCPService


def _cfg(**parameters):
    return NodeConfig(id="n1", name="n", type=NodeType.TASK, parameters=parameters)


# ---------------------------------------------------------------- IAM

def _agent(name, capabilities=(), status="active"):
    return IAMAgentResponse(
        id=uuid.uuid4(), name=name, agent_type="service", capabilities=list(capabilities),
        status=status, created_at="2026-01-01T00:00:00", updated_at="2026-01-01T00:00:00")


def _sent(sender_id, create):
    return IAMMessageResponse(
        id=uuid.uuid4(), sender_id=sender_id, recipient_id=create.recipient_id,
        message_type=create.message_type, content=create.content, status="pending",
        retry_count=0, created_at="2026-01-01T00:00:00")


@pytest.fixture
def iam():
    alice, bob, carol = _agent("alice", ["triage"]), _agent("bob", ["triage"]), _agent("carol")
    service = create_autospec(IAMService, instance=True)
    service.list_agents.return_value = [alice, bob, carol]
    service.discover_agents.return_value = [alice, bob]
    service.get_agent.side_effect = lambda agent_id: next(
        (a for a in (alice, bob, carol) if a.id == agent_id), None)
    service.send_message.side_effect = lambda sender_id, create: _sent(sender_id, create)
    service.get_agent_metrics.return_value = IAMAgentMetrics(
        agent_id=alice.id, messages_sent=2, messages_received=1, avg_response_time_ms=5,
        error_rate=0.0, last_activity="2026-01-01T00:00:00", health_score=1.0)
    service.agents = SimpleNamespace(alice=alice, bob=bob, carol=carol)
    return service


def _iam_node(service, **parameters):
    node = IAMNode(_cfg(**parameters))
    node._iam_service = service
    node._db = MagicMock()
    node._db.scalar = AsyncMock(return_value=True)  # the caller is a superuser
    return node


_RUN = {"user_id": "admin", "tenant_id": "t1"}


@pytest.mark.asyncio
async def test_iam_send_message_stores_a_message_between_two_registered_agents(iam):
    node = _iam_node(iam, target_agent="bob", message="hello", sender_agent="alice")
    out = await node._send_message({}, _RUN)

    sender_id, create = iam.send_message.call_args.args
    assert sender_id == iam.agents.alice.id
    assert create.recipient_id == iam.agents.bob.id
    assert create.content == {"text": "hello"}
    assert out["status"] == "pending" and out["target_agent"] == "bob"


@pytest.mark.asyncio
async def test_iam_send_message_without_a_registered_sender_fails_clearly(iam):
    node = _iam_node(iam, target_agent="bob", message="hello")
    with pytest.raises(ValueError, match="sender is required"):
        await node._send_message({}, {**_RUN, "agent_id": None})


@pytest.mark.asyncio
@pytest.mark.parametrize("context", [{"user_id": "staff"}, {}])
async def test_iam_a_non_superuser_cannot_send_as_or_to_any_agent(iam, context):
    node = _iam_node(iam, target_agent="bob", message="hi", sender_agent="alice")
    node._db.scalar = AsyncMock(return_value=False)
    with pytest.raises(ValueError, match="IAM agent not found"):
        await node._send_message({}, context)
    node = _iam_node(iam, message="hi", sender_agent="alice")
    node._db.scalar = AsyncMock(return_value=False)
    with pytest.raises(ValueError, match="IAM agent not found"):
        await node._broadcast_message({}, context)
    iam.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_iam_unconfigured_delegation_still_skips(iam):
    out = await _iam_node(iam, agent="Coordinator", prompt="coordinate")._send_message({}, {})
    assert out["status"] == "skipped" and out["executed"] is False
    iam.send_message.assert_not_called()


@pytest.mark.asyncio
async def test_iam_broadcast_by_capability_sends_one_message_per_other_agent(iam):
    node = _iam_node(iam, message="ping", sender_agent="alice",
                     agent_filter={"capability": "triage"})
    out = await node._broadcast_message({}, _RUN)

    iam.discover_agents.assert_awaited_once_with(["triage"])
    assert [r["agent_id"] for r in out["delivery_results"]] == [str(iam.agents.bob.id)]
    assert out["broadcast_stats"] == {"total_agents": 1, "messages_sent": 1, "failed": 0}


@pytest.mark.asyncio
async def test_iam_broadcast_by_name_pattern_and_status(iam):
    node = _iam_node(iam, message="ping", sender_agent="alice", agent_filter={"pattern": "c*"})
    out = await node._broadcast_message({}, _RUN)
    assert [r["agent_id"] for r in out["delivery_results"]] == [str(iam.agents.carol.id)]

    await _iam_node(iam, message="ping", sender_agent="alice",
                    agent_filter={"status": "active"})._broadcast_message({}, _RUN)
    assert iam.list_agents.call_args.kwargs["status"] == "active"


@pytest.mark.asyncio
async def test_iam_discover_agents(iam):
    out = await _iam_node(iam, filters={"capability": "triage"})._discover_agents()
    assert out["count"] == 2 and out["agents"][0]["name"] == "alice"

    out = await _iam_node(iam, include_details=True)._discover_agents()
    assert out["count"] == 3
    assert iam.get_agent.await_count == 3


@pytest.mark.asyncio
async def test_iam_get_agent_info_with_metrics_and_capabilities(iam):
    node = _iam_node(iam, agent_id=str(iam.agents.alice.id),
                     include_metrics=True, include_capabilities=True)
    out = (await node._get_agent_info({}))["agent"]
    assert out["name"] == "alice"
    assert out["capabilities"] == ["triage"]
    assert out["metrics"]["messages_sent"] == 2


@pytest.mark.asyncio
async def test_iam_get_agent_info_unknown_agent_fails(iam):
    with pytest.raises(ValueError, match="not found"):
        await _iam_node(iam, agent_id="nobody")._get_agent_info({})


@pytest.mark.asyncio
@pytest.mark.parametrize("operation, params", [
    ("request", {"target_agent": "bob", "request": "?"}),
    ("subscribe", {"topic": "t"}),
    ("publish", {"topic": "t", "message": "m"}),
])
async def test_iam_operations_without_backing_fail_instead_of_returning_a_result(iam, operation, params):
    node = _iam_node(iam, operation=operation, **params)
    handler = {"request": node._send_request, "subscribe": node._subscribe_to_topic,
               "publish": node._publish_to_topic}[operation]
    with pytest.raises(IAMOperationUnavailable, match="not available"):
        await handler({}, {})


@pytest.mark.asyncio
async def test_iam_tag_filters_fail_clearly(iam):
    with pytest.raises(IAMOperationUnavailable, match="tags"):
        await _iam_node(iam, filters={"tag": "x"})._discover_agents()
    with pytest.raises(IAMOperationUnavailable, match="tags"):
        await _iam_node(iam, message="m", sender_agent="alice",
                        agent_filter={"tag": "x"})._broadcast_message({}, _RUN)


@pytest.mark.asyncio
async def test_iam_execute_builds_the_real_service_and_runs_discover(monkeypatch):
    service = create_autospec(IAMService, instance=True)
    service.list_agents.return_value = [_agent("alice")]
    monkeypatch.setattr("services.iam.IAMService", lambda db: service)
    out = await IAMNode(_cfg(operation="discover_agents")).execute({}, {"db": MagicMock()})
    assert out["count"] == 1


# ---------------------------------------------------------------- MCP

def _server(name="files", owner="u1"):
    return SimpleNamespace(id=str(uuid.uuid4()), name=name, owner_user_id=owner)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _Connection:
    def __init__(self, tools=(), resources=()):
        self.tools = [SimpleNamespace(name=t) for t in tools]
        self.resources = [SimpleNamespace(uri=u, name=u, description="d", mime_type="text/plain")
                          for u in resources]
        self.read_resource = AsyncMock(return_value={"contents": [{"text": "hi"}]})


@pytest.fixture
def mcp(monkeypatch):
    server = _server()
    db = MagicMock()
    db.execute = AsyncMock(return_value=_Result([server]))
    db.scalar = AsyncMock(return_value=False)
    client = SimpleNamespace(
        connections={server.id: _Connection(tools=["read_file"], resources=["file:///a"])},
        call_tool=AsyncMock(return_value={"content": [{"type": "text", "text": "done"}], "isError": False}),
    )

    async def get_client(self):
        return client

    monkeypatch.setattr(MCPNode, "_client", get_client)
    service = create_autospec(MCPService, instance=True)
    service.list_tools.return_value = [{"id": "t", "server_id": server.id, "name": "read_file",
                                        "description": None, "schema": {}}]
    return SimpleNamespace(server=server, db=db, client=client, service=service)


def _mcp_node(env, **parameters):
    node = MCPNode(_cfg(**parameters))
    node._db = env.db
    node._context = {"user_id": "u1", "tenant_id": "t1"}
    node._mcp_service = env.service
    return node


@pytest.mark.asyncio
async def test_mcp_discover_tools_lists_registered_tools_through_the_service(mcp):
    out = await _mcp_node(mcp, server_name="files")._discover_tools()

    assert out["count"] == 1 and out["tools"][0]["name"] == "read_file"
    kwargs = mcp.service.list_tools.call_args.kwargs
    assert kwargs["server_id"] == mcp.server.id
    assert kwargs["caller"].user_id == "u1"


@pytest.mark.asyncio
async def test_mcp_execute_tool_calls_the_connected_server(mcp):
    node = _mcp_node(mcp, server_name="files", tool_name="read_file", tool_args={"path": "/a"})
    out = await node._execute_tool({"tool_args": {"mode": "r"}})

    mcp.client.call_tool.assert_awaited_once_with(mcp.server.id, "read_file", {"path": "/a", "mode": "r"})
    assert out["server_name"] == "files" and out["tool_result"]["content"][0]["text"] == "done"


@pytest.mark.asyncio
async def test_mcp_execute_tool_finds_the_server_that_offers_the_tool(mcp):
    out = await _mcp_node(mcp, tool_name="read_file")._execute_tool({})
    assert out["server_name"] == "files"

    with pytest.raises(ValueError, match="No connected MCP server offers tool 'nope'"):
        await _mcp_node(mcp, tool_name="nope")._execute_tool({})


@pytest.mark.asyncio
async def test_mcp_execute_tool_failure_fails_the_node(mcp):
    mcp.client.call_tool.return_value = {"error": "boom"}
    with pytest.raises(RuntimeError, match="boom"):
        await _mcp_node(mcp, server_name="files", tool_name="read_file")._execute_tool({})


@pytest.mark.asyncio
async def test_mcp_a_server_the_user_may_not_use_is_not_found(mcp):
    mcp.server.owner_user_id = "someone-else"
    with pytest.raises(ValueError, match="not found"):
        await _mcp_node(mcp, server_name="files", tool_name="read_file")._execute_tool({})
    mcp.client.call_tool.assert_not_awaited()


@pytest.mark.asyncio
async def test_mcp_an_unconnected_server_says_how_to_connect_it(mcp):
    mcp.client.connections.clear()
    with pytest.raises(ValueError, match="not connected"):
        await _mcp_node(mcp, server_name="files", tool_name="read_file")._execute_tool({})


@pytest.mark.asyncio
async def test_mcp_list_and_get_resource(mcp):
    out = await _mcp_node(mcp, server_name="files")._list_resources()
    assert out["count"] == 1 and out["resources"][0]["uri"] == "file:///a"

    out = await _mcp_node(mcp)._list_resources()
    assert out["resources"][0]["server_name"] == "files"

    out = await _mcp_node(mcp, server_name="files", resource_uri="file:///a")._get_resource({})
    assert out["resource"]["contents"][0]["text"] == "hi"
    mcp.client.connections[mcp.server.id].read_resource.assert_awaited_once_with("file:///a")


@pytest.mark.asyncio
async def test_mcp_aggregate_context_reads_resources_tools_and_static_data(mcp):
    sources = [
        {"type": "resource", "server_name": "files", "uri": "file:///a", "name": "doc"},
        {"type": "tool", "server_name": "files", "tool_name": "read_file", "name": "out"},
        {"type": "static", "name": "k", "data": 1},
    ]
    out = await _mcp_node(mcp, sources=sources)._aggregate_context({})
    assert set(out["context"]) == {"doc", "out", "k"} and out["errors"] == []


@pytest.mark.asyncio
async def test_mcp_aggregate_context_raises_when_every_source_failed(mcp):
    sources = [{"type": "resource", "server_name": "files", "uri": "a"},
               {"type": "prompt", "prompt_name": "p"}]
    mcp.client.connections[mcp.server.id].read_resource.side_effect = RuntimeError("down")
    with pytest.raises(RuntimeError, match="all 2 source"):
        await _mcp_node(mcp, sources=sources)._aggregate_context({})


@pytest.mark.asyncio
@pytest.mark.parametrize("handler, params", [
    ("_get_prompt", {"prompt_name": "p"}),
    ("_list_prompts", {}),
    ("_call_server", {"server_name": "files", "method": "ping"}),
])
async def test_mcp_operations_without_backing_fail_instead_of_returning_a_result(mcp, handler, params):
    with pytest.raises(MCPOperationUnavailable, match="not available"):
        await getattr(_mcp_node(mcp, **params), handler)(*([{}] if handler != "_list_prompts" else []))


@pytest.mark.asyncio
async def test_mcp_category_and_resource_type_filters_fail_clearly(mcp):
    with pytest.raises(MCPOperationUnavailable, match="category"):
        await _mcp_node(mcp, category="x")._discover_tools()
    with pytest.raises(MCPOperationUnavailable, match="resource_type"):
        await _mcp_node(mcp, resource_type="x")._list_resources()


@pytest.mark.asyncio
async def test_mcp_execute_builds_the_real_service_and_runs_discover(mcp, monkeypatch):
    monkeypatch.setattr("services.mcp_service.MCPService", lambda db: mcp.service)
    out = await MCPNode(_cfg(operation="discover_tools", server_name="files")).execute(
        {}, {"db": mcp.db, "user_id": "u1"})
    assert out["count"] == 1
