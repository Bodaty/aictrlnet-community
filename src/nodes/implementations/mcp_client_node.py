"""MCP Client Node for calling tools on an external MCP server."""

import logging
from typing import Any, Dict, Optional

from ..base_node import BaseNode
from ..models import NodeConfig
from adapters.implementations.ai.mcp_client_adapter import (
    MCPConnection,
    MCPServerConfig,
    MCPTransportType,
)
from core.ssrf import validate_outbound_url
from events.event_bus import event_bus


logger = logging.getLogger(__name__)

# What the MCP protocol offers a client: tools. The node used to advertise
# message/quality/workflow operations that the protocol has no notion of, and
# handed them to an adapter speaking a private /mcp/v1/* dialect instead.
SUPPORTED_OPERATIONS = ("tool",)


class MCPClientNode(BaseNode):
    """Node that calls a tool on an external MCP server.

    Parameters:
    - mcp_server_url: URL of the MCP server (HTTP transport)
    - api_key: bearer token for that server (optional)
    - server_name: label used in logs and events (optional)
    - operation: "tool"
    - tool_name / arguments: what to call, also accepted from the node input
    """

    def __init__(self, config: NodeConfig):
        super().__init__(config)

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Call the configured tool and return its result."""
        operation = self.config.parameters.get("operation", "tool")
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Unsupported MCP operation '{operation}'. An MCP server exposes "
                f"tools; use operation='tool' with a tool_name."
            )

        server_url = self.config.parameters.get("mcp_server_url")
        if not server_url:
            raise ValueError("mcp_server_url is required")

        tool_name = self.config.parameters.get("tool_name") or input_data.get("tool_name")
        if not tool_name:
            raise ValueError("tool_name is required for the tool operation")

        arguments = dict(self.config.parameters.get("arguments") or {})
        arguments.update(input_data.get("arguments") or {})

        # Validated here rather than inside the client: the client reports a
        # refused connection as False, which would surface as "could not
        # connect" instead of naming the reason.
        import asyncio

        await asyncio.to_thread(validate_outbound_url, server_url)

        api_key = self.config.parameters.get("api_key")
        server_name = self.config.parameters.get("server_name") or server_url
        connection = MCPConnection(
            MCPServerConfig(
                name=server_name,
                command="",
                transport=MCPTransportType.HTTP_SSE,
                url=server_url,
                headers={"Authorization": f"Bearer {api_key}"} if api_key else None,
            )
        )

        try:
            if not await connection.connect():
                raise RuntimeError(f"Could not connect to MCP server '{server_name}'")

            result = await connection.call_tool(tool_name, arguments)
            if isinstance(result, dict) and (result.get("error") or result.get("isError")):
                # A failed call is a failed node, not a completed one carrying
                # an error payload.
                raise RuntimeError(
                    f"MCP tool '{tool_name}' failed: {result.get('error') or result.get('content')}"
                )

            await event_bus.publish(
                "node.mcp_client.executed",
                {
                    "node_id": self.config.id,
                    "server_name": server_name,
                    "tool_name": tool_name,
                    "success": True,
                },
            )
            return {
                "tool_result": result,
                "tool_name": tool_name,
                "server_name": server_name,
            }
        finally:
            disconnect = getattr(connection, "disconnect", None)
            if callable(disconnect):
                try:
                    await disconnect()
                except Exception:  # noqa: BLE001 - teardown must not mask the result
                    logger.debug("MCP client node: disconnect failed", exc_info=True)

    def validate_config(self) -> bool:
        """Validate node configuration."""
        if not self.config.parameters.get("mcp_server_url"):
            raise ValueError("mcp_server_url is required")

        operation = self.config.parameters.get("operation", "tool")
        if operation not in SUPPORTED_OPERATIONS:
            raise ValueError(
                f"Invalid operation: {operation}. Must be one of {list(SUPPORTED_OPERATIONS)}"
            )
        return True
