"""MCP (Model Context Protocol) node implementation.

Every operation here calls something that exists:

- tool discovery reads the registered tools through ``MCPService.list_tools``;
- tool calls and resource reads go through the MCP client adapter that holds the
  live connections (the one the ``/api/v1/mcp/servers/{id}/connect`` route fills);
- a server a workflow may not use is "not found", the same answer the MCP
  routes give.

An operation with nothing real behind it (prompts, raw server methods, filters
on fields tools do not carry) raises ``MCPOperationUnavailable`` rather than
returning an empty or invented result.
"""

import logging
from typing import Any, Dict, List, Optional
import json

from sqlalchemy import or_, select

from ..base_node import BaseNode
from ..models import NodeConfig
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class MCPOperationUnavailable(RuntimeError):
    """The operation has no implementation behind it on this platform."""


def _unavailable(operation: str, why: str) -> MCPOperationUnavailable:
    return MCPOperationUnavailable(f"MCP operation '{operation}' is not available: {why}")


class MCPNode(BaseNode):
    """Node for Model Context Protocol operations.

    Operations:
    - ``discover_tools`` / ``execute_tool``
    - ``list_resources`` / ``get_resource``
    - ``aggregate_context`` (resource, tool and static sources)

    ``get_prompt``, ``list_prompts`` and ``call_server`` are accepted in a
    workflow definition but fail when run: the MCP client has no prompts support
    and exposes no raw-method call.
    """

    _db = None
    _context: Dict[str, Any] = {}

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MCP node. Returns output dict for BaseNode.run() to wrap."""
        from services.mcp_service import MCPService
        db = context.get('db')
        if not db:
            raise ValueError("Database session not provided in context")
        # Held on the instance because every operation helper below needs them.
        # Node instances are created fresh per NodeRegistry.create_node call, so
        # per-execution state on self is safe.
        self._db = db
        self._context = context
        self._mcp_service = MCPService(db)

        operation = self.config.parameters.get("operation", "execute_tool")

        if operation == "execute_tool":
            output_data = await self._execute_tool(input_data)
        elif operation == "discover_tools":
            output_data = await self._discover_tools()
        elif operation == "get_resource":
            output_data = await self._get_resource(input_data)
        elif operation == "list_resources":
            output_data = await self._list_resources()
        elif operation == "get_prompt":
            output_data = await self._get_prompt(input_data)
        elif operation == "list_prompts":
            output_data = await self._list_prompts()
        elif operation == "aggregate_context":
            output_data = await self._aggregate_context(input_data)
        elif operation == "call_server":
            output_data = await self._call_server(input_data)
        else:
            raise ValueError(f"Unsupported MCP operation: {operation}")

        await event_bus.publish(
            "node.executed",
            {
                "node_id": self.config.id,
                "node_type": "mcp",
                "operation": operation
            }
        )

        return output_data

    # ------------------------------------------------------------------
    # Access and connections
    # ------------------------------------------------------------------

    async def _caller(self):
        """The principal the workflow runs as: the user who started it."""
        from core.mcp_access import MCPCaller
        from models.user import User

        user_id = self._context.get("user_id")
        tenant_id = self._context.get("tenant_id")
        is_super = False
        if user_id:
            is_super = bool(await self._db.scalar(
                select(User.is_superuser).where(User.id == str(user_id))))
        return MCPCaller(
            user_id=str(user_id) if user_id else None,
            tenant_id=str(tenant_id) if tenant_id else None,
            is_superuser=is_super,
        )

    async def _client(self):
        """The MCP client adapter that holds the live server connections."""
        from api.v1.endpoints.mcp import get_mcp_client
        return await get_mcp_client()

    async def _resolve_server(self, ref: str, *, use: bool):
        """The server named by id or name that the workflow's user may read
        (``use=False``) or use (``use=True``). Anything else is "not found"."""
        from core.mcp_access import can_read, can_use
        from models.community_complete import MCPServer

        caller = await self._caller()
        allowed = can_use if use else can_read
        rows = (await self._db.execute(
            select(MCPServer).where(or_(MCPServer.id == str(ref), MCPServer.name == str(ref)))
        )).scalars().all()
        visible = [s for s in rows if allowed(s, caller)]
        if not visible:
            raise ValueError(f"MCP server '{ref}' not found")
        if len(visible) > 1:
            raise ValueError(f"More than one MCP server is named '{ref}'; use its id")
        return visible[0]

    async def _connection(self, server):
        client = await self._client()
        connection = client.connections.get(str(server.id))
        if connection is None:
            raise ValueError(
                f"MCP server '{server.name}' is not connected. An administrator "
                f"connects it with POST /api/v1/mcp/servers/{server.id}/connect."
            )
        return client, connection

    async def _server_for_tool(self, tool_name: str):
        """The one connected server the workflow may use that offers the tool."""
        from core.mcp_access import can_use
        from models.community_complete import MCPServer

        client = await self._client()
        ids = [sid for sid, conn in client.connections.items()
               if any(t.name == tool_name for t in conn.tools)]
        if not ids:
            raise ValueError(f"No connected MCP server offers tool '{tool_name}'")
        caller = await self._caller()
        rows = (await self._db.execute(
            select(MCPServer).where(MCPServer.id.in_(ids)))).scalars().all()
        usable = [s for s in rows if can_use(s, caller)]
        if not usable:
            raise ValueError(f"No MCP server found with tool '{tool_name}'")
        if len(usable) > 1:
            raise ValueError(
                f"Tool '{tool_name}' is offered by {len(usable)} servers; set server_name")
        return usable[0]

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------

    async def _run_tool(self, server_ref: Optional[str], tool_name: str,
                        arguments: Dict[str, Any]) -> Dict[str, Any]:
        if server_ref:
            server = await self._resolve_server(server_ref, use=True)
        else:
            server = await self._server_for_tool(tool_name)
        client, _ = await self._connection(server)

        result = await client.call_tool(str(server.id), tool_name, arguments)
        if isinstance(result, dict) and (result.get("error") or result.get("isError")):
            # A failed call is a failed node, not a completed one carrying an
            # error payload.
            raise RuntimeError(
                f"MCP tool '{tool_name}' failed: {result.get('error') or result.get('content')}"
            )
        return {
            "tool_result": result,
            "tool_name": tool_name,
            "server_name": server.name
        }

    async def _execute_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool."""
        server_name = self.config.parameters.get("server_name")
        tool_name = self.config.parameters.get("tool_name") or input_data.get("tool_name")
        tool_args = dict(self.config.parameters.get("tool_args") or {})

        if not tool_name:
            raise ValueError("tool_name is required for execute_tool operation")

        if "tool_args" in input_data:
            tool_args.update(input_data["tool_args"])

        return await self._run_tool(server_name, tool_name, tool_args)

    async def _discover_tools(self) -> Dict[str, Any]:
        """List the registered MCP tools the workflow's user may see."""
        if self.config.parameters.get("category"):
            raise _unavailable(
                "discover_tools",
                "registered tools carry no category, so a category filter cannot be applied")

        server_id = None
        server_name = self.config.parameters.get("server_name")
        if server_name:
            server_id = str((await self._resolve_server(server_name, use=False)).id)

        tools = await self._mcp_service.list_tools(
            server_id=server_id, caller=await self._caller())
        return {
            "tools": tools,
            "count": len(tools)
        }

    async def _read_resource(self, server_ref: Optional[str], uri: str) -> Dict[str, Any]:
        if not server_ref:
            raise ValueError("server_name is required to read a resource")
        server = await self._resolve_server(server_ref, use=True)
        _, connection = await self._connection(server)

        resource = await connection.read_resource(uri)
        if isinstance(resource, dict) and resource.get("error"):
            raise RuntimeError(f"Resource '{uri}' could not be read: {resource['error']}")
        return resource

    async def _get_resource(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Read an MCP resource."""
        server_name = self.config.parameters.get("server_name")
        resource_uri = self.config.parameters.get("resource_uri") or input_data.get("resource_uri")

        if not resource_uri:
            raise ValueError("resource_uri is required for get_resource operation")

        resource = await self._read_resource(server_name, resource_uri)
        if not resource:
            raise ValueError(f"Resource '{resource_uri}' not found")

        return {
            "resource": resource,
            "uri": resource_uri
        }

    async def _list_resources(self) -> Dict[str, Any]:
        """List the resources of the connected MCP servers the workflow's user may read."""
        from core.mcp_access import can_read
        from models.community_complete import MCPServer

        if self.config.parameters.get("resource_type"):
            raise _unavailable(
                "list_resources",
                "resources carry a mime type, not a type, so a resource_type filter cannot be applied")

        server_name = self.config.parameters.get("server_name")
        if server_name:
            server = await self._resolve_server(server_name, use=False)
            client, _ = await self._connection(server)
            servers = [server]
        else:
            client = await self._client()
            ids = list(client.connections)
            caller = await self._caller()
            rows = (await self._db.execute(
                select(MCPServer).where(MCPServer.id.in_(ids)))).scalars().all() if ids else []
            servers = [s for s in rows if can_read(s, caller)]

        resources = [
            {
                "uri": r.uri,
                "name": r.name,
                "description": r.description,
                "mime_type": r.mime_type,
                "server_name": s.name,
            }
            for s in servers
            for r in client.connections[str(s.id)].resources
        ]
        return {
            "resources": resources,
            "count": len(resources)
        }

    async def _get_prompt(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise _unavailable("get_prompt", "the MCP client does not implement prompts/get")

    async def _list_prompts(self) -> Dict[str, Any]:
        raise _unavailable("list_prompts", "the MCP client does not implement prompts/list")

    async def _call_server(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        raise _unavailable(
            "call_server",
            "the MCP client exposes tool calls and resource reads, not arbitrary server methods")

    async def _aggregate_context(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate context from multiple sources."""
        sources = self.config.parameters.get("sources", [])
        if not sources and "sources" in input_data:
            sources = input_data["sources"]

        if not sources:
            raise ValueError("sources are required for aggregate_context operation")

        aggregated_context = {}
        errors = []

        for source in sources:
            try:
                source_type = source.get("type")

                if source_type == "resource":
                    resource = await self._read_resource(
                        source.get("server_name"), source.get("uri"))
                    if resource:
                        aggregated_context[source.get("name", source.get("uri"))] = resource

                elif source_type == "tool":
                    result = await self._run_tool(
                        source.get("server_name"),
                        source.get("tool_name"),
                        source.get("arguments", {}))
                    if result:
                        aggregated_context[source.get("name", source.get("tool_name"))] = result["tool_result"]

                elif source_type == "prompt":
                    await self._get_prompt({})

                elif source_type == "static":
                    aggregated_context[source.get("name", "static")] = source.get("data")

            except Exception as e:
                errors.append({
                    "source": source,
                    "error": str(e)
                })

        output_format = self.config.parameters.get("output_format", "structured")

        if output_format == "text":
            text_parts = []
            for key, value in aggregated_context.items():
                text_parts.append(f"=== {key} ===")
                if isinstance(value, dict):
                    text_parts.append(json.dumps(value, indent=2))
                else:
                    text_parts.append(str(value))
                text_parts.append("")

            self._require_some_context(aggregated_context, errors, sources)
            return {
                "context": "\n".join(text_parts),
                "errors": errors
            }

        else:
            self._require_some_context(aggregated_context, errors, sources)
            return {
                "context": aggregated_context,
                "errors": errors
            }

    @staticmethod
    def _require_some_context(aggregated_context, errors, sources) -> None:
        """Raise when every source failed.

        Partial failure is tolerated and surfaced in `errors` - some sources
        being unavailable is a normal condition for context aggregation. But
        when nothing at all was aggregated and every source errored, the node
        did no work; returning an empty context under a COMPLETED status hands
        downstream nodes an empty prompt and calls it success.
        """
        if not aggregated_context and errors and len(errors) >= len(sources):
            raise RuntimeError(
                f"Context aggregation produced nothing: all {len(sources)} "
                f"source(s) failed. First error: {errors[0]}"
            )

    def validate_config(self) -> bool:
        """Validate node configuration."""
        operation = self.config.parameters.get("operation")
        if not operation:
            raise ValueError("operation parameter is required")

        valid_operations = [
            "execute_tool", "discover_tools", "get_resource", "list_resources",
            "get_prompt", "list_prompts", "aggregate_context", "call_server"
        ]

        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")

        # Validate operation-specific requirements
        if operation == "execute_tool" and not self.config.parameters.get("tool_name"):
            # Tool name can come from input, so this is OK
            pass
        elif operation == "get_resource" and not self.config.parameters.get("resource_uri"):
            # Resource URI can come from input, so this is OK
            pass
        elif operation == "get_prompt" and not self.config.parameters.get("prompt_name"):
            # Prompt name can come from input, so this is OK
            pass
        elif operation == "aggregate_context" and not self.config.parameters.get("sources"):
            # Sources can come from input, so this is OK
            pass
        elif operation == "call_server" and not self.config.parameters.get("server_name"):
            raise ValueError("server_name is required for call_server operation")

        return True
