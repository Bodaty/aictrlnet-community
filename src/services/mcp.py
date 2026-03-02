"""MCP service for business logic."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import json
import time

from models.community import MCPServer, MCPTool, MCPInvocation


class MCPService:
    """Service for MCP-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_server_info(self) -> Dict[str, Any]:
        """Get overall MCP server information."""
        # Count servers
        server_count_query = select(func.count()).select_from(MCPServer)
        server_count = await self.db.scalar(server_count_query) or 0
        
        # Count tools
        tool_count_query = select(func.count()).select_from(MCPTool)
        tool_count = await self.db.scalar(tool_count_query) or 0
        
        # Standard MCP capabilities
        capabilities = [
            "tools/list",
            "tools/call",
            "prompts/list",
            "prompts/get",
            "resources/list",
            "resources/read",
        ]
        
        return {
            "capabilities": capabilities,
            "server_count": server_count,
            "tool_count": tool_count,
        }
    
    async def list_adapters(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[MCPServer]:
        """List MCP servers (adapters)."""
        query = select(MCPServer).order_by(MCPServer.name).offset(skip).limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def list_servers(
        self,
        skip: int = 0,
        limit: int = 100,
    ) -> List[MCPServer]:
        """List MCP servers (alias for list_adapters)."""
        return await self.list_adapters(skip=skip, limit=limit)
    
    async def get_server(self, server_id: str) -> Optional[MCPServer]:
        """Get a specific MCP server."""
        query = select(MCPServer).where(MCPServer.id == server_id)
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def get_server_tools(self, server_id: str) -> List[MCPTool]:
        """Get all tools for a specific server."""
        query = select(MCPTool).where(MCPTool.server_id == server_id)
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def get_capabilities(
        self,
        adapter_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get MCP capabilities, optionally filtered by adapter."""
        if adapter_id:
            # Get tools for specific adapter
            query = select(MCPTool).where(MCPTool.server_id == adapter_id)
            result = await self.db.execute(query)
            tools = result.scalars().all()
            
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.tool_schema,
                    "adapter_id": tool.server_id,
                }
                for tool in tools
            ]
        else:
            # Get all tools from all servers
            query = select(MCPTool).join(MCPServer).where(MCPServer.status == "active")
            result = await self.db.execute(query)
            tools = result.scalars().all()
            
            return [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.tool_schema,
                    "adapter_id": tool.server_id,
                }
                for tool in tools
            ]
    
    async def execute_command(
        self,
        adapter: str,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a command on an MCP adapter."""
        start_time = time.time()
        
        # Get the server
        server_query = select(MCPServer).where(
            MCPServer.id == adapter,
            MCPServer.status == "active"
        )
        server_result = await self.db.execute(server_query)
        server = server_result.scalar_one_or_none()
        
        if not server:
            raise ValueError(f"MCP server '{adapter}' not found or inactive")
        
        # Find the tool
        tool_query = select(MCPTool).where(
            MCPTool.server_id == adapter,
            MCPTool.name == method
        )
        tool_result = await self.db.execute(tool_query)
        tool = tool_result.scalar_one_or_none()
        
        if not tool:
            raise ValueError(f"Tool '{method}' not found on server '{adapter}'")
        
        # Execute via real MCP protocol for stdio transport
        try:
            result_data = await self._execute_mcp_tool(server, tool, params)
            status = "success"
            error_msg = None

        except Exception as e:
            result_data = None
            status = "error"
            error_msg = str(e)
            raise

        finally:
            # Record the invocation — wrapped so logging errors don't mask tool errors
            duration_ms = int((time.time() - start_time) * 1000)
            try:
                invocation = MCPInvocation(
                    tool_id=tool.id,
                    server_id=server.id,
                    request_data=params,
                    response_data=result_data,
                    status=status,
                    error_message=error_msg,
                    duration_ms=duration_ms
                )

                self.db.add(invocation)
                await self.db.commit()
            except Exception:
                await self.db.rollback()

        return {
            "data": result_data,
            "execution_time_ms": duration_ms,
        }

    async def _execute_mcp_tool(
        self,
        server: MCPServer,
        tool: MCPTool,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool via the real MCP protocol."""
        # Route based on transport_type field
        transport = getattr(server, "transport_type", "stdio") or "stdio"

        if transport == "http_sse" or (server.url and not server.command):
            if not server.url:
                raise ValueError(
                    f"MCP server '{server.name}' is configured for HTTP transport "
                    "but has no URL set."
                )
            return await self._execute_mcp_tool_http(server, tool, params)

        if not server.command:
            raise ValueError(
                f"MCP server '{server.name}' has no command or url configured. "
                "Set 'command' for stdio transport or 'url' for HTTP transport."
            )

        from adapters.implementations.ai.mcp_client_adapter import (
            MCPConnection,
            MCPServerConfig,
            MCPTransportType,
        )

        # Parse args and env_vars from JSON text fields
        args = []
        if server.args:
            try:
                args = json.loads(server.args)
            except (json.JSONDecodeError, TypeError):
                args = []

        env = {}
        if server.env_vars:
            try:
                env = json.loads(server.env_vars)
            except (json.JSONDecodeError, TypeError):
                env = {}

        # Build config and connect
        config = MCPServerConfig(
            name=server.name,
            command=server.command,
            args=args,
            env=env,
            transport=MCPTransportType.STDIO,
        )

        connection = MCPConnection(config)
        try:
            connected = await connection.connect()
            if not connected:
                raise RuntimeError(f"Failed to connect to MCP server '{server.name}'")

            # Call the tool
            result = await connection.call_tool(tool.name, params)

            if "error" in result:
                err = result['error']
                if isinstance(err, dict):
                    err = err.get('message', str(err))
                raise RuntimeError(f"MCP tool error: {err}")

            if result.get("isError"):
                # Tool reported an execution error via MCP protocol
                content = result.get("content") or []
                error_text = ""
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        error_text += item.get("text", "")
                raise RuntimeError(f"MCP tool execution error: {error_text or 'unknown'}")

            return result

        finally:
            await connection.disconnect()

    async def _execute_mcp_tool_http(
        self,
        server: MCPServer,
        tool: MCPTool,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool via HTTP transport using MCPServerAdapter."""
        from adapters.mcp.server_adapter import MCPServerAdapter

        adapter = MCPServerAdapter(
            control_plane_url="",
            mcp_server_url=server.url,
            api_key=server.api_key,
            server_name=server.name,
        )

        try:
            result = await adapter.process_task({
                "payload": {
                    "api_type": "tool",
                    "tool_name": tool.name,
                    "tool_input": params,
                }
            })

            if not result.get("success"):
                raise RuntimeError(
                    f"HTTP MCP tool error: {result.get('error', 'unknown error')}"
                )

            # Return in MCP-standard content format, matching stdio path
            tool_result = result.get("result", {})
            if isinstance(tool_result, str):
                text_content = tool_result
            else:
                text_content = json.dumps(tool_result)

            return {
                "content": [{"type": "text", "text": text_content}],
                "isError": False,
            }
        finally:
            await adapter.close()

    async def create_server(
        self,
        name: str,
        url: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> MCPServer:
        """Create a new MCP server."""
        # MCPServer stores extra info in server_info (JSON text field)
        server_info = {}
        if description:
            server_info["description"] = description
        if config:
            server_info["config"] = config

        server = MCPServer(
            name=name,
            url=url,
            server_info=json.dumps(server_info) if server_info else None,
            transport_type="http_sse" if url else "stdio",
            status="active"
        )
        
        self.db.add(server)
        await self.db.commit()
        await self.db.refresh(server)
        
        return server
    
    async def create_tool(
        self,
        server_id: str,
        name: str,
        description: Optional[str] = None,
        tool_schema: Optional[Dict[str, Any]] = None
    ) -> MCPTool:
        """Create a new MCP tool."""
        tool = MCPTool(
            server_id=server_id,
            name=name,
            description=description,
            tool_schema=tool_schema or {}
        )
        
        self.db.add(tool)
        await self.db.commit()
        await self.db.refresh(tool)
        
        return tool
    
    async def get_invocation_history(
        self,
        server_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[MCPInvocation]:
        """Get MCP invocation history."""
        query = select(MCPInvocation).order_by(MCPInvocation.created_at.desc())
        
        if server_id:
            query = query.where(MCPInvocation.server_id == server_id)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        return result.scalars().all()