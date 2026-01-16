"""MCP service for business logic."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
import json
import time
import asyncio

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
        
        # Simulate MCP execution (in real implementation, this would call the actual MCP server)
        try:
            # Mock execution based on common MCP patterns
            result_data = await self._mock_mcp_execution(server, tool, params)
            status = "success"
            error_msg = None
            
        except Exception as e:
            result_data = None
            status = "error"
            error_msg = str(e)
            raise
            
        finally:
            # Record the invocation
            duration_ms = int((time.time() - start_time) * 1000)
            
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
        
        return {
            "data": result_data,
            "execution_time_ms": duration_ms,
        }
    
    async def _mock_mcp_execution(
        self,
        server: MCPServer,
        tool: MCPTool,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mock MCP execution for testing purposes."""
        # Simulate network delay
        await asyncio.sleep(0.1)
        
        # Return mock data based on tool name
        tool_name = tool.name.lower()
        
        if "list" in tool_name:
            return {
                "items": [
                    {
                        "id": f"item_{i}",
                        "name": f"Mock Item {i}",
                        "type": "resource"
                    }
                    for i in range(3)
                ]
            }
        elif "get" in tool_name or "read" in tool_name:
            return {
                "content": f"Mock content for {params.get('id', 'unknown')}",
                "type": "text",
                "metadata": {
                    "source": server.name,
                    "timestamp": time.time()
                }
            }
        elif "call" in tool_name or "execute" in tool_name:
            return {
                "result": f"Executed {tool.name} with params: {json.dumps(params)}",
                "success": True,
                "metadata": {
                    "server": server.name,
                    "tool": tool.name
                }
            }
        else:
            return {
                "message": f"Mock response from {server.name}.{tool.name}",
                "parameters": params,
                "server_info": {
                    "name": server.name,
                    "status": server.status
                }
            }
    
    async def create_server(
        self,
        name: str,
        url: Optional[str] = None,
        description: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> MCPServer:
        """Create a new MCP server."""
        server = MCPServer(
            name=name,
            url=url,
            description=description,
            config=config or {},
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