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