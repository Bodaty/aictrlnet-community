"""
DEPRECATED: This file is replaced by mcp_unified.py which uses the adapter pattern.

DO NOT USE THIS FILE - Use services.mcp_unified.UnifiedMCPService instead.

This file is kept temporarily for reference but should be removed after
verifying all functionality works with the unified approach.

============================================

MCP (Model Context Protocol) service implementation."""

import json
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError

from models.community import MCPServer, MCPTool, MCPInvocation, MCPServerCapability, TaskMCP, MCPContextStorage
from schemas.mcp import (
    MCPServerCreate, MCPServerResponse, MCPServerUpdate,
    MCPExecuteRequest, MCPExecuteResponse,
    MCPCapability
)
from adapters.mcp.dispatcher import MCPDispatcher

logger = logging.getLogger(__name__)


class MCPTokenizer:
    """Handles token counting and optimization for MCP contexts."""
    
    @staticmethod
    def count_tokens(text: str) -> int:
        """Count tokens in text (simplified)."""
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    
    @staticmethod
    def compress_context(messages: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
        """Compress context to fit within token limit."""
        # Simple implementation: keep most recent messages
        compressed = []
        total_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = MCPTokenizer.count_tokens(str(msg))
            if total_tokens + msg_tokens <= max_tokens:
                compressed.insert(0, msg)
                total_tokens += msg_tokens
            else:
                break
                
        return compressed


class MCPService:
    """Service for managing MCP servers and operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        # Use real MCP dispatcher with control plane URL
        # In production, this URL would come from config
        control_plane_url = "http://localhost:8000"
        self.dispatcher = MCPDispatcher(control_plane_url)
        self.tokenizer = MCPTokenizer()
    
    # Server Management
    
    async def get_server(self, server_id: str) -> Optional[MCPServerResponse]:
        """Get MCP server by ID."""
        result = await self.db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if server:
            return self._server_to_response(server)
        return None
    
    async def list_servers(
        self, 
        provider: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0, 
        limit: int = 100
    ) -> List[MCPServerResponse]:
        """List all MCP servers with optional filters."""
        query = select(MCPServer)
        
        if provider:
            query = query.where(MCPServer.service_type == provider)
        if status:
            query = query.where(MCPServer.status == status)
        
        query = query.offset(skip).limit(limit)
        result = await self.db.execute(query)
        servers = result.scalars().all()
        
        return [self._server_to_response(server) for server in servers]
    
    async def update_server(self, server_id: str, update_data: MCPServerUpdate) -> Optional[MCPServerResponse]:
        """Update MCP server."""
        result = await self.db.execute(
            select(MCPServer).where(MCPServer.id == server_id)
        )
        server = result.scalar_one_or_none()
        
        if server:
            if update_data.name is not None:
                server.name = update_data.name
            if update_data.url is not None:
                server.url = update_data.url
            if update_data.status is not None:
                server.status = update_data.status
            if update_data.config is not None:
                server.server_info = json.dumps(update_data.config)
            if update_data.capabilities is not None:
                # Update capabilities in separate table
                # First remove existing capabilities
                await self.db.execute(
                    select(MCPServerCapability).where(
                        MCPServerCapability.server_id == server_id
                    ).delete()
                )
                
                # Add new capabilities
                for cap_name in update_data.capabilities:
                    capability = MCPServerCapability(
                        id=str(uuid.uuid4()),
                        server_id=server.id,
                        capability=cap_name,
                        supported=True,
                        created_at=datetime.utcnow().timestamp()
                    )
                    self.db.add(capability)
            
            server.updated_at = datetime.utcnow().timestamp()
            await self.db.commit()
            await self.db.refresh(server)
            
            return self._server_to_response(server)
        
        return None
    
    async def get_server_capabilities(self, server_id: str) -> List[MCPCapability]:
        """Get capabilities for a specific server."""
        result = await self.db.execute(
            select(MCPServerCapability).where(MCPServerCapability.server_id == server_id)
        )
        capabilities = result.scalars().all()
        
        return [
            MCPCapability(
                name=cap.capability,
                enabled=cap.supported,
                description=f"Capability: {cap.capability}"
            )
            for cap in capabilities
        ]
    
    # Discovery
    
    async def discover_servers(self, protocol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Discover available MCP servers."""
        # Mock discovery - in production would scan network/registry
        servers = await self.list_servers(status="active")
        
        discovered = []
        for server in servers:
            discovered.append({
                "id": server.id,
                "name": server.name,
                "url": server.url,
                "protocol": protocol or "rest",
                "capabilities": await self.get_server_capabilities(server.id),
                "status": server.status
            })
        
        return discovered
    
    # Execution
    
    # Tools
    
    async def list_tools(self, server_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools."""
        query = select(MCPTool)
        if server_id:
            query = query.where(MCPTool.server_id == server_id)
            
        result = await self.db.execute(query)
        tools = result.scalars().all()
        
        return [
            {
                "id": str(tool.id),
                "server_id": str(tool.server_id),
                "name": tool.name,
                "description": tool.description,
                "schema": tool.tool_schema
            }
            for tool in tools
        ]
    
    # Task Management
    
    async def create_mcp_task(self, task_data: Dict[str, Any]) -> TaskMCP:
        """Create a new MCP-enabled task."""
        task = TaskMCP(
            source_id=task_data.get("source_id", str(uuid.uuid4())),
            destination=task_data.get("destination", "default"),
            payload=json.dumps(task_data.get("payload", {})),
            status=task_data.get("status", "pending"),
            mcp_enabled=True,
            mcp_metadata=json.dumps(task_data.get("mcp_metadata", {}))
        )
        
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        
        return task
    
    async def get_mcp_task(self, task_id: str) -> Optional[TaskMCP]:
        """Get MCP task by ID."""
        result = await self.db.execute(
            select(TaskMCP).where(TaskMCP.task_id == task_id)
        )
        return result.scalar_one_or_none()
    
    # Context Storage
    
    async def store_context(self, task_id: str, context_type: str, content: Any, role: Optional[str] = None) -> MCPContextStorage:
        """Store context for MCP operations."""
        context = MCPContextStorage(
            task_id=task_id,
            context_type=context_type,
            content=json.dumps(content),
            role=role
        )
        
        self.db.add(context)
        await self.db.commit()
        await self.db.refresh(context)
        
        return context
    
    async def get_task_contexts(self, task_id: str, context_type: Optional[str] = None) -> List[MCPContextStorage]:
        """Get contexts for a task."""
        query = select(MCPContextStorage).where(MCPContextStorage.task_id == task_id)
        
        if context_type:
            query = query.where(MCPContextStorage.context_type == context_type)
        
        query = query.order_by(MCPContextStorage.created_at)
        result = await self.db.execute(query)
        
        return result.scalars().all()
    
    # Helper methods
    
    def _server_to_response(self, server: MCPServer) -> MCPServerResponse:
        """Convert server model to response schema."""
        # Parse server info as config
        config = {}
        if server.server_info:
            try:
                config = json.loads(server.server_info)
            except:
                pass
        
        # Convert timestamps
        created_at = datetime.fromtimestamp(server.created_at) if isinstance(server.created_at, (int, float)) else server.created_at
        updated_at = datetime.fromtimestamp(server.updated_at) if isinstance(server.updated_at, (int, float)) else server.updated_at
        
        return MCPServerResponse(
            id=server.id,
            name=server.name,
            url=server.url,
            description=server.description,
            status=server.status,
            config=config,
            created_at=created_at,
            updated_at=updated_at
        )