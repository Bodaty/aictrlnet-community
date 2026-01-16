"""MCP Dispatcher for routing tasks to appropriate MCP servers."""

from typing import Dict, Any, List, Optional
from collections import defaultdict
import random
import logging
import uuid
from sqlalchemy import select

from .server_adapter import MCPServerAdapter
from models import MCPServer, MCPServerCapability
from core.database import get_session_maker
from core.exceptions import AdapterError

logger = logging.getLogger(__name__)


class MCPDispatcher:
    """Routes tasks to appropriate MCP servers based on capabilities"""
    
    def __init__(self, control_plane_url: str):
        self.control_plane_url = control_plane_url
        self.servers: Dict[str, MCPServerAdapter] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize dispatcher by loading registered servers from database"""
        if self._initialized:
            return
            
        SessionLocal = get_session_maker()
        async with SessionLocal() as db:
            result = await db.execute(
                select(MCPServer).filter_by(status="active")
            )
            servers = result.scalars().all()
            
            for server in servers:
                try:
                    adapter = MCPServerAdapter(
                        control_plane_url=self.control_plane_url,
                        mcp_server_url=server.url,
                        api_key=server.api_key,
                        server_name=server.name
                    )
                    adapter.server_id = server.id
                    
                    # Load capabilities from database
                    result = await db.execute(
                        select(MCPServerCapability).filter_by(
                            server_id=server.id,
                            supported=True
                        )
                    )
                    capabilities = result.scalars().all()
                    
                    adapter.capabilities = {
                        cap.capability: True for cap in capabilities
                    }
                    
                    # Index by both name and ID for compatibility
                    self.servers[server.name] = adapter
                    self.servers[server.id] = adapter  # Also index by ID
                    logger.info(f"Loaded MCP server: {server.name} (ID: {server.id})")
                    
                except Exception as e:
                    logger.error(f"Failed to load MCP server {server.name}: {str(e)}")
        
        self._initialized = True
        logger.info(f"MCP Dispatcher initialized with {len(self.servers)} servers")
    
    async def register_server(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        server_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Register a new MCP server"""
        adapter = MCPServerAdapter(
            control_plane_url=self.control_plane_url,
            mcp_server_url=server_url,
            api_key=api_key,
            server_name=server_name
        )
        
        try:
            result = await adapter.register_mcp_server()
            self.servers[adapter.server_name] = adapter
            
            logger.info(f"Registered MCP server: {adapter.server_name}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to register MCP server: {str(e)}")
            raise AdapterError(f"Server registration failed: {str(e)}")
    
    async def dispatch(self, server_id: str, method: str, params: Dict[str, Any]) -> Any:
        """Compatibility method for MCPService - dispatches to a specific server."""
        await self.initialize()
        
        # Convert to task format expected by dispatch_task
        # Merge params directly into payload for compatibility with server adapter
        payload = params.copy() if params else {}
        payload["api_type"] = method
        payload["server_name"] = server_id  # Use server_id as server_name
        
        logger.debug(f"Dispatch - method: {method}, server_id: {server_id}, params: {params}")
        logger.debug(f"Dispatch - created payload: {payload}")
        
        task = {
            "task_id": str(uuid.uuid4()),
            "payload": payload
        }
        
        # Use dispatch_task internally
        result = await self.dispatch_task(task)
        
        # If dispatch_task returns error format, raise exception
        if isinstance(result, dict) and not result.get("success", True):
            raise AdapterError(result.get("error", "MCP dispatch failed"))
        
        # Return the result in format expected by MCPService
        return result
    
    async def dispatch_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Route task to appropriate MCP server"""
        await self.initialize()
        
        logger.debug(f"Dispatcher.dispatch_task received task: {task}")
        
        # Check if specific server requested
        server_name = task.get("payload", {}).get("server_name")
        if server_name and server_name in self.servers:
            server = self.servers[server_name]
            logger.info(f"Routing task to requested server: {server_name}")
        else:
            # Select server based on capabilities
            server = await self._select_server(task)
            
        if not server:
            return {
                "success": False,
                "error": "No MCP server available for this task"
            }
        
        try:
            result = await server.process_task(task)
            return result
        except Exception as e:
            logger.error(f"Task processing failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "server_name": server.server_name
            }
    
    async def _select_server(self, task: Dict[str, Any]) -> Optional[MCPServerAdapter]:
        """Select best server based on capabilities"""
        api_type = task.get("payload", {}).get("api_type", "message")
        
        # Get servers with required capability
        capable_servers = [
            (name, adapter) for name, adapter in self.servers.items()
            if adapter.capabilities.get(api_type, False)
        ]
        
        if not capable_servers:
            logger.warning(f"No servers found with capability: {api_type}")
            return None
        
        # For Community edition, just randomly select from capable servers
        # Business edition would add load balancing here
        selected = random.choice(capable_servers)
        logger.info(f"Selected server {selected[0]} for {api_type} task")
        
        return selected[1]
    
    async def list_available_servers(
        self,
        capability: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List available MCP servers, optionally filtered by capability"""
        await self.initialize()
        
        servers = []
        
        for name, adapter in self.servers.items():
            if capability and not adapter.capabilities.get(capability, False):
                continue
                
            servers.append({
                "name": name,
                "url": adapter.mcp_server_url,
                "capabilities": list(adapter.capabilities.keys()),
                "status": "active",
                "server_id": adapter.server_id
            })
        
        return servers
    
    async def get_server_health(self, server_name: str) -> Dict[str, Any]:
        """Get health status of a specific server"""
        await self.initialize()
        
        if server_name not in self.servers:
            return {
                "status": "not_found",
                "error": f"Server {server_name} not found"
            }
        
        server = self.servers[server_name]
        return await server.check_server_health()
    
    async def remove_server(self, server_name: str) -> bool:
        """Remove a server from the dispatcher"""
        if server_name in self.servers:
            server = self.servers[server_name]
            await server.close()
            del self.servers[server_name]
            
            # Update database
            SessionLocal = get_session_maker()
            async with SessionLocal() as db:
                result = await db.execute(
                    select(MCPServer).filter_by(name=server_name)
                )
                mcp_server = result.scalar_one_or_none()
                if mcp_server:
                    mcp_server.status = "inactive"
                    await db.commit()
            
            logger.info(f"Removed MCP server: {server_name}")
            return True
        
        return False
    
    async def close_all(self):
        """Close all server connections"""
        for server in self.servers.values():
            await server.close()
        self.servers.clear()
        self._initialized = False