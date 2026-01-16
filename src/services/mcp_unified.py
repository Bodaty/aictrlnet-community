"""Unified MCP Service that uses the adapter pattern.

This replaces the duplicate mcp_wrapper.py and mcp_service.py implementations
by using the MCP adapters from the adapter system.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from adapters.registry import adapter_registry

logger = logging.getLogger(__name__)


class UnifiedMCPService:
    """Unified MCP service using adapter pattern."""
    
    def __init__(self):
        """Initialize the unified MCP service."""
        self.mcp_adapter = None
        self.connected_servers = {}
        self.active_connections = {}
        
    async def initialize(self):
        """Initialize the MCP adapter."""
        try:
            # Get MCP client adapter class from registry
            adapter_class = adapter_registry.get_adapter_class("mcp-client-1.0.0")
            if not adapter_class:
                # Try without version as fallback
                adapter_class = adapter_registry.get_adapter_class("mcp-client")
            
            if adapter_class:
                # Create adapter instance with proper config
                from adapters.models import AdapterConfig, AdapterCategory
                config = AdapterConfig(
                    name="mcp-client",
                    adapter_type="mcp-client",
                    category=AdapterCategory.AI,
                    version="1.0.0",
                    description="MCP Protocol Client",
                    required_edition="community",
                    custom_config={"discovery_only": False}
                )
                self.mcp_adapter = adapter_class(config)
            
            if not self.mcp_adapter:
                logger.warning("MCP client adapter not available")
                return False
            
            # Initialize the adapter
            await self.mcp_adapter.initialize()
            logger.info("MCP adapter initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP adapter: {e}")
            return False
    
    async def connect(self, server_url: str) -> Dict[str, Any]:
        """Connect to an MCP server.
        
        Args:
            server_url: URL of the MCP server
            
        Returns:
            Connection info including connection_id and capabilities
        """
        if not self.mcp_adapter:
            await self.initialize()
        
        if not self.mcp_adapter:
            raise RuntimeError("MCP adapter not available")
        
        try:
            # Use adapter to connect
            result = await self.mcp_adapter.execute({
                "operation": "connect",
                "server_url": server_url
            })
            
            connection_id = result.data.get("connection_id")
            self.active_connections[connection_id] = {
                "server_url": server_url,
                "connected_at": datetime.utcnow(),
                "capabilities": result.data.get("capabilities", {})
            }
            
            return {
                "connection_id": connection_id,
                "status": "connected",
                "capabilities": result.data.get("capabilities", {})
            }
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {e}")
            raise
    
    async def list_tools(self, connection_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available tools from connected MCP servers.
        
        Args:
            connection_id: Optional specific connection to query
            
        Returns:
            List of available tools
        """
        if not self.mcp_adapter:
            return []
        
        try:
            # If specific connection requested
            if connection_id:
                result = await self.mcp_adapter.execute({
                    "operation": "list_tools",
                    "connection_id": connection_id
                })
                return result.data.get("tools", [])
            
            # Otherwise get tools from all connections
            all_tools = []
            for conn_id in self.active_connections:
                result = await self.mcp_adapter.execute({
                    "operation": "list_tools", 
                    "connection_id": conn_id
                })
                tools = result.data.get("tools", [])
                for tool in tools:
                    tool["connection_id"] = conn_id
                all_tools.extend(tools)
            
            return all_tools
            
        except Exception as e:
            logger.error(f"Failed to list tools: {e}")
            return []
    
    async def execute_tool(
        self,
        connection_id: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a tool on an MCP server.
        
        Args:
            connection_id: Connection to use
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            
        Returns:
            Tool execution result
        """
        if not self.mcp_adapter:
            raise RuntimeError("MCP adapter not available")
        
        if connection_id not in self.active_connections:
            raise ValueError(f"Invalid connection ID: {connection_id}")
        
        try:
            result = await self.mcp_adapter.execute({
                "operation": "execute_tool",
                "connection_id": connection_id,
                "tool_name": tool_name,
                "arguments": arguments
            })
            
            return {
                "status": "success",
                "result": result.data.get("result"),
                "execution_time": result.data.get("execution_time", 0)
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def disconnect(self, connection_id: str):
        """Disconnect from an MCP server.
        
        Args:
            connection_id: Connection to disconnect
        """
        if connection_id not in self.active_connections:
            return
        
        if self.mcp_adapter:
            try:
                await self.mcp_adapter.execute({
                    "operation": "disconnect",
                    "connection_id": connection_id
                })
            except Exception as e:
                logger.error(f"Error disconnecting: {e}")
        
        del self.active_connections[connection_id]
    
    async def get_connection_status(self) -> Dict[str, Any]:
        """Get status of all MCP connections.
        
        Returns:
            Status information for all connections
        """
        return {
            "adapter_initialized": self.mcp_adapter is not None,
            "active_connections": len(self.active_connections),
            "connections": [
                {
                    "connection_id": conn_id,
                    "server_url": info["server_url"],
                    "connected_at": info["connected_at"].isoformat(),
                    "capabilities": info["capabilities"]
                }
                for conn_id, info in self.active_connections.items()
            ]
        }
    
    async def cleanup(self):
        """Clean up all connections."""
        for conn_id in list(self.active_connections.keys()):
            await self.disconnect(conn_id)
        
        if self.mcp_adapter:
            await self.mcp_adapter.shutdown()
            self.mcp_adapter = None