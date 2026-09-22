"""MCP Client Node for consuming external MCP services in workflows."""

import logging
from typing import Any, Dict, Optional
from datetime import datetime

from ..base_node import BaseNode
from ..models import NodeConfig
from adapters.mcp.dispatcher import MCPDispatcher
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class MCPClientNode(BaseNode):
    """Node that connects to external MCP servers.
    
    This node allows workflows to consume services from external MCP servers,
    enabling integration with third-party AI providers, tools, and services.
    
    Parameters:
    - mcp_server_url: URL of the external MCP server
    - api_key: API key for authentication (optional)
    - server_name: Name to identify the server (default: "external_mcp")
    - operation: Type of MCP operation (message, quality, workflow)
    - timeout: Request timeout in seconds (default: 30)
    """
    
    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self.mcp_dispatcher: Optional[MCPDispatcher] = None
        self._initialized = False
    
    async def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize MCP connection.

        Registering used to write the node's URL and api_key into
        ``mcp_servers`` as an owner-less row — visible to every tenant — and to
        flip any existing row with the same URL to active, on a workflow run by
        any user. It also probed the URL before any of that. The node never
        worked past this point either: the dispatch call below it passes a
        `timeout` argument the dispatcher has never accepted (ledger A-26).

        So it stops here, loudly, instead of leaving that trail. The working
        client over the MCP protocol is the next change in this series.
        """
        raise NotImplementedError(
            "The mcpClient node is being rebuilt on the MCP protocol client and "
            "is unavailable in this version; use the MCP server API to register "
            "a server, or the mcp node's tool operation."
        )
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MCP client request. Returns output dict for BaseNode.run() to wrap."""
        # Ensure initialization
        if not self._initialized:
            await self.initialize(context)

        # Get operation type and timeout
        operation = self.config.parameters.get("operation", "message")
        timeout = self.config.parameters.get("timeout", 30)

        # Build MCP task based on operation
        mcp_task = self._build_mcp_task(operation, input_data)

        # Dispatch task to external MCP server
        result = await self.mcp_dispatcher.dispatch_task(mcp_task, timeout=timeout)

        # Publish execution event
        await event_bus.publish(
            "node.mcp_client.executed",
            {
                "node_id": self.config.id,
                "server_name": self.config.parameters.get("server_name"),
                "operation": operation,
                "success": True
            }
        )

        return result
    
    def _build_mcp_task(self, operation: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build MCP task structure based on operation type."""
        base_task = {
            "destination": "mcp",
            "server_name": self.config.parameters.get("server_name", "external_mcp")
        }
        
        if operation == "message":
            # MCP message processing
            messages = input_data.get("messages", [])
            if not messages and "content" in input_data:
                # Convert simple content to message format
                messages = [
                    {"role": "user", "content": input_data["content"]}
                ]
            
            base_task["payload"] = {
                "api_type": "message",
                "messages": messages,
                "parameters": input_data.get("parameters", {})
            }
            
        elif operation == "quality":
            # Quality assessment
            base_task["payload"] = {
                "api_type": "quality",
                "content": input_data.get("content", ""),
                "content_type": input_data.get("content_type", "text"),
                "criteria": input_data.get("criteria", {})
            }
            
        elif operation == "workflow":
            # Workflow creation/execution
            base_task["payload"] = {
                "api_type": "workflow",
                "workflow_definition": input_data.get("workflow_definition", {}),
                "execute_immediately": input_data.get("execute_immediately", False)
            }
            
        elif operation == "tool":
            # Tool execution (if MCP server supports tools)
            base_task["payload"] = {
                "api_type": "tool",
                "tool_name": input_data.get("tool_name"),
                "arguments": input_data.get("arguments", {})
            }
            
        elif operation == "custom":
            # Custom MCP operation
            base_task["payload"] = {
                "api_type": input_data.get("api_type", "custom"),
                **input_data
            }
            
        else:
            raise ValueError(f"Unsupported MCP operation: {operation}")
        
        return base_task
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        # Required parameters
        if not self.config.parameters.get("mcp_server_url"):
            raise ValueError("mcp_server_url is required")
        
        # Validate operation
        operation = self.config.parameters.get("operation", "message")
        valid_operations = ["message", "quality", "workflow", "tool", "custom"]
        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")
        
        # Validate timeout
        timeout = self.config.parameters.get("timeout", 30)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        
        return True
    
    async def cleanup(self) -> None:
        """Cleanup MCP connections."""
        if self.mcp_dispatcher and self._initialized:
            try:
                server_name = self.config.parameters.get("server_name", "external_mcp")
                await self.mcp_dispatcher.unregister_server(server_name)
                logger.info(f"MCP Client Node cleaned up for server: {server_name}")
            except Exception as e:
                logger.error(f"Error during MCP Client Node cleanup: {str(e)}")
        
        self._initialized = False