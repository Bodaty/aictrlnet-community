"""MCP (Model Context Protocol) node implementation."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from ..base_node import BaseNode
from ..models import NodeConfig
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class MCPNode(BaseNode):
    """Node for Model Context Protocol operations.
    
    Enables:
    - Tool discovery and registration
    - Resource management
    - Prompt template management
    - Context aggregation
    - Cross-model communication
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MCP node. Returns output dict for BaseNode.run() to wrap."""
        # Get MCP service from context or create new instance
        from services.mcp_service import MCPService
        db = context.get('db')
        if not db:
            raise ValueError("Database session not provided in context")
        mcp_service = MCPService(db)

        # Get MCP operation type
        operation = self.config.parameters.get("operation", "execute_tool")

        # Execute based on operation type
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

        # Publish completion event
        await event_bus.publish(
            "node.executed",
            {
                "node_id": self.config.id,
                "node_type": "mcp",
                "operation": operation
            }
        )

        return output_data
    
    async def _execute_tool(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an MCP tool."""
        # Get tool configuration
        server_name = self.config.parameters.get("server_name")
        tool_name = self.config.parameters.get("tool_name") or input_data.get("tool_name")
        tool_args = self.config.parameters.get("tool_args", {}).copy()
        
        if not tool_name:
            raise ValueError("tool_name is required for execute_tool operation")
        
        # Merge input arguments
        if "tool_args" in input_data:
            tool_args.update(input_data["tool_args"])
        
        # Get server (if specified) or use default
        if server_name:
            server = await mcp_service.get_server(server_name)
            if not server:
                raise ValueError(f"MCP server '{server_name}' not found")
        else:
            # Find server that has this tool
            server = await mcp_service.find_server_for_tool(tool_name)
            if not server:
                raise ValueError(f"No MCP server found with tool '{tool_name}'")
        
        # Execute tool
        result = await mcp_service.execute_tool(
            server_name=server.name,
            tool_name=tool_name,
            arguments=tool_args
        )
        
        return {
            "tool_result": result,
            "tool_name": tool_name,
            "server_name": server.name
        }
    
    async def _discover_tools(self) -> Dict[str, Any]:
        """Discover available MCP tools."""
        server_name = self.config.parameters.get("server_name")
        category = self.config.parameters.get("category")
        
        if server_name:
            # Get tools from specific server
            tools = await mcp_service.get_server_tools(server_name)
        else:
            # Get all tools
            tools = await mcp_service.get_all_tools()
        
        # Filter by category if specified
        if category:
            tools = [t for t in tools if t.get("category") == category]
        
        return {
            "tools": tools,
            "count": len(tools)
        }
    
    async def _get_resource(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get an MCP resource."""
        server_name = self.config.parameters.get("server_name")
        resource_uri = self.config.parameters.get("resource_uri") or input_data.get("resource_uri")
        
        if not resource_uri:
            raise ValueError("resource_uri is required for get_resource operation")
        
        # Get resource
        if server_name:
            resource = await mcp_service.get_resource(
                server_name=server_name,
                uri=resource_uri
            )
        else:
            # Try all servers
            resource = await mcp_service.get_resource_from_any_server(resource_uri)
        
        if not resource:
            raise ValueError(f"Resource '{resource_uri}' not found")
        
        return {
            "resource": resource,
            "uri": resource_uri
        }
    
    async def _list_resources(self) -> Dict[str, Any]:
        """List available MCP resources."""
        server_name = self.config.parameters.get("server_name")
        resource_type = self.config.parameters.get("resource_type")
        
        if server_name:
            # Get resources from specific server
            resources = await mcp_service.get_server_resources(server_name)
        else:
            # Get all resources
            resources = await mcp_service.get_all_resources()
        
        # Filter by type if specified
        if resource_type:
            resources = [r for r in resources if r.get("type") == resource_type]
        
        return {
            "resources": resources,
            "count": len(resources)
        }
    
    async def _get_prompt(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get an MCP prompt template."""
        server_name = self.config.parameters.get("server_name")
        prompt_name = self.config.parameters.get("prompt_name") or input_data.get("prompt_name")
        prompt_args = self.config.parameters.get("prompt_args", {}).copy()
        
        if not prompt_name:
            raise ValueError("prompt_name is required for get_prompt operation")
        
        # Merge input arguments
        if "prompt_args" in input_data:
            prompt_args.update(input_data["prompt_args"])
        
        # Get prompt
        if server_name:
            prompt = await mcp_service.get_prompt(
                server_name=server_name,
                prompt_name=prompt_name,
                arguments=prompt_args
            )
        else:
            # Try all servers
            prompt = await mcp_service.get_prompt_from_any_server(
                prompt_name=prompt_name,
                arguments=prompt_args
            )
        
        if not prompt:
            raise ValueError(f"Prompt '{prompt_name}' not found")
        
        return {
            "prompt": prompt,
            "prompt_name": prompt_name
        }
    
    async def _list_prompts(self) -> Dict[str, Any]:
        """List available MCP prompts."""
        server_name = self.config.parameters.get("server_name")
        category = self.config.parameters.get("category")
        
        if server_name:
            # Get prompts from specific server
            prompts = await mcp_service.get_server_prompts(server_name)
        else:
            # Get all prompts
            prompts = await mcp_service.get_all_prompts()
        
        # Filter by category if specified
        if category:
            prompts = [p for p in prompts if p.get("category") == category]
        
        return {
            "prompts": prompts,
            "count": len(prompts)
        }
    
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
                    # Get resource content
                    resource = await mcp_service.get_resource_from_any_server(
                        source.get("uri")
                    )
                    if resource:
                        aggregated_context[source.get("name", source.get("uri"))] = resource
                
                elif source_type == "tool":
                    # Execute tool to get context
                    result = await mcp_service.execute_tool(
                        server_name=source.get("server_name"),
                        tool_name=source.get("tool_name"),
                        arguments=source.get("arguments", {})
                    )
                    if result:
                        aggregated_context[source.get("name", source.get("tool_name"))] = result
                
                elif source_type == "prompt":
                    # Get prompt content
                    prompt = await mcp_service.get_prompt_from_any_server(
                        prompt_name=source.get("prompt_name"),
                        arguments=source.get("arguments", {})
                    )
                    if prompt:
                        aggregated_context[source.get("name", source.get("prompt_name"))] = prompt
                
                elif source_type == "static":
                    # Add static data
                    aggregated_context[source.get("name", "static")] = source.get("data")
                
            except Exception as e:
                errors.append({
                    "source": source,
                    "error": str(e)
                })
        
        # Format context based on output format
        output_format = self.config.parameters.get("output_format", "structured")
        
        if output_format == "text":
            # Convert to text format
            text_parts = []
            for key, value in aggregated_context.items():
                text_parts.append(f"=== {key} ===")
                if isinstance(value, dict):
                    text_parts.append(json.dumps(value, indent=2))
                else:
                    text_parts.append(str(value))
                text_parts.append("")
            
            return {
                "context": "\n".join(text_parts),
                "errors": errors
            }
        
        else:
            # Return structured format
            return {
                "context": aggregated_context,
                "errors": errors
            }
    
    async def _call_server(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a direct call to an MCP server."""
        server_name = self.config.parameters.get("server_name")
        method = self.config.parameters.get("method") or input_data.get("method")
        params = self.config.parameters.get("params", {}).copy()
        
        if not server_name:
            raise ValueError("server_name is required for call_server operation")
        if not method:
            raise ValueError("method is required for call_server operation")
        
        # Merge input params
        if "params" in input_data:
            params.update(input_data["params"])
        
        # Make server call
        result = await mcp_service.call_server_method(
            server_name=server_name,
            method=method,
            params=params
        )
        
        return {
            "result": result,
            "server_name": server_name,
            "method": method
        }
    
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