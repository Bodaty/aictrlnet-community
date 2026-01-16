"""MCP Agent Tool Provider - Community Edition.

Provides MCP tools as first-class agent capabilities, allowing agents
to automatically use tools from connected MCP servers.

This is the base implementation for Community edition. Business and
Enterprise editions extend this with ML enhancements and compliance features.
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import logging

from services.mcp_unified import UnifiedMCPService

logger = logging.getLogger(__name__)


class MCPAgentTool:
    """Represents an MCP tool that can be used by agents."""

    def __init__(
        self,
        name: str,
        description: str,
        server_name: str,
        connection_id: str,
        input_schema: Dict[str, Any],
        mcp_service: UnifiedMCPService
    ):
        """Initialize an MCP agent tool.

        Args:
            name: Tool name
            description: Tool description
            server_name: Name of the MCP server providing this tool
            connection_id: Connection ID for the MCP server
            input_schema: JSON schema for tool parameters
            mcp_service: Reference to the MCP service for execution
        """
        self.name = name
        self.description = description
        self.server_name = server_name
        self.connection_id = connection_id
        self.input_schema = input_schema
        self._mcp_service = mcp_service

    async def execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the MCP tool with given arguments.

        Args:
            arguments: Tool arguments matching the input schema

        Returns:
            Execution result from the MCP server
        """
        return await self._mcp_service.execute_tool(
            connection_id=self.connection_id,
            tool_name=self.name,
            arguments=arguments
        )

    def to_agent_tool_format(self) -> Dict[str, Any]:
        """Convert to format expected by agent framework.

        Returns:
            Dict with name, description, parameters, handler, and metadata
        """
        return {
            "name": f"mcp_{self.server_name}_{self.name}",
            "description": f"[MCP:{self.server_name}] {self.description}",
            "parameters": self.input_schema,
            "handler": self.execute,
            "metadata": {
                "source": "mcp",
                "server": self.server_name,
                "original_name": self.name,
                "connection_id": self.connection_id
            }
        }

    def __repr__(self) -> str:
        return f"MCPAgentTool(name={self.name}, server={self.server_name})"


class MCPAgentToolProvider:
    """
    Provides MCP tools as agent capabilities.

    This service bridges the gap between MCP servers and the agent framework,
    automatically generating agent-compatible tools from MCP tool schemas.

    Community Edition Features:
    - Manual tool injection into agents
    - Tool discovery across connected servers
    - Basic caching with TTL
    """

    def __init__(self, mcp_service: UnifiedMCPService):
        """Initialize the tool provider.

        Args:
            mcp_service: Initialized UnifiedMCPService instance
        """
        self.mcp_service = mcp_service
        self._tool_cache: Dict[str, List[MCPAgentTool]] = {}
        self._last_refresh: Optional[datetime] = None
        self._cache_ttl_seconds = 300  # 5 minute cache

    async def get_available_tools(
        self,
        server_filter: Optional[str] = None,
        capability_filter: Optional[str] = None,
        refresh: bool = False
    ) -> List[MCPAgentTool]:
        """
        Get all available MCP tools that can be used by agents.

        Args:
            server_filter: Optional server name to filter tools
            capability_filter: Optional capability to filter by
            refresh: Force refresh of tool cache

        Returns:
            List of MCPAgentTool instances
        """
        # Check cache validity
        if not refresh and self._is_cache_valid():
            return self._get_cached_tools(server_filter, capability_filter)

        # Refresh tools from all connections
        all_tools = []

        try:
            raw_tools = await self.mcp_service.list_tools()

            for tool_data in raw_tools:
                mcp_tool = MCPAgentTool(
                    name=tool_data.get("name", "unknown"),
                    description=tool_data.get("description", ""),
                    server_name=tool_data.get("server_name", "unknown"),
                    connection_id=tool_data.get("connection_id", ""),
                    input_schema=tool_data.get("inputSchema", tool_data.get("input_schema", {})),
                    mcp_service=self.mcp_service
                )
                all_tools.append(mcp_tool)

            # Update cache
            self._update_cache(all_tools)

        except Exception as e:
            logger.error(f"Failed to refresh MCP tools: {e}")
            # Return cached tools if available
            if self._tool_cache:
                return self._get_cached_tools(server_filter, capability_filter)
            raise

        return self._get_cached_tools(server_filter, capability_filter)

    async def inject_tools_into_agent(
        self,
        agent_config: Dict[str, Any],
        server_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Inject MCP tools into an agent's configuration.

        Args:
            agent_config: The agent configuration to enhance
            server_filter: Optional server to limit tools from

        Returns:
            Enhanced agent configuration with MCP tools
        """
        mcp_tools = await self.get_available_tools(server_filter=server_filter)

        # Convert to agent tool format
        agent_tools = [tool.to_agent_tool_format() for tool in mcp_tools]

        # Merge with existing tools
        existing_tools = agent_config.get("tools", [])
        enhanced_config = {
            **agent_config,
            "tools": existing_tools + agent_tools,
            "metadata": {
                **agent_config.get("metadata", {}),
                "mcp_tools_injected": len(agent_tools),
                "mcp_tools_timestamp": datetime.utcnow().isoformat()
            }
        }

        return enhanced_config

    async def create_tool_executor(
        self,
        tool_name: str,
        server_name: Optional[str] = None
    ) -> Optional[Callable]:
        """
        Create an executor function for a specific MCP tool.

        This can be used to dynamically add tools to agents at runtime.

        Args:
            tool_name: Name of the tool to create executor for
            server_name: Optional server name to filter by

        Returns:
            Async callable that executes the tool, or None if not found
        """
        tools = await self.get_available_tools(server_filter=server_name)

        for tool in tools:
            if tool.name == tool_name:
                return tool.execute

        return None

    async def get_tool_by_name(
        self,
        tool_name: str,
        server_name: Optional[str] = None
    ) -> Optional[MCPAgentTool]:
        """
        Get a specific tool by name.

        Args:
            tool_name: Name of the tool
            server_name: Optional server name to filter by

        Returns:
            MCPAgentTool if found, None otherwise
        """
        tools = await self.get_available_tools(server_filter=server_name)

        for tool in tools:
            if tool.name == tool_name:
                return tool

        return None

    async def get_tools_summary(self) -> Dict[str, Any]:
        """
        Get a summary of available MCP tools.

        Returns:
            Dict with tool counts and server breakdown
        """
        tools = await self.get_available_tools()

        # Group by server
        by_server: Dict[str, int] = {}
        for tool in tools:
            server = tool.server_name
            by_server[server] = by_server.get(server, 0) + 1

        return {
            "total_tools": len(tools),
            "servers": len(by_server),
            "tools_by_server": by_server,
            "cache_valid": self._is_cache_valid(),
            "last_refresh": self._last_refresh.isoformat() if self._last_refresh else None
        }

    def _is_cache_valid(self) -> bool:
        """Check if tool cache is still valid."""
        if not self._last_refresh:
            return False

        elapsed = (datetime.utcnow() - self._last_refresh).total_seconds()
        return elapsed < self._cache_ttl_seconds

    def _update_cache(self, tools: List[MCPAgentTool]) -> None:
        """Update the tool cache."""
        self._tool_cache = {}
        for tool in tools:
            server = tool.server_name
            if server not in self._tool_cache:
                self._tool_cache[server] = []
            self._tool_cache[server].append(tool)

        self._last_refresh = datetime.utcnow()

    def _get_cached_tools(
        self,
        server_filter: Optional[str],
        capability_filter: Optional[str]
    ) -> List[MCPAgentTool]:
        """Get tools from cache with optional filtering."""
        if server_filter:
            return self._tool_cache.get(server_filter, [])

        all_tools = []
        for tools in self._tool_cache.values():
            all_tools.extend(tools)

        return all_tools

    def clear_cache(self) -> None:
        """Clear the tool cache."""
        self._tool_cache = {}
        self._last_refresh = None


class MCPEnabledAgentService:
    """
    Enhanced agent service with automatic MCP tool integration.

    This wraps the base agent execution service to add MCP tool
    injection capabilities.
    """

    def __init__(
        self,
        agent_service: Any,  # AgentExecutionService
        mcp_tool_provider: MCPAgentToolProvider
    ):
        """Initialize the MCP-enabled agent service.

        Args:
            agent_service: Base agent execution service
            mcp_tool_provider: MCP tool provider instance
        """
        self.agent_service = agent_service
        self.mcp_tool_provider = mcp_tool_provider

    async def execute_with_mcp_tools(
        self,
        agent_id: str,
        task: Dict[str, Any],
        mcp_servers: Optional[List[str]] = None,
        auto_inject_tools: bool = True
    ) -> Dict[str, Any]:
        """
        Execute an agent task with MCP tools available.

        Args:
            agent_id: The agent to execute
            task: The task to perform
            mcp_servers: Optional list of MCP servers to use tools from
            auto_inject_tools: Whether to automatically inject all available MCP tools

        Returns:
            Agent execution result
        """
        # Get agent configuration
        agent_config = await self.agent_service.get_agent_config(agent_id)

        if not agent_config:
            return {
                "status": "error",
                "error": f"Agent {agent_id} not found"
            }

        # Inject MCP tools if enabled
        if auto_inject_tools:
            if mcp_servers:
                for server in mcp_servers:
                    agent_config = await self.mcp_tool_provider.inject_tools_into_agent(
                        agent_config,
                        server_filter=server
                    )
            else:
                # Inject all available tools
                agent_config = await self.mcp_tool_provider.inject_tools_into_agent(
                    agent_config
                )

        # Execute with enhanced configuration
        return await self.agent_service.execute_agent(
            agent_id=agent_id,
            task=task,
            config_override=agent_config
        )

    async def get_available_mcp_tools(
        self,
        server_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get available MCP tools in agent-friendly format.

        Args:
            server_filter: Optional server to filter by

        Returns:
            List of tools in agent format
        """
        tools = await self.mcp_tool_provider.get_available_tools(
            server_filter=server_filter
        )
        return [tool.to_agent_tool_format() for tool in tools]
