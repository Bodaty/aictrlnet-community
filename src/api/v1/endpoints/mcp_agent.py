"""MCP Agent Integration API Endpoints - Community Edition.

These endpoints allow agents to discover and use MCP tools from
connected MCP servers.

Phase 5 Implementation - See docs/implementation/MCP_IMPLEMENTATION_SPEC_FASTAPI.md
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
import logging

from core.database import get_db
from core.security import get_current_active_user
from services.mcp_unified import UnifiedMCPService
from services.mcp_agent_tool_provider import MCPAgentToolProvider, MCPEnabledAgentService
from schemas.mcp import (
    MCPAgentToolResponse,
    MCPAgentToolListResponse,
    MCPAgentExecutionRequest,
    MCPAgentExecutionResponse,
    MCPToolInjectionRequest,
    MCPToolInjectionResponse,
    MCPToolsSummaryResponse
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp/agent", tags=["MCP Agent Integration"])

# Module-level service cache
_mcp_service: Optional[UnifiedMCPService] = None
_tool_provider: Optional[MCPAgentToolProvider] = None


async def get_mcp_service() -> UnifiedMCPService:
    """Get or create MCP service instance."""
    global _mcp_service
    if _mcp_service is None:
        _mcp_service = UnifiedMCPService()
        await _mcp_service.initialize()
    return _mcp_service


async def get_tool_provider() -> MCPAgentToolProvider:
    """Get or create tool provider instance."""
    global _tool_provider
    if _tool_provider is None:
        mcp_service = await get_mcp_service()
        _tool_provider = MCPAgentToolProvider(mcp_service)
    return _tool_provider


@router.get("/tools", response_model=MCPAgentToolListResponse)
async def list_agent_available_tools(
    server: Optional[str] = Query(None, description="Filter by MCP server name"),
    refresh: bool = Query(False, description="Force refresh tool cache"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """
    List all MCP tools available for agent use.

    Returns tools formatted for agent consumption with metadata
    about their MCP server source.

    Community Edition: Basic tool listing with caching.
    Business Edition: Adds ML-based relevance ranking.
    Enterprise Edition: Adds tenant filtering and approval status.
    """
    try:
        tool_provider = await get_tool_provider()
        tools = await tool_provider.get_available_tools(
            server_filter=server,
            refresh=refresh
        )

        # Get summary for metadata
        summary = await tool_provider.get_tools_summary()

        return MCPAgentToolListResponse(
            tools=[
                MCPAgentToolResponse(
                    name=tool.name,
                    description=tool.description,
                    server_name=tool.server_name,
                    input_schema=tool.input_schema,
                    agent_tool_name=f"mcp_{tool.server_name}_{tool.name}"
                )
                for tool in tools
            ],
            total=len(tools),
            servers=summary["servers"],
            cache_valid=summary["cache_valid"]
        )

    except Exception as e:
        logger.error(f"Failed to list MCP tools: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list MCP tools: {str(e)}"
        )


@router.get("/tools/summary", response_model=MCPToolsSummaryResponse)
async def get_tools_summary(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get a summary of available MCP tools.

    Returns tool counts grouped by server, cache status, and last refresh time.
    """
    try:
        tool_provider = await get_tool_provider()
        summary = await tool_provider.get_tools_summary()

        return MCPToolsSummaryResponse(**summary)

    except Exception as e:
        logger.error(f"Failed to get tools summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tools summary: {str(e)}"
        )


@router.get("/tools/{tool_name}", response_model=MCPAgentToolResponse)
async def get_tool_by_name(
    tool_name: str,
    server: Optional[str] = Query(None, description="Filter by MCP server name"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Get a specific MCP tool by name.

    Returns detailed information about a single tool.
    """
    try:
        tool_provider = await get_tool_provider()
        tool = await tool_provider.get_tool_by_name(tool_name, server_name=server)

        if not tool:
            raise HTTPException(
                status_code=404,
                detail=f"Tool '{tool_name}' not found"
            )

        return MCPAgentToolResponse(
            name=tool.name,
            description=tool.description,
            server_name=tool.server_name,
            input_schema=tool.input_schema,
            agent_tool_name=f"mcp_{tool.server_name}_{tool.name}"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get tool {tool_name}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get tool: {str(e)}"
        )


@router.post("/execute", response_model=MCPAgentExecutionResponse)
async def execute_agent_with_mcp(
    request: MCPAgentExecutionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Execute an agent with MCP tools automatically injected.

    This endpoint allows agents to use MCP tools as first-class
    capabilities without manual configuration.

    Community Edition: Basic tool injection.
    Business Edition: ML-optimized tool selection based on task.
    Enterprise Edition: Compliance-aware tool filtering.
    """
    try:
        # Import here to avoid circular imports
        from services.agent_execution_basic import AgentExecutionService

        tool_provider = await get_tool_provider()
        agent_service = AgentExecutionService(db)

        mcp_agent_service = MCPEnabledAgentService(agent_service, tool_provider)

        result = await mcp_agent_service.execute_with_mcp_tools(
            agent_id=request.agent_id,
            task=request.task,
            mcp_servers=request.mcp_servers,
            auto_inject_tools=request.auto_inject_tools
        )

        return MCPAgentExecutionResponse(
            status=result.get("status", "unknown"),
            agent_id=request.agent_id,
            task_id=result.get("task_id"),
            result=result.get("result"),
            error=result.get("error"),
            mcp_tools_used=result.get("mcp_tools_used", []),
            execution_time_ms=result.get("execution_time_ms"),
            metadata=result.get("metadata", {})
        )

    except Exception as e:
        logger.error(f"Failed to execute agent with MCP tools: {e}")
        return MCPAgentExecutionResponse(
            status="error",
            agent_id=request.agent_id,
            error=str(e),
            metadata={"exception_type": type(e).__name__}
        )


@router.post("/agents/{agent_id}/inject-mcp-tools", response_model=MCPToolInjectionResponse)
async def inject_mcp_tools_into_agent(
    agent_id: str,
    request: MCPToolInjectionRequest = MCPToolInjectionRequest(),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Inject MCP tools into an existing agent's configuration.

    This modifies the agent's tool set to include available MCP tools,
    making them available for future executions.

    Community Edition: Injects all available tools or filtered by server.
    Business Edition: Uses ML to recommend relevant tools.
    Enterprise Edition: Filters by tenant permissions and approval status.
    """
    try:
        # Import here to avoid circular imports
        from services.agent_execution_basic import AgentExecutionService

        tool_provider = await get_tool_provider()
        agent_service = AgentExecutionService(db)

        # Get current agent config
        agent_config = await agent_service.get_agent_config(agent_id)

        if not agent_config:
            raise HTTPException(status_code=404, detail="Agent not found")

        # Track which servers we inject from
        servers_used = []

        # Inject tools
        if request.servers:
            for server in request.servers:
                agent_config = await tool_provider.inject_tools_into_agent(
                    agent_config,
                    server_filter=server
                )
                servers_used.append(server)
        else:
            agent_config = await tool_provider.inject_tools_into_agent(agent_config)
            summary = await tool_provider.get_tools_summary()
            servers_used = list(summary.get("tools_by_server", {}).keys())

        # Save updated config
        await agent_service.update_agent_config(agent_id, agent_config)

        tools_injected = agent_config.get("metadata", {}).get("mcp_tools_injected", 0)

        return MCPToolInjectionResponse(
            agent_id=agent_id,
            tools_injected=tools_injected,
            servers_used=servers_used,
            message=f"Successfully injected {tools_injected} MCP tools into agent"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to inject MCP tools into agent {agent_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to inject MCP tools: {str(e)}"
        )


@router.post("/tools/refresh")
async def refresh_tools_cache(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """
    Force refresh the MCP tools cache.

    Clears the cache and fetches fresh tool information from all
    connected MCP servers.
    """
    try:
        tool_provider = await get_tool_provider()
        tool_provider.clear_cache()

        # Trigger refresh
        await tool_provider.get_available_tools(refresh=True)
        summary = await tool_provider.get_tools_summary()

        return {
            "status": "success",
            "message": "Tool cache refreshed",
            "tools_found": summary["total_tools"],
            "servers": summary["servers"],
            "last_refresh": summary["last_refresh"]
        }

    except Exception as e:
        logger.error(f"Failed to refresh tools cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to refresh cache: {str(e)}"
        )
