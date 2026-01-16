"""MCP (Model Context Protocol) endpoints."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from core.dependencies import get_edition
from services.mcp_service import MCPService
from schemas.mcp import (
    MCPServerCreate, MCPServerResponse, MCPServerUpdate,
    MCPExecuteRequest, MCPExecuteResponse,
    MCPCapability, MCPDiscoveryResponse, MCPInfo
)

router = APIRouter()


# Info endpoint

@router.get("/info", response_model=MCPInfo)
async def get_mcp_info(
    edition: str = Depends(get_edition)
) -> MCPInfo:
    """Get MCP system information."""
    return MCPInfo(
        version="1.0.0",
        supported_providers=["openai", "anthropic", "google", "cohere", "mistral"],
        features=["context-management", "token-optimization", "multi-provider", "streaming"],
        status="active"
    )


# Server management endpoints

@router.post("/servers", response_model=MCPServerResponse)
async def register_mcp_server(
    server_data: MCPServerCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Register a new MCP server."""
    service = MCPService(db)
    
    try:
        server = await service.create_server(server_data)
        return server
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/servers", response_model=List[MCPServerResponse])
async def list_mcp_servers(
    status: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """List all MCP servers."""
    service = MCPService(db)
    servers = await service.list_servers(
        status=status,
        skip=skip,
        limit=limit
    )
    return servers


@router.get("/servers/{server_id}", response_model=MCPServerResponse)
async def get_mcp_server(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get MCP server details."""
    service = MCPService(db)
    server = await service.get_server(server_id)
    
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    return server


@router.patch("/servers/{server_id}", response_model=MCPServerResponse)
async def update_mcp_server(
    server_id: str,
    update_data: MCPServerUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Update MCP server configuration."""
    service = MCPService(db)
    server = await service.update_server(server_id, update_data)
    
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    return server


@router.get("/servers/{server_id}/capabilities", response_model=List[MCPCapability])
async def get_server_capabilities(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get capabilities for a specific MCP server."""
    service = MCPService(db)
    
    # Verify server exists
    server = await service.get_server(server_id)
    if not server:
        raise HTTPException(status_code=404, detail="Server not found")
    
    capabilities = await service.get_server_capabilities(server_id)
    return capabilities


# Discovery endpoint

@router.get("/discovery", response_model=MCPDiscoveryResponse)
async def discover_mcp_servers(
    protocol: Optional[str] = Query(None, description="Filter by protocol"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Discover available MCP servers."""
    service = MCPService(db)
    
    discovered = await service.discover_servers(protocol=protocol)
    
    return MCPDiscoveryResponse(
        servers=discovered,
        total=len(discovered),
        timestamp=None
    )


# Execution endpoint

@router.post("/execute", response_model=MCPExecuteResponse)
async def execute_mcp_method(
    request: MCPExecuteRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Execute a method on an MCP server."""
    service = MCPService(db)
    
    try:
        result = await service.execute_method(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Tools endpoints

@router.get("/tools")
async def list_mcp_tools(
    server_id: Optional[str] = Query(None, description="Filter by server ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """List available MCP tools."""
    service = MCPService(db)
    tools = await service.list_tools(server_id=server_id)
    return {"tools": tools, "total": len(tools)}


# Legacy adapters endpoint for backward compatibility

@router.get("/adapters")
async def list_mcp_adapters(
    current_user: dict = Depends(get_current_active_user)
):
    """List MCP adapters (legacy endpoint for compatibility)."""
    # Return mock adapters for backward compatibility
    return {
        "adapters": [
            {
                "id": "openai-adapter",
                "name": "OpenAI Adapter",
                "provider": "openai",
                "capabilities": ["completion", "embedding", "moderation"],
                "status": "active"
            },
            {
                "id": "anthropic-adapter",
                "name": "Anthropic Adapter",
                "provider": "anthropic",
                "capabilities": ["completion", "chat"],
                "status": "active"
            }
        ],
        "total": 2
    }


# Legacy capabilities endpoint

@router.get("/capabilities")
async def get_mcp_capabilities(
    current_user: dict = Depends(get_current_active_user)
):
    """Get MCP system capabilities."""
    return {
        "capabilities": {
            "providers": ["openai", "anthropic", "google", "cohere", "mistral"],
            "features": ["context-management", "token-optimization", "streaming"],
            "max_context_size": 128000,
            "supported_models": [
                "gpt-4", "gpt-3.5-turbo", "claude-3", "claude-2",
                "gemini-pro", "command", "mistral-large"
            ]
        }
    }