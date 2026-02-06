"""MCP Server Management API endpoints.

This module implements MCP (Model Context Protocol) server management
using the real MCP protocol specification:
- stdio transport (subprocess with stdin/stdout communication)
- JSON-RPC 2.0 message format
- Proper initialization handshake with capability negotiation

See: https://modelcontextprotocol.io/specification/2025-03-26
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List, Optional, Dict, Any
import logging
import json

from core.database import get_db
from core.security import get_current_active_user
from schemas.mcp import (
    MCPServerCreate,
    MCPServerResponse,
    MCPServerUpdate,
    MCPServerList,
    MCPHealthCheck,
    MCPHealthCheckList,
    MCPTestRequest,
    MCPTestResponse,
    MCPTaskCreate,
    MCPTaskResponse,
    MCPServerDiscoveryResponse,
    MCPInfo,
    MCPTransportType,
    # MCP Async Task schemas (SEP-1686)
    MCPAsyncTaskCreate,
    MCPAsyncTaskResponse,
    MCPAsyncTaskUpdate,
    MCPAsyncTaskList,
    MCPTaskState,
    MCPTasksGetRequest,
    MCPTasksCancelRequest,
    # MCP Sampling schemas (SEP-1577)
    MCPCreateSamplingRequest,
    MCPCreateSamplingResponse,
    MCPSamplingCapability,
    # MCP Elicitation schemas (SEP-1036)
    MCPElicitationRequest,
    MCPElicitationResponse,
    MCPElicitationStatus
)
from services.mcp_integration import MCPTaskIntegration
from adapters.mcp.factory import create_mcp_dispatcher
# Use unified MCP service for new endpoints
from services.mcp_unified import UnifiedMCPService
# Import real MCP client adapter
from adapters.implementations.ai.mcp_client_adapter import (
    MCPClientAdapter,
    MCPServerConfig,
    MCPTransportType as AdapterTransportType
)
from adapters.models import AdapterConfig
from models import MCPServer, MCPServerCapability, User
from models.community_complete import MCPAsyncTask, MCPTaskState as MCPTaskStateModel
from datetime import datetime

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level MCP client adapter instance for real MCP connections
_mcp_client: Optional[MCPClientAdapter] = None


async def get_oauth_token_for_mcp_server(
    server: "MCPServer",
    db: AsyncSession,
    user_id: Optional[str] = None
) -> Optional[str]:
    """Get OAuth2 access token for MCP server authentication (SEP-991).

    In Community Edition, this returns None (no OAuth2 support).
    Business/Enterprise editions override this to fetch tokens from
    their OAuth2 provider integration.

    Args:
        server: The MCP server instance
        db: Database session
        user_id: Optional user ID for user-specific tokens

    Returns:
        Access token string if available, None otherwise
    """
    # Community Edition: OAuth2 is a Business feature
    # Return None - Business/Enterprise will override this
    if server.oauth2_provider_id:
        logger.info(
            f"MCP server {server.name} has oauth2_provider_id={server.oauth2_provider_id}, "
            "but OAuth2 is a Business/Enterprise feature"
        )
    return None


async def get_mcp_client() -> MCPClientAdapter:
    """Get or create MCP client adapter instance."""
    global _mcp_client
    if _mcp_client is None:
        from adapters.models import AdapterCategory
        config = AdapterConfig(
            name="mcp_client",
            category=AdapterCategory.AI
        )
        _mcp_client = MCPClientAdapter(config)
        await _mcp_client.initialize()
    return _mcp_client


@router.get("/info", response_model=MCPInfo)
async def get_mcp_info():
    """Get MCP system information"""
    return MCPInfo(
        version="1.0.0",
        supported_providers=["openai", "anthropic", "google", "cohere", "huggingface", "local", "custom"],
        features=["context-management", "token-optimization", "multi-provider", "task-routing"],
        status="active"
    )


@router.post("/servers", response_model=MCPServerResponse)
async def register_mcp_server(
    server_data: MCPServerCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Register a new MCP server.

    Supports two transport types per MCP specification:
    1. stdio: Spawns a subprocess and communicates via stdin/stdout (standard MCP)
    2. http_sse: Connects to a remote server via HTTP with SSE (less common)

    For stdio transport (recommended):
    - command: The executable to run (e.g., "npx", "python", "node")
    - args: Arguments to pass to the command
    - env_vars: Environment variables for the subprocess

    For http_sse transport:
    - url: The server URL
    - api_key: Optional API key for authentication
    """
    try:
        # Validate transport-specific requirements
        if server_data.transport_type == MCPTransportType.stdio:
            if not server_data.command:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="command is required for stdio transport"
                )
        elif server_data.transport_type == MCPTransportType.http_sse:
            if not server_data.url:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="url is required for http_sse transport"
                )

        # Check if server already exists by name
        existing = await db.execute(
            select(MCPServer).filter_by(name=server_data.name)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Server with name '{server_data.name}' already exists"
            )

        # Create server record in database
        server = MCPServer(
            name=server_data.name,
            transport_type=server_data.transport_type.value,
            command=server_data.command,
            args=json.dumps(server_data.args) if server_data.args else None,
            env_vars=json.dumps(server_data.env_vars) if server_data.env_vars else None,
            url=server_data.url,
            api_key=server_data.api_key,
            service_type=server_data.service_type,
            status="registered",
            server_info=json.dumps(server_data.server_metadata) if server_data.server_metadata else None,
            oauth2_provider_id=server_data.oauth2_provider_id,  # SEP-991: OAuth2 integration
            created_at=datetime.utcnow().timestamp(),
            updated_at=datetime.utcnow().timestamp()
        )
        db.add(server)
        await db.commit()
        await db.refresh(server)

        # For stdio transport, attempt to connect and discover capabilities
        connection_result = {}
        if server_data.transport_type == MCPTransportType.stdio and server_data.command:
            try:
                mcp_client = await get_mcp_client()
                mcp_config = MCPServerConfig(
                    name=server_data.name,
                    command=server_data.command,
                    args=server_data.args or [],
                    env=server_data.env_vars or {},
                    transport=AdapterTransportType.STDIO
                )

                connected = await mcp_client.connect_server(mcp_config)

                if connected:
                    connection = mcp_client.connections.get(server_data.name)
                    if connection:
                        # Update server with discovered capabilities
                        server.status = "connected"
                        server.protocol_version = connection._protocol_version
                        server.server_capabilities = json.dumps({
                            "tools": connection.capabilities.tools,
                            "resources": connection.capabilities.resources,
                            "prompts": connection.capabilities.prompts,
                            "logging": connection.capabilities.logging
                        })
                        server.last_checked = datetime.utcnow().timestamp()

                        connection_result = {
                            "connected": True,
                            "tools_count": len(connection.tools),
                            "resources_count": len(connection.resources),
                            "server_info": connection.server_info
                        }
                else:
                    server.status = "connection_failed"
                    connection_result = {"connected": False, "error": "Failed to connect"}

                await db.commit()
                await db.refresh(server)

            except Exception as e:
                logger.warning(f"Failed to connect to MCP server during registration: {e}")
                server.status = "connection_failed"
                await db.commit()

        # Parse stored JSON fields for response
        args_list = json.loads(server.args) if server.args else None
        env_dict = json.loads(server.env_vars) if server.env_vars else None
        caps_dict = json.loads(server.server_capabilities) if server.server_capabilities else None

        return MCPServerResponse(
            id=server.id,
            name=server.name,
            transport_type=server.transport_type,
            command=server.command,
            args=args_list,
            env_vars=env_dict,
            url=server.url,
            service_type=server.service_type,
            status=server.status,
            protocol_version=server.protocol_version,
            server_capabilities=caps_dict,
            last_checked=datetime.fromtimestamp(server.last_checked) if server.last_checked else None,
            server_info=connection_result.get("server_info") if connection_result else None,
            created_at=datetime.fromtimestamp(server.created_at),
            updated_at=datetime.fromtimestamp(server.updated_at) if server.updated_at else None,
            server_metadata=server_data.server_metadata,
            oauth2_provider_id=server.oauth2_provider_id  # SEP-991: OAuth2 integration
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to register MCP server: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register server: {str(e)}"
        )


@router.get("/servers", response_model=MCPServerList)
async def list_mcp_servers(
    capability: Optional[str] = None,
    transport_type: Optional[str] = None,
    server_status: Optional[str] = None,
    page: int = 1,
    per_page: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List all registered MCP servers.

    Supports filtering by:
    - capability: Filter by specific capability support
    - transport_type: Filter by transport type (stdio or http_sse)
    - server_status: Filter by connection status
    """
    try:
        # Build query
        query = select(MCPServer)

        if server_status:
            query = query.filter_by(status=server_status)

        if transport_type:
            query = query.filter_by(transport_type=transport_type)

        # Get total count
        count_query = select(func.count()).select_from(MCPServer)
        if server_status:
            count_query = count_query.filter_by(status=server_status)
        if transport_type:
            count_query = count_query.filter_by(transport_type=transport_type)
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination
        offset = (page - 1) * per_page
        query = query.offset(offset).limit(per_page)

        # Execute query
        result = await db.execute(query)
        servers = result.scalars().all()

        # Convert to response models
        server_responses = []
        for server in servers:
            # Check capability filter if provided
            if capability:
                caps = await db.execute(
                    select(MCPServerCapability).filter_by(
                        server_id=server.id,
                        capability=capability,
                        supported=True
                    )
                )
                if not caps.scalar_one_or_none():
                    continue

            # Parse stored JSON fields
            args_list = json.loads(server.args) if server.args else None
            env_dict = json.loads(server.env_vars) if server.env_vars else None
            caps_dict = json.loads(server.server_capabilities) if server.server_capabilities else None

            server_responses.append(MCPServerResponse(
                id=server.id,
                name=server.name,
                transport_type=server.transport_type,
                command=server.command,
                args=args_list,
                env_vars=env_dict,
                url=server.url,
                service_type=server.service_type,
                status=server.status,
                protocol_version=server.protocol_version,
                server_capabilities=caps_dict,
                last_checked=datetime.fromtimestamp(server.last_checked) if server.last_checked else None,
                server_info=json.loads(server.server_info) if server.server_info else None,
                created_at=datetime.fromtimestamp(server.created_at),
                updated_at=datetime.fromtimestamp(server.updated_at) if server.updated_at else None,
                oauth2_provider_id=server.oauth2_provider_id  # SEP-991: OAuth2 integration
            ))

        return MCPServerList(
            servers=server_responses,
            total=total,
            page=page,
            per_page=per_page
        )

    except Exception as e:
        logger.error(f"Failed to list MCP servers: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list servers: {str(e)}"
        )


@router.get("/servers/{server_id}", response_model=MCPServerResponse)
async def get_mcp_server(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get details of a specific MCP server."""
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    # Parse stored JSON fields
    args_list = json.loads(server.args) if server.args else None
    env_dict = json.loads(server.env_vars) if server.env_vars else None
    caps_dict = json.loads(server.server_capabilities) if server.server_capabilities else None

    return MCPServerResponse(
        id=server.id,
        name=server.name,
        transport_type=server.transport_type,
        command=server.command,
        args=args_list,
        env_vars=env_dict,
        url=server.url,
        service_type=server.service_type,
        status=server.status,
        protocol_version=server.protocol_version,
        server_capabilities=caps_dict,
        last_checked=datetime.fromtimestamp(server.last_checked) if server.last_checked else None,
        server_info=json.loads(server.server_info) if server.server_info else None,
        created_at=datetime.fromtimestamp(server.created_at),
        updated_at=datetime.fromtimestamp(server.updated_at) if server.updated_at else None,
        oauth2_provider_id=server.oauth2_provider_id  # SEP-991: OAuth2 integration
    )


@router.patch("/servers/{server_id}", response_model=MCPServerResponse)
async def update_mcp_server(
    server_id: str,
    update_data: MCPServerUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update an MCP server."""
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        if field == "server_metadata" and value is not None:
            # Store metadata as JSON string
            setattr(server, "server_info", json.dumps(value))
        elif field == "args" and value is not None:
            # Store args as JSON string
            setattr(server, "args", json.dumps(value))
        elif field == "env_vars" and value is not None:
            # Store env_vars as JSON string
            setattr(server, "env_vars", json.dumps(value))
        elif field == "transport_type" and value is not None:
            # Store transport_type as string value
            setattr(server, "transport_type", value.value if hasattr(value, 'value') else value)
        else:
            setattr(server, field, value)

    server.updated_at = datetime.utcnow().timestamp()

    await db.commit()
    await db.refresh(server)

    # Parse stored JSON fields
    args_list = json.loads(server.args) if server.args else None
    env_dict = json.loads(server.env_vars) if server.env_vars else None
    caps_dict = json.loads(server.server_capabilities) if server.server_capabilities else None

    return MCPServerResponse(
        id=server.id,
        name=server.name,
        transport_type=server.transport_type,
        command=server.command,
        args=args_list,
        env_vars=env_dict,
        url=server.url,
        service_type=server.service_type,
        status=server.status,
        protocol_version=server.protocol_version,
        server_capabilities=caps_dict,
        last_checked=datetime.fromtimestamp(server.last_checked) if server.last_checked else None,
        server_info=json.loads(server.server_info) if server.server_info else None,
        created_at=datetime.fromtimestamp(server.created_at),
        updated_at=datetime.fromtimestamp(server.updated_at) if server.updated_at else None,
        oauth2_provider_id=server.oauth2_provider_id  # SEP-991: OAuth2 integration
    )


@router.delete("/servers/{server_id}")
async def delete_mcp_server(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an MCP server"""
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()
    
    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )
    
    # Remove from dispatcher
    dispatcher = create_mcp_dispatcher()
    await dispatcher.remove_server(server.name)
    
    # Delete from database
    await db.delete(server)
    await db.commit()
    
    return {"message": "MCP server deleted successfully"}


@router.get("/servers/{server_id}/health", response_model=MCPHealthCheck)
async def check_server_health(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Check health of a specific MCP server.

    For stdio transport servers, this will check if the subprocess is running
    and the server is responsive to JSON-RPC messages.

    For http_sse transport servers, this will check if the URL is reachable.
    """
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    try:
        mcp_client = await get_mcp_client()
        connection = mcp_client.connections.get(server.name)

        if connection and connection.initialized:
            # Server is connected and initialized
            return MCPHealthCheck(
                server_id=server.id,
                server_name=server.name,
                transport_type=server.transport_type,
                url=server.url,
                command=server.command,
                status="healthy",
                protocol_version=connection._protocol_version,
                checked_at=datetime.utcnow()
            )
        elif server.transport_type == "stdio" and server.command:
            # Try to connect and check health
            try:
                args_list = json.loads(server.args) if server.args else []
                env_dict = json.loads(server.env_vars) if server.env_vars else {}

                mcp_config = MCPServerConfig(
                    name=server.name,
                    command=server.command,
                    args=args_list,
                    env=env_dict,
                    transport=AdapterTransportType.STDIO
                )

                connected = await mcp_client.connect_server(mcp_config)

                if connected:
                    connection = mcp_client.connections.get(server.name)
                    # Update server status in database
                    server.status = "connected"
                    server.protocol_version = connection._protocol_version if connection else None
                    server.last_checked = datetime.utcnow().timestamp()
                    await db.commit()

                    return MCPHealthCheck(
                        server_id=server.id,
                        server_name=server.name,
                        transport_type=server.transport_type,
                        command=server.command,
                        status="healthy",
                        protocol_version=connection._protocol_version if connection else None,
                        checked_at=datetime.utcnow()
                    )
                else:
                    server.status = "unhealthy"
                    server.last_checked = datetime.utcnow().timestamp()
                    await db.commit()

                    return MCPHealthCheck(
                        server_id=server.id,
                        server_name=server.name,
                        transport_type=server.transport_type,
                        command=server.command,
                        status="unhealthy",
                        error="Failed to connect to MCP server",
                        checked_at=datetime.utcnow()
                    )
            except Exception as e:
                return MCPHealthCheck(
                    server_id=server.id,
                    server_name=server.name,
                    transport_type=server.transport_type,
                    command=server.command,
                    status="error",
                    error=str(e),
                    checked_at=datetime.utcnow()
                )
        else:
            # Fallback to old dispatcher for HTTP/SSE or legacy servers
            dispatcher = create_mcp_dispatcher()
            health = await dispatcher.get_server_health(server.name)

            return MCPHealthCheck(
                server_id=server.id,
                server_name=server.name,
                transport_type=server.transport_type,
                url=server.url,
                command=server.command,
                status=health.get("status", "unknown"),
                protocol_version=health.get("protocol_version"),
                error=health.get("error"),
                checked_at=datetime.utcnow()
            )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return MCPHealthCheck(
            server_id=server.id,
            server_name=server.name,
            transport_type=server.transport_type,
            url=server.url,
            command=server.command,
            status="error",
            error=str(e),
            checked_at=datetime.utcnow()
        )


@router.post("/servers/{server_id}/test", response_model=MCPTestResponse)
async def test_server_connection(
    server_id: str,
    test_request: MCPTestRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Test connection to an MCP server"""
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()
    
    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )
    
    try:
        # Create test task
        test_task = MCPTaskIntegration.prepare_mcp_task({
            "task_id": f"test-{server_id}",
            "source_id": str(current_user.id),
            "payload": {
                "messages": [{"role": "user", "content": test_request.test_message}],
                "server_name": server.name,
                "max_tokens": 100
            }
        })
        
        # Route task
        start_time = datetime.utcnow()
        result, status_code = await MCPTaskIntegration.route_task(test_task)
        end_time = datetime.utcnow()
        
        latency_ms = (end_time - start_time).total_seconds() * 1000
        
        if status_code == 200:
            return MCPTestResponse(
                server_id=server_id,
                success=True,
                response=result.get("result", {}).get("response", "Test successful"),
                latency_ms=latency_ms,
                tested_at=datetime.utcnow()
            )
        else:
            return MCPTestResponse(
                server_id=server_id,
                success=False,
                error=result.get("error", "Test failed"),
                latency_ms=latency_ms,
                tested_at=datetime.utcnow()
            )
            
    except Exception as e:
        logger.error(f"Test connection failed: {str(e)}")
        return MCPTestResponse(
            server_id=server_id,
            success=False,
            error=str(e),
            tested_at=datetime.utcnow()
        )


@router.get("/discovery", response_model=MCPServerDiscoveryResponse)
async def discover_mcp_resources(
    current_user: User = Depends(get_current_active_user)
):
    """Discover available MCP resources and capabilities"""
    try:
        capabilities = await MCPTaskIntegration.list_mcp_capabilities()

        return MCPServerDiscoveryResponse(
            servers=[
                MCPServerResponse(**server) for server in capabilities.get("servers", [])
            ],
            total=capabilities.get("total_servers", 0),
            providers=list(set(s.get("service_type", "custom") for s in capabilities.get("servers", []))),
            capabilities=capabilities.get("capabilities", [])
        )

    except Exception as e:
        logger.warning(f"MCP discovery unavailable: {str(e)}")
        # Return empty discovery response when service is unavailable
        return MCPServerDiscoveryResponse(
            servers=[],
            total=0,
            providers=[],
            capabilities=[]
        )


@router.post("/execute", response_model=MCPTaskResponse)
async def execute_mcp_task(
    task_data: MCPTaskCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Execute a task via MCP"""
    try:
        # Prepare task
        task = MCPTaskIntegration.prepare_mcp_task({
            "task_id": f"mcp-{datetime.utcnow().timestamp()}",
            "source_id": str(current_user.id),
            "payload": task_data.dict()
        }, api_type=task_data.api_type)
        
        # Route task
        result, status_code = await MCPTaskIntegration.route_task(task)
        
        if status_code != 200:
            raise HTTPException(
                status_code=status_code,
                detail=result.get("error", "Task execution failed")
            )
        
        return MCPTaskResponse(
            task_id=result["task_id"],
            source_id=result["source_id"],
            destination=result["destination"],
            status=result["status"],
            result=result.get("result"),
            mcp_server_used=result.get("mcp_metadata", {}).get("server_name"),
            input_tokens=result.get("usage", {}).get("prompt_tokens"),
            output_tokens=result.get("usage", {}).get("completion_tokens"),
            created_at=datetime.utcnow(),
            completed_at=datetime.utcnow() if result["status"] == "completed" else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task execution failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task execution failed: {str(e)}"
        )


@router.get("/test")
async def test_mcp_connection(
    server_url: str = "http://mcp-test-server:8080",
    current_user: User = Depends(get_current_active_user)
):
    """Test MCP connection and list available tools."""
    try:
        # Initialize service
        service = UnifiedMCPService()

        # Connect and get tools
        await service.connect(server_url)

        # Get capabilities
        capabilities = await service.list_tools()

        return {
            "status": "connected",
            "server_url": server_url,
            "capabilities": capabilities
        }

    except Exception as e:
        logger.error(f"MCP test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )


# ============================================================
# Real MCP Protocol Endpoints (stdio transport, JSON-RPC 2.0)
# ============================================================

@router.post("/servers/{server_id}/connect")
async def connect_to_mcp_server(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Connect to an MCP server using the real MCP protocol.

    This endpoint:
    1. Spawns the MCP server subprocess (for stdio transport)
    2. Performs the MCP initialization handshake
    3. Discovers available tools and resources
    4. Updates the server status and capabilities in the database

    Returns the discovered tools and resources.
    """
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    if server.transport_type != "stdio":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only stdio transport is currently supported for real MCP connections"
        )

    if not server.command:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Server command is required for stdio transport"
        )

    try:
        mcp_client = await get_mcp_client()

        # Check if already connected
        if server.name in mcp_client.connections:
            connection = mcp_client.connections[server.name]
            return {
                "status": "already_connected",
                "server_name": server.name,
                "tools": [
                    {
                        "name": t.name,
                        "description": t.description,
                        "input_schema": t.input_schema
                    }
                    for t in connection.tools
                ],
                "resources": [
                    {
                        "uri": r.uri,
                        "name": r.name,
                        "description": r.description
                    }
                    for r in connection.resources
                ],
                "capabilities": {
                    "tools": connection.capabilities.tools,
                    "resources": connection.capabilities.resources,
                    "prompts": connection.capabilities.prompts
                }
            }

        # Parse stored JSON fields
        args_list = json.loads(server.args) if server.args else []
        env_dict = json.loads(server.env_vars) if server.env_vars else {}

        # Create MCP server config
        mcp_config = MCPServerConfig(
            name=server.name,
            command=server.command,
            args=args_list,
            env=env_dict,
            transport=AdapterTransportType.STDIO
        )

        # Connect to server
        connected = await mcp_client.connect_server(mcp_config)

        if not connected:
            server.status = "connection_failed"
            server.last_checked = datetime.utcnow().timestamp()
            await db.commit()

            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to connect to MCP server"
            )

        # Get connection details
        connection = mcp_client.connections[server.name]

        # Update server in database
        server.status = "connected"
        server.protocol_version = connection._protocol_version
        server.server_capabilities = json.dumps({
            "tools": connection.capabilities.tools,
            "resources": connection.capabilities.resources,
            "prompts": connection.capabilities.prompts,
            "logging": connection.capabilities.logging
        })
        server.last_checked = datetime.utcnow().timestamp()
        await db.commit()

        return {
            "status": "connected",
            "server_name": server.name,
            "protocol_version": connection._protocol_version,
            "server_info": connection.server_info,
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema
                }
                for t in connection.tools
            ],
            "tools_count": len(connection.tools),
            "resources": [
                {
                    "uri": r.uri,
                    "name": r.name,
                    "description": r.description,
                    "mime_type": r.mime_type
                }
                for r in connection.resources
            ],
            "resources_count": len(connection.resources),
            "capabilities": {
                "tools": connection.capabilities.tools,
                "resources": connection.capabilities.resources,
                "prompts": connection.capabilities.prompts,
                "logging": connection.capabilities.logging
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect to MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to connect: {str(e)}"
        )


@router.post("/servers/{server_id}/disconnect")
async def disconnect_from_mcp_server(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Disconnect from an MCP server.

    This terminates the MCP server subprocess and cleans up resources.
    """
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    try:
        mcp_client = await get_mcp_client()

        if server.name not in mcp_client.connections:
            return {
                "status": "not_connected",
                "server_name": server.name,
                "message": "Server was not connected"
            }

        # Disconnect
        await mcp_client.disconnect_server(server.name)

        # Update server status
        server.status = "disconnected"
        server.last_checked = datetime.utcnow().timestamp()
        await db.commit()

        return {
            "status": "disconnected",
            "server_name": server.name,
            "message": "Successfully disconnected from MCP server"
        }

    except Exception as e:
        logger.error(f"Failed to disconnect from MCP server: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to disconnect: {str(e)}"
        )


@router.get("/servers/{server_id}/tools")
async def list_server_tools(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List tools available from an MCP server.

    The server must be connected first using the /connect endpoint.
    Per MCP 2025-11-25, includes outputSchema and annotations when available.
    """
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    try:
        mcp_client = await get_mcp_client()

        if server.name not in mcp_client.connections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Server is not connected. Call /connect first."
            )

        connection = mcp_client.connections[server.name]

        # Build tool list with MCP 2025-11-25 features
        tools_list = []
        for t in connection.tools:
            tool_info = {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
                "server_name": t.server_name
            }

            # Include outputSchema if available (MCP 2025-11-25)
            if hasattr(t, 'output_schema') and t.output_schema:
                tool_info["output_schema"] = t.output_schema

            # Include annotations if available (MCP 2025-11-25)
            if hasattr(t, 'annotations') and t.annotations:
                tool_info["annotations"] = {
                    "title": getattr(t.annotations, 'title', None),
                    "read_only_hint": getattr(t.annotations, 'read_only_hint', None),
                    "destructive_hint": getattr(t.annotations, 'destructive_hint', None),
                    "idempotent_hint": getattr(t.annotations, 'idempotent_hint', None),
                    "open_world_hint": getattr(t.annotations, 'open_world_hint', None)
                }
                # Remove None values
                tool_info["annotations"] = {
                    k: v for k, v in tool_info["annotations"].items() if v is not None
                }

            tools_list.append(tool_info)

        return {
            "server_name": server.name,
            "tools": tools_list,
            "total": len(connection.tools)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tools: {str(e)}"
        )


@router.post("/servers/{server_id}/tools/{tool_name}/call")
async def call_mcp_tool(
    server_id: str,
    tool_name: str,
    arguments: Dict[str, Any],
    request_structured: bool = False,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Call a tool on an MCP server with optional structured output.

    This sends a tools/call request to the MCP server using JSON-RPC 2.0.
    Per MCP 2025-11-25, supports structured output when tool has outputSchema.

    Args:
        server_id: The server ID
        tool_name: The name of the tool to call
        arguments: The arguments to pass to the tool (as request body)
        request_structured: If True, request structured output (if tool supports it)

    Returns:
        The tool execution result with content array and optional structuredContent.
    """
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    try:
        mcp_client = await get_mcp_client()

        if server.name not in mcp_client.connections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Server is not connected. Call /connect first."
            )

        # Get tool definition to check for outputSchema
        connection = mcp_client.connections[server.name]
        tool_def = next((t for t in connection.tools if t.name == tool_name), None)
        has_output_schema = tool_def and hasattr(tool_def, 'output_schema') and tool_def.output_schema

        # Call the tool
        start_time = datetime.utcnow()
        tool_result = await mcp_client.call_tool(server.name, tool_name, arguments)
        end_time = datetime.utcnow()

        latency_ms = (end_time - start_time).total_seconds() * 1000

        if "error" in tool_result:
            return {
                "success": False,
                "server_name": server.name,
                "tool_name": tool_name,
                "error": tool_result["error"],
                "latency_ms": latency_ms
            }

        # Build response with structured content support
        response = {
            "success": True,
            "server_name": server.name,
            "tool_name": tool_name,
            "content": tool_result.get("content", []),
            "is_error": tool_result.get("isError", False),
            "latency_ms": latency_ms
        }

        # Include structured content if available (MCP 2025-11-25)
        if "structuredContent" in tool_result:
            response["structured_content"] = {
                "data": tool_result["structuredContent"],
                "validated": True,
                "schema_version": "1.0"
            }
            response["output_schema_used"] = tool_name

        # If tool has outputSchema and we got text content, try to parse as structured
        elif has_output_schema and request_structured:
            # Attempt to extract structured data from text content
            text_content = next(
                (c.get("text") for c in tool_result.get("content", []) if c.get("type") == "text"),
                None
            )
            if text_content:
                try:
                    import json as json_module
                    parsed = json_module.loads(text_content)
                    response["structured_content"] = {
                        "data": parsed,
                        "validated": False,  # Not validated against schema
                        "schema_version": "1.0"
                    }
                except (json_module.JSONDecodeError, ValueError):
                    pass  # Content is not JSON, skip structured content

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to call tool {tool_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to call tool: {str(e)}"
        )


@router.get("/servers/{server_id}/resources")
async def list_server_resources(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List resources available from an MCP server.

    The server must be connected first using the /connect endpoint.
    """
    result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    try:
        mcp_client = await get_mcp_client()

        if server.name not in mcp_client.connections:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Server is not connected. Call /connect first."
            )

        connection = mcp_client.connections[server.name]

        return {
            "server_name": server.name,
            "resources": [
                {
                    "uri": r.uri,
                    "name": r.name,
                    "description": r.description,
                    "mime_type": r.mime_type
                }
                for r in connection.resources
            ],
            "total": len(connection.resources)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list resources: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list resources: {str(e)}"
        )


@router.get("/connections")
async def list_active_connections(
    current_user: User = Depends(get_current_active_user)
):
    """List all active MCP server connections.

    Returns information about currently connected MCP servers and their
    available tools and resources.
    """
    try:
        mcp_client = await get_mcp_client()
        health = await mcp_client.health_check()

        return {
            "status": health["status"],
            "connected_servers": health["connected_servers"],
            "protocol_compliant": health.get("protocol_compliant", True),
            "transport": health.get("transport", "stdio"),
            "servers": health["servers"]
        }

    except Exception as e:
        logger.warning(f"MCP connections unavailable: {e}")
        # Return empty connections response when service is unavailable
        return {
            "status": "unavailable",
            "connected_servers": 0,
            "protocol_compliant": True,
            "transport": "stdio",
            "servers": []
        }


# ============================================================
# MCP Async Tasks Endpoints (SEP-1686)
# Per MCP specification 2025-11-25
# See: https://spec.modelcontextprotocol.io/specification/2025-11-25/server/tasks/
# ============================================================

@router.post("/tasks", response_model=MCPAsyncTaskResponse)
async def create_async_task(
    task_data: MCPAsyncTaskCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create a new async task for tracking long-running MCP operations.

    Per MCP SEP-1686, tasks are used when operations take longer than
    a single request/response cycle. The client can poll for status
    using the task token.

    Returns:
        The created task with its unique token.
    """
    import uuid

    try:
        # Verify server exists
        result = await db.execute(
            select(MCPServer).filter_by(id=task_data.server_id)
        )
        server = result.scalar_one_or_none()

        if not server:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="MCP server not found"
            )

        # Generate unique task token
        task_token = f"task_{uuid.uuid4().hex}"

        # Create task record
        task = MCPAsyncTask(
            server_id=task_data.server_id,
            tool_id=task_data.tool_id,
            task_token=task_token,
            method=task_data.method,
            state=MCPTaskStateModel.WORKING,
            request_params=task_data.request_params,
            timeout_seconds=task_data.timeout_seconds or 300,
            client_id=task_data.client_id or str(current_user.id),
            task_metadata=task_data.task_metadata,
            started_at=datetime.utcnow(),
            last_activity_at=datetime.utcnow()
        )
        db.add(task)
        await db.commit()
        await db.refresh(task)

        return MCPAsyncTaskResponse(
            id=task.id,
            task_token=task.task_token,
            server_id=task.server_id,
            tool_id=task.tool_id,
            method=task.method,
            state=MCPTaskState(task.state.value),
            progress=task.progress,
            progress_message=task.progress_message,
            result_content=task.result_content,
            structured_result=task.structured_result,
            error_code=task.error_code,
            error_message=task.error_message,
            error_data=task.error_data,
            started_at=task.started_at,
            completed_at=task.completed_at,
            last_activity_at=task.last_activity_at,
            timeout_seconds=task.timeout_seconds,
            client_id=task.client_id,
            task_metadata=task.task_metadata,
            created_at=task.created_at,
            updated_at=task.updated_at
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to create async task: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create task: {str(e)}"
        )


@router.get("/tasks", response_model=MCPAsyncTaskList)
async def list_async_tasks(
    server_id: Optional[str] = None,
    state: Optional[MCPTaskState] = None,
    page: int = 1,
    per_page: int = 20,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List async tasks with optional filtering.

    Args:
        server_id: Filter by MCP server ID
        state: Filter by task state (working, completed, failed, cancelled)
        page: Page number for pagination
        per_page: Items per page
    """
    try:
        # Build query
        query = select(MCPAsyncTask)

        if server_id:
            query = query.filter_by(server_id=server_id)

        if state:
            query = query.filter_by(state=MCPTaskStateModel(state.value))

        # Get total count
        count_query = select(func.count()).select_from(MCPAsyncTask)
        if server_id:
            count_query = count_query.filter_by(server_id=server_id)
        if state:
            count_query = count_query.filter_by(state=MCPTaskStateModel(state.value))
        total_result = await db.execute(count_query)
        total = total_result.scalar()

        # Apply pagination and ordering
        offset = (page - 1) * per_page
        query = query.order_by(MCPAsyncTask.created_at.desc()).offset(offset).limit(per_page)

        # Execute query
        result = await db.execute(query)
        tasks = result.scalars().all()

        # Convert to response models
        task_responses = [
            MCPAsyncTaskResponse(
                id=task.id,
                task_token=task.task_token,
                server_id=task.server_id,
                tool_id=task.tool_id,
                method=task.method,
                state=MCPTaskState(task.state.value),
                progress=task.progress,
                progress_message=task.progress_message,
                result_content=task.result_content,
                structured_result=task.structured_result,
                error_code=task.error_code,
                error_message=task.error_message,
                error_data=task.error_data,
                started_at=task.started_at,
                completed_at=task.completed_at,
                last_activity_at=task.last_activity_at,
                timeout_seconds=task.timeout_seconds,
                client_id=task.client_id,
                task_metadata=task.task_metadata,
                created_at=task.created_at,
                updated_at=task.updated_at
            )
            for task in tasks
        ]

        return MCPAsyncTaskList(
            tasks=task_responses,
            total=total,
            page=page,
            per_page=per_page
        )

    except Exception as e:
        logger.error(f"Failed to list async tasks: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list tasks: {str(e)}"
        )


# NOTE: This route MUST be defined before /tasks/{task_token} to avoid path conflicts
@router.get("/tasks/server/{server_id}/active", response_model=MCPAsyncTaskList)
async def get_active_tasks_for_server(
    server_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get all active (working) tasks for a specific MCP server.

    This is useful for monitoring the workload of a specific server.
    """
    # Verify server exists
    server_result = await db.execute(
        select(MCPServer).filter_by(id=server_id)
    )
    server = server_result.scalar_one_or_none()

    if not server:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP server not found"
        )

    # Get active tasks
    result = await db.execute(
        select(MCPAsyncTask)
        .filter_by(server_id=server_id, state=MCPTaskStateModel.WORKING)
        .order_by(MCPAsyncTask.started_at.desc())
    )
    tasks = result.scalars().all()

    task_responses = [
        MCPAsyncTaskResponse(
            id=task.id,
            task_token=task.task_token,
            server_id=task.server_id,
            tool_id=task.tool_id,
            method=task.method,
            state=MCPTaskState(task.state.value),
            progress=task.progress,
            progress_message=task.progress_message,
            result_content=task.result_content,
            structured_result=task.structured_result,
            error_code=task.error_code,
            error_message=task.error_message,
            error_data=task.error_data,
            started_at=task.started_at,
            completed_at=task.completed_at,
            last_activity_at=task.last_activity_at,
            timeout_seconds=task.timeout_seconds,
            client_id=task.client_id,
            task_metadata=task.task_metadata,
            created_at=task.created_at,
            updated_at=task.updated_at
        )
        for task in tasks
    ]

    return MCPAsyncTaskList(
        tasks=task_responses,
        total=len(task_responses),
        page=1,
        per_page=len(task_responses)
    )


@router.get("/tasks/{task_token}", response_model=MCPAsyncTaskResponse)
async def get_async_task(
    task_token: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of an async task by its token.

    This is the primary method for clients to poll task status
    per MCP SEP-1686.
    """
    result = await db.execute(
        select(MCPAsyncTask).filter_by(task_token=task_token)
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    return MCPAsyncTaskResponse(
        id=task.id,
        task_token=task.task_token,
        server_id=task.server_id,
        tool_id=task.tool_id,
        method=task.method,
        state=MCPTaskState(task.state.value),
        progress=task.progress,
        progress_message=task.progress_message,
        result_content=task.result_content,
        structured_result=task.structured_result,
        error_code=task.error_code,
        error_message=task.error_message,
        error_data=task.error_data,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_activity_at=task.last_activity_at,
        timeout_seconds=task.timeout_seconds,
        client_id=task.client_id,
        task_metadata=task.task_metadata,
        created_at=task.created_at,
        updated_at=task.updated_at
    )


@router.patch("/tasks/{task_token}", response_model=MCPAsyncTaskResponse)
async def update_async_task(
    task_token: str,
    update_data: MCPAsyncTaskUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Update an async task's status, progress, or result.

    This is used by MCP servers to update task progress or mark
    tasks as completed/failed.
    """
    result = await db.execute(
        select(MCPAsyncTask).filter_by(task_token=task_token)
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Update fields
    update_dict = update_data.model_dump(exclude_unset=True)
    for field, value in update_dict.items():
        if field == "state" and value is not None:
            setattr(task, field, MCPTaskStateModel(value.value))
        else:
            setattr(task, field, value)

    # Update timestamps
    task.last_activity_at = datetime.utcnow()
    if update_data.state in [MCPTaskState.completed, MCPTaskState.failed, MCPTaskState.cancelled]:
        task.completed_at = datetime.utcnow()

    await db.commit()
    await db.refresh(task)

    return MCPAsyncTaskResponse(
        id=task.id,
        task_token=task.task_token,
        server_id=task.server_id,
        tool_id=task.tool_id,
        method=task.method,
        state=MCPTaskState(task.state.value),
        progress=task.progress,
        progress_message=task.progress_message,
        result_content=task.result_content,
        structured_result=task.structured_result,
        error_code=task.error_code,
        error_message=task.error_message,
        error_data=task.error_data,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_activity_at=task.last_activity_at,
        timeout_seconds=task.timeout_seconds,
        client_id=task.client_id,
        task_metadata=task.task_metadata,
        created_at=task.created_at,
        updated_at=task.updated_at
    )


@router.post("/tasks/{task_token}/cancel", response_model=MCPAsyncTaskResponse)
async def cancel_async_task(
    task_token: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Cancel an async task.

    Per MCP SEP-1686, this requests cancellation of a running task.
    The server should honor the cancellation request if possible.
    """
    result = await db.execute(
        select(MCPAsyncTask).filter_by(task_token=task_token)
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Can only cancel working tasks
    if task.state != MCPTaskStateModel.WORKING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel task in state: {task.state.value}"
        )

    # Mark as cancelled
    task.state = MCPTaskStateModel.CANCELLED
    task.completed_at = datetime.utcnow()
    task.last_activity_at = datetime.utcnow()

    await db.commit()
    await db.refresh(task)

    return MCPAsyncTaskResponse(
        id=task.id,
        task_token=task.task_token,
        server_id=task.server_id,
        tool_id=task.tool_id,
        method=task.method,
        state=MCPTaskState(task.state.value),
        progress=task.progress,
        progress_message=task.progress_message,
        result_content=task.result_content,
        structured_result=task.structured_result,
        error_code=task.error_code,
        error_message=task.error_message,
        error_data=task.error_data,
        started_at=task.started_at,
        completed_at=task.completed_at,
        last_activity_at=task.last_activity_at,
        timeout_seconds=task.timeout_seconds,
        client_id=task.client_id,
        task_metadata=task.task_metadata,
        created_at=task.created_at,
        updated_at=task.updated_at
    )


@router.delete("/tasks/{task_token}")
async def delete_async_task(
    task_token: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Delete an async task record.

    Only completed, failed, or cancelled tasks can be deleted.
    """
    result = await db.execute(
        select(MCPAsyncTask).filter_by(task_token=task_token)
    )
    task = result.scalar_one_or_none()

    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )

    # Cannot delete working tasks
    if task.state == MCPTaskStateModel.WORKING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a running task. Cancel it first."
        )

    await db.delete(task)
    await db.commit()

    return {"message": "Task deleted successfully", "task_token": task_token}


# ============================================================
# MCP Sampling Endpoints (SEP-1577 - MCP 2025-11-25)
# Enables agentic workflows where server requests LLM sampling
# ============================================================

@router.post("/sampling/create", response_model=MCPCreateSamplingResponse)
async def create_sampling(
    request: MCPCreateSamplingRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Handle a sampling request from an MCP server.

    Per MCP SEP-1577, this endpoint allows MCP servers to request
    LLM sampling from the client, enabling agentic multi-step workflows.

    The client (our API) performs the sampling using available LLM adapters
    and returns the result to the MCP server.
    """
    try:
        # Build messages for LLM
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})

        for msg in request.messages:
            content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
            messages.append({"role": msg.role, "content": content})

        # Get model preferences
        model_name = None
        if request.model_preferences and request.model_preferences.hints:
            for hint in request.model_preferences.hints:
                if hint.name:
                    model_name = hint.name
                    break

        # Use LLM service for sampling
        # This integrates with existing LLM adapters
        try:
            from llm.service import LLMService
            llm_service = LLMService()

            result = await llm_service.complete(
                messages=messages,
                model=model_name,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stop=request.stop_sequences
            )

            return MCPCreateSamplingResponse(
                role="assistant",
                content=result.get("content", ""),
                model=result.get("model"),
                stop_reason=result.get("stop_reason", "endTurn")
            )

        except ImportError:
            # Fallback if LLM service not available
            logger.warning("LLM service not available for sampling")
            return MCPCreateSamplingResponse(
                role="assistant",
                content="Sampling service temporarily unavailable",
                model=None,
                stop_reason="endTurn"
            )

        except Exception as e:
            logger.warning(f"LLM sampling failed: {e}")
            return MCPCreateSamplingResponse(
                role="assistant",
                content="Sampling service temporarily unavailable",
                model=None,
                stop_reason="endTurn"
            )

    except Exception as e:
        logger.warning(f"Sampling request failed: {e}")
        return MCPCreateSamplingResponse(
            role="assistant",
            content="Sampling service temporarily unavailable",
            model=None,
            stop_reason="endTurn"
        )


@router.get("/sampling/capabilities", response_model=MCPSamplingCapability)
async def get_sampling_capabilities(
    current_user: User = Depends(get_current_active_user)
):
    """Get the sampling capabilities of this MCP client.

    Returns information about supported models and limits.
    """
    try:
        # Check what LLM adapters are available
        available_models = []
        try:
            from adapters.registry import AdapterRegistry
            registry = AdapterRegistry()
            for adapter_name in ["claude", "anthropic", "openai", "ollama"]:
                if registry.get_adapter_class(adapter_name):
                    available_models.append(adapter_name)
        except Exception:
            available_models = ["claude", "openai"]  # Defaults

        return MCPSamplingCapability(
            supported=True,
            models=available_models,
            max_context_tokens=128000  # Default max context
        )

    except Exception as e:
        logger.error(f"Failed to get sampling capabilities: {e}")
        return MCPSamplingCapability(
            supported=False,
            models=[],
            max_context_tokens=0
        )


# ============================================================
# MCP Elicitation Endpoints (SEP-1036 - MCP 2025-11-25)
# Secure out-of-band credential flows
# ============================================================

@router.post("/elicitation/request", response_model=MCPElicitationResponse)
async def create_elicitation_request(
    request: MCPElicitationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Create an elicitation request for secure user interaction.

    Per MCP SEP-1036, this allows servers to request the client
    to open a URL for secure credential input or OAuth flows.

    In a web context, this would trigger the frontend to open
    the URL in a new tab/window for user interaction.
    """
    import uuid

    try:
        # Generate a unique request ID for tracking
        request_id = f"elicit_{uuid.uuid4().hex[:12]}"

        # In production, this would:
        # 1. Store the request in the database
        # 2. Notify the frontend via WebSocket/SSE
        # 3. Wait for callback from the URL
        # 4. Return the result

        # For now, return a pending response that the frontend can poll
        logger.info(f"Elicitation request created: {request_id} -> {request.url}")

        return MCPElicitationResponse(
            status=MCPElicitationStatus.pending,
            result={
                "request_id": request_id,
                "url": request.url,
                "message": request.message,
                "timeout_seconds": request.timeout_seconds
            }
        )

    except Exception as e:
        logger.error(f"Elicitation request failed: {e}")
        return MCPElicitationResponse(
            status=MCPElicitationStatus.cancelled,
            error=str(e)
        )


@router.get("/elicitation/{request_id}/status", response_model=MCPElicitationResponse)
async def get_elicitation_status(
    request_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Check the status of an elicitation request.

    The frontend polls this endpoint to check if the user
    has completed the action at the elicitation URL.
    """
    # In production, this would check the database for the result
    # For now, return pending as a placeholder

    return MCPElicitationResponse(
        status=MCPElicitationStatus.pending,
        result={"request_id": request_id, "message": "Waiting for user action"}
    )


@router.post("/elicitation/{request_id}/complete", response_model=MCPElicitationResponse)
async def complete_elicitation(
    request_id: str,
    result: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
):
    """Complete an elicitation request with the result.

    Called by the callback URL after user completes the action.
    """
    logger.info(f"Elicitation completed: {request_id}")

    return MCPElicitationResponse(
        status=MCPElicitationStatus.completed,
        result=result
    )