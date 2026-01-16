import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
import json
from httpx import AsyncClient

from src.mcp_client.client import MCPClient
from src.mcp_client.models import MCPServer, MCPCapability, MCPRequest, MCPResponse
from src.mcp_client.registry import MCPServerRegistry
from src.mcp_client.services import MCPClientService
from src.schemas.mcp import (
    MCPServerCreate, MCPServerUpdate, MCPServerResponse,
    MCPTaskResponse, MCPMetricsResponse
)


@pytest.fixture
def mock_server():
    """Create a mock MCP server."""
    return MCPServer(
        id="test-server-1",
        name="Test Server",
        url="http://test-server.com",
        api_key="test-api-key",
        capabilities=[MCPCapability.TASK_GENERATION, MCPCapability.DATA_VALIDATION],
        is_healthy=True,
        last_health_check=datetime.utcnow()
    )


@pytest.fixture
def mock_client():
    """Create a mock MCP client."""
    return MCPClient(
        server_url="http://test-server.com",
        api_key="test-api-key",
        timeout=30
    )


@pytest.fixture
async def mock_registry():
    """Create a mock MCP server registry."""
    registry = MCPServerRegistry()
    await registry.initialize()
    return registry


@pytest.fixture
async def mock_service(mock_registry):
    """Create a mock MCP client service."""
    service = MCPClientService()
    service.registry = mock_registry
    return service


class TestMCPClient:
    """Test MCP Client functionality."""
    
    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_client):
        """Test MCP client initialization."""
        assert mock_client.server_url == "http://test-server.com"
        assert mock_client.api_key == "test-api-key"
        assert mock_client.timeout == 30
    
    @pytest.mark.asyncio
    async def test_discover_capabilities(self, mock_client):
        """Test capability discovery."""
        with patch.object(mock_client.session, 'get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    "capabilities": ["task_generation", "data_validation"],
                    "version": "1.0.0"
                }
            )
            mock_get.return_value.__aenter__.return_value.raise_for_status = MagicMock()
            
            capabilities = await mock_client.discover_capabilities()
            
            assert len(capabilities) == 2
            assert MCPCapability.TASK_GENERATION in capabilities
            assert MCPCapability.DATA_VALIDATION in capabilities
    
    @pytest.mark.asyncio
    async def test_send_request(self, mock_client):
        """Test sending MCP request."""
        request = MCPRequest(
            method="task.generate",
            params={"prompt": "Test task"},
            context={"user_id": "test-user"}
        )
        
        with patch.object(mock_client.session, 'post') as mock_post:
            mock_post.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={
                    "result": {"task_id": "123", "status": "completed"},
                    "metadata": {"duration": 1.5}
                }
            )
            mock_post.return_value.__aenter__.return_value.raise_for_status = MagicMock()
            
            response = await mock_client.send_request(request)
            
            assert response.result["task_id"] == "123"
            assert response.result["status"] == "completed"
            assert response.metadata["duration"] == 1.5
    
    @pytest.mark.asyncio
    async def test_health_check(self, mock_client):
        """Test health check functionality."""
        with patch.object(mock_client.session, 'get') as mock_get:
            mock_get.return_value.__aenter__.return_value.json = AsyncMock(
                return_value={"status": "healthy", "uptime": 3600}
            )
            mock_get.return_value.__aenter__.return_value.raise_for_status = MagicMock()
            
            is_healthy = await mock_client.health_check()
            
            assert is_healthy is True


class TestMCPServerRegistry:
    """Test MCP Server Registry functionality."""
    
    @pytest.mark.asyncio
    async def test_register_server(self, mock_registry, mock_server):
        """Test server registration."""
        await mock_registry.register_server(mock_server)
        
        servers = await mock_registry.get_all_servers()
        assert len(servers) == 1
        assert servers[0].id == mock_server.id
    
    @pytest.mark.asyncio
    async def test_get_server_by_id(self, mock_registry, mock_server):
        """Test getting server by ID."""
        await mock_registry.register_server(mock_server)
        
        retrieved = await mock_registry.get_server(mock_server.id)
        assert retrieved is not None
        assert retrieved.id == mock_server.id
        assert retrieved.name == mock_server.name
    
    @pytest.mark.asyncio
    async def test_update_server(self, mock_registry, mock_server):
        """Test server update."""
        await mock_registry.register_server(mock_server)
        
        # Update server
        mock_server.name = "Updated Server"
        mock_server.is_healthy = False
        
        await mock_registry.update_server(mock_server)
        
        retrieved = await mock_registry.get_server(mock_server.id)
        assert retrieved.name == "Updated Server"
        assert retrieved.is_healthy is False
    
    @pytest.mark.asyncio
    async def test_remove_server(self, mock_registry, mock_server):
        """Test server removal."""
        await mock_registry.register_server(mock_server)
        
        await mock_registry.remove_server(mock_server.id)
        
        servers = await mock_registry.get_all_servers()
        assert len(servers) == 0
    
    @pytest.mark.asyncio
    async def test_get_healthy_servers(self, mock_registry):
        """Test getting healthy servers."""
        # Register healthy server
        healthy_server = MCPServer(
            id="healthy-1",
            name="Healthy Server",
            url="http://healthy.com",
            api_key="key1",
            capabilities=[MCPCapability.TASK_GENERATION],
            is_healthy=True,
            last_health_check=datetime.utcnow()
        )
        
        # Register unhealthy server
        unhealthy_server = MCPServer(
            id="unhealthy-1",
            name="Unhealthy Server",
            url="http://unhealthy.com",
            api_key="key2",
            capabilities=[MCPCapability.DATA_VALIDATION],
            is_healthy=False,
            last_health_check=datetime.utcnow()
        )
        
        await mock_registry.register_server(healthy_server)
        await mock_registry.register_server(unhealthy_server)
        
        healthy_servers = await mock_registry.get_healthy_servers()
        assert len(healthy_servers) == 1
        assert healthy_servers[0].id == "healthy-1"


class TestMCPClientService:
    """Test MCP Client Service functionality."""
    
    @pytest.mark.asyncio
    async def test_create_server(self, mock_service):
        """Test creating a server."""
        server_data = MCPServerCreate(
            name="New Server",
            url="http://new-server.com",
            api_key="new-key",
            capabilities=["task_generation", "data_validation"]
        )
        
        with patch('src.mcp_client.client.MCPClient') as MockClient:
            mock_client_instance = MockClient.return_value
            mock_client_instance.discover_capabilities = AsyncMock(
                return_value=[MCPCapability.TASK_GENERATION, MCPCapability.DATA_VALIDATION]
            )
            mock_client_instance.health_check = AsyncMock(return_value=True)
            
            server = await mock_service.create_server(server_data)
            
            assert server.name == "New Server"
            assert server.url == "http://new-server.com"
            assert server.is_healthy is True
            assert len(server.capabilities) == 2
    
    @pytest.mark.asyncio
    async def test_execute_task(self, mock_service, mock_server):
        """Test executing a task on a server."""
        await mock_service.registry.register_server(mock_server)
        
        with patch('src.mcp_client.client.MCPClient') as MockClient:
            mock_client_instance = MockClient.return_value
            mock_client_instance.send_request = AsyncMock(
                return_value=MCPResponse(
                    result={"status": "success", "data": "test-result"},
                    metadata={"duration": 0.5}
                )
            )
            
            result = await mock_service.execute_task(
                server_id=mock_server.id,
                method="task.generate",
                params={"prompt": "Test"},
                context={"user": "test"}
            )
            
            assert result["status"] == "success"
            assert result["data"] == "test-result"
    
    @pytest.mark.asyncio
    async def test_get_task_metrics(self, mock_service):
        """Test getting task metrics."""
        # Mock some task executions
        mock_service._task_count = 10
        mock_service._success_count = 8
        mock_service._failure_count = 2
        mock_service._total_duration = 15.5
        
        metrics = await mock_service.get_metrics()
        
        assert metrics["total_tasks"] == 10
        assert metrics["successful_tasks"] == 8
        assert metrics["failed_tasks"] == 2
        assert metrics["average_duration"] == 1.55
        assert metrics["success_rate"] == 0.8


@pytest.mark.asyncio
async def test_mcp_api_endpoints(async_client: AsyncClient):
    """Test MCP API endpoints."""
    # Test server creation
    server_data = {
        "name": "API Test Server",
        "url": "http://api-test.com",
        "api_key": "test-key",
        "capabilities": ["task_generation"]
    }
    
    response = await async_client.post(
        "/api/v1/mcp-client/servers/",
        json=server_data,
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    created_server = response.json()
    assert created_server["name"] == "API Test Server"
    server_id = created_server["id"]
    
    # Test getting all servers
    response = await async_client.get(
        "/api/v1/mcp-client/servers/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    servers = response.json()
    assert len(servers) >= 1
    
    # Test getting specific server
    response = await async_client.get(
        f"/api/v1/mcp-client/servers/{server_id}/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    server = response.json()
    assert server["id"] == server_id
    
    # Test server health check
    response = await async_client.post(
        f"/api/v1/mcp-client/servers/{server_id}/test/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    
    # Test metrics endpoint
    response = await async_client.get(
        "/api/v1/mcp-client/metrics/summary/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    metrics = response.json()
    assert "total_tasks" in metrics
    assert "average_duration" in metrics