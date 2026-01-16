import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import json
import asyncio
from httpx import AsyncClient

from src.mcp_server.server import MCPServer
from src.mcp_server.handlers import MCPRequestHandler
from src.mcp_server.models import MCPTask, TaskStatus, MCPEndpoint
from src.mcp_server.services.workflow_exposure import WorkflowExposureService
from src.schemas.mcp import (
    MCPTaskCreate, MCPTaskResponse, MCPEndpointResponse,
    WorkflowEndpointCreate, WorkflowEndpointResponse
)


@pytest.fixture
async def mock_mcp_server():
    """Create a mock MCP server."""
    server = MCPServer(port=8080)
    await server.initialize()
    return server


@pytest.fixture
def mock_handler():
    """Create a mock request handler."""
    return MCPRequestHandler()


@pytest.fixture
async def mock_exposure_service():
    """Create a mock workflow exposure service."""
    service = WorkflowExposureService()
    await service.initialize()
    return service


@pytest.fixture
def mock_task():
    """Create a mock MCP task."""
    return MCPTask(
        id="test-task-1",
        method="workflow.execute",
        params={"workflow_id": "wf-123", "input": {"data": "test"}},
        context={"user_id": "test-user"},
        status=TaskStatus.PENDING,
        created_at=datetime.utcnow()
    )


@pytest.fixture
def mock_endpoint():
    """Create a mock MCP endpoint."""
    return MCPEndpoint(
        endpoint_id="ep-123",
        workflow_id="wf-123",
        path="/test/endpoint",
        method="workflow.execute",
        description="Test endpoint",
        input_schema={"type": "object", "properties": {"data": {"type": "string"}}},
        output_schema={"type": "object", "properties": {"result": {"type": "string"}}},
        mode="single",
        active=True,
        created_at=datetime.utcnow()
    )


class TestMCPServer:
    """Test MCP Server functionality."""
    
    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_mcp_server):
        """Test server initialization."""
        assert mock_mcp_server.port == 8080
        assert mock_mcp_server.is_running is False
        assert mock_mcp_server.task_manager is not None
        assert mock_mcp_server.event_bus is not None
    
    @pytest.mark.asyncio
    async def test_register_handler(self, mock_mcp_server, mock_handler):
        """Test handler registration."""
        mock_mcp_server.register_handler("test.method", mock_handler.handle_test)
        
        assert "test.method" in mock_mcp_server.handlers
        assert mock_mcp_server.handlers["test.method"] == mock_handler.handle_test
    
    @pytest.mark.asyncio
    async def test_handle_request(self, mock_mcp_server):
        """Test request handling."""
        # Register a test handler
        async def test_handler(params, context):
            return {"status": "success", "data": params.get("input")}
        
        mock_mcp_server.register_handler("test.method", test_handler)
        
        # Create request
        request = {
            "method": "test.method",
            "params": {"input": "test-data"},
            "context": {"user_id": "test-user"}
        }
        
        response = await mock_mcp_server.handle_request(request)
        
        assert response["result"]["status"] == "success"
        assert response["result"]["data"] == "test-data"
        assert "task_id" in response["metadata"]
    
    @pytest.mark.asyncio
    async def test_task_creation_and_tracking(self, mock_mcp_server):
        """Test task creation and tracking."""
        # Create a task
        task_data = MCPTaskCreate(
            method="workflow.execute",
            params={"workflow_id": "wf-123"},
            context={"user_id": "test-user"}
        )
        
        task = await mock_mcp_server.task_manager.create_task(
            method=task_data.method,
            params=task_data.params,
            context=task_data.context
        )
        
        assert task.id is not None
        assert task.status == TaskStatus.PENDING
        
        # Update task status
        await mock_mcp_server.task_manager.update_task_status(
            task.id,
            TaskStatus.RUNNING
        )
        
        # Get task
        retrieved_task = await mock_mcp_server.task_manager.get_task(task.id)
        assert retrieved_task.status == TaskStatus.RUNNING


class TestMCPRequestHandler:
    """Test MCP Request Handler functionality."""
    
    @pytest.mark.asyncio
    async def test_workflow_execution_handler(self, mock_handler):
        """Test workflow execution handler."""
        with patch('src.services.workflow_service.WorkflowService') as MockService:
            mock_service = MockService.return_value
            mock_service.execute_workflow = AsyncMock(
                return_value={
                    "execution_id": "exec-123",
                    "status": "completed",
                    "output": {"result": "success"}
                }
            )
            
            params = {
                "workflow_id": "wf-123",
                "input": {"data": "test"}
            }
            context = {"user_id": "test-user"}
            
            result = await mock_handler.handle_workflow_execute(params, context)
            
            assert result["execution_id"] == "exec-123"
            assert result["status"] == "completed"
            assert result["output"]["result"] == "success"
    
    @pytest.mark.asyncio
    async def test_quality_validation_handler(self, mock_handler):
        """Test quality validation handler."""
        with patch('src.mcp_server.handlers.validate_data_quality') as mock_validate:
            mock_validate.return_value = {
                "is_valid": True,
                "score": 0.95,
                "issues": []
            }
            
            params = {
                "data": {"field1": "value1", "field2": "value2"},
                "schema": {"type": "object"}
            }
            context = {"user_id": "test-user"}
            
            result = await mock_handler.handle_quality_validate(params, context)
            
            assert result["is_valid"] is True
            assert result["score"] == 0.95
            assert len(result["issues"]) == 0


class TestWorkflowExposureService:
    """Test Workflow Exposure Service functionality."""
    
    @pytest.mark.asyncio
    async def test_register_endpoint(self, mock_exposure_service, mock_endpoint):
        """Test endpoint registration."""
        await mock_exposure_service.register_endpoint(mock_endpoint)
        
        endpoints = await mock_exposure_service.get_all_endpoints()
        assert len(endpoints) == 1
        assert endpoints[0].endpoint_id == mock_endpoint.endpoint_id
    
    @pytest.mark.asyncio
    async def test_get_endpoint_by_path(self, mock_exposure_service, mock_endpoint):
        """Test getting endpoint by path."""
        await mock_exposure_service.register_endpoint(mock_endpoint)
        
        endpoint = await mock_exposure_service.get_endpoint_by_path("/test/endpoint")
        assert endpoint is not None
        assert endpoint.endpoint_id == mock_endpoint.endpoint_id
    
    @pytest.mark.asyncio
    async def test_update_endpoint_stats(self, mock_exposure_service, mock_endpoint):
        """Test updating endpoint statistics."""
        await mock_exposure_service.register_endpoint(mock_endpoint)
        
        # Record some requests
        await mock_exposure_service.record_request(
            mock_endpoint.endpoint_id,
            duration=1.5,
            success=True
        )
        await mock_exposure_service.record_request(
            mock_endpoint.endpoint_id,
            duration=2.0,
            success=True
        )
        await mock_exposure_service.record_request(
            mock_endpoint.endpoint_id,
            duration=0.5,
            success=False
        )
        
        stats = await mock_exposure_service.get_endpoint_stats(mock_endpoint.endpoint_id)
        
        assert stats["total_requests"] == 3
        assert stats["successful_requests"] == 2
        assert stats["failed_requests"] == 1
        assert stats["average_duration"] == 1.33  # (1.5 + 2.0 + 0.5) / 3
        assert stats["success_rate"] == 0.67  # 2/3
    
    @pytest.mark.asyncio
    async def test_deactivate_endpoint(self, mock_exposure_service, mock_endpoint):
        """Test endpoint deactivation."""
        await mock_exposure_service.register_endpoint(mock_endpoint)
        
        await mock_exposure_service.deactivate_endpoint(mock_endpoint.endpoint_id)
        
        endpoint = await mock_exposure_service.get_endpoint(mock_endpoint.endpoint_id)
        assert endpoint.active is False


@pytest.mark.asyncio
async def test_mcp_server_api_endpoints(async_client: AsyncClient):
    """Test MCP Server API endpoints."""
    # Test task creation
    task_data = {
        "method": "workflow.execute",
        "params": {"workflow_id": "wf-123", "input": {"test": "data"}},
        "context": {"user_id": "test-user"}
    }
    
    response = await async_client.post(
        "/api/v1/mcp-server/tasks/",
        json=task_data,
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    created_task = response.json()
    assert created_task["method"] == "workflow.execute"
    task_id = created_task["id"]
    
    # Test getting tasks
    response = await async_client.get(
        "/api/v1/mcp-server/tasks/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    tasks = response.json()
    assert len(tasks) >= 1
    
    # Test getting specific task
    response = await async_client.get(
        f"/api/v1/mcp-server/tasks/{task_id}/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    task = response.json()
    assert task["id"] == task_id
    
    # Test workflow endpoint creation
    endpoint_data = {
        "workflow_id": "wf-123",
        "path": "/api/test/workflow",
        "description": "Test workflow endpoint",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "mode": "single"
    }
    
    response = await async_client.post(
        "/api/v1/mcp-server/workflow/endpoints/",
        json=endpoint_data,
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    created_endpoint = response.json()
    assert created_endpoint["workflow_id"] == "wf-123"
    endpoint_id = created_endpoint["endpoint_id"]
    
    # Test getting workflow endpoints
    response = await async_client.get(
        "/api/v1/mcp-server/workflow/endpoints/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "endpoints" in data
    assert len(data["endpoints"]) >= 1
    
    # Test endpoint statistics
    response = await async_client.get(
        f"/api/v1/mcp-server/workflow/endpoints/{endpoint_id}/stats/",
        headers={"Authorization": "Bearer dev-token-for-testing"}
    )
    
    assert response.status_code == 200
    stats = response.json()
    assert "total_requests" in stats
    assert "success_rate" in stats


class TestMCPIntegration:
    """Test MCP integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow_execution(self, mock_mcp_server, mock_exposure_service):
        """Test end-to-end workflow execution through MCP."""
        # Register workflow endpoint
        endpoint = MCPEndpoint(
            endpoint_id="e2e-test",
            workflow_id="wf-e2e",
            path="/e2e/test",
            method="workflow.execute",
            description="E2E test endpoint",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
            mode="single",
            active=True
        )
        
        await mock_exposure_service.register_endpoint(endpoint)
        
        # Mock workflow execution
        with patch('src.services.workflow_service.WorkflowService') as MockService:
            mock_service = MockService.return_value
            mock_service.execute_workflow = AsyncMock(
                return_value={
                    "execution_id": "exec-e2e",
                    "status": "completed",
                    "output": {"result": "e2e-success"}
                }
            )
            
            # Handle request through MCP server
            request = {
                "method": "workflow.execute",
                "params": {
                    "workflow_id": "wf-e2e",
                    "input": {"test": "e2e-data"}
                },
                "context": {"user_id": "e2e-user"}
            }
            
            # Register handler
            handler = MCPRequestHandler()
            mock_mcp_server.register_handler("workflow.execute", handler.handle_workflow_execute)
            
            # Execute request
            response = await mock_mcp_server.handle_request(request)
            
            assert response["result"]["status"] == "completed"
            assert response["result"]["output"]["result"] == "e2e-success"
            assert "task_id" in response["metadata"]
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self, mock_mcp_server):
        """Test concurrent task execution."""
        # Register handler that simulates work
        async def slow_handler(params, context):
            await asyncio.sleep(0.1)  # Simulate work
            return {"result": f"processed-{params['id']}"}
        
        mock_mcp_server.register_handler("test.slow", slow_handler)
        
        # Create multiple concurrent requests
        requests = [
            {
                "method": "test.slow",
                "params": {"id": i},
                "context": {"user_id": "test-user"}
            }
            for i in range(5)
        ]
        
        # Execute concurrently
        tasks = [mock_mcp_server.handle_request(req) for req in requests]
        responses = await asyncio.gather(*tasks)
        
        # Verify all completed
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response["result"]["result"] == f"processed-{i}"