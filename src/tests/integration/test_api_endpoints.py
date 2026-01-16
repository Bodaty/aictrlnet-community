"""Integration tests for actual API endpoints."""

import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from unittest.mock import Mock, patch, AsyncMock

from main import app  # Import the actual FastAPI app


@pytest.fixture
async def client():
    """Create test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def mock_auth():
    """Mock authentication for tests."""
    mock_user = {
        "id": "test-user-id",
        "username": "testuser",
        "email": "test@example.com",
        "is_active": True,
        "edition": "community"
    }
    
    with patch("core.security.get_current_active_user") as mock:
        mock.return_value = mock_user
        yield mock


@pytest.fixture
def mock_db():
    """Mock database session."""
    with patch("core.database.get_db") as mock:
        mock_session = AsyncMock(spec=AsyncSession)
        mock.return_value = mock_session
        yield mock_session


class TestAdapterEndpoints:
    """Test adapter-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_discover_adapters(self, client: AsyncClient, mock_auth, mock_db):
        """Test the /api/v1/adapters/discover endpoint."""
        # Mock the database query results
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        mock_db.execute.return_value.scalar.return_value = 0
        
        # Add auth header
        headers = {"Authorization": "Bearer test-token"}
        
        response = await client.get("/api/v1/adapters/discover", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "adapters" in data
        assert "total" in data
        assert "edition" in data
        assert "categories" in data
    
    @pytest.mark.asyncio
    async def test_discover_adapters_with_category(self, client: AsyncClient, mock_auth, mock_db):
        """Test adapter discovery with category filter."""
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        mock_db.execute.return_value.scalar.return_value = 0
        
        headers = {"Authorization": "Bearer test-token"}
        
        response = await client.get(
            "/api/v1/adapters/discover?category=ai", 
            headers=headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data["adapters"], list)
    
    @pytest.mark.asyncio
    async def test_check_adapter_availability(self, client: AsyncClient, mock_auth, mock_db):
        """Test the /api/v1/adapters/availability endpoint."""
        headers = {"Authorization": "Bearer test-token"}
        
        # Mock adapter service
        with patch("services.adapter.AdapterService.check_availability") as mock_check:
            mock_check.return_value = {
                "openai": {"available": True, "edition": "community"},
                "claude": {"available": True, "edition": "community"}
            }
            
            response = await client.post(
                "/api/v1/adapters/availability",
                json={"adapter_types": ["openai", "claude"]},
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["results"]["openai"]["available"] is True
            assert data["results"]["claude"]["available"] is True


class TestWorkflowEndpoints:
    """Test workflow-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_workflow(self, client: AsyncClient, mock_auth, mock_db):
        """Test creating a workflow."""
        headers = {"Authorization": "Bearer test-token"}
        
        workflow_data = {
            "name": "Test Workflow",
            "description": "Test workflow description",
            "definition": {
                "nodes": [
                    {
                        "id": "node1",
                        "type": "process",
                        "name": "Start Node"
                    }
                ],
                "edges": []
            }
        }
        
        # Mock the workflow service
        with patch("services.workflow.WorkflowService.create_workflow") as mock_create:
            mock_workflow = Mock()
            mock_workflow.id = "test-workflow-id"
            mock_workflow.name = "Test Workflow"
            mock_create.return_value = mock_workflow
            
            response = await client.post(
                "/api/v1/workflows",
                json=workflow_data,
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-workflow-id"
            assert data["name"] == "Test Workflow"
    
    @pytest.mark.asyncio
    async def test_list_workflows(self, client: AsyncClient, mock_auth, mock_db):
        """Test listing workflows."""
        headers = {"Authorization": "Bearer test-token"}
        
        # Mock query results
        mock_db.execute.return_value.scalars.return_value.all.return_value = []
        mock_db.execute.return_value.scalar.return_value = 0
        
        response = await client.get("/api/v1/workflows", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data
        assert "total" in data
        assert isinstance(data["workflows"], list)


class TestTaskEndpoints:
    """Test task-related endpoints."""
    
    @pytest.mark.asyncio
    async def test_create_task(self, client: AsyncClient, mock_auth, mock_db):
        """Test creating a task."""
        headers = {"Authorization": "Bearer test-token"}
        
        task_data = {
            "name": "Test Task",
            "description": "Test task description",
            "type": "data_processing",
            "config": {"key": "value"}
        }
        
        with patch("services.task.TaskService.create_task") as mock_create:
            mock_task = Mock()
            mock_task.id = "test-task-id"
            mock_task.name = "Test Task"
            mock_task.status = "pending"
            mock_create.return_value = mock_task
            
            response = await client.post(
                "/api/v1/tasks",
                json=task_data,
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-task-id"
            assert data["name"] == "Test Task"
            assert data["status"] == "pending"
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, client: AsyncClient, mock_auth, mock_db):
        """Test getting task status."""
        headers = {"Authorization": "Bearer test-token"}
        task_id = "test-task-id"
        
        with patch("services.task.TaskService.get_task") as mock_get:
            mock_task = Mock()
            mock_task.id = task_id
            mock_task.status = "running"
            mock_task.progress = 50
            mock_get.return_value = mock_task
            
            response = await client.get(
                f"/api/v1/tasks/{task_id}",
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == task_id
            assert data["status"] == "running"
            assert data["progress"] == 50


class TestAnalyticsEndpoints:
    """Test analytics endpoints."""
    
    @pytest.mark.asyncio
    async def test_get_metrics(self, client: AsyncClient, mock_auth, mock_db):
        """Test getting metrics."""
        headers = {"Authorization": "Bearer test-token"}
        
        with patch("services.analytics.AnalyticsService.get_metrics") as mock_metrics:
            mock_metrics.return_value = {
                "total_tasks": 100,
                "completed_tasks": 85,
                "success_rate": 0.85
            }
            
            response = await client.get(
                "/api/v1/analytics/metrics?start_date=2025-01-01",
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_tasks"] == 100
            assert data["success_rate"] == 0.85
    
    @pytest.mark.asyncio
    async def test_get_performance_data(self, client: AsyncClient, mock_auth, mock_db):
        """Test getting performance data."""
        headers = {"Authorization": "Bearer test-token"}
        
        with patch("services.analytics.AnalyticsService.get_performance_data") as mock_perf:
            mock_perf.return_value = {
                "avg_response_time": 250,
                "p95_response_time": 500,
                "error_rate": 0.02
            }
            
            response = await client.get(
                "/api/v1/analytics/performance",
                headers=headers
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["avg_response_time"] == 250
            assert data["error_rate"] == 0.02


class TestAuthEndpoints:
    """Test authentication endpoints."""
    
    @pytest.mark.asyncio
    async def test_login(self, client: AsyncClient, mock_db):
        """Test login endpoint."""
        with patch("services.auth.AuthService.authenticate_user") as mock_auth:
            mock_auth.return_value = {
                "access_token": "test-jwt-token",
                "token_type": "bearer",
                "user": {
                    "username": "testuser",
                    "email": "test@example.com"
                }
            }
            
            response = await client.post(
                "/api/v1/auth/login",
                data={
                    "username": "testuser",
                    "password": "testpass"
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["access_token"] == "test-jwt-token"
            assert data["token_type"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_protected_endpoint_without_auth(self, client: AsyncClient):
        """Test accessing protected endpoint without authentication."""
        response = await client.get("/api/v1/workflows")
        
        # Should return 401 Unauthorized
        assert response.status_code == 401