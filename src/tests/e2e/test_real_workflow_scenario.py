"""End-to-end test for a real API workflow scenario."""

import pytest
from httpx import AsyncClient
from unittest.mock import patch, Mock, AsyncMock
import json

from main import app


@pytest.fixture
async def authenticated_client():
    """Create authenticated test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Mock authentication
        with patch("core.security.get_current_active_user") as mock_auth:
            mock_auth.return_value = {
                "id": "test-user",
                "username": "testuser",
                "email": "test@example.com",
                "is_active": True,
                "edition": "community"
            }
            client.headers["Authorization"] = "Bearer test-token"
            yield client


class TestCompleteWorkflowScenario:
    """Test a complete workflow scenario from creation to execution."""
    
    @pytest.mark.asyncio
    async def test_ai_powered_data_processing_workflow(self, authenticated_client):
        """Test creating and executing an AI-powered data processing workflow."""
        
        # Step 1: Discover available adapters
        with patch("services.adapter.AdapterService.discover_adapters") as mock_discover:
            mock_discover.return_value = (
                [
                    {
                        "type": "openai",
                        "name": "OpenAI GPT",
                        "category": "ai",
                        "available": True,
                        "capabilities": ["chat_completion", "embeddings"]
                    },
                    {
                        "type": "email",
                        "name": "Email Adapter",
                        "category": "communication",
                        "available": True,
                        "capabilities": ["send_email"]
                    }
                ],
                2  # total count
            )
            
            response = await authenticated_client.get("/api/v1/adapters/discover?category=ai")
            assert response.status_code == 200
            adapters = response.json()["adapters"]
            assert len(adapters) == 2
        
        # Step 2: Create a workflow template
        workflow_template = {
            "name": "AI Data Analysis",
            "description": "Analyze data with AI and send results via email",
            "is_template": True,
            "definition": {
                "nodes": [
                    {
                        "id": "start",
                        "type": "trigger",
                        "name": "Data Input"
                    },
                    {
                        "id": "analyze",
                        "type": "adapter",
                        "name": "AI Analysis",
                        "data": {
                            "adapter_type": "openai",
                            "capability": "chat_completion"
                        }
                    },
                    {
                        "id": "notify",
                        "type": "adapter",
                        "name": "Email Results",
                        "data": {
                            "adapter_type": "email",
                            "capability": "send_email"
                        }
                    }
                ],
                "edges": [
                    {"from": "start", "to": "analyze"},
                    {"from": "analyze", "to": "notify"}
                ]
            }
        }
        
        with patch("services.workflow.WorkflowService.create_workflow") as mock_create:
            mock_workflow = Mock()
            mock_workflow.id = "template-123"
            mock_workflow.name = workflow_template["name"]
            mock_workflow.is_template = True
            mock_create.return_value = mock_workflow
            
            response = await authenticated_client.post(
                "/api/v1/workflows",
                json=workflow_template
            )
            assert response.status_code == 200
            template_id = response.json()["id"]
        
        # Step 3: Create a workflow instance from template
        workflow_instance = {
            "name": "Customer Data Analysis - Jan 2025",
            "template_id": template_id,
            "parameters": {
                "input_data": "customer_feedback.csv",
                "ai_prompt": "Analyze customer sentiment and identify key themes",
                "email_to": "team@example.com"
            }
        }
        
        with patch("services.workflow.WorkflowService.create_workflow") as mock_create:
            mock_instance = Mock()
            mock_instance.id = "workflow-456"
            mock_instance.name = workflow_instance["name"]
            mock_instance.status = "created"
            mock_create.return_value = mock_instance
            
            response = await authenticated_client.post(
                "/api/v1/workflows",
                json=workflow_instance
            )
            assert response.status_code == 200
            workflow_id = response.json()["id"]
        
        # Step 4: Execute the workflow
        with patch("services.workflow.WorkflowService.execute_workflow") as mock_execute:
            mock_execution = Mock()
            mock_execution.id = "exec-789"
            mock_execution.workflow_id = workflow_id
            mock_execution.status = "running"
            mock_execute.return_value = mock_execution
            
            response = await authenticated_client.post(
                f"/api/v1/workflows/{workflow_id}/execute",
                json={"input_data": {"file": "customer_feedback.csv"}}
            )
            assert response.status_code == 200
            execution_id = response.json()["id"]
        
        # Step 5: Monitor execution progress
        with patch("services.workflow.WorkflowService.get_execution") as mock_get_exec:
            # Simulate progress updates
            mock_get_exec.side_effect = [
                Mock(status="running", progress=30, current_node="analyze"),
                Mock(status="running", progress=80, current_node="notify"),
                Mock(status="completed", progress=100, result={
                    "sentiment_score": 0.78,
                    "key_themes": ["product quality", "customer service", "pricing"],
                    "email_sent": True
                })
            ]
            
            # Check progress
            response = await authenticated_client.get(
                f"/api/v1/workflows/executions/{execution_id}"
            )
            assert response.status_code == 200
            assert response.json()["status"] == "running"
            
            # Final check - completed
            response = await authenticated_client.get(
                f"/api/v1/workflows/executions/{execution_id}"
            )
            assert response.json()["status"] == "completed"
            assert response.json()["result"]["sentiment_score"] == 0.78
        
        # Step 6: Get analytics for the workflow
        with patch("services.analytics.AnalyticsService.get_workflow_analytics") as mock_analytics:
            mock_analytics.return_value = {
                "total_executions": 10,
                "success_rate": 0.9,
                "avg_duration_ms": 3500,
                "adapter_usage": {
                    "openai": 10,
                    "email": 9  # 1 failed email
                }
            }
            
            response = await authenticated_client.get(
                f"/api/v1/analytics/workflows/{workflow_id}"
            )
            assert response.status_code == 200
            analytics = response.json()
            assert analytics["success_rate"] == 0.9
            assert analytics["avg_duration_ms"] == 3500
    
    @pytest.mark.asyncio
    async def test_task_automation_scenario(self, authenticated_client):
        """Test creating and running automated tasks."""
        
        # Step 1: Create a scheduled task
        task_data = {
            "name": "Daily Report Generation",
            "type": "report_generation",
            "schedule": "0 9 * * *",  # Daily at 9 AM
            "config": {
                "report_type": "sales_summary",
                "recipients": ["manager@example.com"],
                "include_charts": True
            }
        }
        
        with patch("services.task.TaskService.create_task") as mock_create_task:
            mock_task = Mock()
            mock_task.id = "task-001"
            mock_task.name = task_data["name"]
            mock_task.status = "scheduled"
            mock_create_task.return_value = mock_task
            
            response = await authenticated_client.post(
                "/api/v1/tasks",
                json=task_data
            )
            assert response.status_code == 200
            task_id = response.json()["id"]
        
        # Step 2: Manually trigger the task
        with patch("services.task.TaskService.execute_task") as mock_execute_task:
            mock_execution = Mock()
            mock_execution.id = "task-exec-001"
            mock_execution.task_id = task_id
            mock_execution.status = "running"
            mock_execute_task.return_value = mock_execution
            
            response = await authenticated_client.post(
                f"/api/v1/tasks/{task_id}/execute"
            )
            assert response.status_code == 200
        
        # Step 3: Check task execution status
        with patch("services.task.TaskService.get_task_execution") as mock_get_exec:
            mock_get_exec.return_value = Mock(
                id="task-exec-001",
                status="completed",
                result={
                    "report_generated": True,
                    "file_path": "/reports/sales_2025_01_26.pdf",
                    "email_sent": True
                },
                duration_ms=2500
            )
            
            response = await authenticated_client.get(
                f"/api/v1/tasks/{task_id}/executions/task-exec-001"
            )
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "completed"
            assert result["result"]["report_generated"] is True
    
    @pytest.mark.asyncio
    async def test_multi_adapter_integration(self, authenticated_client):
        """Test workflow using multiple adapters together."""
        
        # Create a complex workflow using multiple adapters
        complex_workflow = {
            "name": "Customer Support Automation",
            "description": "Automated customer support with AI and multiple channels",
            "definition": {
                "nodes": [
                    {
                        "id": "receive",
                        "type": "trigger",
                        "name": "Receive Customer Query"
                    },
                    {
                        "id": "analyze",
                        "type": "adapter",
                        "name": "AI Analysis",
                        "data": {
                            "adapter_type": "openai",
                            "capability": "chat_completion",
                            "config": {
                                "model": "gpt-4",
                                "system_prompt": "You are a helpful customer support agent."
                            }
                        }
                    },
                    {
                        "id": "route",
                        "type": "condition",
                        "name": "Route Based on Urgency",
                        "data": {
                            "conditions": [
                                {"field": "urgency", "operator": "eq", "value": "high", "output": "slack"},
                                {"field": "urgency", "operator": "eq", "value": "low", "output": "email"}
                            ]
                        }
                    },
                    {
                        "id": "slack",
                        "type": "adapter",
                        "name": "Slack Alert",
                        "data": {
                            "adapter_type": "slack",
                            "capability": "send_message",
                            "config": {"channel": "#urgent-support"}
                        }
                    },
                    {
                        "id": "email",
                        "type": "adapter",
                        "name": "Email Response",
                        "data": {
                            "adapter_type": "email",
                            "capability": "send_email"
                        }
                    }
                ],
                "edges": [
                    {"from": "receive", "to": "analyze"},
                    {"from": "analyze", "to": "route"},
                    {"from": "route", "to": "slack", "condition": "high"},
                    {"from": "route", "to": "email", "condition": "low"}
                ]
            }
        }
        
        with patch("services.workflow.WorkflowService.create_workflow") as mock_create:
            mock_workflow = Mock()
            mock_workflow.id = "complex-workflow-001"
            mock_workflow.name = complex_workflow["name"]
            mock_create.return_value = mock_workflow
            
            response = await authenticated_client.post(
                "/api/v1/workflows",
                json=complex_workflow
            )
            assert response.status_code == 200
            
        # Verify the workflow was created with all adapters
        with patch("services.workflow.WorkflowService.get_workflow") as mock_get:
            mock_get.return_value = Mock(
                id="complex-workflow-001",
                name=complex_workflow["name"],
                node_count=5,
                adapter_types=["openai", "slack", "email"]
            )
            
            response = await authenticated_client.get(
                "/api/v1/workflows/complex-workflow-001"
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["adapter_types"]) == 3