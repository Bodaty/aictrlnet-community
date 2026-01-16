import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
import json

from src.nodes.implementations.mcp_client_node import MCPClientNode
from src.nodes.implementations.mcp_server_node import MCPServerNode
from src.nodes.base import NodeExecutionResult, NodeExecutionStatus
from src.schemas.workflow import WorkflowExecution
from src.mcp_client.models import MCPCapability


@pytest.fixture
def mock_workflow_context():
    """Create mock workflow execution context."""
    return {
        "workflow_id": "test-workflow-123",
        "execution_id": "exec-123",
        "user_id": "test-user",
        "tenant_id": "test-tenant"
    }


@pytest.fixture
def mcp_client_node():
    """Create MCP Client Node instance."""
    return MCPClientNode()


@pytest.fixture
def mcp_server_node():
    """Create MCP Server Node instance."""
    return MCPServerNode()


class TestMCPClientNode:
    """Test MCP Client Node functionality."""
    
    @pytest.mark.asyncio
    async def test_node_initialization(self, mcp_client_node):
        """Test node initialization."""
        assert mcp_client_node.node_type == "mcpClient"
        assert mcp_client_node.name == "MCP Client"
        assert mcp_client_node.category == "mcp"
        assert len(mcp_client_node.inputs) == 2  # data, config
        assert len(mcp_client_node.outputs) == 1  # result
    
    @pytest.mark.asyncio
    async def test_validate_config(self, mcp_client_node):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            "server_url": "http://test-server.com",
            "api_key": "test-key",
            "operation": "message",
            "timeout": 30
        }
        
        errors = mcp_client_node.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config - missing required fields
        invalid_config = {
            "operation": "message"
        }
        
        errors = mcp_client_node.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("server_url" in error for error in errors)
        assert any("api_key" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_execute_message_operation(self, mcp_client_node, mock_workflow_context):
        """Test executing message operation."""
        input_data = {
            "data": {
                "message": "Test message",
                "metadata": {"priority": "high"}
            },
            "config": {
                "server_url": "http://test-server.com",
                "api_key": "test-key",
                "operation": "message",
                "timeout": 30
            }
        }
        
        with patch('src.mcp_client.client.MCPClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.send_request = AsyncMock(
                return_value=MagicMock(
                    result={"status": "sent", "message_id": "msg-123"},
                    metadata={"timestamp": "2025-01-01T00:00:00Z"}
                )
            )
            
            result = await mcp_client_node.execute(input_data, mock_workflow_context)
            
            assert result.status == NodeExecutionStatus.COMPLETED
            assert result.outputs["result"]["status"] == "sent"
            assert result.outputs["result"]["message_id"] == "msg-123"
            assert "mcp_response" in result.metadata
    
    @pytest.mark.asyncio
    async def test_execute_quality_operation(self, mcp_client_node, mock_workflow_context):
        """Test executing quality validation operation."""
        input_data = {
            "data": {
                "dataset": [{"id": 1, "value": "test"}],
                "schema": {"type": "array"}
            },
            "config": {
                "server_url": "http://test-server.com",
                "api_key": "test-key",
                "operation": "quality",
                "quality_threshold": 0.8
            }
        }
        
        with patch('src.mcp_client.client.MCPClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.send_request = AsyncMock(
                return_value=MagicMock(
                    result={
                        "is_valid": True,
                        "quality_score": 0.95,
                        "issues": []
                    },
                    metadata={"processing_time": 0.5}
                )
            )
            
            result = await mcp_client_node.execute(input_data, mock_workflow_context)
            
            assert result.status == NodeExecutionStatus.COMPLETED
            assert result.outputs["result"]["is_valid"] is True
            assert result.outputs["result"]["quality_score"] == 0.95
    
    @pytest.mark.asyncio
    async def test_execute_workflow_operation(self, mcp_client_node, mock_workflow_context):
        """Test executing workflow operation."""
        input_data = {
            "data": {
                "workflow_params": {"input": "test-data"},
                "workflow_config": {"max_retries": 3}
            },
            "config": {
                "server_url": "http://test-server.com",
                "api_key": "test-key",
                "operation": "workflow",
                "target_workflow_id": "remote-wf-123"
            }
        }
        
        with patch('src.mcp_client.client.MCPClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.send_request = AsyncMock(
                return_value=MagicMock(
                    result={
                        "execution_id": "remote-exec-123",
                        "status": "completed",
                        "output": {"result": "success"}
                    },
                    metadata={"duration": 2.5}
                )
            )
            
            result = await mcp_client_node.execute(input_data, mock_workflow_context)
            
            assert result.status == NodeExecutionStatus.COMPLETED
            assert result.outputs["result"]["execution_id"] == "remote-exec-123"
            assert result.outputs["result"]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, mcp_client_node, mock_workflow_context):
        """Test execution with error handling."""
        input_data = {
            "data": {"message": "test"},
            "config": {
                "server_url": "http://test-server.com",
                "api_key": "test-key",
                "operation": "message"
            }
        }
        
        with patch('src.mcp_client.client.MCPClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.send_request = AsyncMock(
                side_effect=Exception("Connection failed")
            )
            
            result = await mcp_client_node.execute(input_data, mock_workflow_context)
            
            assert result.status == NodeExecutionStatus.FAILED
            assert "Connection failed" in result.error


class TestMCPServerNode:
    """Test MCP Server Node functionality."""
    
    @pytest.mark.asyncio
    async def test_node_initialization(self, mcp_server_node):
        """Test node initialization."""
        assert mcp_server_node.node_type == "mcpServer"
        assert mcp_server_node.name == "MCP Server"
        assert mcp_server_node.category == "mcp"
        assert len(mcp_server_node.inputs) == 2  # data, config
        assert len(mcp_server_node.outputs) == 1  # endpoint_info
    
    @pytest.mark.asyncio
    async def test_validate_config(self, mcp_server_node):
        """Test configuration validation."""
        # Valid config
        valid_config = {
            "endpoint_path": "/api/test",
            "method_name": "test.method",
            "description": "Test endpoint",
            "mode": "single"
        }
        
        errors = mcp_server_node.validate_config(valid_config)
        assert len(errors) == 0
        
        # Invalid config - invalid mode
        invalid_config = {
            "endpoint_path": "/api/test",
            "method_name": "test.method",
            "mode": "invalid-mode"
        }
        
        errors = mcp_server_node.validate_config(invalid_config)
        assert len(errors) > 0
        assert any("mode" in error for error in errors)
    
    @pytest.mark.asyncio
    async def test_execute_single_mode(self, mcp_server_node, mock_workflow_context):
        """Test executing in single mode."""
        input_data = {
            "data": {
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"}
            },
            "config": {
                "endpoint_path": "/api/test",
                "method_name": "test.method",
                "description": "Test endpoint",
                "mode": "single"
            }
        }
        
        with patch('src.mcp_server.services.workflow_exposure.WorkflowExposureService') as MockService:
            mock_service = MockService.return_value
            mock_service.register_workflow_endpoint = AsyncMock(
                return_value={
                    "endpoint_id": "ep-123",
                    "url": "http://localhost:8003/api/test",
                    "active": True
                }
            )
            
            # Mock cache service
            with patch('src.core.cache.get_cache') as mock_get_cache:
                mock_cache = AsyncMock()
                mock_cache._redis_client = MagicMock()
                mock_get_cache.return_value = mock_cache
                
                result = await mcp_server_node.execute(input_data, mock_workflow_context)
                
                assert result.status == NodeExecutionStatus.COMPLETED
                assert result.outputs["endpoint_info"]["endpoint_id"] == "ep-123"
                assert result.outputs["endpoint_info"]["url"] == "http://localhost:8003/api/test"
                assert result.outputs["endpoint_info"]["mode"] == "single"
    
    @pytest.mark.asyncio
    async def test_execute_continuous_mode(self, mcp_server_node, mock_workflow_context):
        """Test executing in continuous mode."""
        input_data = {
            "data": {
                "input_schema": {"type": "object"},
                "output_schema": {"type": "object"},
                "polling_config": {
                    "interval": 60,
                    "max_concurrent": 5
                }
            },
            "config": {
                "endpoint_path": "/api/continuous",
                "method_name": "continuous.process",
                "description": "Continuous endpoint",
                "mode": "continuous"
            }
        }
        
        with patch('src.mcp_server.services.workflow_exposure.WorkflowExposureService') as MockService:
            mock_service = MockService.return_value
            mock_service.register_workflow_endpoint = AsyncMock(
                return_value={
                    "endpoint_id": "ep-continuous",
                    "url": "http://localhost:8003/api/continuous",
                    "active": True
                }
            )
            
            # Mock cache service
            with patch('src.core.cache.get_cache') as mock_get_cache:
                mock_cache = AsyncMock()
                mock_cache._redis_client = MagicMock()
                mock_get_cache.return_value = mock_cache
                
                result = await mcp_server_node.execute(input_data, mock_workflow_context)
                
                assert result.status == NodeExecutionStatus.COMPLETED
                assert result.outputs["endpoint_info"]["mode"] == "continuous"
                assert "polling_interval" in result.outputs["endpoint_info"]
    
    @pytest.mark.asyncio
    async def test_execute_webhook_mode(self, mcp_server_node, mock_workflow_context):
        """Test executing in webhook mode."""
        input_data = {
            "data": {
                "webhook_config": {
                    "callback_url": "http://callback.com/webhook",
                    "retry_policy": {"max_retries": 3}
                }
            },
            "config": {
                "endpoint_path": "/api/webhook",
                "method_name": "webhook.process",
                "description": "Webhook endpoint",
                "mode": "webhook"
            }
        }
        
        with patch('src.mcp_server.services.workflow_exposure.WorkflowExposureService') as MockService:
            mock_service = MockService.return_value
            mock_service.register_workflow_endpoint = AsyncMock(
                return_value={
                    "endpoint_id": "ep-webhook",
                    "url": "http://localhost:8003/api/webhook",
                    "active": True
                }
            )
            
            # Mock cache service
            with patch('src.core.cache.get_cache') as mock_get_cache:
                mock_cache = AsyncMock()
                mock_cache._redis_client = MagicMock()
                mock_get_cache.return_value = mock_cache
                
                result = await mcp_server_node.execute(input_data, mock_workflow_context)
                
                assert result.status == NodeExecutionStatus.COMPLETED
                assert result.outputs["endpoint_info"]["mode"] == "webhook"
                assert result.outputs["endpoint_info"]["callback_url"] == "http://callback.com/webhook"


class TestMCPNodesIntegration:
    """Test integration between MCP nodes."""
    
    @pytest.mark.asyncio
    async def test_client_server_integration(self, mcp_client_node, mcp_server_node, mock_workflow_context):
        """Test client and server nodes working together."""
        # First, create an endpoint with server node
        server_input = {
            "data": {
                "input_schema": {"type": "object", "properties": {"message": {"type": "string"}}},
                "output_schema": {"type": "object", "properties": {"result": {"type": "string"}}}
            },
            "config": {
                "endpoint_path": "/api/integration/test",
                "method_name": "integration.test",
                "description": "Integration test endpoint",
                "mode": "single"
            }
        }
        
        with patch('src.mcp_server.services.workflow_exposure.WorkflowExposureService') as MockService:
            mock_service = MockService.return_value
            mock_service.register_workflow_endpoint = AsyncMock(
                return_value={
                    "endpoint_id": "ep-integration",
                    "url": "http://localhost:8003/api/integration/test",
                    "active": True
                }
            )
            
            # Mock cache service
            with patch('src.core.cache.get_cache') as mock_get_cache:
                mock_cache = AsyncMock()
                mock_cache._redis_client = MagicMock()
                mock_get_cache.return_value = mock_cache
                
                server_result = await mcp_server_node.execute(server_input, mock_workflow_context)
                
                assert server_result.status == NodeExecutionStatus.COMPLETED
                endpoint_url = server_result.outputs["endpoint_info"]["url"]
        
        # Now, use client node to call the endpoint
        client_input = {
            "data": {
                "message": "Hello from client"
            },
            "config": {
                "server_url": endpoint_url,
                "api_key": "test-key",
                "operation": "custom",
                "custom_method": "integration.test"
            }
        }
        
        with patch('src.mcp_client.client.MCPClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.send_request = AsyncMock(
                return_value=MagicMock(
                    result={"result": "Integration successful"},
                    metadata={"endpoint_id": "ep-integration"}
                )
            )
            
            client_result = await mcp_client_node.execute(client_input, mock_workflow_context)
            
            assert client_result.status == NodeExecutionStatus.COMPLETED
            assert client_result.outputs["result"]["result"] == "Integration successful"