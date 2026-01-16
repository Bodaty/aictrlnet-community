"""Integration tests for workflows with adapters."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from workflows.models import (
    WorkflowDefinition, WorkflowInstance, WorkflowStatus,
    NodeDefinition, NodeType
)
from workflows.executor import WorkflowExecutor
from nodes.registry import node_registry
from adapters.registry import adapter_registry
from adapters.factory import AdapterFactory
from adapters.models import AdapterConfig, AdapterCategory
from events.event_bus import event_bus


@pytest.fixture
async def setup_integration_env():
    """Set up integration test environment."""
    # Initialize event bus
    await event_bus.start()
    
    # Initialize registries
    from control_plane.registry import component_registry
    await component_registry.initialize()
    
    # Register test adapter
    test_adapter_config = AdapterConfig(
        adapter_id="test-adapter",
        adapter_type="mock",
        name="Test Adapter",
        category=AdapterCategory.GENERAL,
        credentials={},
        settings={}
    )
    
    # Create workflow executor
    executor = WorkflowExecutor()
    await executor.initialize()
    
    yield {
        "executor": executor,
        "event_bus": event_bus,
        "adapter_config": test_adapter_config
    }
    
    # Cleanup
    await executor.shutdown()
    await event_bus.stop()


@pytest.fixture
def sample_workflow():
    """Create a sample workflow definition."""
    return WorkflowDefinition(
        id="test-workflow",
        name="Test Workflow",
        description="Integration test workflow",
        nodes=[
            NodeDefinition(
                id="start",
                type=NodeType.PROCESS,
                name="Start Node",
                config={
                    "action": "initialize",
                    "parameters": {"message": "Starting workflow"}
                },
                outputs={"next": "api-call"}
            ),
            NodeDefinition(
                id="api-call",
                type=NodeType.API_CALL,
                name="API Call Node",
                config={
                    "adapter_type": "mock",
                    "capability": "test_action",
                    "parameters": {"input": "test data"}
                },
                outputs={"success": "process-data", "error": "error-handler"}
            ),
            NodeDefinition(
                id="process-data",
                type=NodeType.PROCESS,
                name="Process Data",
                config={
                    "action": "transform",
                    "parameters": {"format": "json"}
                },
                outputs={"next": "end"}
            ),
            NodeDefinition(
                id="error-handler",
                type=NodeType.PROCESS,
                name="Error Handler",
                config={
                    "action": "log_error",
                    "parameters": {"severity": "high"}
                },
                outputs={"next": "end"}
            ),
            NodeDefinition(
                id="end",
                type=NodeType.PROCESS,
                name="End Node",
                config={
                    "action": "finalize",
                    "parameters": {}
                }
            )
        ],
        edges=[
            {"source": "start", "target": "api-call", "condition": "next"},
            {"source": "api-call", "target": "process-data", "condition": "success"},
            {"source": "api-call", "target": "error-handler", "condition": "error"},
            {"source": "process-data", "target": "end", "condition": "next"},
            {"source": "error-handler", "target": "end", "condition": "next"}
        ],
        start_node_id="start"
    )


class TestWorkflowAdapterIntegration:
    """Test workflow execution with adapter integration."""
    
    @pytest.mark.asyncio
    async def test_workflow_with_successful_adapter_call(self, setup_integration_env, sample_workflow):
        """Test workflow execution with successful adapter call."""
        env = setup_integration_env
        executor = env["executor"]
        
        # Mock adapter execute method
        with patch('adapters.base_adapter.BaseAdapter.execute') as mock_execute:
            mock_execute.return_value = AsyncMock(
                success=True,
                data={"result": "test success"},
                metadata={"execution_time": 0.1}
            )
            
            # Execute workflow
            instance = await executor.execute_workflow(
                sample_workflow,
                {"initial_data": "test"}
            )
            
            # Verify workflow completed successfully
            assert instance.status == WorkflowStatus.COMPLETED
            assert len(instance.node_executions) == 5  # All nodes executed
            
            # Verify adapter was called
            mock_execute.assert_called_once()
            
            # Verify process data node received adapter output
            process_node = next(
                n for n in instance.node_executions 
                if n.node_id == "process-data"
            )
            assert process_node.status == "completed"
    
    @pytest.mark.asyncio
    async def test_workflow_with_failed_adapter_call(self, setup_integration_env, sample_workflow):
        """Test workflow execution with failed adapter call."""
        env = setup_integration_env
        executor = env["executor"]
        
        # Mock adapter execute to fail
        with patch('adapters.base_adapter.BaseAdapter.execute') as mock_execute:
            mock_execute.side_effect = Exception("Adapter error")
            
            # Execute workflow
            instance = await executor.execute_workflow(
                sample_workflow,
                {"initial_data": "test"}
            )
            
            # Verify workflow still completed (error was handled)
            assert instance.status == WorkflowStatus.COMPLETED
            
            # Verify error handler was executed
            error_node = next(
                n for n in instance.node_executions 
                if n.node_id == "error-handler"
            )
            assert error_node.status == "completed"
            
            # Verify process-data was not executed
            process_nodes = [
                n for n in instance.node_executions 
                if n.node_id == "process-data"
            ]
            assert len(process_nodes) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_multiple_adapters(self, setup_integration_env):
        """Test workflow with multiple adapter calls."""
        env = setup_integration_env
        executor = env["executor"]
        
        # Create workflow with multiple adapter calls
        multi_adapter_workflow = WorkflowDefinition(
            id="multi-adapter-workflow",
            name="Multi Adapter Workflow",
            description="Workflow with multiple adapters",
            nodes=[
                NodeDefinition(
                    id="start",
                    type=NodeType.PROCESS,
                    name="Start",
                    config={"action": "initialize"},
                    outputs={"next": "parallel"}
                ),
                NodeDefinition(
                    id="parallel",
                    type=NodeType.PARALLEL,
                    name="Parallel Adapter Calls",
                    config={
                        "branches": [
                            {
                                "id": "adapter1",
                                "type": NodeType.API_CALL,
                                "config": {
                                    "adapter_type": "mock",
                                    "capability": "action1"
                                }
                            },
                            {
                                "id": "adapter2",
                                "type": NodeType.API_CALL,
                                "config": {
                                    "adapter_type": "mock",
                                    "capability": "action2"
                                }
                            }
                        ]
                    },
                    outputs={"complete": "aggregate"}
                ),
                NodeDefinition(
                    id="aggregate",
                    type=NodeType.PROCESS,
                    name="Aggregate Results",
                    config={
                        "action": "merge_results"
                    },
                    outputs={"next": "end"}
                ),
                NodeDefinition(
                    id="end",
                    type=NodeType.PROCESS,
                    name="End",
                    config={"action": "finalize"}
                )
            ],
            edges=[
                {"source": "start", "target": "parallel"},
                {"source": "parallel", "target": "aggregate"},
                {"source": "aggregate", "target": "end"}
            ],
            start_node_id="start"
        )
        
        # Mock adapter responses
        adapter_responses = [
            AsyncMock(success=True, data={"result": "adapter1 result"}),
            AsyncMock(success=True, data={"result": "adapter2 result"})
        ]
        
        with patch('adapters.base_adapter.BaseAdapter.execute') as mock_execute:
            mock_execute.side_effect = adapter_responses
            
            # Execute workflow
            instance = await executor.execute_workflow(
                multi_adapter_workflow,
                {"test": "data"}
            )
            
            # Verify both adapters were called
            assert mock_execute.call_count == 2
            
            # Verify workflow completed
            assert instance.status == WorkflowStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_workflow_event_emissions(self, setup_integration_env, sample_workflow):
        """Test that workflow execution emits proper events."""
        env = setup_integration_env
        executor = env["executor"]
        event_bus = env["event_bus"]
        
        # Track emitted events
        emitted_events = []
        
        async def event_handler(event):
            emitted_events.append(event)
        
        # Subscribe to workflow events
        await event_bus.subscribe("workflow.*", event_handler)
        await event_bus.subscribe("node.*", event_handler)
        
        # Execute workflow
        with patch('adapters.base_adapter.BaseAdapter.execute') as mock_execute:
            mock_execute.return_value = AsyncMock(
                success=True,
                data={"result": "success"}
            )
            
            instance = await executor.execute_workflow(
                sample_workflow,
                {"test": "data"}
            )
        
        # Verify events were emitted
        event_types = [e.get("type") for e in emitted_events]
        
        assert "workflow.started" in event_types
        assert "workflow.completed" in event_types
        assert any("node.started" in t for t in event_types)
        assert any("node.completed" in t for t in event_types)
    
    @pytest.mark.asyncio
    async def test_workflow_with_conditional_branching(self, setup_integration_env):
        """Test workflow with conditional branching based on adapter response."""
        env = setup_integration_env
        executor = env["executor"]
        
        # Create conditional workflow
        conditional_workflow = WorkflowDefinition(
            id="conditional-workflow",
            name="Conditional Workflow",
            description="Workflow with conditions",
            nodes=[
                NodeDefinition(
                    id="start",
                    type=NodeType.PROCESS,
                    name="Start",
                    config={"action": "initialize"},
                    outputs={"next": "check-condition"}
                ),
                NodeDefinition(
                    id="check-condition",
                    type=NodeType.API_CALL,
                    name="Check Condition",
                    config={
                        "adapter_type": "mock",
                        "capability": "check_status"
                    },
                    outputs={
                        "active": "process-active",
                        "inactive": "process-inactive"
                    }
                ),
                NodeDefinition(
                    id="process-active",
                    type=NodeType.PROCESS,
                    name="Process Active",
                    config={"action": "handle_active"},
                    outputs={"next": "end"}
                ),
                NodeDefinition(
                    id="process-inactive",
                    type=NodeType.PROCESS,
                    name="Process Inactive",
                    config={"action": "handle_inactive"},
                    outputs={"next": "end"}
                ),
                NodeDefinition(
                    id="end",
                    type=NodeType.PROCESS,
                    name="End",
                    config={"action": "finalize"}
                )
            ],
            edges=[
                {"source": "start", "target": "check-condition"},
                {"source": "check-condition", "target": "process-active", "condition": "active"},
                {"source": "check-condition", "target": "process-inactive", "condition": "inactive"},
                {"source": "process-active", "target": "end"},
                {"source": "process-inactive", "target": "end"}
            ],
            start_node_id="start"
        )
        
        # Test active branch
        with patch('adapters.base_adapter.BaseAdapter.execute') as mock_execute:
            mock_execute.return_value = AsyncMock(
                success=True,
                data={"status": "active"},
                metadata={"branch": "active"}
            )
            
            instance = await executor.execute_workflow(
                conditional_workflow,
                {"test": "data"}
            )
            
            # Verify active branch was taken
            active_nodes = [
                n for n in instance.node_executions 
                if n.node_id == "process-active"
            ]
            assert len(active_nodes) == 1
            
            inactive_nodes = [
                n for n in instance.node_executions 
                if n.node_id == "process-inactive"
            ]
            assert len(inactive_nodes) == 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_loop(self, setup_integration_env):
        """Test workflow with loop node processing multiple items."""
        env = setup_integration_env
        executor = env["executor"]
        
        # Create workflow with loop
        loop_workflow = WorkflowDefinition(
            id="loop-workflow",
            name="Loop Workflow",
            description="Workflow with loop",
            nodes=[
                NodeDefinition(
                    id="start",
                    type=NodeType.PROCESS,
                    name="Start",
                    config={
                        "action": "initialize",
                        "parameters": {
                            "items": ["item1", "item2", "item3"]
                        }
                    },
                    outputs={"next": "loop"}
                ),
                NodeDefinition(
                    id="loop",
                    type=NodeType.LOOP,
                    name="Process Items",
                    config={
                        "items_path": "$.items",
                        "process_node": {
                            "type": NodeType.API_CALL,
                            "config": {
                                "adapter_type": "mock",
                                "capability": "process_item"
                            }
                        }
                    },
                    outputs={"complete": "aggregate"}
                ),
                NodeDefinition(
                    id="aggregate",
                    type=NodeType.PROCESS,
                    name="Aggregate",
                    config={"action": "combine_results"},
                    outputs={"next": "end"}
                ),
                NodeDefinition(
                    id="end",
                    type=NodeType.PROCESS,
                    name="End",
                    config={"action": "finalize"}
                )
            ],
            edges=[
                {"source": "start", "target": "loop"},
                {"source": "loop", "target": "aggregate"},
                {"source": "aggregate", "target": "end"}
            ],
            start_node_id="start"
        )
        
        # Mock adapter to process each item
        call_count = 0
        
        async def mock_process_item(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return AsyncMock(
                success=True,
                data={"processed": f"item{call_count}"}
            )
        
        with patch('adapters.base_adapter.BaseAdapter.execute', side_effect=mock_process_item):
            instance = await executor.execute_workflow(
                loop_workflow,
                {"items": ["item1", "item2", "item3"]}
            )
            
            # Verify adapter was called for each item
            assert call_count == 3
            
            # Verify workflow completed
            assert instance.status == WorkflowStatus.COMPLETED