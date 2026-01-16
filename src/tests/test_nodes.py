"""Tests for node system."""

import pytest
import asyncio
from datetime import datetime

from nodes.models import (
    NodeConfig, NodeType, NodeStatus, NodeInstance,
    WorkflowTemplate, WorkflowInstance, NodeConnection
)
from nodes.base_node import BaseNode
from nodes.registry import node_registry
from nodes.executor import NodeExecutor
from nodes.implementations import TaskNode, DecisionNode, TransformNode


@pytest.fixture
def sample_workflow_template():
    """Create a sample workflow template."""
    nodes = [
        NodeConfig(
            id="start",
            name="Start Node",
            type=NodeType.START,
            outputs=["data"]
        ),
        NodeConfig(
            id="validate",
            name="Validate Data",
            type=NodeType.TASK,
            parameters={
                "task_type": "validate",
                "validations": {
                    "amount": {"required": True, "type": "number", "min": 0},
                    "category": {"required": True, "type": "string"}
                }
            },
            inputs=["data"],
            outputs=["validated_data"]
        ),
        NodeConfig(
            id="decision",
            name="Check Amount",
            type=NodeType.DECISION,
            parameters={
                "decision_type": "condition",
                "conditions": [
                    {
                        "expression": "amount > 100",
                        "branch_id": "high_value"
                    }
                ],
                "default_branch": "low_value",
                "branch_mapping": {
                    "high_value": ["transform_high"],
                    "low_value": ["transform_low"]
                }
            },
            inputs=["validated_data"]
        ),
        NodeConfig(
            id="transform_high",
            name="Transform High Value",
            type=NodeType.TRANSFORM,
            parameters={
                "transform_type": "mapping",
                "mapping": {
                    "amount": "amount",
                    "category": "category",
                    "priority": {"source": "amount", "transform": "stringify", "default": "high"}
                }
            },
            inputs=["validated_data"],
            outputs=["transformed_data"]
        ),
        NodeConfig(
            id="transform_low",
            name="Transform Low Value",
            type=NodeType.TRANSFORM,
            parameters={
                "transform_type": "mapping",
                "mapping": {
                    "amount": "amount",
                    "category": "category",
                    "priority": {"source": "amount", "transform": "stringify", "default": "low"}
                }
            },
            inputs=["validated_data"],
            outputs=["transformed_data"]
        ),
        NodeConfig(
            id="end",
            name="End Node",
            type=NodeType.END,
            inputs=["transformed_data"]
        )
    ]
    
    connections = [
        NodeConnection(from_node_id="start", to_node_id="validate"),
        NodeConnection(from_node_id="validate", to_node_id="decision"),
        NodeConnection(from_node_id="decision", to_node_id="transform_high", condition="high_value"),
        NodeConnection(from_node_id="decision", to_node_id="transform_low", condition="low_value"),
        NodeConnection(from_node_id="transform_high", to_node_id="end"),
        NodeConnection(from_node_id="transform_low", to_node_id="end")
    ]
    
    return WorkflowTemplate(
        name="Test Workflow",
        description="Sample workflow for testing",
        nodes=nodes,
        connections=connections
    )


@pytest.fixture
def workflow_instance(sample_workflow_template):
    """Create a workflow instance from template."""
    instance = WorkflowInstance(
        template_id=sample_workflow_template.id,
        name="Test Instance",
        input_data={"amount": 150, "category": "electronics"}
    )
    
    # Create node instances
    for node_config in sample_workflow_template.nodes:
        node_instance = NodeInstance(
            node_config=node_config,
            workflow_instance_id=instance.id
        )
        instance.node_instances[node_config.id] = node_instance
    
    # Set up connections
    for conn in sample_workflow_template.connections:
        from_instance = instance.node_instances[conn.from_node_id]
        to_instance = instance.node_instances[conn.to_node_id]
        
        from_instance.next_nodes.append(conn.to_node_id)
        to_instance.previous_nodes.append(conn.from_node_id)
    
    return instance


@pytest.mark.asyncio
async def test_task_node_execution():
    """Test task node execution."""
    config = NodeConfig(
        name="Test Task",
        type=NodeType.TASK,
        parameters={
            "task_type": "process",
            "transformations": {
                "message": "uppercase"
            }
        }
    )
    
    node = TaskNode(config)
    
    # Execute node
    input_data = {"message": "hello world", "value": 42}
    output = await node.execute(input_data, {})
    
    # Verify transformation
    assert output["message"] == "HELLO WORLD"
    assert output["value"] == 42


@pytest.mark.asyncio
async def test_decision_node_execution():
    """Test decision node execution."""
    config = NodeConfig(
        name="Test Decision",
        type=NodeType.DECISION,
        parameters={
            "decision_type": "condition",
            "conditions": [
                {
                    "expression": "score > 80",
                    "branch_id": "high_score"
                },
                {
                    "expression": "score > 60",
                    "branch_id": "medium_score"
                }
            ],
            "default_branch": "low_score"
        }
    )
    
    node = DecisionNode(config)
    
    # Test high score
    output = await node.execute({"score": 90}, {})
    assert output["selected_branch"] == "high_score"
    
    # Test medium score
    output = await node.execute({"score": 70}, {})
    assert output["selected_branch"] == "medium_score"
    
    # Test low score
    output = await node.execute({"score": 50}, {})
    assert output["selected_branch"] == "low_score"


@pytest.mark.asyncio
async def test_transform_node_execution():
    """Test transform node execution."""
    config = NodeConfig(
        name="Test Transform",
        type=NodeType.TRANSFORM,
        parameters={
            "transform_type": "mapping",
            "mapping": {
                "full_name": {"source": "first_name", "transform": "uppercase"},
                "age_str": {"source": "age", "transform": "stringify"},
                "status": {"source": "active", "default": "unknown"}
            }
        }
    )
    
    node = TransformNode(config)
    
    # Execute transform
    input_data = {"first_name": "john", "age": 30}
    output = await node.execute(input_data, {})
    
    # Verify mapping
    assert output["full_name"] == "JOHN"
    assert output["age_str"] == "30"
    assert output["status"] == "unknown"


@pytest.mark.asyncio
async def test_node_registry():
    """Test node registry functionality."""
    # Get available node types
    types = node_registry.get_available_node_types()
    assert NodeType.START.value in types
    assert NodeType.TASK.value in types
    assert NodeType.DECISION.value in types
    
    # Create node from config
    config = NodeConfig(
        name="Registry Test",
        type=NodeType.TASK
    )
    
    node = node_registry.create_node(config)
    assert isinstance(node, TaskNode)
    
    # Get node info
    info = node_registry.get_node_info(NodeType.TASK.value)
    assert info["type"] == NodeType.TASK.value
    assert info["class"] == "TaskNode"


@pytest.mark.asyncio
async def test_workflow_execution(workflow_instance):
    """Test complete workflow execution."""
    executor = NodeExecutor(max_parallel_nodes=5)
    
    # Execute workflow
    result = await executor.execute_workflow(workflow_instance)
    
    # Verify workflow completed
    assert result.status == "completed"
    assert result.completed_at is not None
    assert result.duration_ms > 0
    
    # Verify output (should have gone through high value branch)
    assert "priority" in result.output_data
    assert result.output_data["amount"] == 150
    assert result.output_data["category"] == "electronics"
    
    # Verify node execution
    validate_node = result.node_instances["validate"]
    assert validate_node.status == NodeStatus.COMPLETED
    assert validate_node.output_data["valid"] is True
    
    decision_node = result.node_instances["decision"]
    assert decision_node.status == NodeStatus.COMPLETED
    assert decision_node.output_data["selected_branch"] == "high_value"
    
    # High value transform should be executed
    high_transform = result.node_instances["transform_high"]
    assert high_transform.status == NodeStatus.COMPLETED
    
    # Low value transform should be skipped
    low_transform = result.node_instances["transform_low"]
    assert low_transform.status == NodeStatus.PENDING  # Not executed


@pytest.mark.asyncio
async def test_node_error_handling():
    """Test node error handling and retry."""
    config = NodeConfig(
        name="Error Test",
        type=NodeType.TASK,
        parameters={"task_type": "calculate", "expression": "1/0"},  # Will cause error
        retry_count=2,
        retry_delay_seconds=1  # Must be int, not float
    )
    
    instance = NodeInstance(
        node_config=config,
        workflow_instance_id="test-workflow"
    )
    
    node = TaskNode(config)
    
    # Execute node (should fail)
    result = await node.run(instance, {})
    
    # Verify error handling
    assert result.status == NodeStatus.PENDING  # Will be retried
    assert instance.attempt_number == 1
    assert instance.last_error is not None
    
    # Execute again (second attempt)
    result = await node.run(instance, {})
    
    # Still fails but now exhausted retries
    assert result.status == NodeStatus.FAILED
    assert instance.attempt_number == 2