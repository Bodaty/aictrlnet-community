"""End-to-end test for AI-powered workflow scenario."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json

from workflows.models import WorkflowDefinition, NodeDefinition, NodeType
from workflows.executor import WorkflowExecutor
from adapters.factory import AdapterFactory
from adapters.models import AdapterConfig, AdapterCategory
from events.event_bus import event_bus
from control_plane.service import control_plane_service


@pytest.fixture
async def e2e_environment():
    """Set up complete E2E test environment."""
    # Initialize all services
    await event_bus.start()
    await control_plane_service.initialize()
    
    # Create workflow executor
    executor = WorkflowExecutor()
    await executor.initialize()
    
    # Register mock adapters
    mock_adapters = {
        "openai": {
            "type": "openai",
            "category": AdapterCategory.AI,
            "capabilities": ["chat_completion", "embeddings"]
        },
        "slack": {
            "type": "slack", 
            "category": AdapterCategory.COMMUNICATION,
            "capabilities": ["send_message", "create_channel"]
        },
        "email": {
            "type": "email",
            "category": AdapterCategory.COMMUNICATION,
            "capabilities": ["send_email"]
        }
    }
    
    yield {
        "executor": executor,
        "adapters": mock_adapters,
        "event_bus": event_bus
    }
    
    # Cleanup
    await executor.shutdown()
    await control_plane_service.shutdown()
    await event_bus.stop()


@pytest.fixture
def customer_support_workflow():
    """Create a customer support automation workflow."""
    return WorkflowDefinition(
        id="customer-support-workflow",
        name="AI Customer Support",
        description="Automated customer support with AI and notifications",
        nodes=[
            # Start: Receive customer query
            NodeDefinition(
                id="receive-query",
                type=NodeType.PROCESS,
                name="Receive Customer Query",
                config={
                    "action": "receive_input",
                    "parameters": {
                        "expected_fields": ["customer_id", "query", "priority"]
                    }
                },
                outputs={"next": "analyze-query"}
            ),
            
            # Analyze query with AI
            NodeDefinition(
                id="analyze-query",
                type=NodeType.AI_PROCESS,
                name="Analyze Query with AI",
                config={
                    "adapter_type": "openai",
                    "capability": "chat_completion",
                    "parameters": {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a customer support analyst. Analyze the query and categorize it."
                            },
                            {
                                "role": "user",
                                "content": "{{query}}"
                            }
                        ],
                        "temperature": 0.3
                    }
                },
                outputs={
                    "success": "check-category",
                    "error": "escalate-to-human"
                }
            ),
            
            # Check category and route
            NodeDefinition(
                id="check-category",
                type=NodeType.PROCESS,
                name="Check Query Category",
                config={
                    "action": "evaluate_conditions",
                    "parameters": {
                        "conditions": [
                            {
                                "field": "category",
                                "operator": "equals",
                                "value": "technical",
                                "output": "technical-response"
                            },
                            {
                                "field": "category",
                                "operator": "equals",
                                "value": "billing",
                                "output": "billing-response"
                            },
                            {
                                "field": "category",
                                "operator": "equals",
                                "value": "general",
                                "output": "general-response"
                            }
                        ],
                        "default_output": "escalate-to-human"
                    }
                },
                outputs={
                    "technical-response": "generate-technical",
                    "billing-response": "generate-billing",
                    "general-response": "generate-general",
                    "escalate-to-human": "escalate-to-human"
                }
            ),
            
            # Generate technical response
            NodeDefinition(
                id="generate-technical",
                type=NodeType.AI_PROCESS,
                name="Generate Technical Response",
                config={
                    "adapter_type": "openai",
                    "capability": "chat_completion",
                    "parameters": {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a technical support expert. Provide a helpful response."
                            },
                            {
                                "role": "user",
                                "content": "{{query}}"
                            }
                        ]
                    }
                },
                outputs={"next": "send-response"}
            ),
            
            # Generate billing response
            NodeDefinition(
                id="generate-billing",
                type=NodeType.AI_PROCESS,
                name="Generate Billing Response",
                config={
                    "adapter_type": "openai",
                    "capability": "chat_completion",
                    "parameters": {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a billing support expert. Provide accurate billing information."
                            },
                            {
                                "role": "user",
                                "content": "{{query}}"
                            }
                        ]
                    }
                },
                outputs={"next": "send-response"}
            ),
            
            # Generate general response
            NodeDefinition(
                id="generate-general",
                type=NodeType.AI_PROCESS,
                name="Generate General Response",
                config={
                    "adapter_type": "openai",
                    "capability": "chat_completion",
                    "parameters": {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful customer support agent."
                            },
                            {
                                "role": "user",
                                "content": "{{query}}"
                            }
                        ]
                    }
                },
                outputs={"next": "send-response"}
            ),
            
            # Send response to customer
            NodeDefinition(
                id="send-response",
                type=NodeType.PARALLEL,
                name="Send Response",
                config={
                    "branches": [
                        {
                            "id": "email-customer",
                            "type": NodeType.NOTIFICATION,
                            "config": {
                                "adapter_type": "email",
                                "capability": "send_email",
                                "parameters": {
                                    "to": "{{customer_email}}",
                                    "subject": "Re: Your Support Query",
                                    "body": "{{response}}"
                                }
                            }
                        },
                        {
                            "id": "log-interaction",
                            "type": NodeType.PROCESS,
                            "config": {
                                "action": "log_to_database",
                                "parameters": {
                                    "table": "support_interactions",
                                    "data": {
                                        "customer_id": "{{customer_id}}",
                                        "query": "{{query}}",
                                        "response": "{{response}}",
                                        "timestamp": "{{timestamp}}"
                                    }
                                }
                            }
                        }
                    ]
                },
                outputs={"complete": "end"}
            ),
            
            # Escalate to human
            NodeDefinition(
                id="escalate-to-human",
                type=NodeType.PARALLEL,
                name="Escalate to Human",
                config={
                    "branches": [
                        {
                            "id": "notify-slack",
                            "type": NodeType.NOTIFICATION,
                            "config": {
                                "adapter_type": "slack",
                                "capability": "send_message",
                                "parameters": {
                                    "channel": "#support-escalations",
                                    "message": "New escalation: {{query}} (Priority: {{priority}})"
                                }
                            }
                        },
                        {
                            "id": "create-ticket",
                            "type": NodeType.PROCESS,
                            "config": {
                                "action": "create_support_ticket",
                                "parameters": {
                                    "customer_id": "{{customer_id}}",
                                    "query": "{{query}}",
                                    "priority": "{{priority}}",
                                    "status": "open"
                                }
                            }
                        }
                    ]
                },
                outputs={"complete": "end"}
            ),
            
            # End
            NodeDefinition(
                id="end",
                type=NodeType.PROCESS,
                name="Complete Workflow",
                config={
                    "action": "finalize",
                    "parameters": {
                        "success": True
                    }
                }
            )
        ],
        edges=[
            {"source": "receive-query", "target": "analyze-query"},
            {"source": "analyze-query", "target": "check-category", "condition": "success"},
            {"source": "analyze-query", "target": "escalate-to-human", "condition": "error"},
            {"source": "check-category", "target": "generate-technical", "condition": "technical-response"},
            {"source": "check-category", "target": "generate-billing", "condition": "billing-response"},
            {"source": "check-category", "target": "generate-general", "condition": "general-response"},
            {"source": "check-category", "target": "escalate-to-human", "condition": "escalate-to-human"},
            {"source": "generate-technical", "target": "send-response"},
            {"source": "generate-billing", "target": "send-response"},
            {"source": "generate-general", "target": "send-response"},
            {"source": "send-response", "target": "end"},
            {"source": "escalate-to-human", "target": "end"}
        ],
        start_node_id="receive-query"
    )


class TestCustomerSupportE2E:
    """End-to-end test for customer support workflow."""
    
    @pytest.mark.asyncio
    async def test_technical_query_flow(self, e2e_environment, customer_support_workflow):
        """Test handling of technical support query."""
        env = e2e_environment
        executor = env["executor"]
        
        # Mock OpenAI responses
        openai_responses = [
            # First call: analyze query
            AsyncMock(
                success=True,
                data={
                    "content": json.dumps({
                        "category": "technical",
                        "urgency": "medium",
                        "topics": ["api", "authentication"]
                    })
                }
            ),
            # Second call: generate technical response
            AsyncMock(
                success=True,
                data={
                    "content": "To resolve your API authentication issue, please check your API key..."
                }
            )
        ]
        
        # Mock email sending
        email_response = AsyncMock(
            success=True,
            data={"message_id": "email-123", "status": "sent"}
        )
        
        with patch('adapters.implementations.ai.openai_adapter.OpenAIAdapter.execute') as mock_openai:
            with patch('adapters.implementations.communication.email_adapter.EmailAdapter.execute') as mock_email:
                mock_openai.side_effect = openai_responses
                mock_email.return_value = email_response
                
                # Execute workflow
                input_data = {
                    "customer_id": "CUST-12345",
                    "customer_email": "customer@example.com",
                    "query": "I'm having trouble authenticating with the API. My requests return 401 errors.",
                    "priority": "medium"
                }
                
                instance = await executor.execute_workflow(
                    customer_support_workflow,
                    input_data
                )
                
                # Verify workflow completed successfully
                assert instance.status == WorkflowStatus.COMPLETED
                
                # Verify AI was called twice
                assert mock_openai.call_count == 2
                
                # Verify email was sent
                mock_email.assert_called_once()
                email_call_args = mock_email.call_args[0][0]
                assert email_call_args.parameters["to"] == "customer@example.com"
                assert "API authentication" in email_call_args.parameters["body"]
    
    @pytest.mark.asyncio
    async def test_escalation_flow(self, e2e_environment, customer_support_workflow):
        """Test escalation to human support."""
        env = e2e_environment
        executor = env["executor"]
        
        # Mock OpenAI to return unknown category
        openai_response = AsyncMock(
            success=True,
            data={
                "content": json.dumps({
                    "category": "unknown",
                    "urgency": "high",
                    "requires_human": True
                })
            }
        )
        
        # Mock Slack notification
        slack_response = AsyncMock(
            success=True,
            data={"message_id": "slack-123", "channel": "#support-escalations"}
        )
        
        with patch('adapters.implementations.ai.openai_adapter.OpenAIAdapter.execute') as mock_openai:
            with patch('adapters.implementations.communication.slack_adapter.SlackAdapter.execute') as mock_slack:
                mock_openai.return_value = openai_response
                mock_slack.return_value = slack_response
                
                # Execute workflow
                input_data = {
                    "customer_id": "CUST-67890",
                    "query": "Complex legal question about data processing agreement",
                    "priority": "high"
                }
                
                instance = await executor.execute_workflow(
                    customer_support_workflow,
                    input_data
                )
                
                # Verify workflow completed
                assert instance.status == WorkflowStatus.COMPLETED
                
                # Verify Slack notification was sent
                mock_slack.assert_called_once()
                slack_call_args = mock_slack.call_args[0][0]
                assert "Complex legal question" in slack_call_args.parameters["message"]
                assert "Priority: high" in slack_call_args.parameters["message"]
    
    @pytest.mark.asyncio
    async def test_ai_error_handling(self, e2e_environment, customer_support_workflow):
        """Test handling of AI service errors."""
        env = e2e_environment
        executor = env["executor"]
        
        # Mock OpenAI to fail
        with patch('adapters.implementations.ai.openai_adapter.OpenAIAdapter.execute') as mock_openai:
            mock_openai.side_effect = Exception("OpenAI API Error: Rate limit exceeded")
            
            # Mock Slack for escalation
            slack_response = AsyncMock(
                success=True,
                data={"message_id": "slack-456"}
            )
            
            with patch('adapters.implementations.communication.slack_adapter.SlackAdapter.execute') as mock_slack:
                mock_slack.return_value = slack_response
                
                # Execute workflow
                input_data = {
                    "customer_id": "CUST-11111",
                    "query": "Simple question that should be handled by AI",
                    "priority": "low"
                }
                
                instance = await executor.execute_workflow(
                    customer_support_workflow,
                    input_data
                )
                
                # Verify workflow still completed via escalation path
                assert instance.status == WorkflowStatus.COMPLETED
                
                # Verify escalation occurred
                mock_slack.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_concurrent_customer_queries(self, e2e_environment, customer_support_workflow):
        """Test handling multiple concurrent customer queries."""
        env = e2e_environment
        executor = env["executor"]
        
        # Create multiple customer queries
        queries = [
            {
                "customer_id": f"CUST-{i}",
                "customer_email": f"customer{i}@example.com",
                "query": f"Technical question {i} about API",
                "priority": "medium"
            }
            for i in range(5)
        ]
        
        # Mock responses
        with patch('adapters.implementations.ai.openai_adapter.OpenAIAdapter.execute') as mock_openai:
            with patch('adapters.implementations.communication.email_adapter.EmailAdapter.execute') as mock_email:
                # OpenAI returns technical category and generates response
                mock_openai.side_effect = [
                    AsyncMock(
                        success=True,
                        data={"content": json.dumps({"category": "technical"})}
                    ),
                    AsyncMock(
                        success=True,
                        data={"content": f"Response for query {i}"}
                    )
                    for i in range(5)
                    for _ in range(2)  # Two calls per query
                ]
                
                mock_email.return_value = AsyncMock(
                    success=True,
                    data={"status": "sent"}
                )
                
                # Execute workflows concurrently
                tasks = [
                    executor.execute_workflow(customer_support_workflow, query)
                    for query in queries
                ]
                
                instances = await asyncio.gather(*tasks)
                
                # Verify all completed successfully
                assert all(inst.status == WorkflowStatus.COMPLETED for inst in instances)
                
                # Verify correct number of AI calls (2 per query)
                assert mock_openai.call_count == 10
                
                # Verify emails sent to all customers
                assert mock_email.call_count == 5
    
    @pytest.mark.asyncio
    async def test_workflow_metrics_and_monitoring(self, e2e_environment, customer_support_workflow):
        """Test workflow metrics and monitoring capabilities."""
        env = e2e_environment
        executor = env["executor"]
        event_bus = env["event_bus"]
        
        # Track metrics
        metrics = {
            "workflows_started": 0,
            "workflows_completed": 0,
            "nodes_executed": 0,
            "ai_calls": 0,
            "notifications_sent": 0
        }
        
        async def metrics_handler(event):
            event_type = event.get("type", "")
            if event_type == "workflow.started":
                metrics["workflows_started"] += 1
            elif event_type == "workflow.completed":
                metrics["workflows_completed"] += 1
            elif event_type == "node.completed":
                metrics["nodes_executed"] += 1
                node_type = event.get("node_type", "")
                if node_type == NodeType.AI_PROCESS:
                    metrics["ai_calls"] += 1
                elif node_type == NodeType.NOTIFICATION:
                    metrics["notifications_sent"] += 1
        
        # Subscribe to events
        await event_bus.subscribe("workflow.*", metrics_handler)
        await event_bus.subscribe("node.*", metrics_handler)
        
        # Mock successful flow
        with patch('adapters.implementations.ai.openai_adapter.OpenAIAdapter.execute') as mock_openai:
            with patch('adapters.implementations.communication.email_adapter.EmailAdapter.execute') as mock_email:
                mock_openai.side_effect = [
                    AsyncMock(success=True, data={"content": json.dumps({"category": "general"})}),
                    AsyncMock(success=True, data={"content": "General response"})
                ]
                mock_email.return_value = AsyncMock(success=True, data={"status": "sent"})
                
                # Execute workflow
                await executor.execute_workflow(
                    customer_support_workflow,
                    {
                        "customer_id": "CUST-METRICS",
                        "customer_email": "metrics@example.com",
                        "query": "General question",
                        "priority": "low"
                    }
                )
                
                # Allow events to propagate
                await asyncio.sleep(0.1)
                
                # Verify metrics
                assert metrics["workflows_started"] == 1
                assert metrics["workflows_completed"] == 1
                assert metrics["nodes_executed"] > 5  # Multiple nodes executed
                assert metrics["ai_calls"] == 2  # Analysis + response generation
                assert metrics["notifications_sent"] >= 1  # Email sent