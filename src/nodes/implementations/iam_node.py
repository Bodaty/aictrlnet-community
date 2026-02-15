"""IAM (Internal Agent Messaging) node implementation."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import uuid

from ..base_node import BaseNode
from ..models import NodeConfig
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class IAMNode(BaseNode):
    """Node for Internal Agent Messaging operations.
    
    Enables:
    - Agent-to-agent communication
    - Message routing and delivery
    - Agent discovery
    - Broadcast messaging
    - Request-response patterns
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the IAM node. Returns output dict for BaseNode.run() to wrap."""
        # Get IAM service from context or create new instance
        from services.iam import IAMService
        db = context.get('db')
        if not db:
            raise ValueError("Database session not provided in context")
        iam_service = IAMService(db)

        # Get IAM operation type
        operation = self.config.parameters.get("operation", "send_message")

        # Execute based on operation type
        if operation == "send_message":
            output_data = await self._send_message(input_data, context)
        elif operation == "broadcast":
            output_data = await self._broadcast_message(input_data, context)
        elif operation == "request":
            output_data = await self._send_request(input_data, context)
        elif operation == "discover_agents":
            output_data = await self._discover_agents()
        elif operation == "get_agent_info":
            output_data = await self._get_agent_info(input_data)
        elif operation == "subscribe":
            output_data = await self._subscribe_to_topic(input_data, context)
        elif operation == "publish":
            output_data = await self._publish_to_topic(input_data, context)
        else:
            raise ValueError(f"Unsupported IAM operation: {operation}")

        # Publish completion event
        await event_bus.publish(
            "node.executed",
            {
                "node_id": self.config.id,
                "node_type": "iam",
                "operation": operation
            }
        )

        return output_data
    
    async def _send_message(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to a specific agent."""
        # Get message configuration
        target_agent = self.config.parameters.get("target_agent") or input_data.get("target_agent")
        message_type = self.config.parameters.get("message_type", "data")
        message_content = self.config.parameters.get("message") or input_data.get("message")
        
        if not target_agent:
            raise ValueError("target_agent is required for send_message operation")
        if not message_content:
            raise ValueError("message content is required")
        
        # Build message
        message = {
            "id": str(uuid.uuid4()),
            "type": message_type,
            "from_agent": context.get("agent_id", "workflow"),
            "to_agent": target_agent,
            "content": message_content,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": context.get("workflow_id"),
            "node_id": self.config.id
        }
        
        # Add optional fields
        if self.config.parameters.get("priority"):
            message["priority"] = self.config.parameters["priority"]
        if self.config.parameters.get("expires_at"):
            message["expires_at"] = self.config.parameters["expires_at"]
        
        # Send message
        delivery_status = await iam_service.send_message(
            to_agent=target_agent,
            message=message
        )
        
        return {
            "message_id": message["id"],
            "delivered": delivery_status.get("delivered", False),
            "delivery_time": delivery_status.get("delivery_time"),
            "target_agent": target_agent
        }
    
    async def _broadcast_message(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Broadcast a message to multiple agents."""
        # Get broadcast configuration
        agent_filter = self.config.parameters.get("agent_filter", {})
        message_type = self.config.parameters.get("message_type", "broadcast")
        message_content = self.config.parameters.get("message") or input_data.get("message")
        
        if not message_content:
            raise ValueError("message content is required for broadcast")
        
        # Build message
        message = {
            "id": str(uuid.uuid4()),
            "type": message_type,
            "from_agent": context.get("agent_id", "workflow"),
            "content": message_content,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": context.get("workflow_id"),
            "node_id": self.config.id
        }
        
        # Apply filters to find target agents
        if agent_filter.get("capability"):
            # Filter by capability
            agents = await iam_service.find_agents_by_capability(
                capability=agent_filter["capability"]
            )
        elif agent_filter.get("tag"):
            # Filter by tag
            agents = await iam_service.find_agents_by_tag(
                tag=agent_filter["tag"]
            )
        elif agent_filter.get("pattern"):
            # Filter by name pattern
            agents = await iam_service.find_agents_by_pattern(
                pattern=agent_filter["pattern"]
            )
        else:
            # Broadcast to all agents
            agents = await iam_service.get_all_agents()
        
        # Send to all matching agents
        delivery_results = []
        for agent in agents:
            try:
                status = await iam_service.send_message(
                    to_agent=agent["id"],
                    message=message
                )
                delivery_results.append({
                    "agent_id": agent["id"],
                    "delivered": status.get("delivered", False)
                })
            except Exception as e:
                delivery_results.append({
                    "agent_id": agent["id"],
                    "delivered": False,
                    "error": str(e)
                })
        
        # Calculate delivery statistics
        total_agents = len(delivery_results)
        successful_deliveries = sum(1 for r in delivery_results if r["delivered"])
        
        return {
            "message_id": message["id"],
            "broadcast_stats": {
                "total_agents": total_agents,
                "successful_deliveries": successful_deliveries,
                "failed_deliveries": total_agents - successful_deliveries
            },
            "delivery_results": delivery_results
        }
    
    async def _send_request(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send a request and wait for response."""
        # Get request configuration
        target_agent = self.config.parameters.get("target_agent") or input_data.get("target_agent")
        request_type = self.config.parameters.get("request_type", "query")
        request_content = self.config.parameters.get("request") or input_data.get("request")
        timeout = self.config.parameters.get("timeout", 30)
        
        if not target_agent:
            raise ValueError("target_agent is required for request operation")
        if not request_content:
            raise ValueError("request content is required")
        
        # Build request
        request = {
            "id": str(uuid.uuid4()),
            "type": f"request.{request_type}",
            "from_agent": context.get("agent_id", "workflow"),
            "to_agent": target_agent,
            "content": request_content,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": context.get("workflow_id"),
            "node_id": self.config.id,
            "expects_response": True
        }
        
        # Send request and wait for response
        response = await iam_service.send_request(
            to_agent=target_agent,
            request=request,
            timeout=timeout
        )
        
        if response:
            return {
                "request_id": request["id"],
                "response_received": True,
                "response": response.get("content"),
                "response_time": response.get("timestamp"),
                "target_agent": target_agent
            }
        else:
            return {
                "request_id": request["id"],
                "response_received": False,
                "timeout": True,
                "target_agent": target_agent
            }
    
    async def _discover_agents(self) -> Dict[str, Any]:
        """Discover available agents."""
        # Get discovery filters
        filters = self.config.parameters.get("filters", {})
        
        # Apply filters
        if filters.get("capability"):
            agents = await iam_service.find_agents_by_capability(
                capability=filters["capability"]
            )
        elif filters.get("tag"):
            agents = await iam_service.find_agents_by_tag(
                tag=filters["tag"]
            )
        elif filters.get("status"):
            agents = await iam_service.find_agents_by_status(
                status=filters["status"]
            )
        else:
            # Get all agents
            agents = await iam_service.get_all_agents()
        
        # Get detailed info if requested
        include_details = self.config.parameters.get("include_details", False)
        if include_details:
            detailed_agents = []
            for agent in agents:
                details = await iam_service.get_agent_details(agent["id"])
                detailed_agents.append(details)
            agents = detailed_agents
        
        return {
            "agents": agents,
            "count": len(agents)
        }
    
    async def _get_agent_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about a specific agent."""
        agent_id = self.config.parameters.get("agent_id") or input_data.get("agent_id")
        
        if not agent_id:
            raise ValueError("agent_id is required for get_agent_info operation")
        
        # Get agent details
        agent_info = await iam_service.get_agent_details(agent_id)
        
        if not agent_info:
            raise ValueError(f"Agent '{agent_id}' not found")
        
        # Get additional info if requested
        include_metrics = self.config.parameters.get("include_metrics", False)
        if include_metrics:
            metrics = await iam_service.get_agent_metrics(agent_id)
            agent_info["metrics"] = metrics
        
        include_capabilities = self.config.parameters.get("include_capabilities", False)
        if include_capabilities:
            capabilities = await iam_service.get_agent_capabilities(agent_id)
            agent_info["capabilities"] = capabilities
        
        return {
            "agent": agent_info
        }
    
    async def _subscribe_to_topic(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Subscribe to a messaging topic."""
        topic = self.config.parameters.get("topic") or input_data.get("topic")
        
        if not topic:
            raise ValueError("topic is required for subscribe operation")
        
        # Subscribe the current agent/workflow to the topic
        subscriber_id = context.get("agent_id", f"workflow-{context.get('workflow_id', 'unknown')}")
        
        subscription = await iam_service.subscribe_to_topic(
            topic=topic,
            subscriber_id=subscriber_id,
            filters=self.config.parameters.get("filters", {})
        )
        
        return {
            "subscription_id": subscription.get("id"),
            "topic": topic,
            "subscriber_id": subscriber_id,
            "active": subscription.get("active", True)
        }
    
    async def _publish_to_topic(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Publish a message to a topic."""
        topic = self.config.parameters.get("topic") or input_data.get("topic")
        message_content = self.config.parameters.get("message") or input_data.get("message")
        
        if not topic:
            raise ValueError("topic is required for publish operation")
        if not message_content:
            raise ValueError("message content is required")
        
        # Build topic message
        message = {
            "id": str(uuid.uuid4()),
            "topic": topic,
            "publisher": context.get("agent_id", "workflow"),
            "content": message_content,
            "timestamp": datetime.utcnow().isoformat(),
            "workflow_id": context.get("workflow_id"),
            "node_id": self.config.id
        }
        
        # Publish to topic
        publish_result = await iam_service.publish_to_topic(
            topic=topic,
            message=message
        )
        
        return {
            "message_id": message["id"],
            "topic": topic,
            "subscribers_notified": publish_result.get("subscribers_notified", 0),
            "publish_time": message["timestamp"]
        }
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        operation = self.config.parameters.get("operation")
        if not operation:
            raise ValueError("operation parameter is required")
        
        valid_operations = [
            "send_message", "broadcast", "request", "discover_agents",
            "get_agent_info", "subscribe", "publish"
        ]
        
        if operation not in valid_operations:
            raise ValueError(f"Invalid operation: {operation}. Must be one of {valid_operations}")
        
        # Validate operation-specific requirements
        if operation == "send_message" and not self.config.parameters.get("target_agent"):
            # Target agent can come from input, so this is OK
            pass
        elif operation == "request" and not self.config.parameters.get("target_agent"):
            # Target agent can come from input, so this is OK
            pass
        elif operation == "get_agent_info" and not self.config.parameters.get("agent_id"):
            # Agent ID can come from input, so this is OK
            pass
        elif operation == "subscribe" and not self.config.parameters.get("topic"):
            # Topic can come from input, so this is OK
            pass
        elif operation == "publish" and not self.config.parameters.get("topic"):
            # Topic can come from input, so this is OK
            pass
        
        return True