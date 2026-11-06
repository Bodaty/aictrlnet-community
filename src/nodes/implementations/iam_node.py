"""IAM (Internal Agent Messaging) node implementation."""

import logging
from typing import Any, Dict, List, Optional
import fnmatch
import uuid

from sqlalchemy import select

from ..base_node import BaseNode
from ..models import NodeConfig
from events.event_bus import event_bus
from schemas.iam import IAMMessageCreate, IAMMessageType


logger = logging.getLogger(__name__)

# One page of the agent registry: the registry is small, and a filter runs over the whole of it.
_AGENT_LIMIT = 1000


class IAMOperationUnavailable(RuntimeError):
    """The operation has no implementation behind it on this platform."""


def _unavailable(operation: str, why: str) -> IAMOperationUnavailable:
    return IAMOperationUnavailable(f"IAM operation '{operation}' is not available: {why}")


class IAMNode(BaseNode):
    """Node for Internal Agent Messaging operations.
    
    Operations: send_message, broadcast, discover_agents, get_agent_info.
    ``request``, ``subscribe`` and ``publish`` are accepted in a workflow
    definition but fail when run: IAM has no reply-waiting request and no topics.
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the IAM node. Returns output dict for BaseNode.run() to wrap."""
        # Get IAM service from context or create new instance
        from services.iam import IAMService
        db = context.get('db')
        if not db:
            raise ValueError("Database session not provided in context")
        self._iam_service = IAMService(db)
        self._db = db

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
    
    # ------------------------------------------------------------------
    # Agents and messages
    # ------------------------------------------------------------------

    async def _resolve_agent(self, ref: Any):
        """An IAM agent by id or by name, or ValueError."""
        try:
            agent_id = ref if isinstance(ref, uuid.UUID) else uuid.UUID(str(ref))
        except ValueError:
            agent_id = None
        if agent_id is not None:
            agent = await self._iam_service.get_agent(agent_id)
            if agent:
                return agent
        for agent in await self._iam_service.list_agents(limit=_AGENT_LIMIT):
            if agent.name == str(ref):
                return agent
        raise ValueError(f"IAM agent '{ref}' not found")

    async def _require_agent_access(self, context: Dict[str, Any]) -> None:
        """Sending as, to, or across IAM agents is a superuser's act.

        IAM agents carry no tenant and no owner, so there is nothing to check a
        non-superuser's claim to an agent against; without this, any workflow
        author could name any agent as the sender and have its messages
        attributed to it. Until agents have owners, anyone else gets the same
        answer as for an agent that does not exist.
        """
        from models.user import User

        user_id = context.get("user_id")
        allowed = bool(user_id) and bool(await self._db.scalar(
            select(User.is_superuser).where(User.id == str(user_id))))
        if not allowed:
            raise ValueError("IAM agent not found")

    async def _sender(self, context: Dict[str, Any]):
        """The IAM agent the message is sent as: ``sender_agent`` or the context's agent."""
        await self._require_agent_access(context)
        ref = self.config.parameters.get("sender_agent") or context.get("agent_id")
        if not ref:
            raise ValueError(
                "A sender is required: set sender_agent to the id or name of a registered "
                "IAM agent (the workflow itself is not an IAM agent)")
        return await self._resolve_agent(ref)

    def _message_create(self, message_content: Any, recipient_id, default_type: str):
        message_type = self.config.parameters.get("message_type", default_type)
        try:
            message_type = IAMMessageType(message_type)
        except ValueError:
            raise ValueError(
                f"Invalid message_type '{message_type}'. "
                f"Must be one of {[t.value for t in IAMMessageType]}")
        if self.config.parameters.get("expires_at"):
            raise ValueError("expires_at is not supported; set ttl_seconds")
        content = message_content if isinstance(message_content, dict) else {"text": message_content}
        return IAMMessageCreate(
            message_type=message_type,
            content=content,
            priority=self.config.parameters.get("priority", 0),
            ttl_seconds=self.config.parameters.get("ttl_seconds"),
            recipient_id=recipient_id,
        )

    @staticmethod
    def _status_of(message) -> str:
        return getattr(message.status, "value", str(message.status))

    async def _send_message(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to a specific agent."""
        # Get message configuration
        target_agent = self.config.parameters.get("target_agent") or input_data.get("target_agent")
        message_content = self.config.parameters.get("message") or input_data.get("message")

        # Auto-generated system templates carry descriptive agentDelegation nodes
        # (prompt/agent/analysis fields, no operation/target_agent). Rather than
        # crashing the whole workflow, an unconfigured delegation is skipped with
        # a clear, audit-visible reason — no message is fabricated or sent.
        if not target_agent or not message_content:
            reason = "no target_agent configured" if not target_agent else "no message content configured"
            logger.info(
                "agentDelegation node %s skipped — %s", self.config.id, reason
            )
            return {
                "status": "skipped",
                "executed": False,
                "operation": "send_message",
                "reason": reason,
                "node_id": self.config.id,
            }

        sender = await self._sender(context)  # checks the caller's access to agents
        recipient = await self._resolve_agent(target_agent)
        sent = await self._iam_service.send_message(
            sender.id, self._message_create(message_content, recipient.id, "notification"))

        # Delivery is asynchronous: the message is stored and queued, and its
        # status moves on from here.
        return {
            "message_id": str(sent.id),
            "status": self._status_of(sent),
            "target_agent": recipient.name
        }

    async def _agents_for(self, capability=None, status=None) -> List[Any]:
        if capability:
            capabilities = [capability] if isinstance(capability, str) else list(capability)
            return await self._iam_service.discover_agents(capabilities)
        return await self._iam_service.list_agents(status=status, limit=_AGENT_LIMIT)

    async def _broadcast_message(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Send a message to each agent matching a filter (all agents by default)."""
        agent_filter = self.config.parameters.get("agent_filter", {})
        message_content = self.config.parameters.get("message") or input_data.get("message")

        if not message_content:
            raise ValueError("message content is required for broadcast")
        if agent_filter.get("tag"):
            raise _unavailable("broadcast", "IAM agents carry capabilities, not tags")

        sender = await self._sender(context)
        agents = await self._agents_for(agent_filter.get("capability"), agent_filter.get("status"))
        if agent_filter.get("pattern"):
            agents = [a for a in agents if fnmatch.fnmatchcase(a.name, agent_filter["pattern"])]
        agents = [a for a in agents if a.id != sender.id]

        results = []
        for agent in agents:
            try:
                sent = await self._iam_service.send_message(
                    sender.id, self._message_create(message_content, agent.id, "broadcast"))
                results.append({
                    "agent_id": str(agent.id),
                    "message_id": str(sent.id),
                    "status": self._status_of(sent),
                })
            except Exception as e:
                results.append({"agent_id": str(agent.id), "status": "failed", "error": str(e)})

        failed = sum(1 for r in results if r["status"] == "failed")
        return {
            "broadcast_stats": {
                "total_agents": len(results),
                "messages_sent": len(results) - failed,
                "failed": failed
            },
            "delivery_results": results
        }

    async def _send_request(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise _unavailable(
            "request", "IAM stores and routes messages but has no request that waits for a reply")

    async def _discover_agents(self) -> Dict[str, Any]:
        """Discover registered agents."""
        filters = self.config.parameters.get("filters", {})
        if filters.get("tag"):
            raise _unavailable("discover_agents", "IAM agents carry capabilities, not tags")

        agents = await self._agents_for(filters.get("capability"), filters.get("status"))

        # Get detailed info if requested
        if self.config.parameters.get("include_details", False):
            agents = [await self._iam_service.get_agent(a.id) for a in agents]

        return {
            "agents": [a.model_dump(mode="json") for a in agents if a],
            "count": len(agents)
        }

    async def _get_agent_info(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get information about a specific agent."""
        agent_id = self.config.parameters.get("agent_id") or input_data.get("agent_id")

        if not agent_id:
            raise ValueError("agent_id is required for get_agent_info operation")

        agent = await self._resolve_agent(agent_id)
        agent_info = agent.model_dump(mode="json")

        if self.config.parameters.get("include_metrics", False):
            metrics = await self._iam_service.get_agent_metrics(agent.id)
            agent_info["metrics"] = metrics.model_dump(mode="json") if metrics else None

        # An agent's capabilities are part of the agent record.
        if self.config.parameters.get("include_capabilities", False):
            agent_info["capabilities"] = list(agent.capabilities or [])

        return {
            "agent": agent_info
        }

    async def _subscribe_to_topic(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise _unavailable("subscribe", "IAM has no topics or subscriptions")

    async def _publish_to_topic(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        raise _unavailable("publish", "IAM has no topics or subscriptions")

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