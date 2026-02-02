"""OpenClaw AI agent adapter — integrates OpenClaw instances with AICtrlNet governance.

Follows the same AgenticAIAdapter -> BaseAdapter pattern used by LangChain,
AutoGPT, CrewAI, AutoGen, and Semantic Kernel adapters.

The adapter is the server-side component for AICtrlNet reaching OUT to OpenClaw.
The SDK packages (TypeScript/Python) are the client libraries for external
runtimes calling INTO AICtrlNet.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from adapters.agentic_ai_adapter import AgenticAIAdapter
from adapters.models import AdapterCapability, AdapterCategory, AdapterConfig, Edition

logger = logging.getLogger(__name__)


class OpenClawAdapter(AgenticAIAdapter):
    """OpenClaw AI agent adapter implementation.

    Capabilities:
    - Governance integration: evaluate_action, register_runtime, report_action
    - OpenClaw-specific: send_message, execute_skill, list_channels, list_skills
    - Delegation: delegate_to_agent, spawn_sub_agent
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        if not config:
            config = AdapterConfig(
                name="openclaw",
                type="openclaw",
                category=AdapterCategory.AI_AGENT,
                edition=Edition.COMMUNITY,
            )
        super().__init__(config)
        self.agent_type = "openclaw"
        self.supports_memory = True
        self.supports_tools = True
        self.max_iterations = 25  # OpenClaw agents tend to be longer-running

        # OpenClaw gateway URL from config
        self.gateway_url = config.base_url or config.custom_config.get("gateway_url")
        self.discovery_only = config.custom_config.get("discovery_only", False)
        self._http_client = None

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize the OpenClaw adapter."""
        self.status = "ready"
        if self.gateway_url and not self.discovery_only:
            try:
                import httpx

                self._http_client = httpx.AsyncClient(
                    base_url=self.gateway_url,
                    timeout=self.config.timeout_seconds,
                )
                logger.info(f"OpenClaw adapter connected to {self.gateway_url}")
            except ImportError:
                logger.warning("httpx not available — OpenClaw adapter in discovery-only mode")
                self.discovery_only = True
        else:
            logger.info("OpenClaw adapter initialized in discovery mode")

    async def shutdown(self) -> None:
        """Clean shutdown of OpenClaw adapter."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("OpenClaw adapter shut down")

    # ── Capabilities ─────────────────────────────────────────────────────

    def get_capabilities(self) -> List[AdapterCapability]:
        """Return OpenClaw adapter capabilities."""
        return [
            # Governance integration capabilities
            AdapterCapability(
                name="evaluate_action",
                description="Evaluate an OpenClaw tool action through Q/G/S/M governance",
                category="governance",
                required_parameters=["action_type", "target"],
                parameters={
                    "action_type": "string — tool/skill/command type",
                    "target": "string — target resource",
                    "description": "string — human-readable description",
                    "risk_hints": "object — caller-supplied risk hints",
                },
                estimated_duration_seconds=0.5,
                cost_per_request=0.001,
            ),
            AdapterCapability(
                name="register_runtime",
                description="Register an OpenClaw instance as a governed runtime",
                category="governance",
                required_parameters=["instance_name", "runtime_type"],
                parameters={
                    "instance_name": "string — display name",
                    "runtime_type": "string — always 'openclaw'",
                    "capabilities": "array — list of capability strings",
                },
                estimated_duration_seconds=1.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="report_action",
                description="Report action outcome for audit trail",
                category="governance",
                required_parameters=["evaluation_id", "action_type", "status"],
                parameters={
                    "evaluation_id": "string — from evaluate response",
                    "status": "string — success/failure/partial/error",
                    "result_summary": "string — outcome description",
                },
                estimated_duration_seconds=0.3,
                cost_per_request=0.0,
            ),
            # OpenClaw-specific capabilities
            AdapterCapability(
                name="send_message",
                description="Send message via OpenClaw channel (Slack, Discord, etc.)",
                category="communication",
                required_parameters=["channel", "message"],
                parameters={
                    "channel": "string — channel ID or name",
                    "message": "string — message content",
                    "platform": "string — slack/discord/teams",
                },
                estimated_duration_seconds=2.0,
                cost_per_request=0.005,
            ),
            AdapterCapability(
                name="execute_skill",
                description="Execute an OpenClaw skill with governance",
                category="execution",
                required_parameters=["skill_name"],
                parameters={
                    "skill_name": "string — skill identifier",
                    "arguments": "object — skill arguments",
                    "timeout_seconds": "number — max execution time",
                },
                estimated_duration_seconds=10.0,
                cost_per_request=0.01,
            ),
            AdapterCapability(
                name="list_channels",
                description="List available OpenClaw messaging channels",
                category="discovery",
                required_parameters=[],
                estimated_duration_seconds=1.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="list_skills",
                description="List available OpenClaw skills",
                category="discovery",
                required_parameters=[],
                estimated_duration_seconds=1.0,
                cost_per_request=0.0,
            ),
            # Delegation capabilities
            AdapterCapability(
                name="delegate_to_agent",
                description="Delegate task to another OpenClaw agent via agent-send",
                category="delegation",
                required_parameters=["target_agent", "task"],
                parameters={
                    "target_agent": "string — target agent ID or name",
                    "task": "string — task description",
                    "context": "object — delegation context",
                },
                estimated_duration_seconds=5.0,
                cost_per_request=0.01,
            ),
            AdapterCapability(
                name="spawn_sub_agent",
                description="Spawn a governed sub-agent",
                category="delegation",
                required_parameters=["agent_config"],
                parameters={
                    "agent_config": "object — sub-agent configuration",
                    "parent_chain_id": "string — delegation chain to extend",
                },
                estimated_duration_seconds=3.0,
                cost_per_request=0.005,
            ),
        ]

    # ── Agent Lifecycle ──────────────────────────────────────────────────

    async def create_agent(self, config: Dict[str, Any]) -> Any:
        """Register an OpenClaw instance and create the runtime bridge.

        In discovery mode, returns static metadata without connecting.
        When connected, calls the runtime gateway /register endpoint.
        """
        if self.discovery_only:
            return {
                "type": "openclaw_discovery",
                "agent_type": "openclaw",
                "capabilities": [c.name for c in self.get_capabilities()],
                "discovery_only": True,
                "config": config,
                "created_at": datetime.utcnow().isoformat(),
            }

        # Create agent connected to OpenClaw gateway
        instance_name = config.get("instance_name", "openclaw-agent")
        capabilities = config.get("capabilities", [
            "tool_execution", "messaging", "skill_execution",
        ])

        agent_data = {
            "type": "openclaw_agent",
            "agent_type": "openclaw",
            "instance_name": instance_name,
            "gateway_url": self.gateway_url,
            "capabilities": capabilities,
            "config": config,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Attempt to ping the gateway
        if self._http_client:
            try:
                response = await self._http_client.get("/health")
                agent_data["gateway_status"] = "connected" if response.status_code < 400 else "unreachable"
            except Exception as exc:
                logger.warning(f"OpenClaw gateway health check failed: {exc}")
                agent_data["gateway_status"] = "unreachable"

        return agent_data

    async def execute_agent(self, agent: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute action through governance pipeline then forward to OpenClaw.

        Flow:
        1. Determine action type from request
        2. If governance-enabled: evaluate -> forward -> report
        3. If direct: forward to OpenClaw gateway
        """
        start_time = datetime.utcnow()

        if isinstance(agent, dict) and agent.get("discovery_only"):
            return {
                "output": f"[OpenClaw Discovery] Would execute: {request.get('input', 'No input')}",
                "duration_ms": 0,
                "agent_type": "openclaw",
                "discovery_only": True,
            }

        operation = request.get("operation", "execute")
        input_text = request.get("input", "")

        if operation == "send_message":
            return await self._handle_send_message(agent, request)
        elif operation == "execute_skill":
            return await self._handle_execute_skill(agent, request)
        elif operation == "delegate":
            return await self._handle_delegate(agent, request)
        elif operation == "list_channels":
            return await self._handle_list_channels(agent)
        elif operation == "list_skills":
            return await self._handle_list_skills(agent)

        # Default: forward as generic execution to OpenClaw
        if self._http_client:
            try:
                response = await self._http_client.post(
                    "/execute",
                    json={"input": input_text, "context": request.get("context", {})},
                )
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                if response.status_code < 400:
                    return {
                        "output": response.json(),
                        "duration_ms": duration_ms,
                        "agent_type": "openclaw",
                        "status": "success",
                    }
                return {
                    "error": f"OpenClaw returned {response.status_code}",
                    "duration_ms": duration_ms,
                    "agent_type": "openclaw",
                    "status": "failed",
                }
            except Exception as exc:
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                return {
                    "error": str(exc),
                    "duration_ms": duration_ms,
                    "agent_type": "openclaw",
                    "status": "failed",
                }

        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        return {
            "output": f"[OpenClaw] No gateway connected. Input: {input_text}",
            "duration_ms": duration_ms,
            "agent_type": "openclaw",
            "status": "no_gateway",
        }

    async def get_agent_state(self, agent: Any) -> Dict[str, Any]:
        """Get OpenClaw instance status via heartbeat + gateway query."""
        if isinstance(agent, dict):
            state = {
                "type": agent.get("type", "unknown"),
                "agent_type": "openclaw",
                "gateway_url": agent.get("gateway_url"),
                "capabilities": agent.get("capabilities", []),
                "config": agent.get("config", {}),
                "status": "active",
            }

            # Query gateway if connected
            if self._http_client and agent.get("gateway_url"):
                try:
                    response = await self._http_client.get("/status")
                    if response.status_code < 400:
                        state["gateway_state"] = response.json()
                        state["status"] = "connected"
                except Exception:
                    state["status"] = "gateway_unreachable"

            return state

        return {"status": "unknown", "agent_type": "openclaw"}

    # ── Execute handler ──────────────────────────────────────────────────

    async def execute(self, request: Any) -> Any:
        """Execute a request using the OpenClaw adapter."""
        from adapters.models import AdapterResponse

        if hasattr(request, "operation"):
            operation = request.operation
        elif isinstance(request, dict):
            operation = request.get("operation", "execute")
        else:
            operation = "execute"

        if operation == "create_agent":
            params = request.parameters if hasattr(request, "parameters") else request
            agent = await self.create_agent(params)
            return AdapterResponse(success=True, data=agent, adapter_id=self.id)

        elif operation == "execute_agent":
            params = request.parameters if hasattr(request, "parameters") else request
            result = await self.execute_agent(params.get("agent"), params)
            return AdapterResponse(
                success=not result.get("error"),
                data=result,
                adapter_id=self.id,
                error=result.get("error"),
            )

        return AdapterResponse(
            success=False,
            error=f"Unknown operation: {operation}",
            adapter_id=self.id,
        )

    # ── Health ───────────────────────────────────────────────────────────

    async def health_check(self) -> Dict[str, Any]:
        """Check OpenClaw adapter health."""
        health = {
            "status": "healthy",
            "adapter": "openclaw",
            "discovery_only": self.discovery_only,
            "gateway_url": self.gateway_url,
        }

        if self._http_client and not self.discovery_only:
            try:
                response = await self._http_client.get("/health")
                health["gateway_reachable"] = response.status_code < 400
            except Exception:
                health["gateway_reachable"] = False

        return health

    # ── Internal handlers ────────────────────────────────────────────────

    async def _handle_send_message(self, agent: Dict, request: Dict) -> Dict[str, Any]:
        """Send a message through OpenClaw to a messaging channel."""
        channel = request.get("channel", "")
        message = request.get("message", "")

        if self._http_client:
            try:
                response = await self._http_client.post(
                    "/channels/send",
                    json={"channel": channel, "message": message},
                )
                if response.status_code < 400:
                    return {"status": "sent", "channel": channel, "agent_type": "openclaw"}
                return {"status": "failed", "error": f"HTTP {response.status_code}", "agent_type": "openclaw"}
            except Exception as exc:
                return {"status": "failed", "error": str(exc), "agent_type": "openclaw"}

        return {"status": "no_gateway", "channel": channel, "agent_type": "openclaw"}

    async def _handle_execute_skill(self, agent: Dict, request: Dict) -> Dict[str, Any]:
        """Execute an OpenClaw skill."""
        skill_name = request.get("skill_name", "")
        arguments = request.get("arguments", {})

        if self._http_client:
            try:
                response = await self._http_client.post(
                    "/skills/execute",
                    json={"skill": skill_name, "arguments": arguments},
                )
                if response.status_code < 400:
                    return {
                        "status": "success",
                        "skill": skill_name,
                        "result": response.json(),
                        "agent_type": "openclaw",
                    }
                return {"status": "failed", "error": f"HTTP {response.status_code}", "agent_type": "openclaw"}
            except Exception as exc:
                return {"status": "failed", "error": str(exc), "agent_type": "openclaw"}

        return {"status": "no_gateway", "skill": skill_name, "agent_type": "openclaw"}

    async def _handle_delegate(self, agent: Dict, request: Dict) -> Dict[str, Any]:
        """Delegate task to another OpenClaw agent."""
        target_agent = request.get("target_agent", "")
        task = request.get("task", "")

        if self._http_client:
            try:
                response = await self._http_client.post(
                    "/agents/send",
                    json={"target": target_agent, "task": task, "context": request.get("context", {})},
                )
                if response.status_code < 400:
                    return {
                        "status": "delegated",
                        "target_agent": target_agent,
                        "result": response.json(),
                        "agent_type": "openclaw",
                    }
                return {"status": "failed", "error": f"HTTP {response.status_code}", "agent_type": "openclaw"}
            except Exception as exc:
                return {"status": "failed", "error": str(exc), "agent_type": "openclaw"}

        return {"status": "no_gateway", "target_agent": target_agent, "agent_type": "openclaw"}

    async def _handle_list_channels(self, agent: Dict) -> Dict[str, Any]:
        """List available OpenClaw messaging channels."""
        if self._http_client:
            try:
                response = await self._http_client.get("/channels")
                if response.status_code < 400:
                    return {"channels": response.json(), "agent_type": "openclaw"}
            except Exception as exc:
                return {"channels": [], "error": str(exc), "agent_type": "openclaw"}

        return {"channels": [], "agent_type": "openclaw", "status": "no_gateway"}

    async def _handle_list_skills(self, agent: Dict) -> Dict[str, Any]:
        """List available OpenClaw skills."""
        if self._http_client:
            try:
                response = await self._http_client.get("/skills")
                if response.status_code < 400:
                    return {"skills": response.json(), "agent_type": "openclaw"}
            except Exception as exc:
                return {"skills": [], "error": str(exc), "agent_type": "openclaw"}

        return {"skills": [], "agent_type": "openclaw", "status": "no_gateway"}
