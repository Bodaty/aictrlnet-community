"""Basic agent execution service for Community Edition.

Provides simple agent execution using the LLM module for consistency.
Limited to fast tier models and basic agents.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import get_settings
from services.agent_config_service import AgentConfigService
from services.llm_helpers import get_user_llm_settings, get_system_llm_settings
from llm import llm_service, UserLLMSettings

settings = get_settings()
logger = logging.getLogger(__name__)


class BasicAgentExecutor:
    """Basic agent executor for Community Edition."""

    def __init__(self):
        self.config_service = AgentConfigService()
        self.execution_count = {}  # Track daily executions

    async def execute_agent(
        self,
        db: AsyncSession,
        user_id: str,
        agent_name: str,
        task: Dict[str, Any],
        mcp_tools: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Execute a basic agent task.
        
        Community Edition limitations:
        - Only fast tier models
        - No memory/context
        - No framework integration
        - Daily execution limits
        """
        start_time = datetime.utcnow()
        
        # Check if user can execute this agent
        if not await self.config_service.can_execute_agent(db, user_id, agent_name):
            return {
                "success": False,
                "error": "Agent not available or not configured",
                "upgrade_prompt": {
                    "message": "Unlock all 33 agents with Business Edition",
                    "url": "/upgrade/business"
                }
            }
        
        # Check daily limit (100 executions for Community)
        today = datetime.utcnow().date().isoformat()
        user_key = f"{user_id}:{today}"
        
        if user_key not in self.execution_count:
            self.execution_count[user_key] = 0
            
        if self.execution_count[user_key] >= 100:
            return {
                "success": False,
                "error": "Daily execution limit reached (100)",
                "upgrade_prompt": {
                    "message": "Get unlimited executions with Business Edition",
                    "url": "/upgrade/business"
                }
            }

        # Cleanup stale date keys (keep only today)
        stale_keys = [k for k in self.execution_count if not k.endswith(f":{today}")]
        for k in stale_keys:
            del self.execution_count[k]

        # Prepare prompt for the agent
        prompt = self._prepare_prompt(agent_name, task)

        # Use LLM service for execution with user preference resolution
        user_settings = await get_user_llm_settings(
            db=db,
            user_id=user_id,
            temperature=0.3,  # Lower temperature for more consistent responses
            max_tokens=500,  # Limit response size for Community
            stream_responses=False
        )

        context = {
            "agent": agent_name,
            "task_type": task.get("type", "general"),
            "edition": "community"
        }

        # Execute using LLM service
        try:
            if mcp_tools:
                # Tool-aware execution: convert MCP tools and use generate_with_tools
                from llm.models import ToolDefinition
                tool_definitions = []
                handler_map = {}
                for tool in mcp_tools:
                    td = ToolDefinition(
                        name=tool["name"],
                        description=tool.get("description", ""),
                        parameters=tool.get("parameters", {}),
                        category="mcp"
                    )
                    tool_definitions.append(td)
                    if "handler" in tool and callable(tool["handler"]):
                        handler_map[tool["name"]] = tool["handler"]

                result = await llm_service.generate_with_tools(
                    prompt=prompt,
                    tools=tool_definitions,
                    user_settings=user_settings,
                    context=context
                )

                # Execute any tool calls the LLM made (single-round)
                tool_results = []
                if result.tool_calls:
                    for tc in result.tool_calls:
                        handler = handler_map.get(tc.name)
                        if handler:
                            try:
                                tool_result = await handler(tc.arguments)
                                tool_results.append({"tool": tc.name, "result": tool_result})
                            except Exception as e:
                                tool_results.append({"tool": tc.name, "error": str(e)})

                output_text = result.text or ""
                if not output_text and tool_results:
                    output_text = f"Executed {len(tool_results)} MCP tool(s)"

                self.execution_count[user_key] += 1
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                return {
                    "success": True,
                    "result": {
                        "output": output_text,
                        "model": result.model_used or "unknown",
                        "tokens": result.tokens_used or 0
                    },
                    "mcp_tools_used": [tr["tool"] for tr in tool_results],
                    "tool_results": tool_results,
                    "agent": agent_name,
                    "execution_time_ms": execution_time,
                    "remaining_executions": 100 - self.execution_count[user_key],
                    "edition": "community"
                }
            else:
                # Standard execution without tools
                result = await llm_service.generate(
                    prompt=prompt,
                    user_settings=user_settings,
                    context=context
                )

                self.execution_count[user_key] += 1
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)

                return {
                    "success": True,
                    "result": {
                        "output": result.text if hasattr(result, 'text') else str(result),
                        "model": result.model_used if hasattr(result, 'model_used') else "unknown",
                        "tokens": result.tokens_used if hasattr(result, 'tokens_used') else 0
                    },
                    "agent": agent_name,
                    "execution_time_ms": execution_time,
                    "remaining_executions": 100 - self.execution_count[user_key],
                    "edition": "community"
                }

        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return {
                "success": False,
                "error": "All connection attempts failed",
                "agent": agent_name
            }
    
    # DEPRECATED: Now using LLM module for all executions
    async def _execute_ollama_deprecated(
        self,
        agent_name: str,
        task: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute using Ollama (local, free)."""
        ollama_url = config.get("ollama_url", "http://localhost:11434")
        
        # Prepare prompt based on agent type
        prompt = self._prepare_prompt(agent_name, task)

        # Use configured default model (fallback for direct Ollama calls)
        model = settings.DEFAULT_LLM_MODEL
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 500  # Limit response size
                    }
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"Ollama error: {response.text}")
            
            result = response.json()
            return {
                "output": result.get("response", ""),
                "model_used": model,
                "provider": "ollama"
            }
    
    # DEPRECATED: Now using LLM module for all executions
    async def _execute_openai_deprecated(
        self,
        agent_name: str,
        task: Dict[str, Any],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute using OpenAI API."""
        api_key = config.get("api_key")
        
        if not api_key:
            raise ValueError("OpenAI API key not configured")
        
        # Prepare prompt
        prompt = self._prepare_prompt(agent_name, task)
        
        # Use cheapest model for Community
        model = "gpt-3.5-turbo"
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": self._get_agent_system_prompt(agent_name)},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 500  # Limit for Community
                }
            )
            
            if response.status_code != 200:
                raise Exception(f"OpenAI error: {response.text}")
            
            result = response.json()
            return {
                "output": result["choices"][0]["message"]["content"],
                "model_used": model,
                "provider": "openai",
                "usage": result.get("usage", {})
            }
    
    def _prepare_prompt(self, agent_name: str, task: Dict[str, Any]) -> str:
        """Prepare prompt based on agent and task."""
        task_type = task.get("type", "general")
        description = task.get("description", "")
        parameters = task.get("parameters", {})
        
        if agent_name == "basic_nlp":
            return f"Analyze the following text and {task_type}: {description}"
        elif agent_name == "basic_workflow":
            return f"Create a simple workflow for: {description}"
        elif agent_name == "basic_assistant":
            return f"Help with the following task: {description}"
        else:
            return description
    
    def _get_agent_system_prompt(self, agent_name: str) -> str:
        """Get system prompt for agent."""
        prompts = {
            "basic_nlp": "You are an NLP assistant. Analyze text and extract key information.",
            "basic_workflow": "You are a workflow designer. Create simple, clear workflows.",
            "basic_assistant": "You are a helpful AI assistant. Provide clear, concise answers."
        }
        return prompts.get(agent_name, "You are a helpful AI assistant.")
    
    async def get_agent_status(
        self,
        db: AsyncSession,
        user_id: str
    ) -> Dict[str, Any]:
        """Get status of available agents for user."""
        config = await self.config_service.get_user_agent_config(db, user_id)
        enabled_agents = config.get("enabled_agents", ["basic_nlp"])
        
        today = datetime.utcnow().date().isoformat()
        user_key = f"{user_id}:{today}"
        executions = self.execution_count.get(user_key, 0)
        
        agents_status = {}
        for agent in self.config_service.ALLOWED_AGENTS:
            agents_status[agent] = {
                "available": agent in enabled_agents,
                "enabled": agent in enabled_agents,
                "description": self._get_agent_description(agent)
            }
        
        return {
            "agents": agents_status,
            "executions_today": executions,
            "remaining_executions": max(0, 100 - executions),
            "daily_limit": 100,
            "enabled_count": len(enabled_agents),
            "max_agents": self.config_service.MAX_AGENTS,
            "edition": "community",
            "upgrade_benefits": self.config_service.get_upgrade_benefits()
        }
    
    def _get_agent_description(self, agent_name: str) -> str:
        """Get description for agent."""
        descriptions = {
            "basic_nlp": "Natural language processing and text analysis",
            "basic_workflow": "Simple workflow generation",
            "basic_assistant": "General AI assistance"
        }
        return descriptions.get(agent_name, "AI agent")


# Module-level singleton — shared across all callers for rate limit enforcement
_basic_executor = BasicAgentExecutor()


class AgentExecutionService:
    """Config load/save/execute wrapper for MCP agent endpoints."""

    def __init__(self, db: AsyncSession, user_id: str = None):
        self.db = db
        self.user_id = user_id
        self._executor = _basic_executor

    async def get_agent_config(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Load per-agent MCP config from User.preferences."""
        from models.user import User

        if not self.user_id:
            return {"agent_id": agent_id, "tools": [], "metadata": {}}

        result = await self.db.execute(
            select(User).where(User.id == self.user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            return None

        prefs = user.preferences or {}
        config = prefs.get("agent_mcp_configs", {}).get(agent_id, {})

        return {
            "agent_id": agent_id,
            "tools": config.get("tools", []),
            "metadata": config.get("metadata", {}),
        }

    async def update_agent_config(self, agent_id: str, config: Dict[str, Any]) -> None:
        """Persist agent config. Strips non-serializable handler functions from tools."""
        from models.user import User

        if not self.user_id:
            return

        result = await self.db.execute(
            select(User).where(User.id == self.user_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            return

        if not user.preferences:
            user.preferences = {}
        if "agent_mcp_configs" not in user.preferences:
            user.preferences["agent_mcp_configs"] = {}

        serializable_tools = [
            {k: v for k, v in tool.items() if k != "handler"}
            for tool in config.get("tools", [])
        ]

        user.preferences["agent_mcp_configs"][agent_id] = {
            "tools": serializable_tools,
            "metadata": config.get("metadata", {}),
        }

        from sqlalchemy.orm.attributes import flag_modified
        flag_modified(user, "preferences")
        await self.db.commit()

    async def execute_agent(
        self,
        agent_id: str,
        task: Dict[str, Any],
        config_override: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Execute agent via BasicAgentExecutor."""
        mcp_tools = None
        if config_override and config_override.get("tools"):
            mcp_tools = config_override["tools"]

        return await self._executor.execute_agent(
            db=self.db,
            user_id=self.user_id or "",
            agent_name=agent_id,
            task=task,
            mcp_tools=mcp_tools
        )