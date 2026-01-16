"""LangChain agent adapter with cascading model source integration."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
import json
import uuid
import httpx

# Import the base AgenticAIAdapter
from adapters.agentic_ai_adapter import AgenticAIAdapter
from adapters.models import AdapterConfig, AdapterCategory, Edition, AdapterRequest
from adapters.registry import adapter_registry

# Try to import usage tracker
try:
    from core.usage_tracker import UsageTracker
    HAS_USAGE_TRACKER = True
except ImportError:
    HAS_USAGE_TRACKER = False
    UsageTracker = None

logger = logging.getLogger(__name__)

# Rate limiting for Community Edition
COMMUNITY_DAILY_LIMIT = 100
COMMUNITY_RATE_LIMIT_KEY = "langchain_community_rate_limit"


class CascadingLLM:
    """LangChain-compatible LLM that uses our cascading adapter system."""
    
    def __init__(self, cascade_order: List[str] = None, config: Dict[str, Any] = None):
        """Initialize with cascade order."""
        self.config = config or {}
        self.cascade_order = cascade_order or [
            "llm-service",      # Priority 1: Internal LLM Service
            "ollama",           # Priority 2: Local models
            "huggingface",      # Priority 3: Model registry
            "openai",           # Priority 4: Cloud providers
            "claude",
            "gemini",
            "mcp-client"        # Priority 5: External MCP (last resort)
        ]
        self.current_adapter = None
        self.failed_adapters = set()  # Track failed adapters to avoid retrying
    
    async def _get_working_adapter(self):
        """Get the first working adapter from cascade."""
        for adapter_name in self.cascade_order:
            if adapter_name in self.failed_adapters:
                continue  # Skip previously failed adapters
                
            try:
                # Get adapter class from registry
                adapter_class = adapter_registry.get_adapter_class(adapter_name)
                if adapter_class:
                    # Create adapter instance
                    adapter = adapter_class(self.config)
                    # Quick health check
                    try:
                        result = await adapter.health_check()
                        if result.get("status") == "healthy":
                            logger.info(f"Using {adapter_name} for LLM operations")
                            return adapter
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Adapter {adapter_name} not available: {e}")
                self.failed_adapters.add(adapter_name)
                continue
        
        raise RuntimeError("No working LLM adapter found in cascade")
    
    async def agenerate(self, prompts: List[str], **kwargs):
        """Async generate for LangChain compatibility."""
        if not self.current_adapter:
            self.current_adapter = await self._get_working_adapter()
        
        responses = []
        for prompt in prompts:
            try:
                result = await self.current_adapter.execute({
                    "operation": "generate",
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000)
                })
                
                # Extract text from result
                if result.success:
                    text = result.data.get("text") or result.data.get("output", "")
                    responses.append(text)
                else:
                    raise Exception(f"Generation failed: {result.error}")
                    
            except Exception as e:
                logger.warning(f"Generation failed with {self.current_adapter.config.name}, trying next: {e}")
                self.failed_adapters.add(self.current_adapter.config.name)
                self.current_adapter = None
                self.current_adapter = await self._get_working_adapter()
                
                # Retry with new adapter
                result = await self.current_adapter.execute({
                    "operation": "generate",
                    "prompt": prompt,
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 1000)
                })
                
                if result.success:
                    text = result.data.get("text") or result.data.get("output", "")
                    responses.append(text)
                else:
                    responses.append("")  # Fallback to empty string
        
        # Return in LangChain expected format
        from langchain.schema import LLMResult, Generation
        generations = [[Generation(text=text) for text in responses]]
        return LLMResult(generations=generations)
    
    def generate(self, prompts: List[str], **kwargs):
        """Sync generate for LangChain compatibility."""
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.agenerate(prompts, **kwargs))
    
    def __call__(self, prompt: str, **kwargs):
        """Make the LLM callable for LangChain."""
        result = self.generate([prompt], **kwargs)
        return result.generations[0][0].text


class LangChainAdapter(AgenticAIAdapter):
    """LangChain agent adapter implementation."""
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        if not config:
            config = AdapterConfig(
                name="langchain",
                category=AdapterCategory.AI_AGENT,
                version="1.0.0",
                description="LangChain AI agent framework adapter for Community Edition",
                required_edition=Edition.COMMUNITY,
                rate_limit_per_minute=10,  # 10 requests per minute for Community
                timeout_seconds=60,
                max_retries=3,
                retry_delay_seconds=1
            )
        super().__init__(config)
        self.agent_type = "langchain"
        self.daily_usage = {}  # Track daily usage in memory
        
        # Framework Service configuration
        self.framework_service_url = os.getenv("FRAMEWORK_SERVICE_URL", "http://ai-agent-framework-service:8004")
        self.framework_service_enabled = os.getenv("FRAMEWORK_SERVICE_ENABLED", "true").lower() == "true"
        self._framework_service_available = False
        self._langchain_available = False
        
    async def initialize(self) -> None:
        """Initialize the LangChain adapter."""
        try:
            # First check if Framework Service is available
            if self.framework_service_enabled:
                try:
                    async with httpx.AsyncClient(timeout=2.0) as client:
                        response = await client.get(f"{self.framework_service_url}/health")
                        if response.status_code == 200:
                            self._framework_service_available = True
                            logger.info("Framework Service is available for LangChain execution")
                except Exception as e:
                    logger.warning(f"Framework Service not available: {e}")
            
            # Check if LangChain is available locally (fallback)
            if not self._framework_service_available:
                try:
                    import langchain
                    from langchain.agents import AgentExecutor
                    from langchain.memory import ConversationBufferMemory
                    from langchain.schema import SystemMessage, HumanMessage
                    self._langchain_available = True
                    logger.info("LangChain library is available locally")
                except ImportError:
                    logger.warning("LangChain not installed locally and Framework Service unavailable")
                    self._langchain_available = False
                
        except Exception as e:
            logger.error(f"Failed to initialize LangChain adapter: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the LangChain adapter."""
        logger.info("Shutting down LangChain adapter")
    
    async def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create a LangChain agent instance with cascading model access."""
        if not self._langchain_available:
            # Try to use cascading LLM even without full LangChain
            try:
                cascading_llm = CascadingLLM(config=config)
                return {
                    "type": "cascading_llm_agent",
                    "llm": cascading_llm,
                    "config": config,
                    "created_at": datetime.utcnow().isoformat()
                }
            except:
                return {
                    "type": "mock_langchain_agent",
                    "config": config,
                    "created_at": datetime.utcnow().isoformat()
                }
        
        try:
            from langchain.agents import initialize_agent, AgentType
            from langchain.memory import ConversationBufferMemory
            
            # Create memory
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Use CascadingLLM instead of FakeListLLM
            # This will automatically try LLM Service, Ollama, HuggingFace, OpenAI, etc.
            cascade_order = config.get("cascade_order")  # Allow custom cascade order
            llm = CascadingLLM(cascade_order=cascade_order, config=config)
            
            # Create tools (empty for now, would be populated based on config)
            tools = []
            
            # Initialize agent
            agent = initialize_agent(
                tools=tools,
                llm=llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=memory,
                verbose=config.get("verbose", False),
                max_iterations=config.get("max_iterations", 3),  # Limited for Community
                handle_parsing_errors=True
            )
            
            return {
                "agent": agent,
                "memory": memory,
                "llm": llm,  # Include reference to cascading LLM
                "config": config,
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating LangChain agent: {str(e)}")
            # Return mock agent on error
            return {
                "type": "mock_langchain_agent",
                "config": config,
                "error": str(e),
                "created_at": datetime.utcnow().isoformat()
            }
    
    async def execute_agent(self, agent: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the LangChain agent with rate limiting for Community Edition."""
        start_time = datetime.utcnow()
        
        # Get IDs for tracking
        user_id = request.get("context", {}).get("user_id", "anonymous")
        tenant_id = request.get("context", {}).get("tenant_id", user_id)
        agent_id = agent.get("id", str(uuid.uuid4())) if isinstance(agent, dict) else str(uuid.uuid4())
        
        # Initialize usage tracker if available
        usage_tracker = None
        if HAS_USAGE_TRACKER and UsageTracker:
            try:
                usage_tracker = UsageTracker()
            except:
                pass
        
        # Check daily limit for Community Edition
        if not await self._check_daily_limit(user_id):
            # Track the failed request
            if usage_tracker:
                await usage_tracker.track_ai_agent_request(
                    tenant_id=tenant_id,
                    adapter_type="langchain",
                    agent_id=agent_id,
                    request_type="execute",
                    duration_ms=0,
                    success=False,
                    error="Daily limit exceeded"
                )
            
            return {
                "error": "Daily limit exceeded for Community Edition (100 requests/day)",
                "limit": COMMUNITY_DAILY_LIMIT,
                "usage": self._get_daily_usage(user_id),
                "reset_time": self._get_reset_time().isoformat()
            }
        
        try:
            # Update usage count
            self._increment_usage(user_id)
            
            input_text = request.get("input", "")
            
            if isinstance(agent, dict) and agent.get("type") == "cascading_llm_agent":
                # Use cascading LLM directly without full LangChain
                cascading_llm = agent.get("llm")
                if cascading_llm:
                    try:
                        output = await cascading_llm.agenerate([input_text])
                        text = output.generations[0][0].text
                        
                        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                        
                        # Track execution
                        if usage_tracker:
                            await usage_tracker.track_ai_agent_request(
                                tenant_id=tenant_id,
                                adapter_type="langchain",
                                agent_id=agent_id,
                                request_type="execute",
                                duration_ms=duration_ms,
                                success=True,
                                metadata={"cascading": True, "adapter_used": getattr(cascading_llm.current_adapter, 'config', {}).get('name', 'unknown')}
                            )
                        
                        return {
                            "output": text,
                            "duration_ms": duration_ms,
                            "model_source": getattr(cascading_llm.current_adapter, 'config', {}).get('name', 'unknown'),
                            "daily_usage": self._get_daily_usage(user_id),
                            "daily_limit": COMMUNITY_DAILY_LIMIT
                        }
                    except Exception as e:
                        logger.error(f"Cascading LLM execution failed: {e}")
                        # Fall through to mock response
                
            if isinstance(agent, dict) and agent.get("type") == "mock_langchain_agent":
                # Mock execution for when nothing works
                await asyncio.sleep(0.5)  # Simulate processing
                
                # Track mock execution
                if usage_tracker:
                    await usage_tracker.track_ai_agent_request(
                        tenant_id=tenant_id,
                        adapter_type="langchain",
                        agent_id=agent_id,
                        request_type="execute",
                        duration_ms=500,
                        success=True,
                        metadata={"mock": True, "input_length": len(input_text)}
                    )
                
                return {
                    "output": f"[Mock LangChain Response] Processed: {input_text}",
                    "duration_ms": 500,
                    "iterations": 1,
                    "tools_used": [],
                    "daily_usage": self._get_daily_usage(user_id),
                    "daily_limit": COMMUNITY_DAILY_LIMIT
                }
            
            # Real LangChain execution
            if self._langchain_available and "agent" in agent:
                langchain_agent = agent["agent"]
                
                # Execute with timeout
                try:
                    result = await asyncio.wait_for(
                        asyncio.to_thread(langchain_agent.run, input_text),
                        timeout=30  # 30 second timeout for Community
                    )
                    
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    # Track successful execution
                    if usage_tracker:
                        await usage_tracker.track_ai_agent_request(
                            tenant_id=tenant_id,
                            adapter_type="langchain",
                            agent_id=agent_id,
                            request_type="execute",
                            duration_ms=duration_ms,
                            success=True,
                            metadata={
                                "iterations": getattr(langchain_agent, "iterations", 1),
                                "input_length": len(input_text)
                            }
                        )
                    
                    return {
                        "output": result,
                        "duration_ms": duration_ms,
                        "iterations": getattr(langchain_agent, "iterations", 1),
                        "tools_used": [],  # Would extract from agent execution
                        "daily_usage": self._get_daily_usage(user_id),
                        "daily_limit": COMMUNITY_DAILY_LIMIT
                    }
                    
                except asyncio.TimeoutError:
                    error_msg = "Execution timeout (30s limit for Community Edition)"
                    if usage_tracker:
                        await usage_tracker.track_ai_agent_request(
                            tenant_id=tenant_id,
                            adapter_type="langchain",
                            agent_id=agent_id,
                            request_type="execute",
                            duration_ms=30000,
                            success=False,
                            error=error_msg
                        )
                    return {
                        "error": error_msg,
                        "duration_ms": 30000
                    }
            
            # Fallback
            return {
                "error": "Agent not properly initialized",
                "duration_ms": 0
            }
            
        except Exception as e:
            logger.error(f"Error executing LangChain agent: {str(e)}")
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Track error
            if usage_tracker:
                await usage_tracker.track_ai_agent_request(
                    tenant_id=tenant_id,
                    adapter_type="langchain",
                    agent_id=agent_id,
                    request_type="execute",
                    duration_ms=duration_ms,
                    success=False,
                    error=str(e)
                )
            
            return {
                "error": str(e),
                "duration_ms": duration_ms
            }
    
    async def get_agent_state(self, agent: Any) -> Dict[str, Any]:
        """Get current state of the LangChain agent."""
        try:
            if isinstance(agent, dict) and agent.get("type") == "mock_langchain_agent":
                return {
                    "type": "mock",
                    "config": agent.get("config", {}),
                    "created_at": agent.get("created_at"),
                    "memory": {"messages": [], "type": "mock"}
                }
            
            if self._langchain_available and "agent" in agent:
                langchain_agent = agent["agent"]
                memory = agent.get("memory")
                
                # Extract memory content
                memory_data = {}
                if memory and hasattr(memory, "chat_memory"):
                    messages = memory.chat_memory.messages
                    memory_data = {
                        "messages": [
                            {
                                "type": msg.__class__.__name__,
                                "content": msg.content
                            }
                            for msg in messages
                        ],
                        "message_count": len(messages)
                    }
                
                return {
                    "type": "langchain",
                    "config": agent.get("config", {}),
                    "created_at": agent.get("created_at"),
                    "memory": memory_data,
                    "max_iterations": getattr(langchain_agent, "max_iterations", 3)
                }
            
            return {"error": "Invalid agent state"}
            
        except Exception as e:
            logger.error(f"Error getting agent state: {str(e)}")
            return {"error": str(e)}
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform LangChain-specific health check."""
        health_data = {
            "langchain_available": self._langchain_available,
            "daily_limit": COMMUNITY_DAILY_LIMIT,
            "rate_limit_per_minute": self.config.rate_limit_per_minute
        }
        
        if self._langchain_available:
            try:
                import langchain
                health_data["langchain_version"] = langchain.__version__
            except:
                pass
        
        return health_data
    
    def _check_daily_limit(self, user_id: str) -> bool:
        """Check if user has exceeded daily limit."""
        usage = self._get_daily_usage(user_id)
        return usage < COMMUNITY_DAILY_LIMIT
    
    def _get_daily_usage(self, user_id: str) -> int:
        """Get current daily usage for user."""
        today = datetime.utcnow().date().isoformat()
        key = f"{user_id}:{today}"
        return self.daily_usage.get(key, 0)
    
    def _increment_usage(self, user_id: str) -> None:
        """Increment usage count for user."""
        today = datetime.utcnow().date().isoformat()
        key = f"{user_id}:{today}"
        self.daily_usage[key] = self.daily_usage.get(key, 0) + 1
        
        # Clean up old entries (simple memory management)
        if len(self.daily_usage) > 1000:
            # Keep only today's entries
            today_keys = [k for k in self.daily_usage.keys() if k.endswith(today)]
            self.daily_usage = {k: v for k, v in self.daily_usage.items() if k in today_keys}
    
    def _get_reset_time(self) -> datetime:
        """Get time when daily limit resets."""
        tomorrow = datetime.utcnow().date() + timedelta(days=1)
        return datetime.combine(tomorrow, datetime.min.time())
    
    def _calculate_cost(self, request: AdapterRequest, result: Dict[str, Any]) -> float:
        """Calculate cost for LangChain operations - free for Community Edition."""
        # Community Edition is free but rate-limited
        return 0.0
    
    def get_capabilities(self) -> List[Any]:
        """Return LangChain adapter capabilities for discovery."""
        from adapters.models import AdapterCapability
        
        # Return capabilities that work in discovery mode
        capabilities = [
            AdapterCapability(
                name="create_agent",
                description="Create a LangChain conversational agent with memory",
                required_parameters=["agent_type"],
                optional_parameters=["memory_type", "verbose", "max_iterations", "tools"]
            ),
            AdapterCapability(
                name="execute_task",
                description="Execute a task using LangChain agent with multi-step reasoning",
                required_parameters=["task"],
                optional_parameters=["max_steps", "temperature", "model"]
            ),
            AdapterCapability(
                name="manage_conversation",
                description="Manage conversational context and memory",
                required_parameters=["message"],
                optional_parameters=["session_id", "memory_key"]
            ),
            AdapterCapability(
                name="chain_operations",
                description="Chain multiple LLM operations together",
                required_parameters=["operations"],
                optional_parameters=["chain_type", "return_intermediate"]
            ),
            AdapterCapability(
                name="use_tools",
                description="Integrate and use external tools with the agent",
                required_parameters=["tools"],
                optional_parameters=["tool_selection_strategy"]
            )
        ]
        
        # In discovery mode, indicate these are available without configuration
        if hasattr(self.config, 'custom_config') and self.config.custom_config.get('discovery_only'):
            for cap in capabilities:
                cap.available_without_config = True
                cap.description += " (Available via cascading to configured LLMs)"
        
        return capabilities