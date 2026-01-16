"""AutoGPT agent adapter with cascading model source integration."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from adapters.agentic_ai_adapter import AgenticAIAdapter
from adapters.models import AdapterConfig, AdapterCategory, Edition
from adapters.registry import adapter_registry

logger = logging.getLogger(__name__)


class CascadingAutoGPTLLM:
    """AutoGPT-compatible LLM that uses our cascading adapter system."""
    
    def __init__(self, cascade_order: List[str] = None):
        """Initialize with cascade order."""
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
        self.failed_adapters = set()
    
    async def _get_working_adapter(self):
        """Get the first working adapter from cascade."""
        for adapter_name in self.cascade_order:
            if adapter_name in self.failed_adapters:
                continue
            
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
                            logger.info(f"AutoGPT using {adapter_name} for LLM operations")
                            return adapter
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Adapter {adapter_name} not available for AutoGPT: {e}")
                self.failed_adapters.add(adapter_name)
                continue
        
        raise RuntimeError("No working LLM adapter found for AutoGPT")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using cascading adapters."""
        if not self.current_adapter:
            self.current_adapter = await self._get_working_adapter()
        
        try:
            result = await self.current_adapter.execute({
                "operation": "generate",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            })
            
            if result.success:
                return result.data.get("text") or result.data.get("output", "")
            else:
                raise Exception(f"Generation failed: {result.error}")
        
        except Exception as e:
            logger.warning(f"AutoGPT generation failed with {self.current_adapter.config.name}, trying next: {e}")
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
                return result.data.get("text") or result.data.get("output", "")
            else:
                return ""  # Fallback to empty string
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat using cascading adapters."""
        if not self.current_adapter:
            self.current_adapter = await self._get_working_adapter()
        
        try:
            result = await self.current_adapter.execute({
                "operation": "chat",
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            })
            
            if result.success:
                return result.data.get("text") or result.data.get("output", "")
            else:
                raise Exception(f"Chat failed: {result.error}")
        
        except Exception as e:
            logger.warning(f"AutoGPT chat failed with {self.current_adapter.config.name}, trying next: {e}")
            self.failed_adapters.add(self.current_adapter.config.name)
            self.current_adapter = None
            self.current_adapter = await self._get_working_adapter()
            
            # Retry with new adapter
            result = await self.current_adapter.execute({
                "operation": "chat",
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000)
            })
            
            if result.success:
                return result.data.get("text") or result.data.get("output", "")
            else:
                return ""  # Fallback to empty string


class AutoGPTAdapter(AgenticAIAdapter):
    """AutoGPT agent adapter implementation."""
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize AutoGPT adapter."""
        if not config:
            config = AdapterConfig(
                name="autogpt",
                type="autogpt",
                category=AdapterCategory.AI_AGENT,
                edition=Edition.COMMUNITY
            )
        super().__init__(config)
        self._autogpt_available = self._check_autogpt_available()
        
    def _check_autogpt_available(self) -> bool:
        """Check if AutoGPT is available."""
        try:
            # Check for AutoGPT installation
            import autogpt
            return True
        except ImportError:
            return False
    
    async def initialize(self) -> None:
        """Initialize the AutoGPT adapter."""
        self.status = "ready"
        logger.info("AutoGPT adapter initialized")
    
    async def shutdown(self):
        """Clean shutdown of AutoGPT adapter."""
        logger.info("Shutting down AutoGPT adapter")
    
    async def get_agent_state(self, agent: Any) -> Dict[str, Any]:
        """Get current state of the AutoGPT agent."""
        if isinstance(agent, dict):
            return {
                "type": agent.get("type", "unknown"),
                "status": "active",
                "memory": agent.get("memory", {}),
                "config": agent.get("config", {})
            }
        return {"status": "unknown"}
    
    def get_capabilities(self) -> List[Any]:
        """Return AutoGPT adapter capabilities."""
        from adapters.models import AdapterCapability
        return [
            AdapterCapability(
                name="create_agent",
                description="Create an AutoGPT agent",
                required_parameters=["config"],
                optional_parameters=["goals", "constraints"]
            ),
            AdapterCapability(
                name="execute_agent",
                description="Execute AutoGPT agent",
                required_parameters=["agent", "input"],
                optional_parameters=["max_iterations"]
            )
        ]
    
    async def execute(self, request: Any) -> Any:
        """Execute a request using AutoGPT adapter."""
        from adapters.models import AdapterRequest, AdapterResponse
        
        # Handle different request types
        if hasattr(request, 'operation'):
            operation = request.operation
        elif isinstance(request, dict):
            operation = request.get('operation', 'execute')
        else:
            operation = 'execute'
        
        if operation == 'create_agent':
            agent = await self.create_agent(request.parameters if hasattr(request, 'parameters') else request)
            return AdapterResponse(
                success=True,
                data=agent,
                adapter_id=self.id
            )
        elif operation == 'execute_agent':
            params = request.parameters if hasattr(request, 'parameters') else request
            result = await self.execute_agent(params.get('agent'), params)
            return AdapterResponse(
                success=not result.get('error'),
                data=result,
                adapter_id=self.id,
                error=result.get('error')
            )
        else:
            return AdapterResponse(
                success=False,
                error=f"Unknown operation: {operation}",
                adapter_id=self.id
            )
    
    async def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create an AutoGPT agent instance with cascading model access."""
        if not self._autogpt_available:
            # Try to use cascading LLM even without full AutoGPT
            try:
                cascading_llm = CascadingAutoGPTLLM()
                return {
                    "type": "cascading_autogpt_agent",
                    "llm": cascading_llm,
                    "config": config,
                    "created_at": datetime.utcnow().isoformat()
                }
            except:
                return {
                    "type": "mock_autogpt_agent",
                    "config": config,
                    "created_at": datetime.utcnow().isoformat()
                }
        
        try:
            # Use cascading LLM for AutoGPT
            llm = CascadingAutoGPTLLM()
            
            # Create AutoGPT-style agent with cascading LLM
            return {
                "type": "autogpt_agent",
                "llm": llm,
                "name": config.get("name", "AutoGPT Agent"),
                "role": config.get("role", "Assistant"),
                "goals": config.get("goals", ["Help the user"]),
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create AutoGPT agent: {e}")
            return {
                "type": "error",
                "error": str(e)
            }
    
    async def execute_agent(self, agent: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an AutoGPT agent task."""
        start_time = datetime.utcnow()
        
        if isinstance(agent, dict) and agent.get("type") == "error":
            return {
                "error": agent.get("error", "Agent creation failed"),
                "status": "failed"
            }
        
        # Handle execution request
        operation = request.get("operation", "execute")
        
        if operation not in ["execute", "chat", "generate"]:
            return {
                "error": f"Unsupported operation: {operation}",
                "status": "failed"
            }
        
        if isinstance(agent, dict) and agent.get("type") == "cascading_autogpt_agent":
            # Use cascading LLM directly
            cascading_llm = agent.get("llm")
            if cascading_llm:
                try:
                    # AutoGPT-style execution with cascading LLM
                    input_text = request.get("input", "")
                    output = await cascading_llm.generate(input_text)
                    
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    return {
                        "output": output,
                        "duration_ms": duration_ms,
                        "model_source": getattr(cascading_llm.current_adapter, 'config', {}).get('name', 'unknown'),
                        "agent_type": "autogpt"
                    }
                except Exception as e:
                    logger.error(f"Cascading AutoGPT execution failed: {e}")
                    # Fall through to mock response
        
        if isinstance(agent, dict) and agent.get("type") == "mock_autogpt_agent":
            # Mock execution for when nothing works
            await asyncio.sleep(0.5)  # Simulate processing
            
            return {
                "output": f"[AutoGPT Mock] Processing: {request.get('input', 'No input')}",
                "duration_ms": 500,
                "agent_type": "autogpt",
                "mock": True
            }
        
        # Real AutoGPT execution would go here
        return {
            "error": "AutoGPT execution not implemented",
            "status": "failed"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check AutoGPT adapter health."""
        return {
            "status": "healthy",
            "autogpt_available": self._autogpt_available,
            "adapter": "autogpt"
        }