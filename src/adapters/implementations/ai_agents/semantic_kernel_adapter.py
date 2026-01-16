"""Semantic Kernel agent adapter with cascading model source integration."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from adapters.agentic_ai_adapter import AgenticAIAdapter
from adapters.models import AdapterConfig, AdapterCategory, Edition
from adapters.registry import adapter_registry

logger = logging.getLogger(__name__)


class CascadingSemanticKernelLLM:
    """Semantic Kernel-compatible LLM that uses our cascading adapter system."""
    
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
        self.kernel_config = {}
    
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
                            logger.info(f"Semantic Kernel using {adapter_name} for LLM operations")
                            return adapter
                    except:
                        pass
            except Exception as e:
                logger.debug(f"Adapter {adapter_name} not available for Semantic Kernel: {e}")
                self.failed_adapters.add(adapter_name)
                continue
        
        raise RuntimeError("No working LLM adapter found for Semantic Kernel")
    
    async def invoke_semantic_function(self, function_name: str, context: Dict[str, Any], **kwargs) -> str:
        """Invoke a semantic function using cascading adapters."""
        if not self.current_adapter:
            self.current_adapter = await self._get_working_adapter()
        
        # Build prompt from semantic function and context
        prompt = self._build_semantic_prompt(function_name, context)
        
        try:
            result = await self.current_adapter.execute({
                "operation": "generate",
                "prompt": prompt,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1000),
                "top_p": kwargs.get("top_p", 0.9)
            })
            
            if result.success:
                return result.data.get("text") or result.data.get("output", "")
            else:
                raise Exception(f"Semantic function failed: {result.error}")
        
        except Exception as e:
            logger.warning(f"Semantic Kernel invocation failed with {self.current_adapter.config.name}, trying next: {e}")
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
    
    def _build_semantic_prompt(self, function_name: str, context: Dict[str, Any]) -> str:
        """Build prompt for semantic function."""
        # Semantic Kernel style prompt building
        prompt_parts = [f"Function: {function_name}"]
        
        if "input" in context:
            prompt_parts.append(f"Input: {context['input']}")
        
        if "variables" in context:
            for key, value in context["variables"].items():
                prompt_parts.append(f"{key}: {value}")
        
        prompt_parts.append("Output:")
        return "\n".join(prompt_parts)


class SemanticKernelAdapter(AgenticAIAdapter):
    """Semantic Kernel agent adapter implementation."""
    
    def __init__(self, config: Optional[AdapterConfig] = None):
        """Initialize Semantic Kernel adapter."""
        if not config:
            config = AdapterConfig(
                name="semantic-kernel",
                type="semantic_kernel",
                category=AdapterCategory.AI_AGENT,
                edition=Edition.COMMUNITY
            )
        super().__init__(config)
        self._sk_available = self._check_sk_available()
        
    def _check_sk_available(self) -> bool:
        """Check if Semantic Kernel is available."""
        try:
            # Check for Semantic Kernel installation
            import semantic_kernel
            return True
        except ImportError:
            return False
    
    async def shutdown(self):
        """Clean shutdown of Semantic Kernel adapter."""
        logger.info("Shutting down Semantic Kernel adapter")
    
    async def create_agent(self, config: Dict[str, Any]) -> Any:
        """Create a Semantic Kernel agent instance with cascading model access."""
        if not self._sk_available:
            # Try to use cascading LLM even without full Semantic Kernel
            try:
                cascading_llm = CascadingSemanticKernelLLM()
                return {
                    "type": "cascading_sk_agent",
                    "llm": cascading_llm,
                    "config": config,
                    "created_at": datetime.utcnow().isoformat()
                }
            except:
                return {
                    "type": "mock_sk_agent",
                    "config": config,
                    "created_at": datetime.utcnow().isoformat()
                }
        
        try:
            # Use cascading LLM for Semantic Kernel
            llm = CascadingSemanticKernelLLM()
            
            # Create SK-style agent with cascading LLM
            return {
                "type": "sk_agent",
                "llm": llm,
                "skills": config.get("skills", []),
                "plugins": config.get("plugins", []),
                "memory": config.get("memory", {}),
                "planner": config.get("planner", None),
                "created_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create Semantic Kernel agent: {e}")
            return {
                "type": "error",
                "error": str(e)
            }
    
    async def execute_agent(self, agent: Any, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a Semantic Kernel agent task."""
        start_time = datetime.utcnow()
        
        if isinstance(agent, dict) and agent.get("type") == "error":
            return {
                "error": agent.get("error", "Agent creation failed"),
                "status": "failed"
            }
        
        # Handle execution request
        operation = request.get("operation", "execute")
        
        if operation not in ["execute", "invoke", "run"]:
            return {
                "error": f"Unsupported operation: {operation}",
                "status": "failed"
            }
        
        if isinstance(agent, dict) and agent.get("type") == "cascading_sk_agent":
            # Use cascading LLM directly
            cascading_llm = agent.get("llm")
            if cascading_llm:
                try:
                    # Semantic Kernel style execution
                    function_name = request.get("function", "default")
                    context = {
                        "input": request.get("input", ""),
                        "variables": request.get("variables", {})
                    }
                    
                    output = await cascading_llm.invoke_semantic_function(function_name, context)
                    
                    duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                    
                    return {
                        "output": output,
                        "duration_ms": duration_ms,
                        "model_source": getattr(cascading_llm.current_adapter, 'config', {}).get('name', 'unknown'),
                        "agent_type": "semantic_kernel"
                    }
                except Exception as e:
                    logger.error(f"Cascading Semantic Kernel execution failed: {e}")
                    # Fall through to mock response
        
        if isinstance(agent, dict) and agent.get("type") == "mock_sk_agent":
            # Mock execution for when nothing works
            await asyncio.sleep(0.5)  # Simulate processing
            
            return {
                "output": f"[Semantic Kernel Mock] Processing: {request.get('input', 'No input')}",
                "duration_ms": 500,
                "agent_type": "semantic_kernel",
                "mock": True
            }
        
        # Real Semantic Kernel execution would go here
        return {
            "error": "Semantic Kernel execution not implemented",
            "status": "failed"
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check Semantic Kernel adapter health."""
        return {
            "status": "healthy",
            "sk_available": self._sk_available,
            "adapter": "semantic_kernel"
        }
    
    def get_capabilities(self) -> List[Any]:
        """Return Semantic Kernel adapter capabilities for discovery."""
        from adapters.models import AdapterCapability
        
        # Return capabilities that work in discovery mode
        capabilities = [
            AdapterCapability(
                name="create_kernel",
                description="Create a Semantic Kernel instance with plugins and memory",
                required_parameters=["kernel_config"],
                optional_parameters=["plugins", "memory_store", "planner_type"]
            ),
            AdapterCapability(
                name="execute_semantic_function",
                description="Execute a semantic function with prompt engineering",
                required_parameters=["function_name", "input"],
                optional_parameters=["context", "settings", "skill_name"]
            ),
            AdapterCapability(
                name="orchestrate_plan",
                description="Create and execute multi-step plans using SK planner",
                required_parameters=["goal"],
                optional_parameters=["max_steps", "planner_type", "available_functions"]
            ),
            AdapterCapability(
                name="manage_plugins",
                description="Load and manage Semantic Kernel plugins/skills",
                required_parameters=["plugin_name"],
                optional_parameters=["plugin_path", "plugin_config"]
            ),
            AdapterCapability(
                name="memory_operations",
                description="Store and retrieve information from semantic memory",
                required_parameters=["operation", "data"],
                optional_parameters=["collection", "relevance_threshold"]
            )
        ]
        
        # In discovery mode, indicate these are available without configuration
        if hasattr(self.config, 'custom_config') and self.config.custom_config.get('discovery_only'):
            for cap in capabilities:
                cap.available_without_config = True
                cap.description += " (Available via cascading to configured LLMs)"
        
        return capabilities