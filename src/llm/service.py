"""Core LLM service for unified text generation."""

import logging
import hashlib
import httpx
from typing import List, Dict, Any, Optional, Tuple, AsyncIterator
from datetime import datetime

from .models import (
    ModelProvider, ModelTier, ModelInfo, UserLLMSettings,
    LLMRequest, LLMResponse, WorkflowStep, CostEstimate, UsageStats,
    ToolCall, ToolResult, LLMToolResponse, ToolDefinition, ToolRecoveryStrategy
)
from .model_selection import (
    select_model_for_task,
    get_model_config,
    estimate_complexity_hybrid,
    classify_model_tier,
    get_provider_from_model
)
from .generation import LLMGenerationEngine
from .caching import LLMCache
from .cost_tracking import CostTracker
from .tier_resolver import get_environment_default_model

logger = logging.getLogger(__name__)


# Providers that support native tool calling (via adapter ToolCallingMixin).
# HuggingFace does not — always uses text-based fallback.
# DeepSeek and DashScope also use text-based fallback (no adapter yet).
_NATIVE_TOOL_PROVIDER_NAMES = {
    "ollama", "anthropic", "openai", "azure_openai", "gemini",
    "vertex_ai", "bedrock", "cohere",
}


class _AdapterProvider:
    """Lazy factory for tool-calling adapter instances.

    Creates adapter instances on first use and caches them. This avoids
    timing issues with app.py lifespan startup — adapters are only created
    when tool calling is actually needed (during request handling, well
    after startup completes).
    """
    _instances: Dict[str, Any] = {}

    @classmethod
    async def get(cls, provider: 'ModelProvider') -> 'Optional[Any]':
        """Get or create a tool-calling adapter for the given provider.

        Returns the adapter if it implements ToolCallingMixin, else None.
        """
        from adapters.tool_calling import ToolCallingMixin

        key = provider.value
        if key not in cls._instances:
            adapter = cls._create_adapter(provider)
            if adapter is None:
                return None
            if not isinstance(adapter, ToolCallingMixin):
                logger.debug(f"Adapter for {key} does not implement ToolCallingMixin")
                return None
            try:
                await adapter.initialize()
                cls._instances[key] = adapter
                logger.info(f"Initialized tool-calling adapter for {key}")
            except Exception as e:
                logger.warning(f"Failed to initialize tool-calling adapter for {key}: {e}")
                return None
        return cls._instances.get(key)

    @classmethod
    def _create_adapter(cls, provider: 'ModelProvider') -> 'Optional[Any]':
        """Create an adapter instance for the given provider."""
        import os
        from adapters.models import AdapterConfig, AdapterCategory

        try:
            if provider == ModelProvider.OLLAMA:
                from adapters.implementations.ai.ollama_adapter import OllamaAdapter
                return OllamaAdapter(AdapterConfig(
                    name="ollama-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    base_url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
                    credentials={},
                    timeout_seconds=300.0,
                ))
            elif provider == ModelProvider.ANTHROPIC:
                from adapters.implementations.ai.claude_adapter import ClaudeAdapter
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
                if not api_key:
                    return None
                return ClaudeAdapter(AdapterConfig(
                    name="claude-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    api_key=api_key,
                    credentials={"api_key": api_key},
                    timeout_seconds=120.0,
                ))
            elif provider == ModelProvider.OPENAI:
                from adapters.implementations.ai.openai_adapter import OpenAIAdapter
                api_key = os.environ.get("OPENAI_API_KEY", "")
                if not api_key:
                    return None
                return OpenAIAdapter(AdapterConfig(
                    name="openai-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    api_key=api_key,
                    credentials={"api_key": api_key},
                    timeout_seconds=120.0,
                ))
            elif provider == ModelProvider.AZURE_OPENAI:
                try:
                    from business_adapters.implementations.ai.azure_openai_adapter import AzureOpenAIAdapter
                except ImportError:
                    return None
                api_key = os.environ.get("AZURE_OPENAI_API_KEY", "")
                endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "")
                if not api_key or not endpoint:
                    return None
                return AzureOpenAIAdapter(AdapterConfig(
                    name="azure-openai-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    api_key=api_key,
                    base_url=endpoint,
                    credentials={"api_key": api_key, "endpoint": endpoint},
                    timeout_seconds=120.0,
                ))
            elif provider == ModelProvider.GEMINI:
                try:
                    from business_adapters.implementations.ai.google_gemini_adapter import GoogleGeminiAdapter
                except ImportError:
                    return None
                api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
                if not api_key:
                    return None
                return GoogleGeminiAdapter(AdapterConfig(
                    name="gemini-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    api_key=api_key,
                    credentials={"api_key": api_key},
                    timeout_seconds=120.0,
                ))
            elif provider == ModelProvider.VERTEX_AI:
                try:
                    from business_adapters.implementations.ai.vertex_ai_adapter import VertexAIAdapter
                except ImportError:
                    return None
                project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_AI_PROJECT")
                if not project_id:
                    return None
                return VertexAIAdapter(AdapterConfig(
                    name="vertex-ai-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    credentials={"project_id": project_id},
                    timeout_seconds=120.0,
                ))
            elif provider == ModelProvider.BEDROCK:
                try:
                    from business_adapters.implementations.ai.aws_bedrock_adapter import AWSBedrockAdapter
                except ImportError:
                    return None
                return AWSBedrockAdapter(AdapterConfig(
                    name="bedrock-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    credentials={},
                    timeout_seconds=120.0,
                ))
            elif provider == ModelProvider.COHERE:
                try:
                    from business_adapters.implementations.ai.cohere_adapter import CohereAdapter
                except ImportError:
                    return None
                api_key = os.environ.get("COHERE_API_KEY", "")
                if not api_key:
                    return None
                return CohereAdapter(AdapterConfig(
                    name="cohere-tools",
                    version="1.0.0",
                    category=AdapterCategory.AI,
                    api_key=api_key,
                    credentials={"api_key": api_key},
                    timeout_seconds=120.0,
                ))
            else:
                return None
        except Exception as e:
            logger.warning(f"Failed to create adapter for {provider.value}: {e}")
            return None

    @classmethod
    def reset(cls):
        """Reset all cached adapter instances (for testing)."""
        cls._instances.clear()


class LLMService:
    """
    Unified LLM service that manages all text generation.
    
    This service:
    - Provides a single interface for all LLM operations
    - Uses existing adapters from Community/Business/Enterprise editions
    - Manages model selection based on task and user preferences
    - Handles caching and cost tracking
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern for LLM service."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the LLM service."""
        if not self._initialized:
            self.generation_engine = LLMGenerationEngine()
            self.cache = LLMCache()
            self.cost_tracker = CostTracker()
            self._initialized = True
            logger.info("LLM Service initialized")
    
    async def generate(
        self,
        prompt: str,
        user_settings: Optional[UserLLMSettings] = None,
        model_override: Optional[str] = None,
        task_type: str = "general",
        complexity: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: Optional[bool] = None,
        system_prompt: Optional[str] = None,
        response_format: str = "text",
        schema: Optional[Dict[str, Any]] = None,
        cache_key: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> LLMResponse:
        """
        Generate text using the best available LLM.
        
        Model Selection Priority:
        1. model_override (explicit override)
        2. user_settings.selected_model (from UI settings)
        3. context['mcp_preferred_model'] (MCP tool preference)
        4. Auto-selection based on task_type and complexity
        
        Args:
            prompt: The prompt to generate from
            user_settings: User's LLM preferences from UI
            model_override: Explicit model override
            task_type: Type of task for model selection
            complexity: Override complexity detection
            temperature: Override temperature setting
            max_tokens: Override max tokens
            stream: Whether to stream the response
            system_prompt: System prompt to use
            response_format: Output format (text/json/structured)
            schema: Schema for structured output
            cache_key: Cache key for response caching
            context: Additional context (e.g., MCP preferences)
            
        Returns:
            LLMResponse with generated text and metadata
        """
        # Create request object
        request = LLMRequest(
            prompt=prompt,
            user_settings=user_settings,
            model_override=model_override,
            task_type=task_type,
            complexity=complexity,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            system_prompt=system_prompt,
            response_format=response_format,
            schema=schema,
            cache_key=cache_key,
            context=context
        )
        
        # Check cache
        if cache_key:
            cached = await self.cache.get(cache_key)
            if cached:
                logger.info(f"Cache hit for key: {cache_key}")
                cached.cache_hit = True
                return cached
        
        # Generate response
        response = await self.generation_engine.generate(request)
        
        # Cache response if requested
        if cache_key:
            await self.cache.set(cache_key, response)
        
        # Track usage
        await self.cost_tracker.track_usage(
            user_id=user_settings.user_id if user_settings else "anonymous",
            model=response.model_used,
            provider=response.provider,
            tokens=response.tokens_used,
            cost=response.cost
        )
        
        return response
    
    async def generate_with_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        user_settings: Optional[UserLLMSettings] = None,
        model_override: Optional[str] = None,
        task_type: str = "tool_use",
        temperature: Optional[float] = 0.3,  # Lower for reliable tool selection
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        tool_choice: str = "auto",  # "auto", "required", or specific tool name
        context: Optional[Dict[str, Any]] = None,
        messages: Optional[List[Dict[str, Any]]] = None  # Multi-turn conversation
    ) -> LLMToolResponse:
        """
        Generate text with tool calling support.

        This method uses a HYBRID APPROACH:
        1. For providers with native tool calling (Ollama llama3.1/3.2, Claude, OpenAI, etc.)
           → Uses constrained API-level tool calling (deterministic, reliable)
        2. For providers without native support
           → Falls back to text-based JSON parsing (less reliable)

        Args:
            prompt: The user's request or query (can be None if messages provided)
            tools: List of ToolDefinition objects available for the LLM to use
            user_settings: User's LLM preferences from UI
            model_override: Explicit model override
            task_type: Type of task (default: "tool_use")
            temperature: Temperature for generation (default: 0.3 for reliability)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt override
            tool_choice: "auto" (LLM decides), "required" (must use tool), or tool name
            context: Additional context for generation
            messages: Optional multi-turn conversation history. When provided,
                used instead of wrapping prompt in a single message. Enables
                iterative agent loops where tool results feed back into the
                conversation. Format: [{"role": "system"|"user"|"assistant"|"tool", "content": ...}]

        Returns:
            LLMToolResponse with either text response or tool_calls to execute
        """
        import json
        import uuid
        from datetime import datetime

        start_time = datetime.utcnow()

        # Determine which model we'll use
        temp_request = LLMRequest(
            prompt=prompt or "",
            user_settings=user_settings,
            model_override=model_override,
            task_type=task_type
        )
        model, tier = await self.generation_engine._select_model(temp_request)
        provider = get_provider_from_model(model)

        logger.info(f"generate_with_tools: Using model={model}, provider={provider.value}")

        # Check if this provider/model supports native tool calling
        native_supported = self._supports_native_tools(provider, model)
        logger.info(f"Native tool calling supported: {native_supported}")

        if native_supported:
            # Use native tool calling - deterministic and reliable
            return await self._generate_with_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                provider=provider,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt or "You are an AI assistant that helps users accomplish tasks.",
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        else:
            # Fallback to text-based tool calling
            logger.info("Using text-based tool calling fallback")
            return await self._generate_with_text_tools(
                prompt=prompt,
                tools=tools,
                user_settings=user_settings,
                model_override=model_override,
                task_type=task_type,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                context=context,
                start_time=start_time,
                messages=messages
            )

    def _supports_native_tools(self, provider: ModelProvider, model: str) -> bool:
        """Check if a provider supports native tool calling via adapter."""
        return provider.value in _NATIVE_TOOL_PROVIDER_NAMES

    async def _generate_with_native_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        model: str,
        provider: ModelProvider,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using native tool calling APIs via adapter framework."""
        try:
            adapter = await _AdapterProvider.get(provider)
            if not adapter:
                logger.info(f"No adapter for {provider.value} - using text-based fallback")
                return await self._generate_with_text_tools(
                    prompt=prompt, tools=tools, user_settings=user_settings,
                    model_override=model, task_type="tool_use", temperature=temperature,
                    max_tokens=max_tokens, system_prompt=system_prompt,
                    tool_choice=tool_choice, context=None, start_time=start_time,
                    messages=messages
                )

            tc_request = self._build_tool_calling_request(
                prompt=prompt, tools=tools, model=model,
                temperature=temperature, max_tokens=max_tokens,
                system_prompt=system_prompt, tool_choice=tool_choice,
                messages=messages
            )
            response = await adapter.chat_with_tools(tc_request)
            return await self._to_llm_tool_response(
                response, provider=provider, tier=tier,
                start_time=start_time, tools=tools, tool_choice=tool_choice,
                model=model, user_settings=user_settings
            )
        except Exception as e:
            logger.error(f"{provider.value} native tool calling failed: {e}")
            logger.info("Falling back to text-based tool calling")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time,
                messages=messages
            )

    # ── Adapter bridge helpers ────────────────────────────────────────────

    def _build_tool_calling_request(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        model: str,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> 'ToolCallingRequest':
        """Build a ToolCallingRequest from LLMService parameters."""
        from adapters.tool_calling import ToolCallingRequest

        # Build messages list
        if messages:
            api_messages = list(messages)
        else:
            api_messages = []
            if system_prompt:
                api_messages.append({"role": "system", "content": system_prompt})
            if prompt:
                api_messages.append({"role": "user", "content": prompt})

        # Convert ToolDefinition objects to JSON Schema dicts
        tool_dicts = []
        for tool in tools:
            tool_dicts.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters if tool.parameters else {
                    "type": "object", "properties": {}, "required": []
                }
            })

        return ToolCallingRequest(
            messages=api_messages,
            tools=tool_dicts,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tool_choice=tool_choice,
        )

    async def _to_llm_tool_response(
        self,
        response: 'ToolCallingResponse',
        provider: ModelProvider,
        tier: ModelTier,
        start_time: datetime,
        tools: List[ToolDefinition],
        tool_choice: str,
        model: str,
        user_settings: Optional[UserLLMSettings],
    ) -> LLMToolResponse:
        """Convert a ToolCallingResponse to LLMToolResponse and track usage."""
        # Convert tool_calls dicts to ToolCall objects
        tool_calls = None
        if response.tool_calls:
            tool_calls = [
                ToolCall(
                    name=tc["name"],
                    arguments=tc.get("arguments", {}),
                    id=tc.get("id")
                )
                for tc in response.tool_calls
            ]

        tool_response = LLMToolResponse(
            text=response.text,
            tool_calls=tool_calls if tool_calls else None,
            execution_plan=None,
            model_used=model,
            provider=provider,
            tier=tier,
            tokens_used=response.tokens_used,
            cost=response.cost,
            response_time=(datetime.utcnow() - start_time).total_seconds(),
            metadata={
                "tools_provided": len(tools),
                "tool_choice": tool_choice,
                "native_tools": True,
                "tool_calls_count": len(response.tool_calls),
                "stop_reason": response.stop_reason,
                "cached_tokens": response.cached_tokens,
            }
        )

        await self.cost_tracker.track_usage(
            user_id=user_settings.user_id if user_settings else "anonymous",
            model=model,
            provider=provider,
            tokens=response.tokens_used,
            cost=response.cost
        )

        return tool_response

    async def _generate_with_text_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        user_settings: Optional[UserLLMSettings],
        model_override: Optional[str],
        task_type: str,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: Optional[str],
        tool_choice: str,
        context: Optional[Dict[str, Any]],
        start_time: datetime,
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using text-based tool calling (fallback for unsupported providers)."""
        import json
        import uuid

        # If messages provided, concatenate into a single prompt string
        effective_prompt = prompt or ""
        if messages:
            parts = []
            for msg in messages:
                role = msg["role"].title()
                content = msg.get("content") or ""
                if msg["role"] == "tool":
                    parts.append(f"Tool Result ({msg.get('name', 'unknown')}): {content}")
                elif msg["role"] == "system":
                    # System content is handled via system_prompt below
                    system_prompt = content
                else:
                    parts.append(f"{role}: {content}")
            effective_prompt = "\n\n".join(parts)

        # Build the tool-aware system prompt
        tools_description = self._build_tools_prompt(tools)
        full_system_prompt = self._build_tool_system_prompt(
            system_prompt or "You are an AI assistant that helps users accomplish tasks.",
            tools_description,
            tool_choice
        )

        # Create request with structured output expectations
        request = LLMRequest(
            prompt=effective_prompt,
            user_settings=user_settings,
            model_override=model_override,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens or 2000,
            system_prompt=full_system_prompt,
            response_format="json",
            context=context
        )

        # Generate response
        response = await self.generation_engine.generate(request)

        # Parse the response to extract tool calls
        tool_calls, text_response, execution_plan = self._parse_tool_response(
            response.text, tools
        )

        # Build and return LLMToolResponse
        tool_response = LLMToolResponse(
            text=text_response,
            tool_calls=tool_calls if tool_calls else None,
            execution_plan=execution_plan,
            model_used=response.model_used,
            provider=response.provider,
            tier=response.tier,
            tokens_used=response.tokens_used,
            cost=response.cost,
            response_time=(datetime.utcnow() - start_time).total_seconds(),
            metadata={
                "tools_provided": len(tools),
                "tool_choice": tool_choice,
                "native_tools": False,
                "raw_response": response.text[:500] if response.text else None
            }
        )

        # Track usage
        await self.cost_tracker.track_usage(
            user_id=user_settings.user_id if user_settings else "anonymous",
            model=response.model_used,
            provider=response.provider,
            tokens=response.tokens_used,
            cost=response.cost
        )

        return tool_response

    def _build_tools_prompt(self, tools: List[ToolDefinition]) -> str:
        """Build a text description of available tools for the prompt."""
        if not tools:
            return "No tools are available."

        # Build list of exact tool names for strict validation
        tool_names = [tool.name for tool in tools]

        lines = [
            "Available tools:",
            "",
            f"IMPORTANT: You may ONLY use these exact tool names: {tool_names}",
            "Do NOT invent or guess tool names. If none of these tools match the request, use 'direct_response'.",
            ""
        ]
        for tool in tools:
            lines.append(tool.to_prompt_format())
            lines.append("")  # Blank line between tools

        return "\n".join(lines)

    def _build_tool_system_prompt(
        self,
        base_prompt: str,
        tools_description: str,
        tool_choice: str
    ) -> str:
        """Build the complete system prompt for tool-enabled generation."""
        tool_instruction = ""
        if tool_choice == "required":
            tool_instruction = "You MUST use one of the available tools to respond."
        elif tool_choice == "auto":
            tool_instruction = "You may use tools when appropriate, or respond directly."
        else:
            # Specific tool name
            tool_instruction = f"You should use the '{tool_choice}' tool if applicable."

        return f"""{base_prompt}

{tools_description}

{tool_instruction}

When you decide to use a tool, respond with a JSON object in this format:
{{
    "action": "tool_call",
    "tool_name": "name_of_tool",
    "tool_arguments": {{ ... arguments ... }},
    "execution_plan": "Brief description of what you're doing and why"
}}

If you want to call multiple tools in sequence, use:
{{
    "action": "tool_chain",
    "tools": [
        {{"tool_name": "first_tool", "tool_arguments": {{...}}}},
        {{"tool_name": "second_tool", "tool_arguments": {{...}}}}
    ],
    "execution_plan": "I'll first do X, then Y..."
}}

If no tool is needed, respond with:
{{
    "action": "direct_response",
    "message": "Your response to the user"
}}

Always respond with valid JSON."""

    def _parse_tool_response(
        self,
        response_text: str,
        tools: List[ToolDefinition]
    ) -> Tuple[Optional[List[ToolCall]], Optional[str], Optional[str]]:
        """
        Parse the LLM response to extract tool calls.

        Validates that tool names match the provided tools list to prevent
        hallucinated tool names from causing errors.

        Returns:
            Tuple of (tool_calls, text_response, execution_plan)
        """
        import json
        import uuid

        if not response_text:
            return None, None, None

        # Build set of valid tool names for validation
        valid_tool_names = {tool.name for tool in tools} if tools else set()

        def _validate_tool_calls(tool_calls: List[ToolCall]) -> List[ToolCall]:
            """Filter out tool calls with invalid/hallucinated tool names."""
            valid_calls = []
            for tc in tool_calls:
                if tc.name in valid_tool_names:
                    valid_calls.append(tc)
                else:
                    logger.warning(f"Filtered out hallucinated tool name: {tc.name}")
            return valid_calls

        # Try to parse as JSON
        try:
            # Extract JSON from potential markdown code blocks
            json_text = self._extract_json_from_text(response_text)
            if json_text:
                data = json_text
            else:
                data = json.loads(response_text)

            # Handle case where extracted JSON is a list (not a dict with action field)
            if isinstance(data, list):
                # If it's a list, treat as tool chain format
                # Check if list items look like tool calls
                if data and isinstance(data[0], dict) and ('tool_name' in data[0] or 'name' in data[0]):
                    # Convert to tool_chain format
                    tool_calls = []
                    for item in data:
                        tool_name = item.get('tool_name') or item.get('name')
                        tool_args = item.get('tool_arguments') or item.get('arguments', {})
                        if tool_name:
                            tool_calls.append(ToolCall(
                                name=tool_name,
                                arguments=tool_args,
                                id=str(uuid.uuid4())
                            ))
                    # Validate tool names before returning
                    valid_calls = _validate_tool_calls(tool_calls)
                    if valid_calls:
                        return valid_calls, None, None
                    # All tools were invalid - return as direct response
                    return None, response_text, None
                # Otherwise treat as direct response
                return None, response_text, None

            action = data.get("action", "direct_response")
            execution_plan = data.get("execution_plan")

            # First, check for presence of tools array (regardless of action value)
            # This handles malformed JSON where LLM duplicates the action key
            if "tools" in data and isinstance(data["tools"], list) and len(data["tools"]) > 0:
                # Multiple tool calls
                tool_calls = []
                for tool_data in data["tools"]:
                    if isinstance(tool_data, dict):
                        tool_name = tool_data.get("tool_name") or tool_data.get("name")
                        tool_args = tool_data.get("tool_arguments") or tool_data.get("arguments", {})
                        if tool_name:
                            tool_call = ToolCall(
                                name=tool_name,
                                arguments=tool_args,
                                id=str(uuid.uuid4())
                            )
                            tool_calls.append(tool_call)

                # Validate tool names before returning
                valid_calls = _validate_tool_calls(tool_calls)
                if valid_calls:
                    return valid_calls, None, execution_plan
                # All tools were invalid
                return None, None, execution_plan

            # Check for single tool call
            if "tool_name" in data:
                tool_name = data.get("tool_name")
                tool_args = data.get("tool_arguments", {})

                if tool_name:
                    tool_call = ToolCall(
                        name=tool_name,
                        arguments=tool_args,
                        id=str(uuid.uuid4())
                    )
                    # Validate the single tool call
                    valid_calls = _validate_tool_calls([tool_call])
                    if valid_calls:
                        return valid_calls, None, execution_plan
                    # Invalid tool name
                    return None, None, execution_plan

            # Check explicit action field
            if action == "tool_call":
                # Single tool call (already handled above, but keep for clarity)
                tool_name = data.get("tool_name")
                tool_args = data.get("tool_arguments", {})

                if tool_name:
                    tool_call = ToolCall(
                        name=tool_name,
                        arguments=tool_args,
                        id=str(uuid.uuid4())
                    )
                    # Validate the single tool call
                    valid_calls = _validate_tool_calls([tool_call])
                    if valid_calls:
                        return valid_calls, None, execution_plan
                    # Invalid tool name
                    return None, None, execution_plan

            elif action == "direct_response" or action not in ["tool_call", "tool_chain"]:
                # Direct text response, no tool call
                message = data.get("message", response_text)
                return None, message, None

        except json.JSONDecodeError:
            # Response is not JSON, treat as direct text response
            pass

        # Fallback: treat as direct response
        return None, response_text, None

    async def generate_workflow_steps(
        self,
        prompt: str,
        user_settings: Optional[UserLLMSettings] = None,
        context: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None
    ) -> List[WorkflowStep]:
        """
        Generate workflow steps from natural language.
        
        Args:
            prompt: Natural language description of workflow
            user_settings: User's LLM preferences
            context: Additional context
            model: Optional model override
            
        Returns:
            List of workflow steps
        """
        # Build domain-aware system prompt with Human+AI collaboration guidance
        industry_context = ""
        if context and context.get("industry"):
            industry_context = f" Focus on {context['industry']} industry terminology and processes."

        system_prompt = f"""You are a workflow generation expert for AICtrlNet, a Human+AI collaboration platform.{industry_context}

CRITICAL - DOMAIN-SPECIFIC NODE NAMES WITH AI CAPABILITIES:
Create step names that are SPECIFIC to the domain while leveraging AI/ML capabilities.

GOOD examples (domain name + AI capability):
- For email marketing:
  * "Audience Segmentation" (AI Agent: ML clustering for customer segments)
  * "Content Personalization" (AI Agent: NLP for personalized messaging)
  * "Campaign Strategy Review" (Human Agent: Marketing manager approval)
  * "Campaign Performance Analysis" (AI Agent: Predictive analytics)

- For invoice processing:
  * "Invoice Data Extraction" (AI Agent: OCR + NLP extraction)
  * "Validation & Anomaly Detection" (AI Agent: ML-based validation)
  * "Finance Manager Approval" (Human Agent: High-value approvals)
  * "Payment Processing" (Integration: ERP/payment system)

BAD examples (DO NOT USE):
- "Document Processing Intelligence AI Agent" - generic, not domain-specific
- "AI Agent 1", "Process Node", "node_1" - meaningless
- "Step 1", "Process Data" - not descriptive

AVAILABLE NODE TYPES:

Basic Control Flow:
- "start" - Workflow entry point (auto-added)
- "end" - Workflow exit point (auto-added)
- "decision" - Conditional branching (CREATES MULTIPLE PATHS)
- "parallel" - Split into parallel execution paths
- "merge" - Join parallel paths back together

AI & Processing:
- "ai_agent" - AI-powered processing (ML, NLP, predictions)
- "human_agent" - Human review/approval required
- "process" - Generic processing step
- "transform" - Data transformation
- "dataSource" - Fetch data from source

Integration:
- "integration" - Connect to external systems
- "apiCall" - Make API calls
- "notification" - Send notifications

Approval & Human-in-Loop:
- "approval" - Formal approval workflow

BRANCHING WORKFLOWS (Critical for real-world processes):
When using "decision" nodes, ALWAYS specify the conditional branches:

Example - Decision with branches:
{{
  "label": "Approval Routing",
  "type": "decision",
  "condition": "amount > 10000",
  "branches": [
    {{"condition": "amount > 10000", "label": "High Value", "target": "Manager Review"}},
    {{"condition": "amount <= 10000", "label": "Standard", "target": "Auto Approve"}}
  ]
}}

Example - Parallel execution:
{{
  "label": "Parallel Processing",
  "type": "parallel",
  "branches": ["Validate Data", "Check Compliance", "Notify Stakeholders"]
}}

WORKFLOW STRUCTURE REQUIREMENTS:
1. Generate 5-12 workflow steps
2. Include at least ONE decision node with branches for non-trivial workflows
3. Use parallel nodes when multiple independent tasks can run simultaneously
4. Include human checkpoints at critical decision points

Node type distribution:
- 35-45% AI Agent nodes (automation, analysis, predictions)
- 15-20% Human Agent nodes (approvals, reviews)
- 15-20% Integration nodes (external systems, APIs)
- 10-15% Decision/routing nodes (with branches!)
- 5-10% Data nodes (dataSource, transform)

HUMAN+AI COLLABORATION (Critical):
- Include both AI and Human agent steps in workflows
- Human steps can be automated later as AI confidence increases
- Pattern: AI does analysis → Human validates critical decisions
- Examples:
  * "AI Pre-Screening" → "Human Final Review" (AI filters, human decides)
  * "AI Risk Scoring" → "Manager Override" (AI recommends, human can override)

OUTPUT FORMAT for each step:
- label: Domain-specific action name (e.g., "Audience Segmentation") - REQUIRED
- type: One of the types listed above - REQUIRED
- description: What this step does
- ai_capability: What AI/ML powers this (if ai_agent type)
- can_automate: true/false - if human_agent, can it be replaced by AI?
- branches: Array of branch definitions (REQUIRED for decision/parallel types!)
- condition: Condition expression (for decision type)
- dependencies: Array of step labels this step depends on (for non-linear flows)"""

        response = await self.generate(
            prompt=prompt,
            user_settings=user_settings,
            model_override=model,
            task_type="workflow_generation",
            context=context,
            response_format="structured",
            system_prompt=system_prompt
        )
        
        # Extract steps from response
        if response.metadata.get("steps"):
            # Use our robust parsing for metadata steps too
            return self._parse_json_steps(response.metadata["steps"])
        
        # Parse steps from text if not in metadata
        return self._parse_workflow_steps(response.text)
    
    async def generate_structured(
        self,
        prompt: str,
        schema: Dict[str, Any],
        model: Optional[str] = None,
        examples: Optional[List[Dict]] = None,
        user_settings: Optional[UserLLMSettings] = None
    ) -> Dict[str, Any]:
        """
        Generate structured output matching a schema.
        
        Args:
            prompt: The prompt to generate from
            schema: JSON schema or Pydantic model schema
            model: Optional model override
            examples: Optional examples for few-shot learning
            user_settings: User's LLM preferences
            
        Returns:
            Structured data matching the schema
        """
        # Build enhanced prompt with schema
        enhanced_prompt = self._build_structured_prompt(prompt, schema, examples)
        
        response = await self.generate(
            prompt=enhanced_prompt,
            user_settings=user_settings,
            model_override=model,
            task_type="structured_generation",
            response_format="json",
            schema=schema,
            system_prompt="Generate JSON output that matches the provided schema exactly."
        )
        
        # Parse and validate JSON with robust extraction
        import json
        import re

        text = response.text

        # Step 1: Strip markdown code fences if present
        if '```' in text:
            # Extract content between code fences
            code_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', text)
            if code_block_match:
                text = code_block_match.group(1).strip()
            else:
                # Remove opening fence if no closing fence
                text = re.sub(r'^```(?:json)?\s*', '', text).strip()

        # Step 2: Ensure text starts with { and ends with }
        # LLM sometimes returns just the inner content without outer braces
        text = text.strip()
        if text and not text.startswith('{'):
            text = '{' + text
        if text and not text.endswith('}'):
            text = text + '}'

        # Step 3: Try json_repair library first (most robust)
        try:
            from json_repair import repair_json
            repaired = repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                logger.debug(f"json_repair successfully parsed structured output")
                return repaired
        except ImportError:
            logger.debug("json_repair library not available")
        except Exception as e:
            logger.debug(f"json_repair failed: {e}")

        # Step 4: Try standard JSON parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON from response: {response.text[:200]}")
            return {}
    
    async def get_available_models(self) -> List[ModelInfo]:
        """
        Get all available models across all providers.
        
        Returns:
            List of available models with their information
        """
        return await self.generation_engine.get_available_models()
    
    async def estimate_cost(
        self,
        prompt: str,
        model: str
    ) -> CostEstimate:
        """
        Estimate cost before generation.
        
        Args:
            prompt: The prompt to estimate cost for
            model: The model to use
            
        Returns:
            Cost estimate with token count and price
        """
        return await self.generation_engine.estimate_cost(prompt, model)
    
    async def get_usage_stats(
        self,
        user_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> UsageStats:
        """
        Get token usage and cost statistics.
        
        Args:
            user_id: Optional user ID to filter by
            date_range: Optional date range to filter by
            
        Returns:
            Usage statistics
        """
        return await self.cost_tracker.get_stats(user_id, date_range)
    
    def _parse_workflow_steps(self, text: str) -> List[WorkflowStep]:
        """
        Parse workflow steps from text with robust JSON extraction.

        Args:
            text: Text containing workflow steps (JSON or plain text)

        Returns:
            List of parsed workflow steps
        """
        import json
        import re

        # Handle None or empty text
        if not text:
            logger.warning("_parse_workflow_steps received empty or None text")
            return []

        # First, try to extract JSON from the text
        json_data = self._extract_json_from_text(text)
        
        if json_data:
            # Handle JSON response
            if isinstance(json_data, list):
                return self._parse_json_steps(json_data)
            elif isinstance(json_data, dict):
                # Check for common wrapper keys
                if "steps" in json_data:
                    return self._parse_json_steps(json_data["steps"])
                elif "workflow" in json_data:
                    return self._parse_json_steps(json_data["workflow"])
                elif "tasks" in json_data:
                    return self._parse_json_steps(json_data["tasks"])
                else:
                    # Single step as dict
                    return self._parse_json_steps([json_data])
        
        # Fallback to plain text parsing
        steps = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Remove numbering if present
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
            
            # Try to extract label and description
            if ':' in line:
                label, description = line.split(':', 1)
                steps.append(WorkflowStep(
                    label=label.strip(),
                    description=description.strip()
                ))
            else:
                steps.append(WorkflowStep(
                    label=line,
                    description=""
                ))
        
        return steps
    
    def _extract_json_from_text(self, text: str) -> Optional[Any]:
        """Extract JSON from text that might contain markdown or other formatting."""
        import json
        import re
        
        # Try direct JSON parsing first
        try:
            return json.loads(text.strip())
        except:
            pass
        
        # Try to extract from markdown code blocks
        patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\[[\s\S]*\]',  # Just the array
            r'\{[\s\S]*\}'   # Just an object
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except:
                    continue
        
        return None
    
    def _parse_json_steps(self, steps_data: List[Any]) -> List[WorkflowStep]:
        """Parse JSON steps into WorkflowStep objects with branching support."""
        steps = []

        for i, step_data in enumerate(steps_data):
            if isinstance(step_data, dict):
                # Extract fields with fallbacks
                label = (
                    step_data.get('label') or
                    step_data.get('name') or
                    step_data.get('title') or
                    step_data.get('action') or
                    f"Step {i+1}"
                )

                description = (
                    step_data.get('description') or
                    step_data.get('details') or
                    step_data.get('text') or
                    ""
                )

                # Extract node_type with fallbacks
                node_type = (
                    step_data.get('node_type') or
                    step_data.get('type') or
                    step_data.get('nodeType') or
                    None
                )

                # Extract branches array for decision/parallel nodes
                branches = (
                    step_data.get('branches') or
                    step_data.get('options') or  # Alternative key
                    step_data.get('paths') or    # Alternative key
                    None
                )

                # Extract condition expression
                condition = (
                    step_data.get('condition') or
                    step_data.get('conditional') or
                    step_data.get('when') or
                    None
                )

                # Extract capability
                capability = (
                    step_data.get('capability') or
                    step_data.get('function') or
                    None
                )

                steps.append(WorkflowStep(
                    label=str(label),
                    description=str(description),
                    action=step_data.get('action'),
                    input_data=step_data.get('input_data') or step_data.get('inputs'),
                    output_schema=step_data.get('output_schema') or step_data.get('outputs'),
                    dependencies=step_data.get('dependencies', []),
                    agent=step_data.get('agent'),
                    template=step_data.get('template'),
                    node_type=node_type,
                    branches=branches,
                    condition=condition,
                    capability=capability
                ))
            elif isinstance(step_data, str):
                # Plain string step
                steps.append(WorkflowStep(
                    label=step_data,
                    description=""
                ))

        return steps
    
    def _build_structured_prompt(
        self,
        prompt: str,
        schema: Dict[str, Any],
        examples: Optional[List[Dict]] = None
    ) -> str:
        """
        Build an enhanced prompt for structured generation.
        
        Args:
            prompt: Original prompt
            schema: Target schema
            examples: Optional examples
            
        Returns:
            Enhanced prompt with schema and examples
        """
        import json
        
        enhanced = f"{prompt}\n\n"
        enhanced += f"Generate JSON output matching this schema:\n```json\n{json.dumps(schema, indent=2)}\n```\n"
        
        if examples:
            enhanced += "\nExamples:\n"
            for i, example in enumerate(examples[:3], 1):
                enhanced += f"\nExample {i}:\n```json\n{json.dumps(example, indent=2)}\n```\n"
        
        enhanced += "\nGenerate valid JSON only, no additional text."
        
        return enhanced


# Global instance for easy access
llm_service = LLMService()