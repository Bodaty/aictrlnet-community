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


# Provider support for native tool calling
NATIVE_TOOL_PROVIDERS = {
    "anthropic": True,
    "openai": True,
    "azure_openai": True,
    "google_gemini": True,
    "vertex_ai": True,
    "aws_bedrock": True,  # Claude models support native, Titan uses text fallback
    "cohere": True,  # Command R+ supports native, older Command models use text fallback
    "deepseek": True,  # DeepSeek V3/R1 support native tool calling
    "dashscope": True,  # Qwen models support tool calling
    "ollama": "model_dependent",  # Check model name (llama3.1, llama3.2, mistral, mixtral, qwen2.5)
    "huggingface": False,  # No native tool support - always text-based fallback
}

# Ollama models that support native tool calling.
# Most modern Ollama models support tools — always try native first and
# fall back to text-based if it fails (_generate_with_ollama_native_tools
# already has a text-based fallback on exception).
OLLAMA_NATIVE_TOOL_MODELS = "all"  # Legacy sentinel; see _supports_native_tools


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
        """Check if a provider/model combination supports native tool calling."""
        provider_name = provider.value.lower()

        # Check provider support
        support = NATIVE_TOOL_PROVIDERS.get(provider_name, False)

        if support == True:
            return True
        elif support == "model_dependent":
            # Ollama: always try native tool calling first.
            # Most modern models (llama3.x, qwen, deepseek, mistral, etc.) support it.
            # If the model doesn't, _generate_with_ollama_native_tools catches the
            # exception and falls back to text-based tool calling automatically.
            return True
        else:
            return False

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
        """Generate using native tool calling APIs - routes to provider-specific implementations."""
        import json
        import uuid

        # Route to provider-specific native tool calling implementation
        if provider == ModelProvider.OLLAMA:
            return await self._generate_with_ollama_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        elif provider == ModelProvider.ANTHROPIC:
            return await self._generate_with_anthropic_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        elif provider == ModelProvider.OPENAI:
            return await self._generate_with_openai_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        elif provider == ModelProvider.AZURE_OPENAI:
            return await self._generate_with_azure_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        elif provider == ModelProvider.GEMINI:
            return await self._generate_with_gemini_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        elif provider == ModelProvider.VERTEX_AI:
            return await self._generate_with_vertex_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        elif provider == ModelProvider.AWS_BEDROCK:
            return await self._generate_with_bedrock_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        elif provider == ModelProvider.COHERE:
            return await self._generate_with_cohere_native_tools(
                prompt=prompt,
                tools=tools,
                model=model,
                tier=tier,
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                start_time=start_time,
                user_settings=user_settings,
                messages=messages
            )
        else:
            # Fallback to text-based for unknown providers
            logger.info(f"No native tool implementation for {provider.value} - using text-based fallback")
            return await self._generate_with_text_tools(
                prompt=prompt,
                tools=tools,
                user_settings=user_settings,
                model_override=model,
                task_type="tool_use",
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                context=None,
                start_time=start_time,
                messages=messages
            )

    async def _generate_with_ollama_native_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using Ollama's native tool calling API."""
        import json
        import uuid

        ollama_url = self.generation_engine.ollama_url

        # Convert tools to Ollama format
        ollama_tools = []
        for tool in tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters if tool.parameters else {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            ollama_tools.append(ollama_tool)

        # Build messages array — use provided messages or construct from prompt
        if messages:
            api_messages = []
            for msg in messages:
                if msg["role"] == "tool":
                    # Ollama uses role=tool with content
                    api_messages.append({
                        "role": "tool",
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant" and msg.get("tool_calls"):
                    # Reconstruct assistant message with tool_calls
                    api_messages.append({
                        "role": "assistant",
                        "content": msg.get("content") or "",
                        "tool_calls": [
                            {"function": {"name": tc["name"], "arguments": tc["arguments"]}}
                            for tc in msg["tool_calls"]
                        ]
                    })
                else:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

        # Build request payload
        payload = {
            "model": model,
            "messages": api_messages,
            "tools": ollama_tools,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or 2000
            }
        }

        logger.info(f"Calling Ollama with native tools: {[t['function']['name'] for t in ollama_tools]}")

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    f"{ollama_url}/api/chat",
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"Ollama error: {response.status_code} - {response.text}")
                    raise Exception(f"Ollama API error: {response.status_code}")

                result = response.json()

            # Extract tool calls from response
            tool_calls = []
            text_response = ""
            message = result.get("message", {})

            # Check for tool_calls in the response
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    function = tc.get("function", {})
                    tool_name = function.get("name")
                    tool_args = function.get("arguments", {})

                    # Arguments might be a string that needs JSON parsing
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    if tool_name:
                        tool_calls.append(ToolCall(
                            name=tool_name,
                            arguments=tool_args,
                            id=str(uuid.uuid4())
                        ))
                        logger.info(f"Native tool call extracted: {tool_name}")

            # Extract text content
            if "content" in message:
                text_response = message["content"] or ""

            # Calculate tokens
            tokens_used = result.get("eval_count", 0) + result.get("prompt_eval_count", 0)

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=model,
                provider=ModelProvider.OLLAMA,
                tier=tier,
                tokens_used=tokens_used,
                cost=0.0,  # Local models are free
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls)
                }
            )

            # Track usage
            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=model,
                provider=ModelProvider.OLLAMA,
                tokens=tokens_used,
                cost=0.0
            )

            return tool_response

        except Exception as e:
            logger.error(f"Ollama native tool calling failed: {e}")
            # Fall back to text-based
            logger.info("Falling back to text-based tool calling")
            return await self._generate_with_text_tools(
                prompt=prompt,
                tools=tools,
                user_settings=user_settings,
                model_override=model,
                task_type="tool_use",
                temperature=temperature,
                max_tokens=max_tokens,
                system_prompt=system_prompt,
                tool_choice=tool_choice,
                context=None,
                start_time=start_time
            )

    async def _generate_with_anthropic_native_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using Anthropic/Claude's native tool calling API."""
        import json
        import uuid
        import os

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not set, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        # Convert tools to Anthropic format
        anthropic_tools = []
        for tool in tools:
            anthropic_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters if tool.parameters else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            anthropic_tools.append(anthropic_tool)

        # Build messages — Anthropic uses separate 'system' field
        if messages:
            api_messages = []
            system_text = system_prompt
            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                elif msg["role"] == "tool":
                    # Anthropic format: role=user with tool_result content blocks
                    api_messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_use_id", ""), "content": msg["content"]}]
                    })
                elif msg["role"] == "assistant" and msg.get("tool_calls"):
                    # Anthropic format: assistant with tool_use content blocks
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    for tc in msg["tool_calls"]:
                        content.append({"type": "tool_use", "id": tc.get("id", ""), "name": tc["name"], "input": tc["arguments"]})
                    api_messages.append({"role": "assistant", "content": content})
                else:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            api_messages = [{"role": "user", "content": prompt}]
            system_text = system_prompt

        # Build request payload
        payload = {
            "model": model,
            "max_tokens": max_tokens or 4096,
            "system": system_text,
            "messages": api_messages,
            "tools": anthropic_tools if tools else [],
            "temperature": temperature
        }

        # Handle tool_choice
        if tool_choice == "required":
            payload["tool_choice"] = {"type": "any"}
        elif tool_choice != "auto":
            payload["tool_choice"] = {"type": "tool", "name": tool_choice}

        logger.info(f"Calling Anthropic with native tools: {[t['name'] for t in anthropic_tools]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    },
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"Anthropic error: {response.status_code} - {response.text}")
                    raise Exception(f"Anthropic API error: {response.status_code}")

                result = response.json()

            # Extract tool calls from response
            tool_calls = []
            text_response = ""

            for content_block in result.get("content", []):
                if content_block.get("type") == "tool_use":
                    tool_calls.append(ToolCall(
                        name=content_block.get("name"),
                        arguments=content_block.get("input", {}),
                        id=content_block.get("id", str(uuid.uuid4()))
                    ))
                    logger.info(f"Anthropic tool call: {content_block.get('name')}")
                elif content_block.get("type") == "text":
                    text_response += content_block.get("text", "")

            # Calculate tokens and cost
            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            tokens_used = input_tokens + output_tokens

            # Anthropic pricing (approximate)
            cost = (input_tokens * 0.003 / 1000) + (output_tokens * 0.015 / 1000)

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=model,
                provider=ModelProvider.ANTHROPIC,
                tier=tier,
                tokens_used=tokens_used,
                cost=cost,
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls),
                    "stop_reason": result.get("stop_reason")
                }
            )

            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=model,
                provider=ModelProvider.ANTHROPIC,
                tokens=tokens_used,
                cost=cost
            )

            return tool_response

        except Exception as e:
            logger.error(f"Anthropic native tool calling failed: {e}")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

    async def _generate_with_openai_native_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using OpenAI's native tool calling API."""
        import json
        import uuid
        import os

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in tools:
            openai_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters if tool.parameters else {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            openai_tools.append(openai_tool)

        # Build messages — use provided or construct from prompt
        if messages:
            api_messages = []
            for msg in messages:
                if msg["role"] == "tool":
                    # OpenAI format: role=tool with tool_call_id
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", msg.get("tool_use_id", "")),
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant" and msg.get("tool_calls"):
                    # OpenAI format: assistant with tool_calls array
                    tc_list = []
                    for tc in msg["tool_calls"]:
                        tc_list.append({
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}
                        })
                    api_messages.append({
                        "role": "assistant",
                        "content": msg.get("content"),
                        "tool_calls": tc_list
                    })
                else:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

        # Build request payload
        payload = {
            "model": model,
            "messages": api_messages,
            "tools": openai_tools,
            "temperature": temperature
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        # Handle tool_choice
        if tool_choice == "required":
            payload["tool_choice"] = "required"
        elif tool_choice == "auto":
            payload["tool_choice"] = "auto"
        else:
            payload["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        logger.info(f"Calling OpenAI with native tools: {[t['function']['name'] for t in openai_tools]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"OpenAI error: {response.status_code} - {response.text}")
                    raise Exception(f"OpenAI API error: {response.status_code}")

                result = response.json()

            # Extract tool calls from response
            tool_calls = []
            text_response = ""

            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})

            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    function = tc.get("function", {})
                    tool_args = function.get("arguments", "{}")
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    tool_calls.append(ToolCall(
                        name=function.get("name"),
                        arguments=tool_args,
                        id=tc.get("id", str(uuid.uuid4()))
                    ))
                    logger.info(f"OpenAI tool call: {function.get('name')}")

            if "content" in message and message["content"]:
                text_response = message["content"]

            # Calculate tokens and cost
            usage = result.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            tokens_used = prompt_tokens + completion_tokens

            # OpenAI pricing (approximate for GPT-4)
            cost = (prompt_tokens * 0.03 / 1000) + (completion_tokens * 0.06 / 1000)

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=model,
                provider=ModelProvider.OPENAI,
                tier=tier,
                tokens_used=tokens_used,
                cost=cost,
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls),
                    "finish_reason": choice.get("finish_reason")
                }
            )

            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=model,
                provider=ModelProvider.OPENAI,
                tokens=tokens_used,
                cost=cost
            )

            return tool_response

        except Exception as e:
            logger.error(f"OpenAI native tool calling failed: {e}")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

    async def _generate_with_azure_native_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using Azure OpenAI's native tool calling API."""
        import json
        import uuid
        import os

        api_key = os.environ.get("AZURE_OPENAI_API_KEY")
        endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
        deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", model)
        api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")

        if not api_key or not endpoint:
            logger.warning("Azure OpenAI credentials not set, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        # Convert tools to OpenAI format (same as OpenAI)
        azure_tools = []
        for tool in tools:
            azure_tool = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters if tool.parameters else {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            azure_tools.append(azure_tool)

        # Build messages — use provided or construct from prompt (same format as OpenAI)
        if messages:
            api_messages = []
            for msg in messages:
                if msg["role"] == "tool":
                    api_messages.append({
                        "role": "tool",
                        "tool_call_id": msg.get("tool_call_id", msg.get("tool_use_id", "")),
                        "content": msg["content"]
                    })
                elif msg["role"] == "assistant" and msg.get("tool_calls"):
                    tc_list = []
                    for tc in msg["tool_calls"]:
                        tc_list.append({
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}
                        })
                    api_messages.append({
                        "role": "assistant",
                        "content": msg.get("content"),
                        "tool_calls": tc_list
                    })
                else:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            api_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

        payload = {
            "messages": api_messages,
            "tools": azure_tools,
            "temperature": temperature
        }

        if max_tokens:
            payload["max_tokens"] = max_tokens

        if tool_choice == "required":
            payload["tool_choice"] = "required"
        elif tool_choice == "auto":
            payload["tool_choice"] = "auto"
        else:
            payload["tool_choice"] = {"type": "function", "function": {"name": tool_choice}}

        url = f"{endpoint}/openai/deployments/{deployment}/chat/completions?api-version={api_version}"
        logger.info(f"Calling Azure OpenAI with native tools: {[t['function']['name'] for t in azure_tools]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers={
                        "api-key": api_key,
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"Azure OpenAI error: {response.status_code} - {response.text}")
                    raise Exception(f"Azure OpenAI API error: {response.status_code}")

                result = response.json()

            tool_calls = []
            text_response = ""

            choice = result.get("choices", [{}])[0]
            message = choice.get("message", {})

            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    function = tc.get("function", {})
                    tool_args = function.get("arguments", "{}")
                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    tool_calls.append(ToolCall(
                        name=function.get("name"),
                        arguments=tool_args,
                        id=tc.get("id", str(uuid.uuid4()))
                    ))
                    logger.info(f"Azure tool call: {function.get('name')}")

            if "content" in message and message["content"]:
                text_response = message["content"]

            usage = result.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            cost = tokens_used * 0.00003  # Approximate Azure pricing

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=model,
                provider=ModelProvider.AZURE_OPENAI,
                tier=tier,
                tokens_used=tokens_used,
                cost=cost,
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls)
                }
            )

            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=model,
                provider=ModelProvider.AZURE_OPENAI,
                tokens=tokens_used,
                cost=cost
            )

            return tool_response

        except Exception as e:
            logger.error(f"Azure OpenAI native tool calling failed: {e}")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

    async def _generate_with_gemini_native_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using Google Gemini's native tool calling API."""
        import json
        import uuid
        import os

        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not set, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        # Convert tools to Gemini format
        gemini_functions = []
        for tool in tools:
            gemini_func = {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters if tool.parameters else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            gemini_functions.append(gemini_func)

        # Build contents from messages or single prompt
        effective_system_prompt = system_prompt
        if messages:
            contents = []
            for msg in messages:
                if msg["role"] == "system":
                    # System messages handled via systemInstruction
                    effective_system_prompt = msg["content"]
                elif msg["role"] == "user":
                    contents.append({"role": "user", "parts": [{"text": msg["content"]}]})
                elif msg["role"] == "assistant":
                    parts = []
                    if msg.get("content"):
                        parts.append({"text": msg["content"]})
                    if msg.get("tool_calls"):
                        for tc in msg["tool_calls"]:
                            parts.append({"functionCall": {
                                "name": tc["name"],
                                "args": tc.get("arguments", {})
                            }})
                    if parts:
                        contents.append({"role": "model", "parts": parts})
                elif msg["role"] == "tool":
                    contents.append({"role": "user", "parts": [{"functionResponse": {
                        "name": msg.get("name", "tool"),
                        "response": {"result": msg["content"]}
                    }}]})
        else:
            contents = [{"role": "user", "parts": [{"text": prompt}]}]

        # Build request payload
        payload = {
            "contents": contents,
            "tools": [{"function_declarations": gemini_functions}],
            "generationConfig": {
                "temperature": temperature
            }
        }

        if effective_system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": effective_system_prompt}]}

        if max_tokens:
            payload["generationConfig"]["maxOutputTokens"] = max_tokens

        # Handle tool_choice
        if tool_choice == "required":
            payload["toolConfig"] = {"functionCallingConfig": {"mode": "ANY"}}
        elif tool_choice != "auto":
            payload["toolConfig"] = {
                "functionCallingConfig": {
                    "mode": "ANY",
                    "allowedFunctionNames": [tool_choice]
                }
            }

        # Use environment default model if not a Gemini model
        # This centralizes model selection via DEFAULT_LLM_MODEL env var
        default_model = get_environment_default_model()
        # Strip -vertex suffix for direct Gemini API (it uses model names like gemini-2.0-flash, not gemini-2.0-flash-vertex)
        gemini_default = default_model.replace('-vertex', '') if 'vertex' in default_model.lower() else default_model
        gemini_model = model if "gemini" in model.lower() else gemini_default
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={api_key}"

        logger.info(f"Calling Gemini with native tools: {[f['name'] for f in gemini_functions]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"Gemini error: {response.status_code} - {response.text}")
                    raise Exception(f"Gemini API error: {response.status_code}")

                result = response.json()

            tool_calls = []
            text_response = ""

            candidates = result.get("candidates", [])
            if candidates:
                content = candidates[0].get("content", {})
                for part in content.get("parts", []):
                    if "functionCall" in part:
                        fc = part["functionCall"]
                        tool_calls.append(ToolCall(
                            name=fc.get("name"),
                            arguments=fc.get("args", {}),
                            id=str(uuid.uuid4())
                        ))
                        logger.info(f"Gemini tool call: {fc.get('name')}")
                    elif "text" in part:
                        text_response += part["text"]

            # Gemini usage metadata
            usage_metadata = result.get("usageMetadata", {})
            tokens_used = usage_metadata.get("totalTokenCount", 0)
            cost = tokens_used * 0.00001  # Approximate Gemini pricing

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=gemini_model,
                provider=ModelProvider.GEMINI,
                tier=tier,
                tokens_used=tokens_used,
                cost=cost,
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls)
                }
            )

            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=gemini_model,
                provider=ModelProvider.GEMINI,
                tokens=tokens_used,
                cost=cost
            )

            return tool_response

        except Exception as e:
            logger.error(f"Gemini native tool calling failed: {e}")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

    async def _generate_with_vertex_native_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using Google Vertex AI's native tool calling API."""
        import json
        import uuid
        import os

        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("VERTEX_AI_PROJECT") or os.environ.get("VERTEX_PROJECT_ID")
        location = os.environ.get("VERTEX_AI_LOCATION") or os.environ.get("VERTEX_LOCATION", "us-central1")

        if not project_id:
            logger.warning("GOOGLE_CLOUD_PROJECT not set, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        try:
            from google.cloud import aiplatform
            from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration, Content, Part, ToolConfig
        except ImportError:
            logger.warning("google-cloud-aiplatform not installed, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)

        # Convert tools to Vertex AI format
        vertex_functions = []
        for tool in tools:
            func_decl = FunctionDeclaration(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters if tool.parameters else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
            vertex_functions.append(func_decl)

        vertex_tools = Tool(function_declarations=vertex_functions)

        # Use gemini model on Vertex - get from centralized environment config
        default_model = get_environment_default_model()
        # Strip -vertex suffix for Vertex SDK (the SDK uses model names like "gemini-2.0-flash" not "gemini-2.0-flash-vertex")
        vertex_default = default_model.replace('-vertex', '') if 'vertex' in default_model.lower() else default_model
        # Also strip -vertex from the provided model name
        stripped_model = model.replace('-vertex', '') if 'vertex' in model.lower() else model
        vertex_model = stripped_model if "gemini" in model.lower() else vertex_default

        logger.info(f"Calling Vertex AI with native tools: {[t.name for t in tools]}")

        try:
            # Extract system prompt from messages if present
            effective_system_prompt = system_prompt
            if messages:
                for msg in messages:
                    if msg["role"] == "system":
                        effective_system_prompt = msg["content"]
                        break

            model_instance = GenerativeModel(
                vertex_model,
                tools=[vertex_tools],
                system_instruction=effective_system_prompt if effective_system_prompt else None
            )

            generation_config = {
                "temperature": temperature,
                "max_output_tokens": max_tokens or 4096
            }

            # Build tool_config from tool_choice
            tool_config = None
            if tool_choice == "required":
                tool_config = ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY
                    )
                )
            elif tool_choice and tool_choice != "auto":
                # Specific tool name
                tool_config = ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.ANY,
                        allowed_function_names=[tool_choice]
                    )
                )

            # Build contents from messages or single prompt
            if messages:
                contents = []
                for msg in messages:
                    if msg["role"] == "system":
                        continue  # Handled via system_instruction
                    elif msg["role"] == "user":
                        contents.append(Content(role="user", parts=[Part.from_text(msg["content"])]))
                    elif msg["role"] == "assistant":
                        parts = []
                        if msg.get("content"):
                            parts.append(Part.from_text(msg["content"]))
                        if msg.get("tool_calls"):
                            for tc in msg["tool_calls"]:
                                parts.append(Part.from_dict({
                                    "function_call": {
                                        "name": tc["name"],
                                        "args": tc.get("arguments", {})
                                    }
                                }))
                        if parts:
                            contents.append(Content(role="model", parts=parts))
                    elif msg["role"] == "tool":
                        contents.append(Content(role="user", parts=[
                            Part.from_function_response(
                                name=msg.get("name", "tool"),
                                response={"result": msg["content"]}
                            )
                        ]))
                response = model_instance.generate_content(
                    contents, generation_config=generation_config, tool_config=tool_config
                )
            else:
                response = model_instance.generate_content(
                    prompt, generation_config=generation_config, tool_config=tool_config
                )

            tool_calls = []
            text_response = ""

            # Extract tool calls from response
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        tool_calls.append(ToolCall(
                            name=fc.name,
                            arguments=dict(fc.args) if fc.args else {},
                            id=str(uuid.uuid4())
                        ))
                        logger.info(f"Vertex AI tool call: {fc.name}")
                    elif hasattr(part, 'text') and part.text:
                        text_response += part.text

            # Get usage metadata
            usage = response.usage_metadata if hasattr(response, 'usage_metadata') else None
            tokens_used = 0
            if usage:
                tokens_used = (usage.prompt_token_count or 0) + (usage.candidates_token_count or 0)

            cost = tokens_used * 0.000015  # Approximate Vertex AI pricing

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=vertex_model,
                provider=ModelProvider.VERTEX_AI,
                tier=tier,
                tokens_used=tokens_used,
                cost=cost,
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls)
                }
            )

            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=vertex_model,
                provider=ModelProvider.VERTEX_AI,
                tokens=tokens_used,
                cost=cost
            )

            return tool_response

        except Exception as e:
            logger.error(f"Vertex AI native tool calling failed: {e}")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

    async def _generate_with_bedrock_native_tools(
        self,
        prompt: Optional[str],
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using AWS Bedrock's native tool calling API (Claude on Bedrock)."""
        import json
        import uuid
        import os

        # Check for AWS credentials
        aws_region = os.environ.get("AWS_REGION", "us-east-1")

        try:
            import boto3
        except ImportError:
            logger.warning("boto3 not installed, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        # Convert tools to Anthropic format (Bedrock uses Claude's format)
        bedrock_tools = []
        for tool in tools:
            bedrock_tool = {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters if tool.parameters else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
            bedrock_tools.append(bedrock_tool)

        # Build messages — Bedrock uses Anthropic format (separate system field)
        if messages:
            api_messages = []
            system_text = system_prompt
            for msg in messages:
                if msg["role"] == "system":
                    system_text = msg["content"]
                elif msg["role"] == "tool":
                    api_messages.append({
                        "role": "user",
                        "content": [{"type": "tool_result", "tool_use_id": msg.get("tool_use_id", ""), "content": msg["content"]}]
                    })
                elif msg["role"] == "assistant" and msg.get("tool_calls"):
                    content = []
                    if msg.get("content"):
                        content.append({"type": "text", "text": msg["content"]})
                    for tc in msg["tool_calls"]:
                        content.append({"type": "tool_use", "id": tc.get("id", ""), "name": tc["name"], "input": tc["arguments"]})
                    api_messages.append({"role": "assistant", "content": content})
                else:
                    api_messages.append({"role": msg["role"], "content": msg["content"]})
        else:
            api_messages = [{"role": "user", "content": prompt}]
            system_text = system_prompt

        # Build request body (Anthropic format for Bedrock)
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens or 4096,
            "system": system_text,
            "messages": api_messages,
            "tools": bedrock_tools if tools else [],
            "temperature": temperature
        }

        if tool_choice == "required":
            body["tool_choice"] = {"type": "any"}
        elif tool_choice != "auto":
            body["tool_choice"] = {"type": "tool", "name": tool_choice}

        # Use Claude on Bedrock
        bedrock_model = model if "anthropic" in model.lower() else "anthropic.claude-3-sonnet-20240229-v1:0"

        logger.info(f"Calling Bedrock with native tools: {[t['name'] for t in bedrock_tools]}")

        try:
            client = boto3.client("bedrock-runtime", region_name=aws_region)

            response = client.invoke_model(
                modelId=bedrock_model,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body)
            )

            result = json.loads(response["body"].read())

            tool_calls = []
            text_response = ""

            for content_block in result.get("content", []):
                if content_block.get("type") == "tool_use":
                    tool_calls.append(ToolCall(
                        name=content_block.get("name"),
                        arguments=content_block.get("input", {}),
                        id=content_block.get("id", str(uuid.uuid4()))
                    ))
                    logger.info(f"Bedrock tool call: {content_block.get('name')}")
                elif content_block.get("type") == "text":
                    text_response += content_block.get("text", "")

            usage = result.get("usage", {})
            tokens_used = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            cost = tokens_used * 0.00003  # Approximate Bedrock pricing

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=bedrock_model,
                provider=ModelProvider.AWS_BEDROCK,
                tier=tier,
                tokens_used=tokens_used,
                cost=cost,
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls)
                }
            )

            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=bedrock_model,
                provider=ModelProvider.AWS_BEDROCK,
                tokens=tokens_used,
                cost=cost
            )

            return tool_response

        except Exception as e:
            logger.error(f"Bedrock native tool calling failed: {e}")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

    async def _generate_with_cohere_native_tools(
        self,
        prompt: str,
        tools: List[ToolDefinition],
        model: str,
        tier: ModelTier,
        temperature: float,
        max_tokens: Optional[int],
        system_prompt: str,
        tool_choice: str,
        start_time: datetime,
        user_settings: Optional[UserLLMSettings],
        messages: Optional[List[Dict[str, Any]]] = None
    ) -> LLMToolResponse:
        """Generate using Cohere's native tool calling API."""
        import json
        import uuid
        import os

        api_key = os.environ.get("COHERE_API_KEY")
        if not api_key:
            logger.warning("COHERE_API_KEY not set, falling back to text-based")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

        # Convert tools to Cohere format
        cohere_tools = []
        for tool in tools:
            # Convert JSON schema to Cohere's parameter_definitions format
            params = tool.parameters or {"type": "object", "properties": {}, "required": []}
            parameter_definitions = {}

            for prop_name, prop_schema in params.get("properties", {}).items():
                parameter_definitions[prop_name] = {
                    "description": prop_schema.get("description", ""),
                    "type": prop_schema.get("type", "string"),
                    "required": prop_name in params.get("required", [])
                }

            cohere_tool = {
                "name": tool.name,
                "description": tool.description,
                "parameter_definitions": parameter_definitions
            }
            cohere_tools.append(cohere_tool)

        # Build request payload
        payload = {
            "model": model if model else "command-r-plus",
            "message": prompt or "",
            "tools": cohere_tools,
            "temperature": temperature
        }

        if system_prompt:
            payload["preamble"] = system_prompt

        # Build chat_history from messages for multi-turn conversations
        if messages:
            chat_history = []
            last_user_message = prompt or ""
            for msg in messages:
                if msg["role"] == "system":
                    payload["preamble"] = msg["content"]
                elif msg["role"] == "user":
                    last_user_message = msg["content"]
                elif msg["role"] == "assistant":
                    chat_history.append({
                        "role": "CHATBOT",
                        "message": msg.get("content") or ""
                    })
                elif msg["role"] == "tool":
                    chat_history.append({
                        "role": "TOOL",
                        "tool_results": [{
                            "call": {"name": msg.get("name", "tool")},
                            "outputs": [{"result": msg["content"]}]
                        }]
                    })
            # Cohere uses message for the current turn, chat_history for prior turns
            payload["message"] = last_user_message
            if chat_history:
                payload["chat_history"] = chat_history

        # Reinforce tool use requirement via preamble (Cohere v1 has no tool_choice param)
        # Applied after messages processing so system message overrides don't erase it
        if tool_choice == "required":
            tool_instruction = "\n\nIMPORTANT: You MUST use at least one of the available tools to answer. Do not respond with text alone."
            payload["preamble"] = (payload.get("preamble") or "") + tool_instruction

        if max_tokens:
            payload["max_tokens"] = max_tokens

        logger.info(f"Calling Cohere with native tools: {[t['name'] for t in cohere_tools]}")

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    "https://api.cohere.ai/v1/chat",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json"
                    },
                    json=payload
                )

                if response.status_code != 200:
                    logger.error(f"Cohere error: {response.status_code} - {response.text}")
                    raise Exception(f"Cohere API error: {response.status_code}")

                result = response.json()

            tool_calls = []
            text_response = result.get("text", "")

            # Extract tool calls from Cohere response
            for tc in result.get("tool_calls", []):
                tool_calls.append(ToolCall(
                    name=tc.get("name"),
                    arguments=tc.get("parameters", {}),
                    id=tc.get("id", str(uuid.uuid4()))
                ))
                logger.info(f"Cohere tool call: {tc.get('name')}")

            # Token usage
            meta = result.get("meta", {})
            tokens = meta.get("tokens", {})
            tokens_used = tokens.get("input_tokens", 0) + tokens.get("output_tokens", 0)
            cost = tokens_used * 0.000015  # Approximate Cohere pricing

            tool_response = LLMToolResponse(
                text=text_response,
                tool_calls=tool_calls if tool_calls else None,
                execution_plan=None,
                model_used=model or "command-r-plus",
                provider=ModelProvider.COHERE,
                tier=tier,
                tokens_used=tokens_used,
                cost=cost,
                response_time=(datetime.utcnow() - start_time).total_seconds(),
                metadata={
                    "tools_provided": len(tools),
                    "tool_choice": tool_choice,
                    "native_tools": True,
                    "tool_calls_count": len(tool_calls),
                    "finish_reason": result.get("finish_reason")
                }
            )

            await self.cost_tracker.track_usage(
                user_id=user_settings.user_id if user_settings else "anonymous",
                model=model or "command-r-plus",
                provider=ModelProvider.COHERE,
                tokens=tokens_used,
                cost=cost
            )

            return tool_response

        except Exception as e:
            logger.error(f"Cohere native tool calling failed: {e}")
            return await self._generate_with_text_tools(
                prompt=prompt, tools=tools, user_settings=user_settings,
                model_override=model, task_type="tool_use", temperature=temperature,
                max_tokens=max_tokens, system_prompt=system_prompt,
                tool_choice=tool_choice, context=None, start_time=start_time
            )

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