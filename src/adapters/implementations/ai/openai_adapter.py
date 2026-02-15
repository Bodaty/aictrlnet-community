"""OpenAI adapter implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
import httpx
import json
from datetime import datetime

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from adapters.tool_calling import ToolCallingMixin, ToolCallingRequest, ToolCallingResponse
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class OpenAIAdapter(BaseAdapter, ToolCallingMixin):
    """Adapter for OpenAI API integration."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.AI
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "https://api.openai.com/v1"
        self.api_key = config.api_key or config.credentials.get("api_key")
        
        # Check if we're in discovery-only mode
        self.discovery_only = config.custom_config.get("discovery_only", False)
        
        if not self.api_key and not self.discovery_only:
            raise ValueError("OpenAI API key is required")
    
    async def initialize(self) -> None:
        """Initialize the OpenAI adapter."""
        # Skip initialization in discovery-only mode
        if self.discovery_only:
            logger.info("OpenAI adapter initialized in discovery-only mode")
            return
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout_seconds
        )
        
        # Test connection
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            logger.info("OpenAI adapter initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI adapter: {str(e)}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("OpenAI adapter shutdown")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available OpenAI models.
        
        In discovery mode, returns a static list of known models.
        With API key, fetches live model list from OpenAI.
        """
        if self.discovery_only:
            # Return static list of known OpenAI models
            return [
                {"id": "gpt-4", "name": "GPT-4", "capabilities": ["chat", "completion"], "max_tokens": 8192},
                {"id": "gpt-4-turbo", "name": "GPT-4 Turbo", "capabilities": ["chat", "completion", "vision"], "max_tokens": 128000},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo", "capabilities": ["chat", "completion"], "max_tokens": 16385},
                {"id": "gpt-3.5-turbo-16k", "name": "GPT-3.5 Turbo 16K", "capabilities": ["chat", "completion"], "max_tokens": 16385},
                {"id": "text-embedding-ada-002", "name": "Ada Embeddings v2", "capabilities": ["embeddings"], "max_tokens": 8191},
                {"id": "dall-e-3", "name": "DALL-E 3", "capabilities": ["image-generation"], "max_tokens": None},
                {"id": "dall-e-2", "name": "DALL-E 2", "capabilities": ["image-generation"], "max_tokens": None},
                {"id": "whisper-1", "name": "Whisper", "capabilities": ["audio-transcription"], "max_tokens": None},
                {"id": "tts-1", "name": "Text-to-Speech", "capabilities": ["text-to-speech"], "max_tokens": None},
                {"id": "tts-1-hd", "name": "Text-to-Speech HD", "capabilities": ["text-to-speech"], "max_tokens": None}
            ]
        
        # With API key, fetch live model list
        if self.client:
            try:
                response = await self.client.get("/models")
                response.raise_for_status()
                data = response.json()
                models = []
                for model in data.get("data", []):
                    models.append({
                        "id": model["id"],
                        "name": model.get("id", "").replace("-", " ").title(),
                        "capabilities": self._infer_capabilities(model["id"]),
                        "max_tokens": self._get_max_tokens(model["id"])
                    })
                return models
            except Exception as e:
                logger.warning(f"Failed to fetch live models, using static list: {e}")
                # Fall back to static list
                return await self.list_models()  # Will use discovery_only path
        
        return []
    
    def _infer_capabilities(self, model_id: str) -> List[str]:
        """Infer model capabilities from model ID."""
        capabilities = []
        if "gpt" in model_id or "turbo" in model_id:
            capabilities.extend(["chat", "completion"])
        if "vision" in model_id or "gpt-4" in model_id:
            capabilities.append("vision")
        if "embedding" in model_id:
            capabilities.append("embeddings")
        if "dall-e" in model_id:
            capabilities.append("image-generation")
        if "whisper" in model_id:
            capabilities.append("audio-transcription")
        if "tts" in model_id:
            capabilities.append("text-to-speech")
        return capabilities or ["completion"]
    
    def _get_max_tokens(self, model_id: str) -> Optional[int]:
        """Get max tokens for a model."""
        token_limits = {
            "gpt-4": 8192,
            "gpt-4-turbo": 128000,
            "gpt-3.5-turbo": 16385,
            "gpt-3.5-turbo-16k": 16385,
            "text-embedding-ada-002": 8191
        }
        for key, limit in token_limits.items():
            if key in model_id:
                return limit
        return 4096  # Default
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return OpenAI adapter capabilities."""
        return [
            AdapterCapability(
                name="chat_completion",
                description="Generate chat completions using OpenAI models",
                category="text_generation",
                parameters={
                    "model": {"type": "string", "description": "Model to use (e.g., gpt-4, gpt-3.5-turbo)"},
                    "messages": {"type": "array", "description": "Array of message objects"},
                    "temperature": {"type": "number", "description": "Sampling temperature (0-2)", "default": 1.0},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens to generate"},
                    "stream": {"type": "boolean", "description": "Stream the response", "default": False}
                },
                required_parameters=["model", "messages"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.01  # Approximate
            ),
            AdapterCapability(
                name="embeddings",
                description="Generate embeddings for text",
                category="embeddings",
                parameters={
                    "model": {"type": "string", "description": "Model to use (e.g., text-embedding-ada-002)"},
                    "input": {"type": "string|array", "description": "Text to embed"}
                },
                required_parameters=["model", "input"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0001
            ),
            AdapterCapability(
                name="image_generation",
                description="Generate images from text prompts",
                category="image_generation",
                parameters={
                    "prompt": {"type": "string", "description": "Text prompt for image generation"},
                    "model": {"type": "string", "description": "Model to use (e.g., dall-e-3)", "default": "dall-e-3"},
                    "size": {"type": "string", "description": "Image size", "default": "1024x1024"},
                    "quality": {"type": "string", "description": "Image quality", "default": "standard"},
                    "n": {"type": "integer", "description": "Number of images", "default": 1}
                },
                required_parameters=["prompt"],
                async_supported=True,
                estimated_duration_seconds=10.0,
                cost_per_request=0.04
            ),
            AdapterCapability(
                name="moderation",
                description="Check content for policy violations",
                category="moderation",
                parameters={
                    "input": {"type": "string", "description": "Text to moderate"}
                },
                required_parameters=["input"],
                async_supported=True,
                estimated_duration_seconds=0.2,
                cost_per_request=0.0
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to OpenAI."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        if request.capability == "chat_completion":
            return await self._handle_chat_completion(request)
        elif request.capability == "embeddings":
            return await self._handle_embeddings(request)
        elif request.capability == "image_generation":
            return await self._handle_image_generation(request)
        elif request.capability == "moderation":
            return await self._handle_moderation(request)
        else:
            raise ValueError(f"Unknown capability: {request.capability}")
    
    async def _handle_chat_completion(self, request: AdapterRequest) -> AdapterResponse:
        """Handle chat completion requests."""
        start_time = datetime.utcnow()
        
        try:
            # Prepare request data
            data = {
                "model": request.parameters["model"],
                "messages": request.parameters["messages"],
                "temperature": request.parameters.get("temperature", 1.0),
                "max_tokens": request.parameters.get("max_tokens"),
                "stream": request.parameters.get("stream", False)
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            # Handle streaming if requested
            if data.get("stream"):
                return await self._handle_streaming_chat(request, data, start_time)
            
            # Make API request
            response = await self.client.post("/chat/completions", json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Calculate metrics
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            
            # Estimate cost (rough approximation)
            cost = self._estimate_cost(
                request.parameters["model"],
                tokens_used
            )
            
            # Publish completion event
            await event_bus.publish(
                "adapter.openai.completion",
                {
                    "model": request.parameters["model"],
                    "tokens": tokens_used,
                    "duration_ms": duration_ms
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "choices": result.get("choices", []),
                    "usage": result.get("usage", {}),
                    "model": result.get("model")
                },
                duration_ms=duration_ms,
                cost=cost,
                tokens_used=tokens_used,
                metadata={
                    "openai_id": result.get("id"),
                    "created": result.get("created")
                }
            )
            
        except httpx.HTTPStatusError as e:
            error_data = e.response.json() if e.response.content else {}
            error_message = error_data.get("error", {}).get("message", str(e))
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=error_message,
                error_code=f"HTTP_{e.response.status_code}",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_streaming_chat(
        self,
        request: AdapterRequest,
        data: Dict[str, Any],
        start_time: datetime
    ) -> AdapterResponse:
        """Handle streaming chat completion."""
        # For streaming, we'll collect chunks and return them
        chunks = []
        
        try:
            async with self.client.stream("POST", "/chat/completions", json=data) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]  # Remove "data: " prefix
                        if chunk_data == "[DONE]":
                            break
                        
                        try:
                            chunk = json.loads(chunk_data)
                            chunks.append(chunk)
                        except json.JSONDecodeError:
                            continue
            
            # Combine chunks into final response
            combined_content = ""
            for chunk in chunks:
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                if "content" in delta:
                    combined_content += delta["content"]
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": combined_content
                        },
                        "finish_reason": "stop"
                    }],
                    "stream_chunks": len(chunks)
                },
                duration_ms=duration_ms,
                metadata={"streaming": True}
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_embeddings(self, request: AdapterRequest) -> AdapterResponse:
        """Handle embeddings requests."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "model": request.parameters["model"],
                "input": request.parameters["input"]
            }
            
            response = await self.client.post("/embeddings", json=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Calculate cost
            tokens_used = result.get("usage", {}).get("total_tokens", 0)
            cost = tokens_used * 0.0001 / 1000  # Rough estimate
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "embeddings": result.get("data", []),
                    "usage": result.get("usage", {})
                },
                duration_ms=duration_ms,
                cost=cost,
                tokens_used=tokens_used
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_image_generation(self, request: AdapterRequest) -> AdapterResponse:
        """Handle image generation requests."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "model": request.parameters.get("model", "dall-e-3"),
                "prompt": request.parameters["prompt"],
                "size": request.parameters.get("size", "1024x1024"),
                "quality": request.parameters.get("quality", "standard"),
                "n": request.parameters.get("n", 1)
            }
            
            response = await self.client.post("/images/generations", json=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Estimate cost based on model and quality
            cost = 0.04 if data["quality"] == "standard" else 0.08
            cost *= data["n"]
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "images": result.get("data", []),
                    "created": result.get("created")
                },
                duration_ms=duration_ms,
                cost=cost
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_moderation(self, request: AdapterRequest) -> AdapterResponse:
        """Handle content moderation requests."""
        start_time = datetime.utcnow()
        
        try:
            data = {"input": request.parameters["input"]}
            
            response = await self.client.post("/moderations", json=data)
            response.raise_for_status()
            
            result = response.json()
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "results": result.get("results", []),
                    "id": result.get("id")
                },
                duration_ms=duration_ms,
                cost=0.0  # Moderation is free
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    def _estimate_cost(self, model: str, tokens: int) -> float:
        """Estimate cost based on model and tokens."""
        # Rough estimates per 1K tokens
        pricing = {
            "gpt-4": 0.03,
            "gpt-4-32k": 0.06,
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.003,
        }
        
        # Default pricing if model not found
        price_per_1k = pricing.get(model, 0.002)
        return (tokens / 1000) * price_per_1k
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform OpenAI-specific health check."""
        try:
            # Check models endpoint
            response = await self.client.get("/models")
            response.raise_for_status()

            models = response.json().get("data", [])
            model_ids = [m["id"] for m in models]

            return {
                "status": "healthy",
                "available_models": len(models),
                "models": model_ids[:5]  # First 5 models
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    # ── ToolCallingMixin implementation ──────────────────────────────────

    async def chat_with_tools(self, request: ToolCallingRequest) -> ToolCallingResponse:
        """Execute tool-augmented chat via OpenAI's Chat Completions API."""
        import os
        import uuid

        api_key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise Exception("OPENAI_API_KEY not available")

        # Convert tools to OpenAI format
        openai_tools = []
        for tool in request.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {
                        "type": "object", "properties": {}, "required": []
                    })
                }
            })

        # Build messages
        api_messages = []
        for msg in request.messages:
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

        payload = {
            "model": request.model,
            "messages": api_messages,
            "tools": openai_tools,
            "temperature": request.temperature
        }

        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        # Handle tool_choice
        if request.tool_choice == "required":
            payload["tool_choice"] = "required"
        elif request.tool_choice == "auto":
            payload["tool_choice"] = "auto"
        else:
            payload["tool_choice"] = {"type": "function", "function": {"name": request.tool_choice}}

        logger.info(f"OpenAI chat_with_tools: {[t['function']['name'] for t in openai_tools]}")

        # Use existing client or create temporary one
        client = self.client
        close_client = False
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url or "https://api.openai.com/v1",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                timeout=120.0,
            )
            close_client = True

        try:
            response = await client.post("/chat/completions", json=payload)

            if response.status_code != 200:
                logger.error(f"OpenAI tool calling error: {response.status_code} - {response.text}")
                raise Exception(f"OpenAI API error: {response.status_code}")

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

                    tool_calls.append({
                        "name": function.get("name"),
                        "arguments": tool_args,
                        "id": tc.get("id", str(uuid.uuid4()))
                    })
                    logger.info(f"OpenAI tool call: {function.get('name')}")

            if "content" in message and message["content"]:
                text_response = message["content"]

            usage = result.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            # OpenAI pricing (approximate for GPT-4)
            cost = (input_tokens * 0.03 / 1000) + (output_tokens * 0.06 / 1000)

            return ToolCallingResponse(
                text=text_response,
                tool_calls=tool_calls,
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                stop_reason=choice.get("finish_reason"),
            )
        finally:
            if close_client:
                await client.aclose()