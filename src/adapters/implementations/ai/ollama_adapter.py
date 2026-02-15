"""Ollama adapter implementation for local LLM support."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
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


class OllamaAdapter(BaseAdapter, ToolCallingMixin):
    """Adapter for Ollama local LLM API."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.AI
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        
        # Check if we're in discovery-only mode
        self.discovery_only = config.custom_config.get("discovery_only", False)
        
        # Ollama API configuration
        self.base_url = config.base_url or config.credentials.get("base_url", "http://localhost:11434")
        # Try host.docker.internal if in Docker environment
        if "localhost" in self.base_url and not self.discovery_only:
            self.base_url = self.base_url.replace("localhost", "host.docker.internal")
        self.timeout = config.timeout_seconds or 300.0  # Longer timeout for local models
    
    async def initialize(self) -> None:
        """Initialize the Ollama adapter."""
        # Skip full initialization in discovery-only mode
        if self.discovery_only:
            logger.info("Ollama adapter initialized in discovery-only mode")
            # Still create client for potential connection attempts
            self.client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=5.0,  # Short timeout for discovery
                limits=httpx.Limits(max_keepalive_connections=1, max_connections=2)
            )
            return
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0
            )
        )
        
        # Test connection
        try:
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            models = response.json()
            logger.info(f"Ollama adapter initialized. Available models: {len(models.get('models', []))}")
        except Exception as e:
            logger.warning(f"Failed to connect to Ollama: {str(e)}")
        
        logger.info("Ollama adapter initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Ollama adapter shutdown")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Ollama models.
        
        Tries to connect to local Ollama instance.
        Falls back to common models list if connection fails.
        """
        # Try to get actual models from Ollama
        if self.client:
            try:
                response = await self.client.get("/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    models = []
                    for model in data.get("models", []):
                        model_name = model.get("name", "unknown")
                        models.append({
                            "id": model_name,
                            "name": model_name.replace(":", " ").title(),
                            "capabilities": ["chat", "completion"],
                            "max_tokens": 4096,
                            "size": model.get("size", 0),
                            "modified": model.get("modified_at", "")
                        })
                    if models:
                        logger.info(f"Discovered {len(models)} Ollama models")
                        return models
            except Exception as e:
                logger.debug(f"Could not connect to Ollama: {e}")
        
        # Return common Ollama models as fallback
        return [
            {"id": "llama2", "name": "Llama 2", "capabilities": ["chat", "completion"], "max_tokens": 4096},
            {"id": "llama2:13b", "name": "Llama 2 13B", "capabilities": ["chat", "completion"], "max_tokens": 4096},
            {"id": "llama2:70b", "name": "Llama 2 70B", "capabilities": ["chat", "completion"], "max_tokens": 4096},
            {"id": "mistral", "name": "Mistral", "capabilities": ["chat", "completion"], "max_tokens": 8192},
            {"id": "mixtral", "name": "Mixtral", "capabilities": ["chat", "completion"], "max_tokens": 32768},
            {"id": "codellama", "name": "Code Llama", "capabilities": ["chat", "completion", "code"], "max_tokens": 4096},
            {"id": "deepseek-coder", "name": "DeepSeek Coder", "capabilities": ["chat", "completion", "code"], "max_tokens": 16384},
            {"id": "phi", "name": "Phi-2", "capabilities": ["chat", "completion"], "max_tokens": 2048},
            {"id": "neural-chat", "name": "Neural Chat", "capabilities": ["chat", "completion"], "max_tokens": 4096},
            {"id": "starling-lm", "name": "Starling LM", "capabilities": ["chat", "completion"], "max_tokens": 8192}
        ]
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Ollama adapter capabilities."""
        return [
            AdapterCapability(
                name="generate",
                description="Generate text using Ollama models",
                category="text_generation",
                parameters={
                    "model": {"type": "string", "description": "Model name (e.g., llama2, mistral)"},
                    "prompt": {"type": "string", "description": "Input prompt"},
                    "system": {"type": "string", "description": "System prompt"},
                    "template": {"type": "string", "description": "Prompt template"},
                    "context": {"type": "array", "description": "Conversation context"},
                    "options": {"type": "object", "description": "Model options (temperature, top_p, etc.)"},
                    "stream": {"type": "boolean", "description": "Stream the response", "default": False}
                },
                required_parameters=["model", "prompt"],
                async_supported=True,
                estimated_duration_seconds=10.0,
                cost_per_request=0.0  # Local models are free
            ),
            AdapterCapability(
                name="chat",
                description="Chat with Ollama models",
                category="chat",
                parameters={
                    "model": {"type": "string", "description": "Model name"},
                    "messages": {"type": "array", "description": "Chat messages"},
                    "options": {"type": "object", "description": "Model options"},
                    "stream": {"type": "boolean", "description": "Stream the response", "default": False}
                },
                required_parameters=["model", "messages"],
                async_supported=True,
                estimated_duration_seconds=10.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="embeddings",
                description="Generate embeddings using Ollama models",
                category="embeddings",
                parameters={
                    "model": {"type": "string", "description": "Model name"},
                    "prompt": {"type": "string", "description": "Text to embed"},
                    "options": {"type": "object", "description": "Model options"}
                },
                required_parameters=["model", "prompt"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="list_models",
                description="List available Ollama models",
                category="utility",
                parameters={},
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="show_model",
                description="Get information about a specific model",
                category="utility",
                parameters={
                    "name": {"type": "string", "description": "Model name"}
                },
                required_parameters=["name"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="pull_model",
                description="Download a model from the Ollama library",
                category="model_management",
                parameters={
                    "name": {"type": "string", "description": "Model name to pull"},
                    "stream": {"type": "boolean", "description": "Stream progress updates", "default": True}
                },
                required_parameters=["name"],
                async_supported=True,
                estimated_duration_seconds=300.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="delete_model",
                description="Delete a local model",
                category="model_management",
                parameters={
                    "name": {"type": "string", "description": "Model name to delete"}
                },
                required_parameters=["name"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="health_check",
                description="Check Ollama server health",
                category="monitoring",
                parameters={},
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute an Ollama operation."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        capability_handlers = {
            "generate": self._handle_generate,
            "chat": self._handle_chat,
            "embeddings": self._handle_embeddings,
            "list_models": self._handle_list_models,
            "show_model": self._handle_show_model,
            "pull_model": self._handle_pull_model,
            "delete_model": self._handle_delete_model,
            "health_check": self._handle_health_check
        }
        
        handler = capability_handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")
        
        return await handler(request)
    
    async def _handle_generate(self, request: AdapterRequest) -> AdapterResponse:
        """Handle text generation."""
        start_time = datetime.utcnow()
        
        try:
            # Build request payload
            payload = {
                "model": request.parameters["model"],
                "prompt": request.parameters["prompt"]
            }
            
            # Add optional parameters
            if "system" in request.parameters:
                payload["system"] = request.parameters["system"]
            
            if "template" in request.parameters:
                payload["template"] = request.parameters["template"]
            
            if "context" in request.parameters:
                payload["context"] = request.parameters["context"]
            
            if "options" in request.parameters:
                payload["options"] = request.parameters["options"]
            
            stream = request.parameters.get("stream", False)
            payload["stream"] = stream
            
            # Make request
            if stream:
                # Handle streaming response
                response_text = ""
                async with self.client.stream("POST", "/api/generate", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "response" in data:
                                response_text += data["response"]
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return AdapterResponse(
                    request_id=request.id,
                    capability=request.capability,
                    status="success",
                    data={
                        "response": response_text,
                        "model": request.parameters["model"]
                    },
                    duration_ms=duration_ms,
                    cost=0.0
                )
            else:
                # Non-streaming response
                response = await self.client.post("/api/generate", json=payload)
                response.raise_for_status()
                result = response.json()
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Publish event
                await event_bus.publish(
                    "adapter.ollama.generation_completed",
                    {
                        "model": request.parameters["model"],
                        "prompt_length": len(request.parameters["prompt"]),
                        "response_length": len(result.get("response", "")),
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
                        "response": result.get("response", ""),
                        "model": result.get("model"),
                        "created_at": result.get("created_at"),
                        "context": result.get("context"),
                        "total_duration": result.get("total_duration"),
                        "eval_count": result.get("eval_count")
                    },
                    duration_ms=duration_ms,
                    cost=0.0
                )
                
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_chat(self, request: AdapterRequest) -> AdapterResponse:
        """Handle chat completion."""
        start_time = datetime.utcnow()
        
        try:
            # Build request payload
            payload = {
                "model": request.parameters["model"],
                "messages": request.parameters["messages"]
            }
            
            if "options" in request.parameters:
                payload["options"] = request.parameters["options"]
            
            stream = request.parameters.get("stream", False)
            payload["stream"] = stream
            
            # Make request
            if stream:
                # Handle streaming response
                response_content = ""
                async with self.client.stream("POST", "/api/chat", json=payload) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            if "message" in data and "content" in data["message"]:
                                response_content += data["message"]["content"]
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return AdapterResponse(
                    request_id=request.id,
                    capability=request.capability,
                    status="success",
                    data={
                        "message": {"role": "assistant", "content": response_content},
                        "model": request.parameters["model"]
                    },
                    duration_ms=duration_ms,
                    cost=0.0
                )
            else:
                # Non-streaming response
                response = await self.client.post("/api/chat", json=payload)
                response.raise_for_status()
                result = response.json()
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return AdapterResponse(
                    request_id=request.id,
                    capability=request.capability,
                    status="success",
                    data={
                        "message": result.get("message", {}),
                        "model": result.get("model"),
                        "created_at": result.get("created_at"),
                        "total_duration": result.get("total_duration"),
                        "eval_count": result.get("eval_count")
                    },
                    duration_ms=duration_ms,
                    cost=0.0
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
        """Handle embeddings generation."""
        start_time = datetime.utcnow()
        
        try:
            # Build request payload
            payload = {
                "model": request.parameters["model"],
                "prompt": request.parameters["prompt"]
            }
            
            if "options" in request.parameters:
                payload["options"] = request.parameters["options"]
            
            # Make request
            response = await self.client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            result = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "embedding": result.get("embedding", []),
                    "model": result.get("model"),
                    "dimensions": len(result.get("embedding", []))
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_list_models(self, request: AdapterRequest) -> AdapterResponse:
        """Handle listing available models."""
        start_time = datetime.utcnow()
        
        try:
            # Get models
            response = await self.client.get("/api/tags")
            response.raise_for_status()
            result = response.json()
            
            # Process models
            models = []
            for model in result.get("models", []):
                models.append({
                    "name": model.get("name"),
                    "size": model.get("size"),
                    "digest": model.get("digest"),
                    "modified_at": model.get("modified_at")
                })
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "models": models,
                    "model_count": len(models)
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_show_model(self, request: AdapterRequest) -> AdapterResponse:
        """Handle getting model information."""
        start_time = datetime.utcnow()
        
        try:
            model_name = request.parameters["name"]
            
            # Get model info
            response = await self.client.post("/api/show", json={"name": model_name})
            response.raise_for_status()
            result = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "modelfile": result.get("modelfile"),
                    "template": result.get("template"),
                    "parameters": result.get("parameters"),
                    "license": result.get("license")
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_pull_model(self, request: AdapterRequest) -> AdapterResponse:
        """Handle pulling a model."""
        start_time = datetime.utcnow()
        
        try:
            model_name = request.parameters["name"]
            stream = request.parameters.get("stream", True)
            
            if stream:
                # Stream progress updates
                progress_data = []
                async with self.client.stream("POST", "/api/pull", json={"name": model_name, "stream": True}) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            progress_data.append(data)
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return AdapterResponse(
                    request_id=request.id,
                    capability=request.capability,
                    status="success",
                    data={
                        "model": model_name,
                        "status": "completed",
                        "progress": progress_data[-1] if progress_data else {}
                    },
                    duration_ms=duration_ms,
                    cost=0.0
                )
            else:
                # Non-streaming pull
                response = await self.client.post("/api/pull", json={"name": model_name, "stream": False})
                response.raise_for_status()
                
                duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                return AdapterResponse(
                    request_id=request.id,
                    capability=request.capability,
                    status="success",
                    data={
                        "model": model_name,
                        "status": "completed"
                    },
                    duration_ms=duration_ms,
                    cost=0.0
                )
                
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_delete_model(self, request: AdapterRequest) -> AdapterResponse:
        """Handle deleting a model."""
        start_time = datetime.utcnow()
        
        try:
            model_name = request.parameters["name"]
            
            # Delete model
            response = await self.client.delete("/api/delete", json={"name": model_name})
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "model": model_name,
                    "deleted": True
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_health_check(self, request: AdapterRequest) -> AdapterResponse:
        """Handle health check."""
        return await self._perform_health_check_response(request)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Ollama health check."""
        try:
            # Check server status
            response = await self.client.get("/")
            response.raise_for_status()
            
            # Get available models
            models_response = await self.client.get("/api/tags")
            models_response.raise_for_status()
            models = models_response.json()
            
            return {
                "status": "healthy",
                "server_url": self.base_url,
                "model_count": len(models.get("models", [])),
                "models": [m["name"] for m in models.get("models", [])]
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "server_url": self.base_url
            }
    
    async def _perform_health_check_response(self, request: AdapterRequest) -> AdapterResponse:
        """Perform health check and return as response."""
        start_time = datetime.utcnow()

        health_status = await self._perform_health_check()
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="success" if health_status["status"] == "healthy" else "error",
            data=health_status,
            duration_ms=duration_ms,
            cost=0.0
        )

    # ── ToolCallingMixin implementation ──────────────────────────────────

    async def chat_with_tools(self, request: ToolCallingRequest) -> ToolCallingResponse:
        """Execute tool-augmented chat via Ollama's native tool calling API."""
        import uuid

        # Convert tools to Ollama format (OpenAI-style)
        ollama_tools = []
        for tool in request.tools:
            ollama_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", tool.get("function", {}).get("name", "")),
                    "description": tool.get("description", tool.get("function", {}).get("description", "")),
                    "parameters": tool.get("parameters", tool.get("function", {}).get("parameters", {
                        "type": "object", "properties": {}, "required": []
                    }))
                }
            })

        # Build messages array
        api_messages = []
        for msg in request.messages:
            if msg["role"] == "tool":
                api_messages.append({
                    "role": "tool",
                    "content": msg["content"]
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
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

        # If no system message in messages and system_prompt provided, prepend it
        has_system = any(m["role"] == "system" for m in api_messages)
        if not has_system and request.system_prompt:
            api_messages.insert(0, {"role": "system", "content": request.system_prompt})

        payload = {
            "model": request.model,
            "messages": api_messages,
            "tools": ollama_tools,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens or 2000
            }
        }

        logger.info(f"Ollama chat_with_tools: {[t['function']['name'] for t in ollama_tools]}")

        # Use existing client or create a temporary one
        client = self.client
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
            )

        try:
            response = await client.post("/api/chat", json=payload)

            if response.status_code != 200:
                logger.error(f"Ollama tool calling error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")

            result = response.json()

            # Extract tool calls
            tool_calls = []
            text_response = ""
            message = result.get("message", {})

            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    function = tc.get("function", {})
                    tool_name = function.get("name")
                    tool_args = function.get("arguments", {})

                    if isinstance(tool_args, str):
                        try:
                            tool_args = json.loads(tool_args)
                        except json.JSONDecodeError:
                            tool_args = {}

                    if tool_name:
                        tool_calls.append({
                            "name": tool_name,
                            "arguments": tool_args,
                            "id": str(uuid.uuid4())
                        })
                        logger.info(f"Ollama tool call: {tool_name}")

            if "content" in message:
                text_response = message["content"] or ""

            input_tokens = result.get("prompt_eval_count", 0)
            output_tokens = result.get("eval_count", 0)

            return ToolCallingResponse(
                text=text_response,
                tool_calls=tool_calls,
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=0.0,  # Local models are free
                stop_reason=result.get("done_reason"),
            )
        finally:
            # Only close if we created a temporary client
            if client is not self.client:
                await client.aclose()