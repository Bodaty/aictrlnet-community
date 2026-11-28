"""Claude/Anthropic adapter implementation."""

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional
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
from llm.tier_resolver import PROVIDER_DEFAULT_MODELS

CLAUDE_DEFAULT_MODEL = PROVIDER_DEFAULT_MODELS["anthropic"]
# The GEO answer-engine series is measured on this model with web_search_20250305; it moves
# only by a deliberate switch recorded in the GEO ledger, not with the default model.
GEO_ANSWER_MODEL = "claude-opus-4-8"
# The cheapest current model: the health check only needs a 200.
HEALTH_CHECK_MODEL = "claude-haiku-5-5"

# Per million tokens (input, output), Anthropic first-party prices; matched by prefix,
# longest first. A model not listed is priced as the default model.
MODEL_PRICES = {
    "claude-fable-5-2": (8.0, 40.0),
    "claude-fable-5-1": (10.0, 50.0),
    "claude-fable-5": (10.0, 50.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-4-8": (5.0, 25.0),
    "claude-opus-4-7": (5.0, 25.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.0, 5.0),
}

# Models whose effort defaults to medium; the adapter asks for high, the depth the
# models it replaced ran at by default. Others default to high (or reject effort).
_MEDIUM_DEFAULT_EFFORT = ("claude-opus-5-5", "claude-haiku-5-5")


def _price_for(model: str):
    for prefix in sorted(MODEL_PRICES, key=len, reverse=True):
        if (model or "").startswith(prefix):
            return MODEL_PRICES[prefix]
    return MODEL_PRICES[CLAUDE_DEFAULT_MODEL]


def _output_config(model: str, requested_effort: Optional[str]) -> Optional[Dict[str, Any]]:
    if requested_effort:
        return {"effort": requested_effort}
    if (model or "").startswith(_MEDIUM_DEFAULT_EFFORT):
        return {"effort": "high"}
    return None


def _forced_tool_instruction(tool_choice: Optional[str]) -> Optional[str]:
    if not tool_choice or tool_choice in ("auto", "none"):
        return None
    if tool_choice == "required":
        return "You must call one of the tools before you answer."
    return f"You must call the {tool_choice} tool before you answer."


def _strict_compatible(schema: Dict[str, Any]) -> bool:
    """strict needs an object schema that closes its properties and names the required ones."""
    return (isinstance(schema, dict) and schema.get("type") == "object"
            and schema.get("additionalProperties") is False and "required" in schema)


logger = logging.getLogger(__name__)


class ClaudeAdapter(BaseAdapter, ToolCallingMixin):
    """Adapter for Anthropic Claude API integration."""

    PHI_PROVIDERS = frozenset(['anthropic'])

    def _phi_provider(self):
        return "anthropic"

    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.AI
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "https://api.anthropic.com/v1"
        self._assert_phi_egress_allowed()
        # Env fallback (Bodaty free-tier key) so per-execution instances built
        # with no stored credential still authenticate — mirrors the perplexity
        # adapter and the tiered GEO model.
        self.api_key = (
            config.api_key
            or config.credentials.get("api_key")
            or os.environ.get("ANTHROPIC_API_KEY")
        )
        
        # Check if we're in discovery-only mode
        self.discovery_only = config.custom_config.get("discovery_only", False)
        
        if not self.api_key and not self.discovery_only:
            raise ValueError("Anthropic API key is required")
    
    async def initialize(self) -> None:
        """Initialize the Claude adapter."""
        # Skip initialization in discovery-only mode
        if self.discovery_only:
            logger.info("Claude adapter initialized in discovery-only mode")
            return
        
        # Create HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"
            },
            timeout=self.config.timeout_seconds
        )
        
        logger.info("Claude adapter initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Claude adapter shutdown")
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available Claude models.
        
        In discovery mode, returns a static list of known models.
        Anthropic doesn't provide a models endpoint, so always returns static list.
        """
        # Claude models are relatively static, return known models
        return [
            {"id": "claude-3-opus-20240229", "name": "Claude 3 Opus", "capabilities": ["chat", "completion", "vision"], "max_tokens": 200000},
            {"id": "claude-3-sonnet-20240229", "name": "Claude 3 Sonnet", "capabilities": ["chat", "completion", "vision"], "max_tokens": 200000},
            {"id": "claude-3-haiku-20240307", "name": "Claude 3 Haiku", "capabilities": ["chat", "completion", "vision"], "max_tokens": 200000},
            {"id": "claude-2.1", "name": "Claude 2.1", "capabilities": ["chat", "completion"], "max_tokens": 200000},
            {"id": "claude-2.0", "name": "Claude 2.0", "capabilities": ["chat", "completion"], "max_tokens": 100000},
            {"id": "claude-instant-1.2", "name": "Claude Instant 1.2", "capabilities": ["chat", "completion"], "max_tokens": 100000}
        ]
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Claude adapter capabilities."""
        return [
            AdapterCapability(
                name="answer",
                description=(
                    "Answer a query with Claude's web_search server tool and "
                    "return the answer text plus the source URLs cited (GEO "
                    "answer-engine contract)"
                ),
                category="answer_engine",
                parameters={
                    "query": {"type": "string", "description": "The question to answer"},
                    "model": {"type": "string", "default": GEO_ANSWER_MODEL},
                    "max_tokens": {"type": "integer"},
                },
                required_parameters=["query"],
                async_supported=True,
                estimated_duration_seconds=8.0,
            ),
            AdapterCapability(
                name="chat_completion",
                description="Generate chat completions using Claude models",
                category="text_generation",
                parameters={
                    "model": {"type": "string", "description": "Model to use (e.g., claude-3-opus-20240229)"},
                    "messages": {"type": "array", "description": "Array of message objects"},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens to generate", "default": 1024},
                    "effort": {"type": "string", "description": "low, medium, high, xhigh or max (models that support effort)"},
                    "system": {"type": "string", "description": "System prompt"},
                    "stream": {"type": "boolean", "description": "Stream the response", "default": False}
                },
                required_parameters=["model", "messages"],
                async_supported=True,
                estimated_duration_seconds=3.0,
                cost_per_request=0.015  # Approximate for Claude 3
            ),
            AdapterCapability(
                name="text_completion",
                description="Generate text completions (legacy format)",
                category="text_generation",
                parameters={
                    "model": {"type": "string", "description": "Model to use"},
                    "prompt": {"type": "string", "description": "Text prompt"},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens", "default": 1024}
                },
                required_parameters=["model", "prompt"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.01
            ),
            AdapterCapability(
                name="analyze_image",
                description="Analyze images with Claude Vision",
                category="vision",
                parameters={
                    "model": {"type": "string", "description": "Vision-capable model"},
                    "messages": {"type": "array", "description": "Messages with image content"},
                    "max_tokens": {"type": "integer", "description": "Maximum tokens", "default": 1024}
                },
                required_parameters=["model", "messages"],
                async_supported=True,
                estimated_duration_seconds=5.0,
                cost_per_request=0.02
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to Claude."""
        self._assert_phi_egress_allowed(request)
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        if request.capability == "answer":
            return await self._handle_answer(request)
        elif request.capability == "chat_completion":
            return await self._handle_chat_completion(request)
        elif request.capability == "text_completion":
            return await self._handle_text_completion(request)
        elif request.capability == "analyze_image":
            return await self._handle_image_analysis(request)
        else:
            raise ValueError(f"Unknown capability: {request.capability}")
    
    async def _handle_chat_completion(self, request: AdapterRequest) -> AdapterResponse:
        """Handle chat completion requests."""
        return await self.__chat_completion_impl(request)

    async def _handle_answer(self, request: AdapterRequest) -> AdapterResponse:
        """Answer a query with Claude's web_search server tool; normalise to the
        GEO contract {content, citations, search_results, model}.

        Uses the Messages API web_search tool so the response carries text-block
        citations (url/title) and web_search_tool_result blocks, which we map to
        citations — the same shape as the perplexity/openai/gemini adapters.
        No temperature is sent (current Opus models reject sampling params).
        """
        start_time = datetime.utcnow()
        params = request.parameters or {}
        query = params.get("query") or params.get("prompt")
        if not query and isinstance(params.get("messages"), list):
            for m in reversed(params["messages"]):
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                    query = m["content"] if isinstance(m["content"], str) else None
                    if query:
                        break
        if not query:
            return AdapterResponse(
                request_id=request.id, capability=request.capability, status="error",
                error="answer capability requires a 'query' parameter",
                error_code="MISSING_QUERY",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

        model = params.get("model") or GEO_ANSWER_MODEL
        data = {
            "model": model,
            "max_tokens": params.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": query}],
            "tools": [{"type": "web_search_20250305", "name": "web_search"}],
        }

        try:
            response = await self.client.post("/messages", json=data)
            response.raise_for_status()
            result = response.json()

            content_text = ""
            citations, search_results = [], []

            def _add(url, title):
                if url:
                    citations.append(url)
                    search_results.append({"title": title, "url": url})

            for block in result.get("content", []):
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    content_text += block.get("text", "") or ""
                    for c in (block.get("citations") or []):
                        if isinstance(c, dict):
                            _add(c.get("url"), c.get("title"))
                elif btype == "web_search_tool_result":
                    inner = block.get("content")
                    if isinstance(inner, list):
                        for r in inner:
                            if isinstance(r, dict) and r.get("type") == "web_search_result":
                                _add(r.get("url"), r.get("title"))

            citations = list(dict.fromkeys(citations))
            seen, deduped = set(), []
            for s in search_results:
                if s["url"] not in seen:
                    seen.add(s["url"])
                    deduped.append(s)
            usage = result.get("usage") or {}

            return AdapterResponse(
                request_id=request.id, capability=request.capability, status="success",
                data={
                    "content": content_text,
                    "citations": citations,
                    "search_results": deduped,
                    "model": result.get("model") or model,
                    "usage": usage,
                },
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                cost=0.0,
                tokens_used=(usage.get("input_tokens", 0) + usage.get("output_tokens", 0)) or None,
                metadata={"claude_id": result.get("id")},
            )
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json()
                msg = error_data.get("error", {}).get("message", str(e))
            except Exception:
                msg = e.response.text or str(e)
            return AdapterResponse(
                request_id=request.id, capability=request.capability, status="error",
                error=f"Claude HTTP {e.response.status_code}: {msg}",
                error_code=f"HTTP_{e.response.status_code}",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id, capability=request.capability, status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def __chat_completion_impl(self, request: AdapterRequest) -> AdapterResponse:
        start_time = datetime.utcnow()

        try:
            # Convert messages to Claude format
            messages = self._convert_messages(request.parameters["messages"])
            
            # Extract system message if present
            system = request.parameters.get("system")
            if not system:
                # Check if first message is system
                if messages and messages[0]["role"] == "system":
                    system = messages[0]["content"]
                    messages = messages[1:]
            
            # Prepare request data
            data = {
                "model": request.parameters["model"],
                "messages": messages,
                "max_tokens": request.parameters.get("max_tokens", 1024),
                "stream": request.parameters.get("stream", False)
            }
            # No temperature/top_p/top_k: current Claude models reject sampling parameters.
            output_config = _output_config(data["model"], request.parameters.get("effort"))
            if output_config:
                data["output_config"] = output_config
            
            if system:
                data["system"] = system
            
            # Handle streaming if requested
            if data.get("stream"):
                return await self._handle_streaming_chat(request, data, start_time)
            
            # Make API request
            response = await self.client.post("/messages", json=data)
            response.raise_for_status()
            
            result = response.json()
            
            # Calculate metrics
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            input_tokens = result.get("usage", {}).get("input_tokens", 0)
            output_tokens = result.get("usage", {}).get("output_tokens", 0)
            total_tokens = input_tokens + output_tokens
            
            # Estimate cost
            cost = self._estimate_cost(
                request.parameters["model"],
                input_tokens,
                output_tokens
            )
            
            # Publish completion event
            await event_bus.publish(
                "adapter.claude.completion",
                {
                    "model": request.parameters["model"],
                    "tokens": total_tokens,
                    "duration_ms": duration_ms
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            # Convert response to OpenAI-compatible format
            content = result.get("content", [])
            text_content = " ".join([c["text"] for c in content if c["type"] == "text"])
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": text_content
                        },
                        "finish_reason": result.get("stop_reason", "stop")
                    }],
                    "usage": {
                        "prompt_tokens": input_tokens,
                        "completion_tokens": output_tokens,
                        "total_tokens": total_tokens
                    },
                    "model": result.get("model")
                },
                duration_ms=duration_ms,
                cost=cost,
                tokens_used=total_tokens,
                metadata={
                    "claude_id": result.get("id"),
                    "stop_reason": result.get("stop_reason")
                }
            )
            
        except httpx.HTTPStatusError as e:
            try:
                error_data = e.response.json() if e.response.content else {}
            except ValueError:
                error_data = {}
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
        chunks = []
        
        try:
            async with self.client.stream("POST", "/messages", json=data) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        chunk_data = line[6:]
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
                if chunk.get("type") == "content_block_delta":
                    delta = chunk.get("delta", {})
                    if delta.get("type") == "text_delta":
                        combined_content += delta.get("text", "")
            
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
    
    async def _handle_text_completion(self, request: AdapterRequest) -> AdapterResponse:
        """Handle legacy text completion format."""
        # Convert to chat format
        chat_request = AdapterRequest(
            capability="chat_completion",
            parameters={
                "model": request.parameters["model"],
                "messages": [{"role": "user", "content": request.parameters["prompt"]}],
                "max_tokens": request.parameters.get("max_tokens", 1024),
                "effort": request.parameters.get("effort"),
            },
            context=request.context
        )
        
        return await self._handle_chat_completion(chat_request)
    
    async def _handle_image_analysis(self, request: AdapterRequest) -> AdapterResponse:
        """Handle image analysis requests."""
        # Similar to chat completion but with image content
        return await self._handle_chat_completion(request)
    
    def _convert_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert messages to Claude format."""
        claude_messages = []
        
        for msg in messages:
            role = msg["role"]
            
            # Claude uses "user" and "assistant" roles
            if role == "system":
                # System messages handled separately
                continue
            
            claude_msg = {
                "role": role if role in ["user", "assistant"] else "user",
                "content": msg["content"]
            }
            
            # Handle image content if present
            if isinstance(msg.get("content"), list):
                claude_content = []
                for item in msg["content"]:
                    if item["type"] == "text":
                        claude_content.append({
                            "type": "text",
                            "text": item["text"]
                        })
                    elif item["type"] == "image_url":
                        claude_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": item["image_url"]["url"].split(",")[1]  # Remove data:image/jpeg;base64,
                            }
                        })
                claude_msg["content"] = claude_content
            
            claude_messages.append(claude_msg)
        
        return claude_messages
    
    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost from the model's per-million-token price (MODEL_PRICES)."""
        input_price, output_price = _price_for(model)
        return (input_tokens / 1_000_000) * input_price + (output_tokens / 1_000_000) * output_price

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Claude-specific health check."""
        try:
            # Simple test message
            test_data = {
                "model": HEALTH_CHECK_MODEL,
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 10
            }

            response = await self.client.post("/messages", json=test_data)

            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "test_response": "success"
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code
                }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }

    # ── ToolCallingMixin implementation ──────────────────────────────────

    async def chat_with_tools(self, request: ToolCallingRequest) -> ToolCallingResponse:
        """Execute tool-augmented chat via Anthropic's Messages API."""
        import os
        import uuid

        api_key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise Exception("ANTHROPIC_API_KEY not available")

        # Convert tools to Anthropic format (input_schema, not parameters)
        anthropic_tools = []
        for tool in request.tools:
            anthropic_tools.append({
                "name": tool.get("name", ""),
                "description": tool.get("description", ""),
                "input_schema": tool.get("parameters", tool.get("input_schema", {
                    "type": "object", "properties": {}, "required": []
                }))
            })

        # Build messages — Anthropic uses separate 'system' field
        api_messages = []
        system_text = request.system_prompt or ""
        for msg in request.messages:
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

        # Build request payload
        payload = {
            "model": request.model,
            "max_tokens": request.max_tokens or 4096,
            "messages": api_messages,
            "tools": anthropic_tools if request.tools else [],
        }
        # No temperature: current Claude models reject sampling parameters.
        output_config = _output_config(request.model, None)
        if output_config:
            payload["output_config"] = output_config

        # A forced tool_choice ("any" or a named tool) is a 400 on current models: ask
        # for it in the system prompt, with auto, and keep arguments schema-valid with
        # strict where the tool's schema allows it.
        forced = _forced_tool_instruction(request.tool_choice)
        if forced and request.tools:
            system_text = f"{system_text}\n\n{forced}" if system_text else forced
            for tool in anthropic_tools:
                if _strict_compatible(tool["input_schema"]):
                    tool["strict"] = True

        # Add system prompt (with prompt caching if enabled)
        if system_text:
            if request.cache_system_prefix:
                payload["system"] = [
                    {
                        "type": "text",
                        "text": system_text,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]
            else:
                payload["system"] = system_text

        # Handle tool_choice
        if request.tool_choice == "none":
            payload["tool_choice"] = {"type": "none"}
        elif forced and request.tools:
            payload["tool_choice"] = {"type": "auto"}

        logger.info(f"Claude chat_with_tools: {[t['name'] for t in anthropic_tools]}")

        # Use existing client or create temporary one
        client = self.client
        close_client = False
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url or "https://api.anthropic.com/v1",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "anthropic-beta": "prompt-caching-2024-07-31",
                    "content-type": "application/json"
                },
                timeout=120.0,
            )
            close_client = True

        try:
            response = await client.post("/messages", json=payload)

            if response.status_code != 200:
                logger.error(f"Anthropic tool calling error: {response.status_code} - {response.text}")
                raise Exception(f"Anthropic API error: {response.status_code}")

            result = response.json()

            # Extract tool calls and text
            tool_calls = []
            text_response = ""
            for content_block in result.get("content", []):
                if content_block.get("type") == "tool_use":
                    tool_calls.append({
                        "name": content_block.get("name"),
                        "arguments": content_block.get("input", {}),
                        "id": content_block.get("id", str(uuid.uuid4()))
                    })
                    logger.info(f"Claude tool call: {content_block.get('name')}")
                elif content_block.get("type") == "text":
                    text_response += content_block.get("text", "")

            usage = result.get("usage", {})
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            cached_tokens = usage.get("cache_read_input_tokens", 0) + usage.get("cache_creation_input_tokens", 0)

            cost = self._estimate_cost(request.model, input_tokens, output_tokens)

            return ToolCallingResponse(
                text=text_response,
                tool_calls=tool_calls,
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
                cached_tokens=cached_tokens,
                stop_reason=result.get("stop_reason"),
            )
        finally:
            if close_client:
                await client.aclose()