"""vLLM adapter implementation.

vLLM (https://github.com/vllm-project/vllm) serves an OpenAI-compatible HTTP API:
  POST /v1/chat/completions
  GET  /v1/models

Tool calling works natively when the server is started with
  --enable-auto-tool-choice --tool-call-parser <hermes|mistral|llama3_json|...>

Authentication is optional (vLLM accepts an --api-key flag but typically runs without one).

This adapter mirrors openai_adapter.py (same wire format) and borrows the
localhost -> host.docker.internal translation from ollama_adapter.py since
vLLM is self-hosted alongside the AICtrlNet stack.
"""

import logging
from typing import Any, Dict, List, Optional
import httpx
import json
import uuid
from datetime import datetime

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from adapters.tool_calling import ToolCallingMixin, ToolCallingRequest, ToolCallingResponse
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


def _strip_vllm_prefix(model: Optional[str]) -> Optional[str]:
    """Strip the routing prefix so vLLM sees the bare served-model-name."""
    if model and model.startswith("vllm:"):
        return model[len("vllm:"):]
    return model


class VLLMAdapter(BaseAdapter, ToolCallingMixin):
    """Adapter for a self-hosted vLLM OpenAI-compatible inference server."""

    def __init__(self, config: AdapterConfig):
        config.category = AdapterCategory.AI
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None
        self.discovery_only = config.custom_config.get("discovery_only", False)

        raw = config.base_url or config.credentials.get("base_url", "http://localhost:8000")
        if "localhost" in raw and not self.discovery_only:
            raw = raw.replace("localhost", "host.docker.internal")
        raw = raw.rstrip("/")
        self.base_url = raw if raw.endswith("/v1") else f"{raw}/v1"

        self.api_key = config.api_key or config.credentials.get("api_key")
        self.timeout = config.timeout_seconds or 120.0

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def initialize(self) -> None:
        """Initialize the vLLM adapter."""
        if self.discovery_only:
            logger.info("vLLM adapter initialized in discovery-only mode")
            return

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=self._build_headers(),
            timeout=self.timeout,
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10,
                keepalive_expiry=30.0,
            ),
        )

        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            data = response.json()
            count = len(data.get("data", []))
            logger.info(f"vLLM adapter initialized at {self.base_url} ({count} model(s) served)")
        except Exception as e:
            logger.warning(f"Failed to reach vLLM server at {self.base_url}: {e}")

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("vLLM adapter shutdown")

    async def list_models(self) -> List[Dict[str, Any]]:
        """List models served by the vLLM instance.

        vLLM only serves the model(s) the operator loaded at startup, so a
        static fallback would be misleading. In discovery_only mode we
        return an empty list rather than fabricate entries.
        """
        if self.discovery_only or not self.client:
            return []

        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            data = response.json()
            models = []
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if not model_id:
                    continue
                models.append({
                    "id": model_id,
                    "name": model_id,
                    "capabilities": ["chat", "completion"],
                    "owned_by": model.get("owned_by"),
                })
            return models
        except Exception as e:
            logger.warning(f"Failed to list vLLM models: {e}")
            return []

    def get_capabilities(self) -> List[AdapterCapability]:
        return [
            AdapterCapability(
                name="chat_completion",
                description="Generate chat completions via a vLLM OpenAI-compatible server",
                category="text_generation",
                parameters={
                    "model": {"type": "string", "description": "Model name as known to vLLM (or vllm: prefix)"},
                    "messages": {"type": "array", "description": "Array of OpenAI-format message objects"},
                    "temperature": {"type": "number", "default": 1.0},
                    "max_tokens": {"type": "integer"},
                    "stream": {"type": "boolean", "default": False},
                    "tools": {"type": "array", "description": "Optional OpenAI-format tool definitions"},
                    "tool_choice": {"type": "string", "default": "auto"},
                },
                required_parameters=["model", "messages"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="list_models",
                description="List models currently served by the vLLM instance",
                category="metadata",
                parameters={},
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=0.2,
                cost_per_request=0.0,
            ),
        ]

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Dispatch capability requests.

        Accepts both ``chat_completion`` (canonical) and ``chat`` so the
        workflow-generation path in llm/generation.py can call this
        adapter with the same capability name it uses for Ollama.
        """
        if request.capability in ("chat_completion", "chat"):
            return await self._handle_chat_completion(request)
        if request.capability == "list_models":
            return await self._handle_list_models(request)
        raise ValueError(f"Unknown capability: {request.capability}")

    async def _handle_list_models(self, request: AdapterRequest) -> AdapterResponse:
        start_time = datetime.utcnow()
        models = await self.list_models()
        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="success",
            data={"models": models},
            duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            cost=0.0,
        )

    async def _handle_chat_completion(self, request: AdapterRequest) -> AdapterResponse:
        start_time = datetime.utcnow()

        try:
            data: Dict[str, Any] = {
                "model": _strip_vllm_prefix(request.parameters["model"]),
                "messages": request.parameters["messages"],
                "temperature": request.parameters.get("temperature", 1.0),
                "max_tokens": request.parameters.get("max_tokens"),
                "stream": request.parameters.get("stream", False),
            }
            tools = request.parameters.get("tools")
            if tools:
                data["tools"] = tools
                data["tool_choice"] = request.parameters.get("tool_choice", "auto")
            data = {k: v for k, v in data.items() if v is not None}

            if data.get("stream"):
                return await self._handle_streaming_chat(request, data, start_time)

            response = await self.client.post("/chat/completions", json=data)
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            tokens_used = result.get("usage", {}).get("total_tokens", 0)

            await event_bus.publish(
                "adapter.vllm.completion",
                {
                    "model": data["model"],
                    "tokens": tokens_used,
                    "duration_ms": duration_ms,
                },
                source_id=self.id,
                source_type="adapter",
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "choices": result.get("choices", []),
                    "usage": result.get("usage", {}),
                    "model": result.get("model"),
                },
                duration_ms=duration_ms,
                cost=0.0,
                tokens_used=tokens_used,
                metadata={
                    "vllm_id": result.get("id"),
                    "created": result.get("created"),
                },
            )

        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except Exception:
                pass
            error_message = error_data.get("error", {}).get("message") if isinstance(error_data, dict) else None
            error_message = error_message or e.response.text or str(e)
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=f"vLLM HTTP {e.response.status_code}: {error_message}",
                error_code=f"HTTP_{e.response.status_code}",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _handle_streaming_chat(
        self,
        request: AdapterRequest,
        data: Dict[str, Any],
        start_time: datetime,
    ) -> AdapterResponse:
        """Stream a chat completion via SSE.

        vLLM follows the OpenAI streaming convention. We pass
        ``stream_options.include_usage`` so the final chunk carries token
        usage (otherwise vLLM omits the ``usage`` field for streaming
        responses).
        """
        data = dict(data)
        data["stream_options"] = {"include_usage": True}

        chunks: List[Dict[str, Any]] = []
        usage: Dict[str, Any] = {}

        try:
            async with self.client.stream("POST", "/chat/completions", json=data) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk_data = line[6:]
                    if chunk_data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(chunk_data)
                    except json.JSONDecodeError:
                        continue
                    chunks.append(chunk)
                    if chunk.get("usage"):
                        usage = chunk["usage"]

            combined_content = ""
            for chunk in chunks:
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {}) or {}
                if "content" in delta and delta["content"]:
                    combined_content += delta["content"]

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            tokens_used = usage.get("total_tokens", 0)

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "choices": [{
                        "message": {"role": "assistant", "content": combined_content},
                        "finish_reason": "stop",
                    }],
                    "usage": usage,
                    "stream_chunks": len(chunks),
                },
                duration_ms=duration_ms,
                cost=0.0,
                tokens_used=tokens_used,
                metadata={"streaming": True},
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _perform_health_check(self) -> Dict[str, Any]:
        try:
            response = await self.client.get("/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            return {
                "status": "healthy",
                "available_models": len(models),
                "models": [m.get("id") for m in models[:5]],
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    # ── ToolCallingMixin implementation ──────────────────────────────────

    async def chat_with_tools(self, request: ToolCallingRequest) -> ToolCallingResponse:
        """Execute a tool-augmented chat via vLLM's OpenAI-compatible endpoint.

        Fails loudly with a clear remediation message if the server returns
        4xx for the tool-calling request — usually because vLLM was started
        without --enable-auto-tool-choice and a --tool-call-parser.
        """
        model = _strip_vllm_prefix(request.model)

        openai_tools = []
        for tool in request.tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {
                        "type": "object", "properties": {}, "required": [],
                    }),
                },
            })

        api_messages: List[Dict[str, Any]] = []
        for msg in request.messages:
            if msg["role"] == "tool":
                api_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id", msg.get("tool_use_id", "")),
                    "content": msg["content"],
                })
            elif msg["role"] == "assistant" and msg.get("tool_calls"):
                tc_list = []
                for tc in msg["tool_calls"]:
                    tc_list.append({
                        "id": tc.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc.get("arguments", {})),
                        },
                    })
                api_messages.append({
                    "role": "assistant",
                    "content": msg.get("content"),
                    "tool_calls": tc_list,
                })
            else:
                api_messages.append({"role": msg["role"], "content": msg["content"]})

        payload: Dict[str, Any] = {
            "model": model,
            "messages": api_messages,
            "tools": openai_tools,
            "temperature": request.temperature,
        }
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens

        if request.tool_choice == "required":
            payload["tool_choice"] = "required"
        elif request.tool_choice == "auto":
            payload["tool_choice"] = "auto"
        else:
            payload["tool_choice"] = {
                "type": "function",
                "function": {"name": request.tool_choice},
            }

        logger.info(
            f"vLLM chat_with_tools model={model} tools={[t['function']['name'] for t in openai_tools]} "
            f"tool_choice={request.tool_choice}"
        )

        client = self.client
        close_client = False
        if client is None:
            client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=self._build_headers(),
                timeout=self.timeout,
            )
            close_client = True

        try:
            response = await client.post("/chat/completions", json=payload)

            if response.status_code >= 400:
                body = response.text
                logger.error(f"vLLM tool calling error: {response.status_code} - {body}")
                if response.status_code == 400:
                    raise Exception(
                        f"vLLM server at {self.base_url} did not accept the tool-calling request. "
                        f"Verify the server was started with --enable-auto-tool-choice "
                        f"--tool-call-parser <hermes|mistral|llama3_json|...>. "
                        f"Original error: {body}"
                    )
                raise Exception(f"vLLM API error {response.status_code}: {body}")

            result = response.json()

            tool_calls: List[Dict[str, Any]] = []
            text_response = ""

            choice = (result.get("choices") or [{}])[0]
            message = choice.get("message", {}) or {}

            if message.get("tool_calls"):
                for tc in message["tool_calls"]:
                    function = tc.get("function", {}) or {}
                    raw_args = function.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        if not raw_args.strip():
                            tool_args: Dict[str, Any] = {}
                        else:
                            try:
                                tool_args = json.loads(raw_args)
                            except json.JSONDecodeError:
                                tool_args = {}
                    else:
                        tool_args = raw_args or {}

                    tool_calls.append({
                        "name": function.get("name"),
                        "arguments": tool_args,
                        "id": tc.get("id", str(uuid.uuid4())),
                    })
                    logger.info(f"vLLM tool call: {function.get('name')}")

            if message.get("content"):
                text_response = message["content"]

            usage = result.get("usage", {}) or {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)

            return ToolCallingResponse(
                text=text_response or None,
                tool_calls=tool_calls,
                tokens_used=input_tokens + output_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=0.0,
                stop_reason=choice.get("finish_reason"),
            )
        finally:
            if close_client:
                await client.aclose()
