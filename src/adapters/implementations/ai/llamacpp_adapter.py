"""llama.cpp adapter implementation.

llama-server (https://github.com/ggml-org/llama.cpp/tree/master/tools/server)
serves an OpenAI-compatible HTTP API on the deployment's own machine:
  POST /v1/chat/completions   (images as image_url content parts)
  GET  /health

Used for local OCR and vision models (e.g. HunyuanOCR) so a page image never
leaves the machine. The endpoint comes from LLAMACPP_URL; there is no default,
so an unconfigured adapter refuses every call instead of guessing a host.
"""

import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import httpx

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)


logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = "Parse this document into Markdown."
DEFAULT_OCR_MAX_TOKENS = 4096
CHAT_ROUTE = "/v1/chat/completions"
HEALTH_ROUTE = "/health"
_UNCONFIGURED = (
    "llama.cpp endpoint is not configured. Required: set LLAMACPP_URL to the "
    "llama-server address on this machine (e.g. http://host.docker.internal:8089)."
)


def _image_part(image_b64: str) -> Dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}


class RemoteImageRefused(ValueError):
    """A chat message carried an image by reference rather than inline."""


def _assert_inline_images(messages: List[Dict[str, Any]]) -> None:
    # llama-server built with curl fetches http(s) image URLs itself: egress
    # the endpoint guard never sees. Only inline data: URIs are sent.
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or "image_url" not in part:
                continue
            ref = part["image_url"]
            url = ref.get("url") if isinstance(ref, dict) else ref
            if not isinstance(url, str) or not url.startswith("data:"):
                raise RemoteImageRefused(
                    "llama.cpp chat refuses an image passed by URL: llama-server would fetch "
                    "it itself. Required: send the image inline as a data: URI (or in "
                    "request.images)."
                )


class LlamaCppAdapter(BaseAdapter):
    """Adapter for a local llama-server (llama.cpp) OpenAI-compatible endpoint."""

    PHI_PROVIDERS = frozenset(['llamacpp'])

    def _phi_provider(self):
        return "llamacpp"

    def __init__(self, config: AdapterConfig):
        config.category = AdapterCategory.AI
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None

        raw = (
            config.base_url
            or config.credentials.get("base_url")
            or os.environ.get("LLAMACPP_URL")
        )
        if raw:
            raw = raw.rstrip("/")
            if raw.endswith("/v1"):
                raw = raw[: -len("/v1")]
            if "localhost" in raw:
                raw = raw.replace("localhost", "host.docker.internal")
        self.base_url: Optional[str] = raw or None
        self._assert_phi_egress_allowed()

        self.api_key = config.api_key or config.credentials.get("api_key")
        self.timeout = config.timeout_seconds or 300.0

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def initialize(self) -> None:
        if not self.base_url:
            logger.warning(f"llama.cpp adapter not configured: {_UNCONFIGURED}")
            return
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            follow_redirects=False,
            headers=self._build_headers(),
            timeout=self.timeout,
        )
        logger.info(f"llama.cpp adapter initialized at {self.base_url}")

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("llama.cpp adapter shutdown")

    def get_capabilities(self) -> List[AdapterCapability]:
        return [
            AdapterCapability(
                name="ocr_page",
                description="Read one page image into Markdown with a local llama.cpp vision/OCR model",
                category="vision",
                parameters={
                    "image_b64": {"type": "string", "description": "Base64-encoded page image"},
                    "prompt": {"type": "string", "default": DEFAULT_OCR_PROMPT},
                    "max_tokens": {"type": "integer", "default": DEFAULT_OCR_MAX_TOKENS},
                    "model": {"type": "string", "description": "Optional model name; llama-server serves the model it was started with"},
                },
                required_parameters=["image_b64"],
                async_supported=True,
                estimated_duration_seconds=20.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="chat",
                description="Chat completion against a local llama-server",
                category="text_generation",
                parameters={
                    "messages": {"type": "array", "description": "Array of OpenAI-format message objects"},
                    "max_tokens": {"type": "integer"},
                    "model": {"type": "string"},
                },
                required_parameters=["messages"],
                async_supported=True,
                estimated_duration_seconds=5.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="health_check",
                description="Report whether the llama-server endpoint is up and its model is loaded",
                category="metadata",
                parameters={},
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=0.2,
                cost_per_request=0.0,
            ),
        ]

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        self._assert_phi_egress_allowed(request)
        if request.capability not in ("ocr_page", "chat", "health_check"):
            raise ValueError(f"Unknown capability: {request.capability}")
        start_time = datetime.utcnow()
        if not self.base_url or self.client is None:
            return self._error(request, start_time, _UNCONFIGURED, "NOT_CONFIGURED")
        if request.capability == "health_check":
            health = await self._probe_health()
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data=health,
                duration_ms=self._elapsed_ms(start_time),
                cost=0.0,
            )
        if request.capability == "ocr_page":
            payload = self._ocr_payload(request)
        else:
            try:
                payload = self._chat_payload(request)
            except RemoteImageRefused as e:
                return self._error(request, start_time, str(e), "REMOTE_IMAGE_REFUSED")
        return await self._post_chat(request, payload, start_time)

    def _ocr_payload(self, request: AdapterRequest) -> Dict[str, Any]:
        params = request.parameters
        image_b64 = params.get("image_b64")
        if not image_b64 and request.images:
            image_b64 = request.images[0]
        if not image_b64:
            raise ValueError("ocr_page requires 'image_b64': the base64-encoded page image")
        payload: Dict[str, Any] = {
            "messages": [{
                "role": "user",
                "content": [
                    _image_part(image_b64),
                    {"type": "text", "text": params.get("prompt") or DEFAULT_OCR_PROMPT},
                ],
            }],
            "temperature": request.temperature if request.temperature is not None else 0,
            "max_tokens": params.get("max_tokens") or DEFAULT_OCR_MAX_TOKENS,
        }
        if params.get("model"):
            payload["model"] = params["model"]
        return payload

    def _chat_payload(self, request: AdapterRequest) -> Dict[str, Any]:
        params = request.parameters
        messages = [dict(m) for m in params["messages"]]
        _assert_inline_images(messages)
        if request.images:
            for message in reversed(messages):
                if message.get("role") == "user":
                    text = message.get("content") or ""
                    message["content"] = [_image_part(img) for img in request.images] + [
                        {"type": "text", "text": text}
                    ]
                    break
            else:
                raise ValueError("chat with images requires a user message to attach them to")
        payload: Dict[str, Any] = {"messages": messages}
        temperature = request.temperature if request.temperature is not None else params.get("temperature")
        if temperature is not None:
            payload["temperature"] = temperature
        if params.get("max_tokens"):
            payload["max_tokens"] = params["max_tokens"]
        if params.get("model"):
            payload["model"] = params["model"]
        if request.response_format is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": request.response_format},
            }
        return payload

    async def _post_chat(
        self, request: AdapterRequest, payload: Dict[str, Any], start_time: datetime
    ) -> AdapterResponse:
        try:
            response = await self.client.post(CHAT_ROUTE, json=payload)
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPStatusError as e:
            return self._error(
                request, start_time,
                f"llama.cpp HTTP {e.response.status_code}: {e.response.text or e}",
                f"HTTP_{e.response.status_code}",
            )
        except httpx.HTTPError as e:
            return self._error(
                request, start_time,
                f"llama.cpp server at {self.base_url} unreachable: {type(e).__name__}: {e}",
                "UNREACHABLE",
            )

        choices = result.get("choices") or []
        if not choices:
            return self._error(request, start_time, "llama.cpp returned no choices", "EMPTY_RESPONSE")
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = result.get("usage") or {}
        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="success",
            data={"text": text, "model": result.get("model"), "usage": usage},
            duration_ms=self._elapsed_ms(start_time),
            cost=0.0,
            tokens_used=usage.get("total_tokens", 0),
            metadata={"llamacpp_id": result.get("id")},
        )

    async def _probe_health(self) -> Dict[str, Any]:
        try:
            response = await self.client.get(HEALTH_ROUTE)
        except httpx.HTTPError as e:
            return {"healthy": False, "detail": f"unreachable at {self.base_url}: {type(e).__name__}: {e}"}
        if response.status_code == 200:
            return {"healthy": True, "detail": f"llama-server at {self.base_url} is up"}
        return {
            "healthy": False,
            "detail": f"llama-server at {self.base_url} answered HTTP {response.status_code}: {response.text[:200]}",
        }

    async def _perform_health_check(self) -> Dict[str, Any]:
        if not self.base_url or self.client is None:
            return {"status": "unhealthy", "healthy": False, "detail": _UNCONFIGURED}
        health = await self._probe_health()
        return {"status": "healthy" if health["healthy"] else "unhealthy", **health}

    @staticmethod
    def _elapsed_ms(start_time: datetime) -> float:
        return (datetime.utcnow() - start_time).total_seconds() * 1000

    def _error(self, request: AdapterRequest, start_time: datetime, message: str, code: str) -> AdapterResponse:
        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="error",
            error=message,
            error_code=code,
            duration_ms=self._elapsed_ms(start_time),
        )
