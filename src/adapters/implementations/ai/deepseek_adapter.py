"""DeepSeek Platform API adapter.

DeepSeek API is OpenAI-compatible (https://api.deepseek.com).
Models: deepseek-chat (V3), deepseek-reasoner (R1).
"""

import logging
import time
from typing import Any, Dict, List, Optional

import httpx

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability,
    AdapterConfig,
    AdapterCategory,
    AdapterResponse,
)

logger = logging.getLogger(__name__)


class DeepSeekAdapter(BaseAdapter):
    """Adapter for DeepSeek Platform API (OpenAI-compatible)."""

    def __init__(self, config: AdapterConfig):
        config.category = AdapterCategory.AI
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "https://api.deepseek.com"
        self.api_key = config.api_key or config.credentials.get("api_key", "")

        self.discovery_only = config.custom_config.get("discovery_only", False)

        if not self.api_key and not self.discovery_only:
            import os
            self.api_key = os.environ.get("DEEPSEEK_API_KEY", "")

    async def initialize(self) -> None:
        if self.discovery_only:
            logger.info("DeepSeek adapter initialized in discovery-only mode")
            return

        if not self.api_key:
            logger.warning("DeepSeek adapter: no API key configured")
            return

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout_seconds,
        )
        logger.info("DeepSeek adapter initialized")

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("DeepSeek adapter shutdown")

    async def list_models(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "deepseek-chat",
                "name": "DeepSeek V3",
                "capabilities": ["chat", "completion", "code-generation"],
                "max_tokens": 65536,
            },
            {
                "id": "deepseek-reasoner",
                "name": "DeepSeek R1",
                "capabilities": ["chat", "completion", "reasoning"],
                "max_tokens": 65536,
            },
        ]

    def get_capabilities(self) -> List[AdapterCapability]:
        return [
            AdapterCapability(
                name="chat_completion",
                description="Generate chat completions using DeepSeek models",
                category="text_generation",
                parameters={
                    "model": {
                        "type": "string",
                        "description": "Model to use (deepseek-chat, deepseek-reasoner)",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Array of message objects",
                    },
                    "temperature": {
                        "type": "number",
                        "description": "Sampling temperature (0-2)",
                        "default": 1.0,
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens to generate",
                    },
                    "stream": {
                        "type": "boolean",
                        "description": "Stream the response",
                        "default": False,
                    },
                },
                required_parameters=["model", "messages"],
                async_supported=True,
                estimated_duration_seconds=3.0,
                cost_per_request=0.001,
            ),
        ]

    async def execute(self, request) -> AdapterResponse:
        """Execute a DeepSeek API request (OpenAI-compatible)."""
        start = time.time()
        request_id = getattr(request, "id", "unknown")
        capability = getattr(request, "capability", "chat_completion")

        if not self.client:
            await self.initialize()

        if not self.client:
            return AdapterResponse(
                request_id=request_id,
                capability=capability,
                status="error",
                error="DeepSeek adapter not initialized (missing API key?)",
                duration_ms=(time.time() - start) * 1000,
            )

        params = request.parameters if hasattr(request, "parameters") else request
        model = params.get("model", "deepseek-chat")
        messages = params.get("messages", [])

        payload = {
            "model": model,
            "messages": messages,
            "temperature": params.get("temperature", 1.0),
            "max_tokens": params.get("max_tokens", 4096),
            "stream": False,
        }

        try:
            response = await self.client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            text = choices[0]["message"]["content"] if choices else ""
            usage = data.get("usage", {})

            return AdapterResponse(
                request_id=request_id,
                capability=capability,
                status="success",
                duration_ms=(time.time() - start) * 1000,
                tokens_used=usage.get("total_tokens"),
                data={
                    "text": text,
                    "model": data.get("model", model),
                    "usage": usage,
                    "finish_reason": choices[0].get("finish_reason") if choices else None,
                },
            )
        except Exception as e:
            logger.error(f"DeepSeek API error: {e}")
            return AdapterResponse(
                request_id=request_id,
                capability=capability,
                status="error",
                error=str(e),
                duration_ms=(time.time() - start) * 1000,
            )

    async def health_check(self) -> Dict[str, Any]:
        if self.discovery_only or not self.api_key:
            return {"status": "discovery_only", "provider": "deepseek"}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    f"{self.base_url}/v1/models",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code == 200:
                    return {"status": "healthy", "provider": "deepseek"}
                return {"status": "unhealthy", "provider": "deepseek", "error": f"HTTP {resp.status_code}"}
        except Exception as e:
            return {"status": "unhealthy", "provider": "deepseek", "error": str(e)}
