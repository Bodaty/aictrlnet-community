"""Perplexity adapter — answer engine with citations.

Perplexity's Sonar models (https://docs.perplexity.ai) expose an
OpenAI-compatible endpoint that, unlike a plain chat completion, performs a
live web search and returns the sources it used:

  POST https://api.perplexity.ai/chat/completions
  -> choices[0].message.content  (the answer)
  -> citations[]                 (source URLs)
  -> search_results[]            ({title, url, date, ...})

This adapter exposes a single **answer** capability that normalises that shape
to `{content, citations, search_results, model}` — the engine-agnostic contract
the GEO audit (and any future answer-engine consumer) reads. OpenAI/Gemini will
add the same `answer` capability in Phase B3.

Auth is Bearer with the Perplexity API key. Resolution order mirrors how vLLM
resolves its endpoint, so per-execution adapter instances that carry no explicit
key still pick up the operator-configured one:
  AdapterConfig.api_key -> credentials["api_key"] -> PERPLEXITY_API_KEY env.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from datetime import datetime
import httpx

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory,
)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.perplexity.ai"
DEFAULT_MODEL = "sonar"


class PerplexityAdapter(BaseAdapter):
    """Perplexity Sonar answer-engine adapter."""

    node_type = "perplexity"

    def __init__(self, config: AdapterConfig):
        config.category = AdapterCategory.AI
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None

        raw = (
            config.base_url
            or config.credentials.get("base_url")
            or os.environ.get("PERPLEXITY_BASE_URL")
            or DEFAULT_BASE_URL
        )
        self.base_url = raw.rstrip("/")

        self.api_key = (
            config.api_key
            or config.credentials.get("api_key")
            or os.environ.get("PERPLEXITY_API_KEY")
        )
        self.default_model = (
            config.custom_config.get("model")
            or config.credentials.get("model")
            or DEFAULT_MODEL
        )
        self.timeout = config.timeout_seconds or 120.0

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def initialize(self) -> None:
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
        if not self.api_key:
            # Don't fabricate success — surface the misconfiguration. The
            # request will fail loud (401) rather than silently return nothing.
            logger.warning(
                "Perplexity adapter initialized without an API key "
                "(set PERPLEXITY_API_KEY or configure the adapter credential)"
            )

    async def shutdown(self) -> None:
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Perplexity adapter shutdown")

    def get_capabilities(self) -> List[AdapterCapability]:
        return [
            AdapterCapability(
                name="answer",
                description=(
                    "Answer a query via Perplexity Sonar (live web search) and "
                    "return the answer text plus the source URLs it cited"
                ),
                category="answer_engine",
                parameters={
                    "query": {"type": "string", "description": "The question to answer"},
                    "model": {"type": "string", "default": DEFAULT_MODEL},
                    "max_tokens": {"type": "integer"},
                    "temperature": {"type": "number"},
                    "system": {"type": "string", "description": "Optional system prompt"},
                    "search_recency_filter": {"type": "string", "description": "e.g. month/week/day"},
                },
                required_parameters=["query"],
                async_supported=True,
                estimated_duration_seconds=8.0,
                cost_per_request=0.0,
            ),
        ]

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        if request.capability == "answer":
            return await self._handle_answer(request)
        raise ValueError(f"Unknown capability: {request.capability}")

    def _extract_query(self, params: Dict[str, Any]) -> Optional[str]:
        """Accept query / prompt / messages so the node mapping is forgiving."""
        q = params.get("query") or params.get("prompt")
        if q:
            return q
        messages = params.get("messages")
        if isinstance(messages, list):
            for m in reversed(messages):
                if isinstance(m, dict) and m.get("role") == "user" and m.get("content"):
                    return m["content"]
        return None

    async def _handle_answer(self, request: AdapterRequest) -> AdapterResponse:
        start_time = datetime.utcnow()
        params = request.parameters or {}
        query = self._extract_query(params)
        if not query:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error="answer capability requires a 'query' parameter",
                error_code="MISSING_QUERY",
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

        if self.client is None:
            await self.initialize()

        messages = []
        system = params.get("system")
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": query})

        body: Dict[str, Any] = {
            "model": params.get("model") or self.default_model,
            "messages": messages,
        }
        if params.get("max_tokens") is not None:
            body["max_tokens"] = params["max_tokens"]
        if params.get("temperature") is not None:
            body["temperature"] = params["temperature"]
        if params.get("search_recency_filter"):
            body["search_recency_filter"] = params["search_recency_filter"]

        try:
            resp = await self.client.post("/chat/completions", json=body)
            resp.raise_for_status()
            result = resp.json()

            choices = result.get("choices") or []
            content = ""
            if choices and isinstance(choices[0], dict):
                content = (choices[0].get("message") or {}).get("content", "") or ""

            citations = [c for c in (result.get("citations") or []) if isinstance(c, str)]
            search_results = [
                sr for sr in (result.get("search_results") or []) if isinstance(sr, dict)
            ]
            usage = result.get("usage") or {}

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "content": content,
                    "citations": citations,
                    "search_results": search_results,
                    "model": result.get("model") or body["model"],
                    "usage": usage,
                },
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
                cost=0.0,
                tokens_used=usage.get("total_tokens"),
                metadata={"perplexity_id": result.get("id")},
            )
        except httpx.HTTPStatusError as e:
            error_data = {}
            try:
                error_data = e.response.json()
            except Exception:
                pass
            msg = None
            if isinstance(error_data, dict):
                err = error_data.get("error")
                msg = err.get("message") if isinstance(err, dict) else err
            msg = msg or e.response.text or str(e)
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=f"Perplexity HTTP {e.response.status_code}: {msg}",
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
