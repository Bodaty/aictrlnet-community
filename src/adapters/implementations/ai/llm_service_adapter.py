"""LLM Service Adapter for internal LLM service."""

import json
import logging
from typing import Dict, Any, List, Optional
import httpx
from datetime import datetime

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability,
    AdapterMetrics,
    AdapterConfig,
    AdapterResponse as AdapterResult,
    AdapterCategory
)

logger = logging.getLogger(__name__)


class LLMServiceAdapter(BaseAdapter):
    """Adapter for internal LLM service at port 8000.
    
    This adapter connects to our internal LLM service that provides:
    - Unified LLM access
    - Response caching
    - Cost tracking
    - Model routing
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        import os

        # Check for discovery mode
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False

        # Resolve service URL: env var > config > default
        # On GCP Cloud Run, the LLM service is the same process on port 8080
        default_url = os.environ.get("LLM_SERVICE_INTERNAL_URL", "http://localhost:8000")

        # Handle both direct attributes and parameters dict
        if hasattr(config, 'parameters') and config.parameters:
            self.service_url = config.parameters.get("service_url", default_url)
            self.timeout = config.parameters.get("timeout", 30)
            self.api_key = config.parameters.get("api_key", "dev-token-for-testing")
        else:
            self.service_url = getattr(config, 'service_url', default_url)
            self.timeout = getattr(config, 'timeout', 30)
            self.api_key = getattr(config, 'api_key', "dev-token-for-testing")
        self.api_prefix = "/api/v1/llm"
        self._client = None
    
    @property
    def adapter_type(self) -> AdapterCategory:
        return AdapterCategory.AI
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Lazy initialization of HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.service_url,
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._client
    
    async def initialize(self) -> None:
        """Initialize connection to LLM service."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            self.initialized = True
            logger.info("LLM Service Adapter initialized in discovery mode")
            return
            
        try:
            # Test connection
            response = await self.client.get(f"{self.api_prefix}/status")
            if response.status_code == 200:
                self.initialized = True
                logger.info(f"LLM Service Adapter initialized: {self.service_url}")
            else:
                raise ConnectionError(f"LLM Service returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM Service Adapter: {e}")
            raise
    
    async def execute(self, task) -> AdapterResult:
        """Execute LLM task through internal service.

        Accepts either a plain dict (legacy) or an AdapterRequest object
        (used by workflow nodes via _call_adapter).
        """
        if not self.initialized:
            await self.initialize()

        start_time = datetime.utcnow()

        # Preserve AdapterRequest metadata before normalizing to dict
        request_id = getattr(task, 'id', 'llm-service-internal')
        capability = getattr(task, 'capability', 'generate')

        # Normalise: convert AdapterRequest to dict
        if hasattr(task, 'capability'):
            operation = task.capability
            params = task.parameters if hasattr(task, 'parameters') else {}
            task = {"operation": operation, **params}

        try:
            operation = task.get("operation", "generate")

            if operation == "generate":
                result = await self._generate(task)
            elif operation == "chat":
                result = await self._chat(task)
            elif operation == "embedding":
                result = await self._embedding(task)
            else:
                raise ValueError(f"Unsupported operation: {operation}")

            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_used = datetime.utcnow()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return AdapterResult(
                request_id=request_id,
                capability=capability,
                status="success",
                data=result,
                duration_ms=duration_ms,
                cost=result.get("cost", 0.0) if isinstance(result, dict) else 0.0,
                metadata={
                    "adapter": "llm_service",
                    "operation": operation,
                    "cached": result.get("cached", False) if isinstance(result, dict) else False
                }
            )

        except Exception as e:
            logger.error(f"LLM Service execution failed: {e}")
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            return AdapterResult(
                request_id=request_id,
                capability=capability,
                status="error",
                error=str(e),
                duration_ms=duration_ms,
                metadata={"adapter": "llm_service"}
            )
    
    async def _generate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using LLM service.

        First tries direct Python call (same-process, no auth needed).
        Falls back to HTTP if direct call is unavailable.
        """
        prompt = task.get("prompt", "")
        model = task.get("model", "auto")
        max_tokens = task.get("max_tokens", 1000)
        temperature = task.get("temperature", 0.7)

        # Try direct Python call first (avoids HTTP auth issues on Cloud Run)
        try:
            from llm.service import llm_service
            # Don't pass model_override — let the LLM service use its own
            # model selection (system default). The llm-service adapter is
            # a bridge, not a model-specific adapter.
            result = await llm_service.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # LLMResponse has: text, model_used, tokens_used, cost, cache_hit
            return {
                "text": result.text,
                "model": result.model_used,
                "usage": {"total_tokens": result.tokens_used},
                "cached": result.cache_hit,
                "cost": result.cost,
            }
        except Exception as direct_err:
            logger.warning(f"Direct LLM call failed ({type(direct_err).__name__}: {direct_err}), falling back to HTTP")

        # Fallback: HTTP call
        payload = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": task.get("top_p", 0.9),
            "stream": task.get("stream", False)
        }
        if "user_id" in task:
            payload["user_id"] = task["user_id"]

        response = await self.client.post(
            f"{self.api_prefix}/generate",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _chat(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Chat completion using LLM service."""
        messages = task.get("messages", [])
        if not messages and "prompt" in task:
            # Convert single prompt to messages format
            messages = [{"role": "user", "content": task["prompt"]}]
        
        payload = {
            "messages": messages,
            "model": task.get("model", "auto"),
            "max_tokens": task.get("max_tokens", 1000),
            "temperature": task.get("temperature", 0.7),
            "stream": task.get("stream", False)
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/chat",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _embedding(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate embeddings using LLM service."""
        payload = {
            "text": task.get("text", ""),
            "model": task.get("model", "text-embedding-ada-002")
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/embedding",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Get adapter capabilities."""
        return [
            AdapterCapability(
                name="model.llm.generation",
                enabled=True,
                description="Text generation via internal LLM service"
            ),
            AdapterCapability(
                name="model.llm.chat",
                enabled=True,
                description="Chat completion via internal LLM service"
            ),
            AdapterCapability(
                name="model.llm.embedding",
                enabled=True,
                description="Text embeddings via internal LLM service"
            ),
            AdapterCapability(
                name="caching",
                enabled=True,
                description="Response caching for efficiency"
            ),
            AdapterCapability(
                name="cost_tracking",
                enabled=True,
                description="Track API usage costs"
            ),
            AdapterCapability(
                name="model_routing",
                enabled=True,
                description="Automatic model selection"
            )
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of LLM service."""
        try:
            response = await self.client.get(f"{self.api_prefix}/health")
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "service": "llm_service",
                    "url": self.service_url,
                    "details": response.json()
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": "llm_service",
                    "url": self.service_url,
                    "error": f"Status code {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "llm_service",
                "url": self.service_url,
                "error": str(e)
            }
    
    async def cleanup(self) -> None:
        """Cleanup resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self.initialized = False
    
    async def shutdown(self) -> None:
        """Shutdown the adapter cleanly."""
        await self.cleanup()