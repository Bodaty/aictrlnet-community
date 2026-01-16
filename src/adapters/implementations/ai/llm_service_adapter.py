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
        
        # Check for discovery mode
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False
        
        # Handle both direct attributes and parameters dict
        if hasattr(config, 'parameters') and config.parameters:
            self.service_url = config.parameters.get("service_url", "http://localhost:8000")
            self.timeout = config.parameters.get("timeout", 30)
            self.api_key = config.parameters.get("api_key", "dev-token-for-testing")
        else:
            self.service_url = getattr(config, 'service_url', "http://localhost:8000")
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
    
    async def execute(self, task: Dict[str, Any]) -> AdapterResult:
        """Execute LLM task through internal service."""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Determine operation type
            operation = task.get("operation", "generate")
            
            if operation == "generate":
                result = await self._generate(task)
            elif operation == "chat":
                result = await self._chat(task)
            elif operation == "embedding":
                result = await self._embedding(task)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_used = datetime.utcnow()
            
            return AdapterResult(
                success=True,
                data=result,
                metadata={
                    "adapter": "llm_service",
                    "operation": operation,
                    "cached": result.get("cached", False)
                }
            )
            
        except Exception as e:
            logger.error(f"LLM Service execution failed: {e}")
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            
            return AdapterResult(
                success=False,
                error=str(e),
                metadata={"adapter": "llm_service"}
            )
    
    async def _generate(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Generate text using LLM service."""
        payload = {
            "prompt": task.get("prompt", ""),
            "model": task.get("model", "auto"),
            "max_tokens": task.get("max_tokens", 1000),
            "temperature": task.get("temperature", 0.7),
            "top_p": task.get("top_p", 0.9),
            "stream": task.get("stream", False)
        }

        # Add user_id if provided (for personalized model selection)
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