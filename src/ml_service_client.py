"""ML Service client for communicating with the ML microservice."""

import httpx
import logging
from typing import Dict, Any, Optional, List
from core.config import settings

logger = logging.getLogger(__name__)


class MLServiceClient:
    """Client for ML microservice communication."""
    
    def __init__(self):
        self.base_url = settings.ML_SERVICE_URL if hasattr(settings, 'ML_SERVICE_URL') else "http://ml-service:8003"
        self.timeout = httpx.Timeout(30.0)
    
    async def get_ai_models(self) -> List[Dict[str, Any]]:
        """Get available AI models from ML service."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/ml/models")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get AI models: {e}")
            return []
    
    async def get_model_capabilities(self, model_id: str) -> Dict[str, Any]:
        """Get capabilities of a specific model."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{self.base_url}/ml/models/{model_id}/capabilities")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get model capabilities: {e}")
            return {}
    
    async def execute_model(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a model with given input."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/ml/models/{model_id}/execute",
                    json=input_data
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to execute model: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> bool:
        """Check ML service health."""
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                response = await client.get(f"{self.base_url}/health")
                return response.status_code == 200
        except Exception:
            return False