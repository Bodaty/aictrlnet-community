"""ML Service Adapter for internal ML microservice."""

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


class MLServiceAdapter(BaseAdapter):
    """Adapter for internal ML service at port 8003.
    
    This adapter connects to our ML microservice that provides:
    - Risk assessment models
    - Quality verification
    - Time series forecasting
    - Anomaly detection
    - Classification and regression
    - Model training and fine-tuning
    """
    
    def __init__(self, config: AdapterConfig):
        super().__init__(config)
        
        # Check for discovery mode
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False
        
        self.service_url = config.base_url or "http://ml-service:8003"
        self.timeout = config.timeout_seconds or 60  # ML ops can take longer
        self.api_prefix = "/ml"
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
                timeout=self.timeout
            )
        return self._client
    
    async def initialize(self) -> None:
        """Initialize connection to ML service."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            self.initialized = True
            logger.info("ML Service Adapter initialized in discovery mode")
            return
            
        try:
            # Test connection
            response = await self.client.get(f"{self.api_prefix}/status")
            if response.status_code == 200:
                self.initialized = True
                logger.info(f"ML Service Adapter initialized: {self.service_url}")
            else:
                raise ConnectionError(f"ML Service returned status {response.status_code}")
        except Exception as e:
            logger.error(f"Failed to initialize ML Service Adapter: {e}")
            raise
    
    async def execute(self, task: Dict[str, Any]) -> AdapterResult:
        """Execute ML task through internal service."""
        if not self.initialized:
            await self.initialize()
        
        import uuid
        from datetime import datetime
        start_time = datetime.utcnow()
        request_id = str(uuid.uuid4())
        
        try:
            # Determine operation type
            operation = task.get("operation", "predict")
            
            if operation == "risk_assessment":
                result = await self._assess_risk(task)
            elif operation == "quality_verification":
                result = await self._verify_quality(task)
            elif operation == "forecast":
                result = await self._forecast(task)
            elif operation == "anomaly_detection":
                result = await self._detect_anomalies(task)
            elif operation == "classify":
                result = await self._classify(task)
            elif operation == "regression":
                result = await self._regression(task)
            elif operation == "train":
                result = await self._train_model(task)
            elif operation == "predict":
                result = await self._predict(task)
            else:
                raise ValueError(f"Unsupported operation: {operation}")
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.successful_requests += 1
            self.metrics.last_used = datetime.utcnow()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResult(
                request_id=request_id,
                capability=f"ml.{operation}",
                status="success",
                data=result,
                duration_ms=duration_ms,
                metadata={
                    "adapter": "ml_service",
                    "operation": operation,
                    "model_type": result.get("model_type", "unknown")
                }
            )
            
        except Exception as e:
            logger.error(f"ML Service execution failed: {e}")
            self.metrics.total_requests += 1
            self.metrics.failed_requests += 1
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResult(
                request_id=request_id,
                capability="ml.unknown",
                status="error",
                error=str(e),
                duration_ms=duration_ms,
                metadata={"adapter": "ml_service"}
            )
    
    async def _assess_risk(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Perform risk assessment."""
        payload = {
            "task_id": task.get("task_id", ""),
            "task_data": task.get("data", {}),
            "risk_factors": task.get("risk_factors", [])
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/risk/assess",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _verify_quality(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Verify data/model quality."""
        payload = {
            "data": task.get("data", {}),
            "quality_metrics": task.get("metrics", ["completeness", "accuracy"]),
            "threshold": task.get("threshold", 0.8)
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/quality/verify",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _forecast(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Time series forecasting."""
        payload = {
            "data": task.get("data", []),
            "periods": task.get("periods", 10),
            "model": task.get("model", "auto"),
            "confidence_level": task.get("confidence_level", 0.95)
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/forecast",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _detect_anomalies(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Detect anomalies in data."""
        payload = {
            "data": task.get("data", []),
            "method": task.get("method", "isolation_forest"),
            "contamination": task.get("contamination", 0.1),
            "features": task.get("features", None)
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/anomaly/detect",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _classify(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Classification task."""
        payload = {
            "data": task.get("data", {}),
            "model": task.get("model", "random_forest"),
            "features": task.get("features", None),
            "return_probabilities": task.get("return_probabilities", True)
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/classify",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _regression(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Regression task."""
        payload = {
            "data": task.get("data", {}),
            "model": task.get("model", "gradient_boosting"),
            "features": task.get("features", None),
            "return_confidence": task.get("return_confidence", True)
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/regression",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _train_model(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Train a new model."""
        payload = {
            "training_data": task.get("training_data", {}),
            "model_type": task.get("model_type", "auto"),
            "hyperparameters": task.get("hyperparameters", {}),
            "validation_split": task.get("validation_split", 0.2)
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/train",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    async def _predict(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """General prediction endpoint."""
        payload = {
            "data": task.get("data", {}),
            "model_id": task.get("model_id", None),
            "model_type": task.get("model_type", "auto")
        }
        
        response = await self.client.post(
            f"{self.api_prefix}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Get adapter capabilities."""
        return [
            AdapterCapability(
                name="model.ml.classification",
                enabled=True,
                description="Classification models (RandomForest, XGBoost, etc.)"
            ),
            AdapterCapability(
                name="model.ml.regression", 
                enabled=True,
                description="Regression models (Linear, GradientBoosting, etc.)"
            ),
            AdapterCapability(
                name="model.timeseries.forecasting",
                enabled=True,
                description="Time series forecasting (ARIMA, Prophet, LSTM)"
            ),
            AdapterCapability(
                name="model.anomaly_detection",
                enabled=True,
                description="Anomaly detection (IsolationForest, Autoencoder)"
            ),
            AdapterCapability(
                name="risk_assessment",
                enabled=True,
                description="AI governance risk assessment"
            ),
            AdapterCapability(
                name="quality_verification",
                enabled=True,
                description="Data and model quality checks"
            ),
            AdapterCapability(
                name="training",
                enabled=True,
                description="Model training capabilities"
            ),
            AdapterCapability(
                name="fine_tuning",
                enabled=True,
                description="Model fine-tuning support"
            ),
            AdapterCapability(
                name="a_b_testing",
                enabled=True,
                description="A/B testing for model comparison"
            )
        ]
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of ML service."""
        try:
            response = await self.client.get(f"{self.api_prefix}/health")
            if response.status_code == 200:
                health_data = response.json()
                return {
                    "status": "healthy",
                    "service": "ml_service",
                    "url": self.service_url,
                    "models_loaded": health_data.get("models_loaded", 0),
                    "details": health_data
                }
            else:
                return {
                    "status": "unhealthy",
                    "service": "ml_service",
                    "url": self.service_url,
                    "error": f"Status code {response.status_code}"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "service": "ml_service",
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