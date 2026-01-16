"""Base adapter class for all adapters."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import time
from contextlib import asynccontextmanager

from .models import (
    AdapterConfig, AdapterCapability, AdapterRequest, 
    AdapterResponse, AdapterStatus, AdapterMetrics, AdapterInfo
)
from events.event_bus import event_bus
from control_plane.services import control_plane_service
from control_plane.models import ComponentType


logger = logging.getLogger(__name__)


class BaseAdapter(ABC):
    """Base class for all adapters in AICtrlNet."""
    
    def __init__(self, config: AdapterConfig):
        self.config = config
        self.id = f"{config.name}-{config.version}"
        self.status = AdapterStatus.INITIALIZING
        self.metrics = AdapterMetrics()
        self._rate_limiter = None
        self._initialized = False
        self._component_id = None
        
        # Set up rate limiting if configured
        if config.rate_limit_per_minute:
            self._rate_limiter = asyncio.Semaphore(config.rate_limit_per_minute)
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the adapter (connect to services, validate config, etc)."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """Shutdown the adapter cleanly."""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return list of capabilities this adapter provides."""
        pass
    
    @abstractmethod
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request using this adapter."""
        pass
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()
    
    async def start(self) -> None:
        """Start the adapter and register with control plane."""
        try:
            # Initialize adapter
            await self.initialize()
            
            # Register with control plane
            capabilities = self.get_capabilities()
            from control_plane.models import ComponentRegistrationRequest, ComponentCapability as CPCapability
            
            registration_request = ComponentRegistrationRequest(
                name=self.config.name,
                type=ComponentType.ADAPTER,
                version=self.config.version,
                description=self.config.description or f"{self.config.name} adapter",
                capabilities=[
                    CPCapability(
                        name=cap.name,
                        description=cap.description,
                        parameters=cap.parameters or {}
                    )
                    for cap in capabilities
                ],
                edition=self.config.required_edition,
                config={
                    "category": self.config.category.value,
                    "base_url": self.config.base_url
                }
            )
            
            registration = await control_plane_service.register_component(
                registration_request,
                user_id="system"
            )
            
            self._component_id = registration.component.id
            self.status = AdapterStatus.READY
            self._initialized = True
            
            # Publish ready event
            await event_bus.publish(
                "adapter.ready",
                {
                    "adapter_id": self.id,
                    "name": self.config.name,
                    "category": self.config.category.value,
                    "capabilities": [cap.name for cap in capabilities],
                    "source_id": self.id,
                    "source_type": "adapter"
                }
            )
            
            logger.info(f"Adapter {self.config.name} started successfully")
            
        except Exception as e:
            self.status = AdapterStatus.ERROR
            logger.error(f"Failed to start adapter {self.config.name}: {str(e)}")
            raise
    
    async def stop(self) -> None:
        """Stop the adapter and cleanup resources."""
        try:
            # Update status
            self.status = AdapterStatus.DISABLED
            
            # Shutdown adapter
            await self.shutdown()
            
            # Update control plane
            if self._component_id:
                await control_plane_service.update_component_status(
                    self._component_id,
                    "inactive",
                    "Adapter stopped"
                )
            
            # Publish stopped event
            await event_bus.publish(
                "adapter.stopped",
                {
                    "adapter_id": self.id,
                    "name": self.config.name,
                    "source_id": self.id,
                    "source_type": "adapter"
                }
            )
            
            logger.info(f"Adapter {self.config.name} stopped successfully")
            
        except Exception as e:
            logger.error(f"Error stopping adapter {self.config.name}: {str(e)}")
    
    async def handle_request(self, request: AdapterRequest) -> AdapterResponse:
        """Handle a request with rate limiting, metrics, and error handling."""
        if not self._initialized:
            raise RuntimeError(f"Adapter {self.config.name} not initialized")
        
        if self.status != AdapterStatus.READY:
            raise RuntimeError(f"Adapter {self.config.name} not ready: {self.status}")
        
        start_time = time.time()
        
        try:
            # Rate limiting
            if self._rate_limiter:
                async with self._rate_limiter:
                    response = await self._execute_with_retry(request)
            else:
                response = await self._execute_with_retry(request)
            
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            self._update_metrics(success=True, duration_ms=duration_ms, cost=response.cost)
            
            # Record success with control plane
            if self._component_id:
                await control_plane_service.record_component_result(
                    self._component_id,
                    success=True,
                    metrics={
                        "duration_ms": duration_ms,
                        "capability": request.capability
                    }
                )
            
            return response
            
        except Exception as e:
            # Update metrics
            duration_ms = (time.time() - start_time) * 1000
            self._update_metrics(success=False, duration_ms=duration_ms, error=str(e))
            
            # Record failure with control plane
            if self._component_id:
                await control_plane_service.record_component_result(
                    self._component_id,
                    success=False,
                    error_message=str(e)
                )
            
            # Create error response
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=duration_ms
            )
    
    async def _execute_with_retry(self, request: AdapterRequest) -> AdapterResponse:
        """Execute request with retry logic."""
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Set busy status
                self.status = AdapterStatus.BUSY
                
                # Execute request
                response = await asyncio.wait_for(
                    self.execute(request),
                    timeout=request.timeout_override or self.config.timeout_seconds
                )
                
                # Reset status
                self.status = AdapterStatus.READY
                
                return response
                
            except asyncio.TimeoutError:
                last_error = "Request timed out"
                logger.warning(f"Attempt {attempt + 1} timed out for {self.config.name}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Attempt {attempt + 1} failed for {self.config.name}: {str(e)}")
            
            # Reset status
            self.status = AdapterStatus.READY
            
            # Wait before retry
            if attempt < self.config.max_retries - 1:
                await asyncio.sleep(self.config.retry_delay_seconds * (attempt + 1))
        
        raise RuntimeError(f"All retry attempts failed: {last_error}")
    
    def _update_metrics(
        self,
        success: bool,
        duration_ms: float,
        cost: Optional[float] = None,
        error: Optional[str] = None
    ):
        """Update adapter metrics."""
        self.metrics.total_requests += 1
        
        if success:
            self.metrics.successful_requests += 1
        else:
            self.metrics.failed_requests += 1
            self.metrics.last_error = error
            self.metrics.last_error_time = datetime.utcnow()
        
        # Update response time
        self.metrics.last_response_time_ms = duration_ms
        if self.metrics.average_response_time_ms == 0:
            self.metrics.average_response_time_ms = duration_ms
        else:
            # Running average
            self.metrics.average_response_time_ms = (
                (self.metrics.average_response_time_ms * (self.metrics.total_requests - 1) + duration_ms)
                / self.metrics.total_requests
            )
        
        # Update cost
        if cost:
            self.metrics.total_cost += cost
        
        # Update last used
        self.metrics.last_used = datetime.utcnow()
        
        # Update rate limiting metrics
        now = datetime.utcnow()
        if (now - self.metrics.minute_start).total_seconds() > 60:
            self.metrics.requests_this_minute = 1
            self.metrics.minute_start = now
        else:
            self.metrics.requests_this_minute += 1
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the adapter."""
        try:
            # Adapter-specific health check
            health_data = await self._perform_health_check()
            
            # Send heartbeat to control plane
            if self._component_id:
                health_score = 100.0 if self.status == AdapterStatus.READY else 50.0
                await control_plane_service.process_heartbeat({
                    "component_id": self._component_id,
                    "health_score": health_score,
                    "metrics": {
                        "total_requests": self.metrics.total_requests,
                        "success_rate": (
                            self.metrics.successful_requests / self.metrics.total_requests
                            if self.metrics.total_requests > 0 else 1.0
                        ),
                        "average_response_time_ms": self.metrics.average_response_time_ms
                    }
                })
            
            return {
                "status": self.status.value,
                "metrics": self.metrics.model_dump(),
                "health_data": health_data
            }
            
        except Exception as e:
            logger.error(f"Health check failed for {self.config.name}: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform adapter-specific health check. Override in subclasses."""
        return {"status": "ok"}
    
    def get_info(self) -> AdapterInfo:
        """Get adapter information."""
        return AdapterInfo(
            id=self.id,
            name=self.config.name,
            category=self.config.category,
            version=self.config.version,
            description=self.config.description,
            status=self.status,
            capabilities=self.get_capabilities(),
            required_edition=self.config.required_edition,
            metrics=self.metrics
        )
    
    def validate_request(self, request: AdapterRequest) -> None:
        """Validate a request against adapter capabilities."""
        # Check if capability exists
        capabilities = {cap.name: cap for cap in self.get_capabilities()}
        
        if request.capability not in capabilities:
            raise ValueError(f"Unknown capability: {request.capability}")
        
        # Validate required parameters
        capability = capabilities[request.capability]
        for param in capability.required_parameters:
            if param not in request.parameters:
                raise ValueError(f"Missing required parameter: {param}")
    
    @asynccontextmanager
    async def rate_limit(self):
        """Context manager for rate limiting."""
        if self._rate_limiter:
            async with self._rate_limiter:
                yield
        else:
            yield