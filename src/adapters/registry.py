"""Adapter registry for managing adapter instances."""

import asyncio
import logging
from typing import Dict, List, Optional, Type, Any
from datetime import datetime

from .base_adapter import BaseAdapter
from .models import AdapterInfo, AdapterCategory, AdapterStatus, AdapterConfig
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Registry for managing adapter instances."""
    
    def __init__(self):
        # Registered adapter classes
        self._adapter_classes: Dict[str, Type[BaseAdapter]] = {}
        
        # Active adapter instances
        self._adapters: Dict[str, BaseAdapter] = {}
        
        # Adapter metadata
        self._adapter_info: Dict[str, AdapterInfo] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
        
        # Background tasks
        self._health_check_task: Optional[asyncio.Task] = None
    
    def register_adapter_class(
        self,
        name: str,
        adapter_class: Type[BaseAdapter],
        category: AdapterCategory,
        description: Optional[str] = None
    ):
        """Register an adapter class that can be instantiated."""
        if not issubclass(adapter_class, BaseAdapter):
            raise ValueError(f"{adapter_class} must be a subclass of BaseAdapter")
        
        self._adapter_classes[name] = adapter_class
        logger.info(f"Registered adapter class: {name}")
    
    def get_adapter_class(self, name: str) -> Optional[Type[BaseAdapter]]:
        """Get a registered adapter class by name."""
        return self._adapter_classes.get(name)
    
    async def create_adapter(
        self,
        name: str,
        config: AdapterConfig
    ) -> BaseAdapter:
        """Create and start an adapter instance."""
        async with self._lock:
            # Check if adapter class exists
            if name not in self._adapter_classes:
                raise ValueError(f"Unknown adapter class: {name}")
            
            # Check if already instantiated
            adapter_id = f"{config.name}-{config.version}"
            if adapter_id in self._adapters:
                raise ValueError(f"Adapter {adapter_id} already exists")
            
            # Create instance
            adapter_class = self._adapter_classes[name]
            adapter = adapter_class(config)
            
            # Start adapter
            try:
                await adapter.start()
                
                # Store adapter
                self._adapters[adapter_id] = adapter
                self._adapter_info[adapter_id] = adapter.get_info()
                
                # Publish creation event
                await event_bus.publish(
                    "adapter.created",
                    {
                        "adapter_id": adapter_id,
                        "name": config.name,
                        "category": config.category.value,
                        "source_id": "adapter_registry",
                        "source_type": "system"
                    }
                )
                
                return adapter
                
            except Exception as e:
                logger.error(f"Failed to create adapter {adapter_id}: {str(e)}")
                raise
    
    async def get_adapter(self, adapter_id: str) -> Optional[BaseAdapter]:
        """Get an adapter instance by ID."""
        return self._adapters.get(adapter_id)
    
    async def get_adapters_by_category(
        self,
        category: AdapterCategory,
        status: Optional[AdapterStatus] = None
    ) -> List[BaseAdapter]:
        """Get all adapters in a category."""
        adapters = []
        
        for adapter in self._adapters.values():
            if adapter.config.category == category:
                if status is None or adapter.status == status:
                    adapters.append(adapter)
        
        return adapters
    
    async def get_adapters_by_capability(
        self,
        capability: str,
        edition: Optional[str] = None
    ) -> List[BaseAdapter]:
        """Get adapters that provide a specific capability."""
        adapters = []
        
        for adapter in self._adapters.values():
            # Check capabilities
            capabilities = {cap.name for cap in adapter.get_capabilities()}
            if capability in capabilities:
                # Check edition
                if edition:
                    edition_hierarchy = ["community", "business", "enterprise"]
                    if edition in edition_hierarchy:
                        adapter_idx = edition_hierarchy.index(adapter.config.required_edition)
                        required_idx = edition_hierarchy.index(edition)
                        if adapter_idx <= required_idx:
                            adapters.append(adapter)
                else:
                    adapters.append(adapter)
        
        return adapters
    
    async def remove_adapter(self, adapter_id: str) -> bool:
        """Remove an adapter instance."""
        async with self._lock:
            adapter = self._adapters.get(adapter_id)
            if not adapter:
                return False
            
            # Stop adapter
            try:
                await adapter.stop()
            except Exception as e:
                logger.error(f"Error stopping adapter {adapter_id}: {str(e)}")
            
            # Remove from registry
            del self._adapters[adapter_id]
            del self._adapter_info[adapter_id]
            
            # Publish removal event
            await event_bus.publish(
                "adapter.removed",
                {
                    "adapter_id": adapter_id,
                    "source_id": "adapter_registry",
                    "source_type": "system"
                }
            )
            
            return True
    
    async def start_health_checks(self, interval_seconds: int = 60):
        """Start background health check task."""
        if not self._health_check_task:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop(interval_seconds)
            )
            logger.info("Started adapter health checks")
    
    async def stop_health_checks(self):
        """Stop background health check task."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
            logger.info("Stopped adapter health checks")
    
    async def _health_check_loop(self, interval_seconds: int):
        """Background task to perform health checks."""
        while True:
            try:
                await asyncio.sleep(interval_seconds)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
    
    async def _perform_health_checks(self):
        """Perform health checks on all adapters."""
        tasks = []
        
        for adapter_id, adapter in self._adapters.items():
            task = asyncio.create_task(self._check_adapter_health(adapter_id, adapter))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_adapter_health(self, adapter_id: str, adapter: BaseAdapter):
        """Check health of a single adapter."""
        try:
            health_data = await adapter.health_check()
            
            # Update adapter info
            self._adapter_info[adapter_id] = adapter.get_info()
            self._adapter_info[adapter_id].last_health_check = datetime.utcnow()
            
            # Publish health event if status changed
            if health_data.get("status") == "error":
                await event_bus.publish(
                    "adapter.health_check_failed",
                    {
                        "adapter_id": adapter_id,
                        "error": health_data.get("error")
                    },
                    source_id="adapter_registry",
                    source_type="system"
                )
                
        except Exception as e:
            logger.error(f"Health check failed for adapter {adapter_id}: {str(e)}")
    
    async def get_adapter_info(self, adapter_id: str) -> Optional[AdapterInfo]:
        """Get adapter information."""
        adapter = self._adapters.get(adapter_id)
        if adapter:
            return adapter.get_info()
        return self._adapter_info.get(adapter_id)
    
    async def list_adapters(
        self,
        category: Optional[AdapterCategory] = None,
        status: Optional[AdapterStatus] = None,
        edition: Optional[str] = None
    ) -> List[AdapterInfo]:
        """List all registered adapters with filtering."""
        adapters = []
        
        for adapter in self._adapters.values():
            info = adapter.get_info()
            
            # Apply filters
            if category and info.category != category:
                continue
            
            if status and info.status != status:
                continue
            
            if edition:
                edition_hierarchy = ["community", "business", "enterprise"]
                if edition in edition_hierarchy:
                    adapter_idx = edition_hierarchy.index(info.required_edition)
                    required_idx = edition_hierarchy.index(edition)
                    if adapter_idx > required_idx:
                        continue
            
            adapters.append(info)
        
        return adapters
    
    def get_available_adapter_classes(self) -> Dict[str, Type[BaseAdapter]]:
        """Get all registered adapter classes."""
        return self._adapter_classes.copy()
    
    async def shutdown(self):
        """Shutdown all adapters and cleanup."""
        # Stop health checks
        await self.stop_health_checks()
        
        # Stop all adapters
        tasks = []
        for adapter_id in list(self._adapters.keys()):
            task = asyncio.create_task(self.remove_adapter(adapter_id))
            tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("Adapter registry shutdown complete")
    
    @property
    def adapter_classes(self) -> Dict[str, Type[BaseAdapter]]:
        """Get registered adapter classes."""
        return self._adapter_classes.copy()


# Global adapter registry
adapter_registry = AdapterRegistry()