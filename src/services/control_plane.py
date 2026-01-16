"""Basic Control Plane service for Community Edition.

Provides core control plane functionality without database persistence.
Advanced features like quality reviews, policies, and metrics are
available in Business Edition.
"""

import json
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path
import uuid
import os

from core.config import settings


logger = logging.getLogger(__name__)


class ControlPlaneService:
    """Basic control plane service for Community Edition.
    
    Provides:
    - Component registration and discovery
    - Basic health checking
    - Simple coordination
    
    Advanced features available in Business Edition:
    - Quality reviews with multi-step approval
    - Control policies
    - Performance metrics
    - Persistent component registry
    - Advanced routing and load balancing
    """
    
    def __init__(self):
        """Initialize the basic control plane service."""
        # In-memory component registry
        self._components = {}
        self._health_status = {}
        
        # File-based persistence for component registry
        try:
            data_path = os.environ.get('DATA_PATH', '/tmp/aictrlnet')
            self.registry_file = Path(data_path) / "control_plane" / "components.json"
            self._load_registry()
        except Exception as e:
            logger.warning(f"Failed to initialize control plane registry: {e}")
            self.registry_file = None
    
    def _load_registry(self):
        """Load component registry from file."""
        if self.registry_file.exists():
            try:
                data = json.loads(self.registry_file.read_text())
                self._components = data.get("components", {})
            except Exception as e:
                logger.warning(f"Failed to load component registry: {e}")
        else:
            self.registry_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_registry()
    
    def _save_registry(self):
        """Save component registry to file."""
        if not self.registry_file:
            return
        try:
            data = {
                "components": self._components,
                "updated_at": datetime.utcnow().isoformat()
            }
            self.registry_file.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"Failed to save component registry: {e}")
    
    async def register_component(
        self,
        component_data
    ) -> Dict[str, Any]:
        """Register a control plane component."""
        component_id = str(uuid.uuid4())
        
        component = {
            "id": component_id,
            "name": component_data.name,
            "type": component_data.type.value,
            "endpoint": component_data.endpoint,
            "capabilities": component_data.capabilities,
            "description": component_data.description,
            "health_check_endpoint": component_data.health_check_endpoint or f"{component_data.endpoint}/health",
            "version": component_data.version or "1.0.0",
            "metadata": component_data.metadata or {},
            "status": "active",
            "created_at": datetime.utcnow().isoformat(),
            "last_health_check": None
        }
        
        self._components[component_id] = component
        self._save_registry()
        
        logger.info(f"Registered component: {component_data.name} ({component_data.type}) at {component_data.endpoint}")
        
        return component
    
    async def list_components(
        self,
        type: Optional[str] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """List registered components."""
        components = []
        
        for component in self._components.values():
            # Filter by type if specified
            if type and component["type"] != type:
                continue
            
            # Filter by status if specified
            if status and component["status"] != status:
                continue
            
            # Add health status if available
            component_with_health = component.copy()
            if component["id"] in self._health_status:
                component_with_health["health"] = self._health_status[component["id"]]
            
            components.append(component_with_health)
        
        # Apply skip and limit
        return components[skip:skip+limit]
    
    async def get_component(self, component_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific component."""
        component = self._components.get(component_id)
        if component:
            component_with_health = component.copy()
            if component_id in self._health_status:
                component_with_health["health"] = self._health_status[component_id]
            return component_with_health
        return None
    
    async def update_component_status(
        self,
        component_id: str,
        status: str,
        message: Optional[str] = None
    ) -> bool:
        """Update component status."""
        if component_id not in self._components:
            return False
        
        self._components[component_id]["status"] = status
        self._components[component_id]["status_message"] = message
        self._components[component_id]["updated_at"] = datetime.utcnow().isoformat()
        
        self._save_registry()
        
        return True
    
    async def deregister_component(self, component_id: str) -> bool:
        """Deregister a component."""
        if component_id in self._components:
            component_name = self._components[component_id]["name"]
            del self._components[component_id]
            
            # Remove health status
            if component_id in self._health_status:
                del self._health_status[component_id]
            
            self._save_registry()
            
            logger.info(f"Deregistered component: {component_name} ({component_id})")
            return True
        
        return False
    
    async def update_health_status(
        self,
        component_id: str,
        healthy: bool,
        latency_ms: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update component health status."""
        if component_id not in self._components:
            return False
        
        self._health_status[component_id] = {
            "healthy": healthy,
            "latency_ms": latency_ms,
            "details": details or {},
            "last_check": datetime.utcnow().isoformat()
        }
        
        self._components[component_id]["last_health_check"] = datetime.utcnow().isoformat()
        
        # Update component status based on health
        if not healthy:
            self._components[component_id]["status"] = "unhealthy"
        elif self._components[component_id]["status"] == "unhealthy":
            self._components[component_id]["status"] = "active"
        
        return True
    
    async def find_components_by_capability(
        self,
        capability: str
    ) -> List[Dict[str, Any]]:
        """Find components that have a specific capability."""
        matching_components = []
        
        for component in self._components.values():
            if capability in component.get("capabilities", []):
                # Only return active/healthy components
                if component["status"] in ["active", "healthy"]:
                    matching_components.append(component)
        
        return matching_components
    
    async def get_control_plane_status(self) -> Dict[str, Any]:
        """Get overall control plane status."""
        total_components = len(self._components)
        active_components = sum(
            1 for c in self._components.values() 
            if c["status"] in ["active", "healthy"]
        )
        unhealthy_components = sum(
            1 for c in self._components.values() 
            if c["status"] == "unhealthy"
        )
        
        component_types = {}
        for component in self._components.values():
            comp_type = component["type"]
            if comp_type not in component_types:
                component_types[comp_type] = 0
            component_types[comp_type] += 1
        
        return {
            "status": "healthy" if unhealthy_components == 0 else "degraded",
            "total_components": total_components,
            "active_components": active_components,
            "unhealthy_components": unhealthy_components,
            "component_types": component_types,
            "features": {
                "component_registry": True,
                "health_checking": True,
                "capability_discovery": True,
                "quality_reviews": False,
                "control_policies": False,
                "performance_metrics": False
            },
            "upgrade_available": True,
            "upgrade_benefits": [
                "Persistent component registry",
                "Quality review workflows",
                "Control policies",
                "Performance metrics",
                "Advanced routing and load balancing"
            ]
        }
    
    # Simplified interface for basic coordination
    
    async def coordinate_task(
        self,
        task_type: str,
        task_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simple task coordination (Community Edition)."""
        # Find components that can handle this task type
        capable_components = await self.find_components_by_capability(task_type)
        
        if not capable_components:
            return {
                "success": False,
                "error": f"No components available to handle task type: {task_type}"
            }
        
        # Simple round-robin selection (advanced load balancing in Business Edition)
        selected_component = capable_components[0]
        
        return {
            "success": True,
            "component": {
                "id": selected_component["id"],
                "name": selected_component["name"],
                "endpoint": selected_component["endpoint"]
            },
            "task_id": str(uuid.uuid4())
        }