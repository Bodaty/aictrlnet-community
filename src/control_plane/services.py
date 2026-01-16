"""Control plane service for component management."""

from typing import List, Optional, Dict, Any
from datetime import datetime
import logging

from .models import (
    Component, ComponentRegistrationRequest, ComponentRegistrationResponse,
    ComponentStatus, ComponentType, ComponentHeartbeat, ComponentEvent
)
from .registry import component_registry
from .auth import component_auth
from events.event_bus import event_bus  # Will be created next


logger = logging.getLogger(__name__)


class ControlPlaneService:
    """Service for control plane operations."""
    
    def __init__(self):
        self.registry = component_registry
        self.auth = component_auth
        self.event_bus = event_bus
    
    async def register_component(
        self,
        request: ComponentRegistrationRequest,
        user_id: str
    ) -> ComponentRegistrationResponse:
        """Register a new component."""
        try:
            # Create component from request
            component = Component(
                name=request.name,
                type=request.type,
                version=request.version,
                description=request.description,
                capabilities=request.capabilities,
                metadata=request.metadata,
                edition=request.edition,
                endpoint_url=request.endpoint_url,
                webhook_url=request.webhook_url,
                config=request.config,
                required_edition=request.edition,
                registered_by=user_id
            )
            
            # Register in registry
            registered_component, token, expires_at = await self.registry.register_component(
                component,
                user_id
            )
            
            # Publish registration event
            await self.event_bus.publish(
                "component.registered",
                {
                    "component_id": registered_component.id,
                    "name": registered_component.name,
                    "type": registered_component.type.value,
                    "edition": registered_component.edition,
                    "capabilities": [cap.name for cap in registered_component.capabilities]
                }
            )
            
            logger.info(f"Component {registered_component.name} registered successfully")
            
            return ComponentRegistrationResponse(
                component=registered_component,
                token=token,
                expires_at=expires_at
            )
            
        except Exception as e:
            logger.error(f"Failed to register component: {str(e)}")
            raise
    
    async def process_heartbeat(self, heartbeat: ComponentHeartbeat) -> Component:
        """Process a component heartbeat."""
        try:
            # Update heartbeat in registry
            component = await self.registry.update_heartbeat(
                heartbeat.component_id,
                heartbeat.health_score,
                heartbeat.metrics
            )
            
            # Publish heartbeat event
            await self.event_bus.publish(
                "component.heartbeat",
                {
                    "component_id": component.id,
                    "health_score": component.health_score,
                    "status": component.status.value,
                    "metrics": heartbeat.metrics
                }
            )
            
            return component
            
        except Exception as e:
            logger.error(f"Failed to process heartbeat: {str(e)}")
            raise
    
    async def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID."""
        return await self.registry.get_component(component_id)
    
    async def get_components(
        self,
        component_type: Optional[ComponentType] = None,
        status: Optional[ComponentStatus] = None,
        edition: Optional[str] = None,
        capability: Optional[str] = None
    ) -> List[Component]:
        """Get components with filtering."""
        if capability:
            components = await self.registry.get_components_by_capability(
                capability, status
            )
        elif component_type:
            components = await self.registry.get_components_by_type(
                component_type, status, edition
            )
        else:
            # Get all components
            components = list(self.registry.components.values())
            
            # Apply filters
            if status:
                components = [c for c in components if c.status == status]
            if edition:
                edition_hierarchy = ["community", "business", "enterprise"]
                if edition in edition_hierarchy:
                    max_edition_idx = edition_hierarchy.index(edition)
                    components = [
                        c for c in components
                        if edition_hierarchy.index(c.required_edition) <= max_edition_idx
                    ]
        
        return components
    
    async def update_component_status(
        self,
        component_id: str,
        status: ComponentStatus,
        reason: Optional[str] = None
    ) -> Component:
        """Update component status."""
        component = await self.registry.update_component_status(
            component_id, status, reason
        )
        
        # Publish status change event
        await self.event_bus.publish(
            "component.status_changed",
            {
                "component_id": component.id,
                "new_status": status.value,
                "reason": reason
            }
        )
        
        return component
    
    async def record_component_result(
        self,
        component_id: str,
        success: bool,
        error_message: Optional[str] = None,
        metrics: Optional[Dict[str, Any]] = None
    ) -> Component:
        """Record the result of a component operation."""
        component = await self.registry.record_component_result(
            component_id, success, error_message
        )
        
        # Publish result event
        event_type = "component.success" if success else "component.error"
        event_data = {
            "component_id": component.id,
            "reputation_score": component.reputation_score,
            "success_count": component.success_count,
            "error_count": component.error_count
        }
        
        if error_message:
            event_data["error"] = error_message
        if metrics:
            event_data["metrics"] = metrics
        
        await self.event_bus.publish(event_type, event_data)
        
        return component
    
    async def get_component_events(
        self,
        component_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ComponentEvent]:
        """Get component events."""
        return await self.registry.get_component_events(
            component_id, event_type, limit
        )
    
    async def refresh_component_token(
        self,
        component_id: str,
        old_token: str
    ) -> tuple[str, datetime]:
        """Refresh a component's JWT token."""
        # Verify old token first
        try:
            payload = self.auth.verify_component_token(old_token)
            if payload["sub"] != component_id:
                raise ValueError("Token does not match component ID")
        except Exception as e:
            raise ValueError(f"Invalid token: {str(e)}")
        
        # Get component
        component = await self.get_component(component_id)
        if not component:
            raise ValueError(f"Component {component_id} not found")
        
        # Generate new token
        new_token, expires_at = self.auth.create_component_token(component)
        
        # Update component
        component.token = new_token
        component.token_expires_at = expires_at
        
        return new_token, expires_at
    
    async def cleanup_inactive_components(self) -> List[str]:
        """Clean up inactive components."""
        inactive_ids = await self.registry.cleanup_inactive_components()
        
        # Publish cleanup events
        for component_id in inactive_ids:
            await self.event_bus.publish(
                "component.inactive",
                {"component_id": component_id}
            )
        
        return inactive_ids
    
    async def get_component_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary of all components."""
        components = await self.get_components()
        
        summary = {
            "total_components": len(components),
            "by_status": {},
            "by_type": {},
            "average_health_score": 0,
            "average_reputation_score": 0
        }
        
        # Count by status
        for status in ComponentStatus:
            count = len([c for c in components if c.status == status])
            if count > 0:
                summary["by_status"][status.value] = count
        
        # Count by type
        for comp_type in ComponentType:
            count = len([c for c in components if c.type == comp_type])
            if count > 0:
                summary["by_type"][comp_type.value] = count
        
        # Calculate averages
        if components:
            summary["average_health_score"] = sum(c.health_score for c in components) / len(components)
            summary["average_reputation_score"] = sum(c.reputation_score for c in components) / len(components)
        
        return summary


# Global service instance
control_plane_service = ControlPlaneService()