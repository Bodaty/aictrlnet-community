"""Component registry for the control plane."""

from datetime import datetime
from typing import Dict, List, Optional, Any
import asyncio
from collections import defaultdict

from .models import Component, ComponentStatus, ComponentType, ComponentEvent
from .auth import component_auth


class ComponentRegistry:
    """In-memory component registry with persistence support."""
    
    def __init__(self):
        # Components by ID
        self.components: Dict[str, Component] = {}
        
        # Components by type for quick lookup
        self.components_by_type: Dict[ComponentType, List[str]] = defaultdict(list)
        
        # Components by capability
        self.components_by_capability: Dict[str, List[str]] = defaultdict(list)
        
        # Component events
        self.events: List[ComponentEvent] = []
        
        # Lock for thread-safe operations
        self._lock = asyncio.Lock()
        
    async def register_component(
        self, 
        component: Component,
        user_id: str
    ) -> tuple[Component, str, datetime]:
        """Register a new component and return it with JWT token."""
        async with self._lock:
            # Check if component already exists
            existing = self._find_component_by_name_and_type(
                component.name, 
                component.type
            )
            if existing and existing.status == ComponentStatus.ACTIVE:
                raise ValueError(f"Component {component.name} already registered")
            
            # Set registration details
            component.registered_by = user_id
            component.status = ComponentStatus.ACTIVE
            
            # Generate JWT token
            token, expires_at = component_auth.create_component_token(component)
            component.token = token
            component.token_expires_at = expires_at
            
            # Store component
            self.components[component.id] = component
            self.components_by_type[component.type].append(component.id)
            
            # Index by capabilities
            for capability in component.capabilities:
                self.components_by_capability[capability.name].append(component.id)
            
            # Record event
            event = ComponentEvent(
                component_id=component.id,
                event_type="registered",
                data={
                    "name": component.name,
                    "type": component.type.value,
                    "version": component.version
                }
            )
            self.events.append(event)
            
            return component, token, expires_at
    
    async def update_heartbeat(
        self, 
        component_id: str,
        health_score: float = 100.0,
        metrics: Dict[str, Any] = None
    ) -> Component:
        """Update component heartbeat."""
        async with self._lock:
            component = self.components.get(component_id)
            if not component:
                raise ValueError(f"Component {component_id} not found")
            
            component.last_heartbeat = datetime.utcnow()
            component.health_score = health_score
            
            if metrics:
                component.metadata.update({"last_metrics": metrics})
            
            # Update status based on health
            if health_score < 50:
                component.status = ComponentStatus.FAILED
            elif health_score < 80:
                component.status = ComponentStatus.INACTIVE
            else:
                component.status = ComponentStatus.ACTIVE
            
            return component
    
    async def get_component(self, component_id: str) -> Optional[Component]:
        """Get a component by ID."""
        async with self._lock:
            return self.components.get(component_id)
    
    async def get_components_by_type(
        self, 
        component_type: ComponentType,
        status: Optional[ComponentStatus] = None,
        edition: Optional[str] = None
    ) -> List[Component]:
        """Get components by type with optional filtering."""
        async with self._lock:
            component_ids = self.components_by_type.get(component_type, [])
            components = [
                self.components[cid] 
                for cid in component_ids 
                if cid in self.components
            ]
            
            # Apply filters
            if status:
                components = [c for c in components if c.status == status]
            
            if edition:
                # Filter by edition hierarchy
                edition_hierarchy = ["community", "business", "enterprise"]
                if edition in edition_hierarchy:
                    max_edition_idx = edition_hierarchy.index(edition)
                    components = [
                        c for c in components
                        if edition_hierarchy.index(c.required_edition) <= max_edition_idx
                    ]
            
            return components
    
    async def get_components_by_capability(
        self,
        capability: str,
        status: Optional[ComponentStatus] = None
    ) -> List[Component]:
        """Get components that provide a specific capability."""
        async with self._lock:
            component_ids = self.components_by_capability.get(capability, [])
            components = [
                self.components[cid]
                for cid in component_ids
                if cid in self.components
            ]
            
            if status:
                components = [c for c in components if c.status == status]
            
            return components
    
    async def update_component_status(
        self,
        component_id: str,
        status: ComponentStatus,
        reason: Optional[str] = None
    ) -> Component:
        """Update component status."""
        async with self._lock:
            component = self.components.get(component_id)
            if not component:
                raise ValueError(f"Component {component_id} not found")
            
            old_status = component.status
            component.status = status
            
            # Record event
            event = ComponentEvent(
                component_id=component_id,
                event_type="status_changed",
                data={
                    "old_status": old_status.value,
                    "new_status": status.value,
                    "reason": reason
                },
                severity="warning" if status == ComponentStatus.FAILED else "info"
            )
            self.events.append(event)
            
            return component
    
    async def record_component_result(
        self,
        component_id: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> Component:
        """Record the result of a component operation."""
        async with self._lock:
            component = self.components.get(component_id)
            if not component:
                raise ValueError(f"Component {component_id} not found")
            
            if success:
                component.success_count += 1
            else:
                component.error_count += 1
                
                # Record error event
                event = ComponentEvent(
                    component_id=component_id,
                    event_type="error",
                    data={"error": error_message},
                    severity="error"
                )
                self.events.append(event)
            
            # Update reputation score
            total_ops = component.success_count + component.error_count
            if total_ops > 0:
                success_rate = component.success_count / total_ops
                component.reputation_score = min(100.0, success_rate * 100)
            
            return component
    
    async def get_component_events(
        self,
        component_id: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[ComponentEvent]:
        """Get component events with optional filtering."""
        async with self._lock:
            events = self.events
            
            if component_id:
                events = [e for e in events if e.component_id == component_id]
            
            if event_type:
                events = [e for e in events if e.event_type == event_type]
            
            # Return most recent events
            return sorted(events, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    async def cleanup_inactive_components(
        self,
        inactive_threshold_minutes: int = 30
    ) -> List[str]:
        """Clean up components that haven't sent heartbeat recently."""
        async with self._lock:
            now = datetime.utcnow()
            inactive_ids = []
            
            for component_id, component in self.components.items():
                if component.last_heartbeat:
                    time_since_heartbeat = (now - component.last_heartbeat).total_seconds() / 60
                    if time_since_heartbeat > inactive_threshold_minutes:
                        component.status = ComponentStatus.INACTIVE
                        inactive_ids.append(component_id)
            
            return inactive_ids
    
    def _find_component_by_name_and_type(
        self, 
        name: str, 
        component_type: ComponentType
    ) -> Optional[Component]:
        """Find a component by name and type."""
        for component in self.components.values():
            if component.name == name and component.type == component_type:
                return component
        return None


# Global registry instance
component_registry = ComponentRegistry()