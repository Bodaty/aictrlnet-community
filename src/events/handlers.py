"""Event handler registry and common handlers."""

import logging
from typing import Dict, List, Any
from datetime import datetime

from .models import Event, EventPriority
from .event_bus import event_bus


logger = logging.getLogger(__name__)


class EventHandlerRegistry:
    """Registry for managing event handlers."""
    
    def __init__(self):
        self.handlers = {}
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """Register default system event handlers."""
        # Component lifecycle handlers
        event_bus.register_handler(
            ["component.registered", "component.updated"],
            self._handle_component_lifecycle,
            priority=EventPriority.HIGH,
            name="component_lifecycle_handler"
        )
        
        # Error event handlers
        event_bus.register_handler(
            ["*.error", "*.failed"],
            self._handle_error_events,
            priority=EventPriority.CRITICAL,
            name="error_handler"
        )
        
        # Workflow event handlers
        event_bus.register_handler(
            ["workflow.*"],
            self._handle_workflow_events,
            priority=EventPriority.NORMAL,
            name="workflow_handler"
        )
        
        # Metrics collection handler
        event_bus.register_handler(
            ["*.metrics", "component.heartbeat"],
            self._handle_metrics_events,
            priority=EventPriority.LOW,
            name="metrics_handler"
        )
    
    async def _handle_component_lifecycle(self, event: Event):
        """Handle component lifecycle events."""
        logger.info(f"Component lifecycle event: {event.type} - {event.data.get('name')}")
        
        # Update any dependent systems
        if event.type == "component.registered":
            # Notify other components about new registration
            await event_bus.publish(
                "system.component_available",
                {
                    "component_id": event.data.get("component_id"),
                    "capabilities": event.data.get("capabilities", [])
                },
                source_id="system",
                source_type="system"
            )
    
    async def _handle_error_events(self, event: Event):
        """Handle error events from any component."""
        logger.error(f"Error event: {event.type} - {event.data.get('error')}")
        
        # Track error patterns
        component_id = event.source_id
        if component_id:
            # Would update error tracking in a real system
            pass
        
        # Alert on critical errors
        if event.priority == EventPriority.CRITICAL:
            await event_bus.publish(
                "alert.critical_error",
                {
                    "original_event": event.type,
                    "component": component_id,
                    "error": event.data.get("error"),
                    "timestamp": event.timestamp.isoformat()
                },
                source_id="system",
                source_type="system",
                priority=EventPriority.CRITICAL
            )
    
    async def _handle_workflow_events(self, event: Event):
        """Handle workflow-related events."""
        workflow_id = event.data.get("workflow_id")
        
        if event.type == "workflow.started":
            logger.info(f"Workflow {workflow_id} started")
        elif event.type == "workflow.completed":
            logger.info(f"Workflow {workflow_id} completed")
            # Calculate and publish workflow metrics
            duration = event.data.get("duration_seconds")
            if duration:
                await event_bus.publish(
                    "metrics.workflow_duration",
                    {
                        "workflow_id": workflow_id,
                        "duration_seconds": duration
                    },
                    source_id="system",
                    source_type="system"
                )
        elif event.type == "workflow.failed":
            logger.error(f"Workflow {workflow_id} failed: {event.data.get('error')}")
    
    async def _handle_metrics_events(self, event: Event):
        """Collect and process metrics events."""
        metrics = event.data.get("metrics", {})
        
        # In a real system, would send to monitoring system
        if metrics:
            logger.debug(f"Metrics from {event.source_id}: {metrics}")


# Common event publishing helpers
async def publish_component_event(
    event_type: str,
    component_id: str,
    data: Dict[str, Any],
    priority: EventPriority = EventPriority.NORMAL
):
    """Helper to publish component-related events."""
    await event_bus.publish(
        f"component.{event_type}",
        {
            "component_id": component_id,
            **data
        },
        source_id=component_id,
        source_type="component",
        priority=priority
    )


async def publish_workflow_event(
    event_type: str,
    workflow_id: str,
    data: Dict[str, Any],
    priority: EventPriority = EventPriority.NORMAL
):
    """Helper to publish workflow-related events."""
    await event_bus.publish(
        f"workflow.{event_type}",
        {
            "workflow_id": workflow_id,
            **data
        },
        source_id=workflow_id,
        source_type="workflow",
        priority=priority
    )


async def publish_system_event(
    event_type: str,
    data: Dict[str, Any],
    priority: EventPriority = EventPriority.NORMAL
):
    """Helper to publish system-level events."""
    await event_bus.publish(
        f"system.{event_type}",
        data,
        source_id="system",
        source_type="system",
        priority=priority
    )


# Initialize the registry
handler_registry = EventHandlerRegistry()