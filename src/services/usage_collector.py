"""Service for collecting and tracking usage metrics automatically."""

import logging
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from decimal import Decimal

from core.usage_tracker import get_usage_tracker
from core.tenant_context import get_current_tenant_id
from events.event_bus import event_bus

logger = logging.getLogger(__name__)


class UsageCollectorService:
    """Service that listens to events and collects usage metrics."""
    
    def __init__(self):
        self.tracker = None
        self._setup_event_listeners()
    
    def _setup_event_listeners(self):
        """Set up event listeners for usage tracking."""
        
        # Workflow events
        event_bus.subscribe("workflow.created", self._on_workflow_created)
        event_bus.subscribe("workflow.executed", self._on_workflow_executed)
        event_bus.subscribe("workflow.deleted", self._on_workflow_deleted)
        
        # Task events
        event_bus.subscribe("task.created", self._on_task_created)
        event_bus.subscribe("task.executed", self._on_task_executed)
        
        # API events
        event_bus.subscribe("api.call", self._on_api_call)
        
        # User events
        event_bus.subscribe("user.created", self._on_user_created)
        event_bus.subscribe("user.login", self._on_user_login)
        
        # Adapter events
        event_bus.subscribe("adapter.created", self._on_adapter_created)
        event_bus.subscribe("adapter.executed", self._on_adapter_executed)
        
        # Platform events
        event_bus.subscribe("platform.credential.created", self._on_platform_credential_created)
        event_bus.subscribe("platform.execution.completed", self._on_platform_execution)
        
        logger.info("Usage collector service initialized with event listeners")
    
    async def _get_tracker(self, db: AsyncSession):
        """Get or create usage tracker."""
        if not self.tracker:
            self.tracker = await get_usage_tracker(db)
        return self.tracker
    
    async def _on_workflow_created(self, event_data: Dict[str, Any]):
        """Track workflow creation."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="workflows",
                value=1.0,
                metadata={"workflow_id": event_data.get("workflow_id")}
            )
    
    async def _on_workflow_executed(self, event_data: Dict[str, Any]):
        """Track workflow execution."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="executions",
                value=1.0,
                metadata={
                    "workflow_id": event_data.get("workflow_id"),
                    "execution_id": event_data.get("execution_id"),
                    "duration_ms": event_data.get("duration_ms")
                }
            )
    
    async def _on_workflow_deleted(self, event_data: Dict[str, Any]):
        """Track workflow deletion."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="workflows",
                value=-1.0,  # Decrement
                metadata={"workflow_id": event_data.get("workflow_id")}
            )
    
    async def _on_task_created(self, event_data: Dict[str, Any]):
        """Track task creation."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="tasks",
                value=1.0,
                metadata={"task_id": event_data.get("task_id")}
            )
    
    async def _on_task_executed(self, event_data: Dict[str, Any]):
        """Track task execution."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            
            # Track execution
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="task_executions",
                value=1.0,
                metadata={
                    "task_id": event_data.get("task_id"),
                    "duration_ms": event_data.get("duration_ms")
                }
            )
            
            # Track compute time
            duration_ms = event_data.get("duration_ms", 0)
            if duration_ms > 0:
                compute_hours = duration_ms / (1000 * 60 * 60)  # Convert to hours
                await tracker.track_usage(
                    tenant_id=tenant_id,
                    metric_type="compute_hours",
                    value=float(compute_hours),
                    metadata={"task_id": event_data.get("task_id")}
                )
    
    async def _on_api_call(self, event_data: Dict[str, Any]):
        """Track API calls."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="api_calls",
                value=1.0,
                metadata={
                    "endpoint": event_data.get("endpoint"),
                    "method": event_data.get("method"),
                    "status_code": event_data.get("status_code")
                }
            )
    
    async def _on_user_created(self, event_data: Dict[str, Any]):
        """Track user creation."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="users",
                value=1.0,
                metadata={"user_id": event_data.get("user_id")}
            )
    
    async def _on_user_login(self, event_data: Dict[str, Any]):
        """Track user login."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="user_logins",
                value=1.0,
                metadata={
                    "user_id": event_data.get("user_id"),
                    "ip_address": event_data.get("ip_address")
                }
            )
    
    async def _on_adapter_created(self, event_data: Dict[str, Any]):
        """Track adapter creation."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="adapters",
                value=1.0,
                metadata={
                    "adapter_id": event_data.get("adapter_id"),
                    "adapter_type": event_data.get("adapter_type")
                }
            )
    
    async def _on_adapter_executed(self, event_data: Dict[str, Any]):
        """Track adapter execution."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="adapter_executions",
                value=1.0,
                metadata={
                    "adapter_id": event_data.get("adapter_id"),
                    "adapter_type": event_data.get("adapter_type"),
                    "duration_ms": event_data.get("duration_ms")
                }
            )
    
    async def _on_platform_credential_created(self, event_data: Dict[str, Any]):
        """Track platform credential creation."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="platform_nodes",
                value=1.0,
                metadata={
                    "credential_id": event_data.get("credential_id"),
                    "platform_type": event_data.get("platform_type")
                }
            )
    
    async def _on_platform_execution(self, event_data: Dict[str, Any]):
        """Track platform execution."""
        tenant_id = event_data.get("tenant_id") or get_current_tenant_id()
        db = event_data.get("db")
        
        if db:
            tracker = await self._get_tracker(db)
            
            # Track execution
            await tracker.track_usage(
                tenant_id=tenant_id,
                metric_type="platform_executions",
                value=1.0,
                metadata={
                    "execution_id": event_data.get("execution_id"),
                    "platform_type": event_data.get("platform_type"),
                    "duration_ms": event_data.get("duration_ms")
                }
            )
            
            # Track storage if provided
            storage_bytes = event_data.get("storage_bytes", 0)
            if storage_bytes > 0:
                storage_gb = storage_bytes / (1024 * 1024 * 1024)  # Convert to GB
                await tracker.track_usage(
                    tenant_id=tenant_id,
                    metric_type="storage_gb",
                    value=float(storage_gb),
                    metadata={
                        "execution_id": event_data.get("execution_id"),
                        "type": "platform_execution"
                    }
                )


# Create global instance
usage_collector = UsageCollectorService()