"""TaskRabbit adapter implementation for physical world task management."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, AsyncGenerator
import httpx
import json
from datetime import datetime, timedelta
import time

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory, AdapterStatus
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class TaskRabbitAdapter(BaseAdapter):
    """Adapter for TaskRabbit API integration."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.HUMAN
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        
        # Extract configuration
        self.client_id = config.credentials.get("client_id")
        self.client_secret = config.credentials.get("client_secret") 
        self.access_token = config.credentials.get("access_token")
        self.base_url = config.base_url or "https://api.taskrabbit.com/v3"
        
        # TaskRabbit uses OAuth 2.0
        if not self.client_id or not self.client_secret:
            raise ValueError("TaskRabbit OAuth credentials required (client_id, client_secret)")
    
    async def initialize(self) -> None:
        """Initialize the TaskRabbit adapter."""
        # Create HTTP client with OAuth headers
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "AICtrlNet/1.0"
        }
        
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0
        )
        
        await self._emit_event("taskrabbit_initialized", {"adapter_id": self.config.id})
        logger.info("TaskRabbit adapter initialized")
    
    async def shutdown(self) -> None:
        """Shutdown the TaskRabbit adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("TaskRabbit adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return TaskRabbit adapter capabilities."""
        return [
            AdapterCapability(
                name="search_taskers",
                description="Search for available taskers in an area",
                category="tasker_search", 
                parameters={
                    "location": {"type": "string", "description": "Location (address, zip code, or coordinates)"},
                    "category": {"type": "string", "description": "Task category (mounting, furniture_assembly, delivery, etc.)"},
                    "skills": {"type": "array", "description": "Required skills"},
                    "radius": {"type": "integer", "description": "Search radius in miles", "default": 10}
                }
            ),
            AdapterCapability(
                name="create_task",
                description="Create a new task on TaskRabbit",
                category="task_management",
                parameters={
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string", "description": "Detailed task description"},
                    "location": {"type": "string", "description": "Task location"},
                    "category": {"type": "string", "description": "Task category"},
                    "budget": {"type": "number", "description": "Task budget in USD"},
                    "scheduled_time": {"type": "string", "description": "Preferred completion time (ISO format)", "optional": True},
                    "required_skills": {"type": "array", "description": "Required skills", "optional": True}
                }
            ),
            AdapterCapability(
                name="get_task_status",
                description="Get the status of a task",
                category="task_management",
                parameters={
                    "task_id": {"type": "string", "description": "TaskRabbit task ID"}
                }
            ),
            AdapterCapability(
                name="update_task",
                description="Update an existing task",
                category="task_management", 
                parameters={
                    "task_id": {"type": "string", "description": "TaskRabbit task ID"},
                    "title": {"type": "string", "description": "Updated title", "optional": True},
                    "description": {"type": "string", "description": "Updated description", "optional": True},
                    "budget": {"type": "number", "description": "Updated budget", "optional": True},
                    "scheduled_time": {"type": "string", "description": "Updated completion time", "optional": True}
                }
            ),
            AdapterCapability(
                name="cancel_task", 
                description="Cancel a task",
                category="task_management",
                parameters={
                    "task_id": {"type": "string", "description": "TaskRabbit task ID"},
                    "reason": {"type": "string", "description": "Cancellation reason", "optional": True}
                }
            ),
            AdapterCapability(
                name="get_categories",
                description="Get available task categories",
                category="metadata",
                parameters={}
            ),
            AdapterCapability(
                name="estimate_cost",
                description="Get cost estimate for a task",
                category="pricing",
                parameters={
                    "category": {"type": "string", "description": "Task category"},
                    "location": {"type": "string", "description": "Task location"},
                    "estimated_hours": {"type": "number", "description": "Estimated hours", "optional": True}
                }
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute TaskRabbit operations."""
        capability = request.capability
        params = request.parameters
        
        try:
            if not self.client:
                await self.initialize()
                
            # Route to appropriate handler
            if capability == "search_taskers":
                result = await self._search_taskers(params)
            elif capability == "create_task":
                result = await self._create_task(params)
            elif capability == "get_task_status":
                result = await self._get_task_status(params)
            elif capability == "update_task":
                result = await self._update_task(params)
            elif capability == "cancel_task":
                result = await self._cancel_task(params)
            elif capability == "get_categories":
                result = await self._get_categories()
            elif capability == "estimate_cost":
                result = await self._estimate_cost(params)
            else:
                raise ValueError(f"Unknown capability: {capability}")
            
            await self._emit_event("taskrabbit_operation_completed", {
                "capability": capability,
                "success": True,
                "result_count": len(result) if isinstance(result, list) else 1
            })
            
            return AdapterResponse(
                success=True,
                data=result,
                message=f"TaskRabbit {capability} completed successfully"
            )
            
        except Exception as e:
            logger.error(f"TaskRabbit {capability} failed: {str(e)}")
            await self._emit_event("taskrabbit_operation_failed", {
                "capability": capability,
                "error": str(e)
            })
            
            return AdapterResponse(
                success=False,
                data=None,
                message=f"TaskRabbit {capability} failed: {str(e)}"
            )
    
    async def _search_taskers(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search for available taskers."""
        location = params.get("location")
        category = params.get("category")
        skills = params.get("skills", [])
        radius = params.get("radius", 10)
        
        # Build query parameters
        query_params = {
            "location": location,
            "radius": radius
        }
        
        if category:
            query_params["category"] = category
        if skills:
            query_params["skills"] = ",".join(skills)
        
        response = await self.client.get("/taskers", params=query_params)
        response.raise_for_status()
        
        data = response.json()
        taskers = data.get("taskers", [])
        
        # Format tasker data
        formatted_taskers = []
        for tasker in taskers:
            formatted_taskers.append({
                "id": tasker.get("id"),
                "name": tasker.get("name"),
                "rating": tasker.get("rating"),
                "reviews_count": tasker.get("reviews_count"),
                "hourly_rate": tasker.get("hourly_rate"),
                "skills": tasker.get("skills", []),
                "availability": tasker.get("availability"),
                "location": tasker.get("location"),
                "profile_picture": tasker.get("profile_picture"),
                "description": tasker.get("description")
            })
        
        return formatted_taskers
    
    async def _create_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task on TaskRabbit."""
        task_data = {
            "title": params.get("title"),
            "description": params.get("description"),
            "location": params.get("location"),
            "category": params.get("category"),
            "budget": params.get("budget")
        }
        
        # Optional parameters
        if params.get("scheduled_time"):
            task_data["scheduled_time"] = params["scheduled_time"]
        if params.get("required_skills"):
            task_data["required_skills"] = params["required_skills"]
        
        response = await self.client.post("/tasks", json=task_data)
        response.raise_for_status()
        
        task = response.json()
        
        return {
            "task_id": task.get("id"),
            "status": task.get("status"),
            "created_at": task.get("created_at"),
            "title": task.get("title"),
            "description": task.get("description"),
            "budget": task.get("budget"),
            "location": task.get("location"),
            "category": task.get("category"),
            "task_url": task.get("url")
        }
    
    async def _get_task_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get task status and details."""
        task_id = params.get("task_id")
        
        response = await self.client.get(f"/tasks/{task_id}")
        response.raise_for_status()
        
        task = response.json()
        
        return {
            "task_id": task.get("id"),
            "status": task.get("status"),
            "title": task.get("title"), 
            "description": task.get("description"),
            "budget": task.get("budget"),
            "location": task.get("location"),
            "category": task.get("category"),
            "created_at": task.get("created_at"),
            "scheduled_time": task.get("scheduled_time"),
            "assigned_tasker": task.get("assigned_tasker"),
            "completion_photos": task.get("completion_photos", []),
            "task_url": task.get("url")
        }
    
    async def _update_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update an existing task."""
        task_id = params.get("task_id")
        update_data = {}
        
        # Only include provided parameters
        for field in ["title", "description", "budget", "scheduled_time"]:
            if field in params:
                update_data[field] = params[field]
        
        response = await self.client.patch(f"/tasks/{task_id}", json=update_data)
        response.raise_for_status()
        
        task = response.json()
        
        return {
            "task_id": task.get("id"),
            "status": task.get("status"),
            "updated_at": task.get("updated_at"),
            "changes": update_data
        }
    
    async def _cancel_task(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Cancel a task."""
        task_id = params.get("task_id")
        reason = params.get("reason", "Task cancelled by user")
        
        cancel_data = {
            "status": "cancelled",
            "cancellation_reason": reason
        }
        
        response = await self.client.patch(f"/tasks/{task_id}", json=cancel_data)
        response.raise_for_status()
        
        task = response.json()
        
        return {
            "task_id": task.get("id"),
            "status": task.get("status"),
            "cancelled_at": task.get("cancelled_at"),
            "cancellation_reason": reason
        }
    
    async def _get_categories(self) -> List[Dict[str, Any]]:
        """Get available task categories."""
        response = await self.client.get("/categories")
        response.raise_for_status()
        
        data = response.json()
        categories = data.get("categories", [])
        
        formatted_categories = []
        for category in categories:
            formatted_categories.append({
                "id": category.get("id"),
                "name": category.get("name"),
                "slug": category.get("slug"),
                "description": category.get("description"),
                "average_hourly_rate": category.get("average_hourly_rate"),
                "popular": category.get("popular", False)
            })
        
        return formatted_categories
    
    async def _estimate_cost(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get cost estimate for a task."""
        category = params.get("category")
        location = params.get("location")
        estimated_hours = params.get("estimated_hours", 2)
        
        query_params = {
            "category": category,
            "location": location,
            "estimated_hours": estimated_hours
        }
        
        response = await self.client.get("/estimates", params=query_params)
        response.raise_for_status()
        
        estimate = response.json()
        
        return {
            "category": category,
            "location": location,
            "estimated_hours": estimated_hours,
            "low_estimate": estimate.get("low_estimate"),
            "high_estimate": estimate.get("high_estimate"),
            "average_estimate": estimate.get("average_estimate"),
            "hourly_rate_range": estimate.get("hourly_rate_range", {}),
            "factors": estimate.get("pricing_factors", [])
        }
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Emit adapter event."""
        try:
            await event_bus.emit(event_type, {
                "adapter_id": self.config.id,
                "adapter_type": "taskrabbit", 
                "timestamp": datetime.utcnow().isoformat(),
                **data
            })
        except Exception as e:
            logger.warning(f"Failed to emit event {event_type}: {e}")
    
    async def stream_execute(self, request: AdapterRequest) -> AsyncGenerator[AdapterResponse, None]:
        """Stream TaskRabbit operations (for long-running task monitoring)."""
        capability = request.capability
        
        if capability == "monitor_task":
            # Stream task status updates
            task_id = request.parameters.get("task_id")
            interval = request.parameters.get("interval", 30)  # seconds
            
            while True:
                try:
                    status_response = await self.execute(AdapterRequest(
                        capability="get_task_status",
                        parameters={"task_id": task_id}
                    ))
                    
                    yield status_response
                    
                    # Check if task is in final state
                    if status_response.success:
                        status = status_response.data.get("status")
                        if status in ["completed", "cancelled", "failed"]:
                            break
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    yield AdapterResponse(
                        success=False,
                        data=None,
                        message=f"Task monitoring failed: {str(e)}"
                    )
                    break
        else:
            # For non-streaming capabilities, just execute once
            response = await self.execute(request)
            yield response
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get adapter health status."""
        return {
            "status": "healthy" if self.client else "disconnected",
            "client_initialized": self.client is not None,
            "base_url": self.base_url,
            "has_credentials": bool(self.client_id and self.client_secret),
            "has_access_token": bool(self.access_token)
        }