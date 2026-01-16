"""Service for exposing workflows as MCP endpoints."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession

from core.cache import get_cache
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class WorkflowMCPService:
    """Service for exposing workflow capabilities through MCP."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def register_workflow_endpoint(
        self,
        workflow_id: str,
        endpoint_name: str,
        endpoint_id: str,
        node_id: str,
        allowed_operations: List[str],
        auth_required: bool = True
    ) -> Dict[str, Any]:
        """Register a workflow as an MCP endpoint."""
        try:
            # Create endpoint metadata
            endpoint_data = {
                "endpoint_id": endpoint_id,
                "endpoint_name": endpoint_name,
                "workflow_id": workflow_id,
                "node_id": node_id,
                "allowed_operations": allowed_operations,
                "auth_required": auth_required,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active",
                "requests_processed": 0,
                "last_request_at": None
            }
            
            # Store in cache
            cache_service = await get_cache()
            await cache_service.set(
                f"mcp:workflow:endpoints:{endpoint_id}",
                endpoint_data,
                expire=86400  # 24 hours
            )
            
            # Add to endpoint index
            if cache_service._redis_client:
                await cache_service._redis_client.sadd(
                    f"mcp:workflow:endpoints:index",
                    endpoint_id
                )
            
            # Add to workflow's endpoint list
            if cache_service._redis_client:
                await cache_service._redis_client.sadd(
                    f"mcp:workflow:{workflow_id}:endpoints",
                    endpoint_id
                )
            
            # Publish registration event
            await event_bus.publish(
                "mcp.workflow.endpoint.registered",
                {
                    "endpoint_id": endpoint_id,
                    "endpoint_name": endpoint_name,
                    "workflow_id": workflow_id,
                    "node_id": node_id
                }
            )
            
            logger.info(f"Registered workflow endpoint: {endpoint_name} ({endpoint_id})")
            
            return {
                "success": True,
                "endpoint_id": endpoint_id,
                "endpoint_name": endpoint_name,
                "status": "active"
            }
            
        except Exception as e:
            logger.error(f"Failed to register workflow endpoint: {str(e)}")
            raise
    
    async def unregister_workflow_endpoint(self, endpoint_id: str) -> Dict[str, Any]:
        """Unregister a workflow MCP endpoint."""
        try:
            # Get endpoint data
            cache_service = await get_cache()
            endpoint_data = await cache_service.get(f"mcp:workflow:endpoints:{endpoint_id}")
            if not endpoint_data:
                return {
                    "success": False,
                    "error": "Endpoint not found"
                }
            
            # Remove from cache
            await cache_service.delete(f"mcp:workflow:endpoints:{endpoint_id}")
            
            # Remove from indices
            if cache_service._redis_client:
                await cache_service._redis_client.srem("mcp:workflow:endpoints:index", endpoint_id)
            
            if endpoint_data.get("workflow_id"):
                if cache_service._redis_client:
                    await cache_service._redis_client.srem(
                        f"mcp:workflow:{endpoint_data['workflow_id']}:endpoints",
                        endpoint_id
                    )
            
            # Publish unregistration event
            await event_bus.publish(
                "mcp.workflow.endpoint.unregistered",
                {
                    "endpoint_id": endpoint_id,
                    "endpoint_name": endpoint_data.get("endpoint_name"),
                    "workflow_id": endpoint_data.get("workflow_id")
                }
            )
            
            logger.info(f"Unregistered workflow endpoint: {endpoint_id}")
            
            return {
                "success": True,
                "endpoint_id": endpoint_id,
                "message": "Endpoint unregistered successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to unregister workflow endpoint: {str(e)}")
            raise
    
    async def get_workflow_endpoints(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get all MCP endpoints for a workflow."""
        try:
            # Get endpoint IDs for workflow
            cache_service = await get_cache()
            endpoint_ids = []
            if cache_service._redis_client:
                endpoint_ids = await cache_service._redis_client.smembers(
                    f"mcp:workflow:{workflow_id}:endpoints"
                )
            
            endpoints = []
            for endpoint_id in endpoint_ids:
                endpoint_data = await cache_service.get(
                    f"mcp:workflow:endpoints:{endpoint_id}"
                )
                if endpoint_data:
                    endpoints.append(endpoint_data)
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to get workflow endpoints: {str(e)}")
            return []
    
    async def get_all_workflow_endpoints(self) -> List[Dict[str, Any]]:
        """Get all registered workflow MCP endpoints."""
        try:
            # Get all endpoint IDs
            cache_service = await get_cache()
            endpoint_ids = []
            if cache_service._redis_client:
                endpoint_ids = await cache_service._redis_client.smembers(
                    "mcp:workflow:endpoints:index"
                )
            
            endpoints = []
            for endpoint_id in endpoint_ids:
                endpoint_data = await cache_service.get(
                    f"mcp:workflow:endpoints:{endpoint_id}"
                )
                if endpoint_data:
                    endpoints.append(endpoint_data)
            
            return endpoints
            
        except Exception as e:
            logger.error(f"Failed to get all workflow endpoints: {str(e)}")
            return []
    
    async def handle_endpoint_request(
        self,
        endpoint_id: str,
        request_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle an incoming request to a workflow MCP endpoint."""
        try:
            # Get endpoint data
            cache_service = await get_cache()
            endpoint_data = await cache_service.get(
                f"mcp:workflow:endpoints:{endpoint_id}"
            )
            
            if not endpoint_data:
                return {
                    "success": False,
                    "error": "Endpoint not found"
                }
            
            if endpoint_data.get("status") != "active":
                return {
                    "success": False,
                    "error": "Endpoint is not active"
                }
            
            # Check if operation is allowed
            operation = request_data.get("operation", "execute")
            allowed_operations = endpoint_data.get("allowed_operations", ["execute"])
            
            if operation not in allowed_operations:
                return {
                    "success": False,
                    "error": f"Operation '{operation}' not allowed",
                    "allowed_operations": allowed_operations
                }
            
            # Update endpoint stats
            endpoint_data["requests_processed"] = endpoint_data.get("requests_processed", 0) + 1
            endpoint_data["last_request_at"] = datetime.utcnow().isoformat()
            
            await cache_service.set(
                f"mcp:workflow:endpoints:{endpoint_id}",
                endpoint_data,
                expire=86400
            )
            
            # Publish request event for the workflow node to handle
            await event_bus.publish(
                f"mcp.endpoint.request:{endpoint_id}",
                {
                    "request_id": request_data.get("request_id"),
                    "client_id": request_data.get("client_id"),
                    "operation": operation,
                    "payload": request_data.get("payload", {}),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
            # For synchronous endpoints, wait for response
            # In production, this would use proper async response handling
            return {
                "success": True,
                "message": "Request received and being processed",
                "request_id": request_data.get("request_id")
            }
            
        except Exception as e:
            logger.error(f"Failed to handle endpoint request: {str(e)}")
            raise
    
    async def get_endpoint_stats(self, endpoint_id: str) -> Dict[str, Any]:
        """Get statistics for a workflow MCP endpoint."""
        try:
            cache_service = await get_cache()
            endpoint_data = await cache_service.get(
                f"mcp:workflow:endpoints:{endpoint_id}"
            )
            
            if not endpoint_data:
                return {
                    "success": False,
                    "error": "Endpoint not found"
                }
            
            return {
                "success": True,
                "endpoint_id": endpoint_id,
                "endpoint_name": endpoint_data.get("endpoint_name"),
                "workflow_id": endpoint_data.get("workflow_id"),
                "status": endpoint_data.get("status"),
                "requests_processed": endpoint_data.get("requests_processed", 0),
                "created_at": endpoint_data.get("created_at"),
                "last_request_at": endpoint_data.get("last_request_at")
            }
            
        except Exception as e:
            logger.error(f"Failed to get endpoint stats: {str(e)}")
            raise