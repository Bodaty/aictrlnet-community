"""MCP Server Node for exposing workflow capabilities as MCP endpoints."""

import logging
import asyncio
from typing import Any, Dict, Optional, List
from datetime import datetime
import uuid

from ..base_node import BaseNode
from ..models import NodeConfig, NodeExecutionResult, NodeStatus
from events.event_bus import event_bus
from core.cache import get_cache


logger = logging.getLogger(__name__)


class MCPServerNode(BaseNode):
    """Node that exposes workflow capabilities as MCP endpoints.
    
    This node allows workflows to act as MCP servers, exposing their functionality
    to external MCP clients. It can operate in different modes:
    - single: Process one request and continue
    - continuous: Keep processing requests until timeout
    - webhook: Register a webhook and process async requests
    
    Parameters:
    - endpoint_name: Name of the MCP endpoint to expose
    - allowed_operations: List of allowed operations (default: ["execute"])
    - auth_required: Whether authentication is required (default: True)
    - mode: Operation mode (single, continuous, webhook)
    - timeout: Timeout in seconds for waiting for requests (default: 300)
    - max_requests: Maximum number of requests to process (continuous mode)
    """
    
    def __init__(self, config: NodeConfig):
        super().__init__(config)
        self.endpoint_id: Optional[str] = None
        self._active = False
        self._request_queue: Optional[asyncio.Queue] = None
    
    async def initialize(self, context: Dict[str, Any]) -> None:
        """Initialize MCP server endpoint."""
        try:
            # Generate unique endpoint ID
            self.endpoint_id = f"mcp-endpoint-{uuid.uuid4().hex[:8]}"
            
            endpoint_name = self.config.parameters.get("endpoint_name")
            if not endpoint_name:
                raise ValueError("endpoint_name is required")
            
            allowed_operations = self.config.parameters.get("allowed_operations", ["execute"])
            auth_required = self.config.parameters.get("auth_required", True)
            workflow_id = context.get("workflow_id")
            
            # Create request queue for this endpoint
            self._request_queue = asyncio.Queue()
            
            # Register endpoint in cache for MCP server to route requests
            endpoint_info = {
                "endpoint_id": self.endpoint_id,
                "endpoint_name": endpoint_name,
                "workflow_id": workflow_id,
                "node_id": self.config.id,
                "allowed_operations": allowed_operations,
                "auth_required": auth_required,
                "created_at": datetime.utcnow().isoformat(),
                "status": "active"
            }
            
            # Store in cache with TTL
            cache_service = await get_cache()
            await cache_service.set(
                f"mcp:endpoints:{self.endpoint_id}",
                endpoint_info,
                expire=3600  # 1 hour TTL
            )
            
            # Also add to endpoint registry for discovery
            # Using Redis set operations directly
            if cache_service._redis_client:
                await cache_service._redis_client.sadd(
                    "mcp:endpoint:registry",
                    self.endpoint_id
                )
            
            self._active = True
            logger.info(f"MCP Server Node initialized endpoint: {endpoint_name} ({self.endpoint_id})")
            
            # Publish initialization event
            await event_bus.publish(
                "mcp.endpoint.created",
                {
                    "endpoint_id": self.endpoint_id,
                    "endpoint_name": endpoint_name,
                    "workflow_id": workflow_id,
                    "node_id": self.config.id
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP Server Node: {str(e)}")
            raise
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute MCP server node - wait for external requests."""
        start_time = datetime.utcnow()
        
        if not self._active:
            await self.initialize(context)
        
        try:
            mode = self.config.parameters.get("mode", "single")
            timeout = self.config.parameters.get("timeout", 300)
            
            if mode == "single":
                # Wait for a single request
                result = await self._process_single_request(timeout, context)
                
            elif mode == "continuous":
                # Process multiple requests
                max_requests = self.config.parameters.get("max_requests", 10)
                result = await self._process_continuous_requests(timeout, max_requests, context)
                
            elif mode == "webhook":
                # Register webhook and return immediately
                result = await self._register_webhook(context)
                
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            # Calculate duration
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return NodeExecutionResult(
                node_instance_id=self.config.id,
                status=NodeStatus.COMPLETED,
                output_data=result,
                metadata={
                    "endpoint_id": self.endpoint_id,
                    "mode": mode,
                    "duration_ms": duration_ms
                }
            )
            
        except asyncio.TimeoutError:
            # Timeout is not necessarily an error for server nodes
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return NodeExecutionResult(
                node_instance_id=self.config.id,
                status=NodeStatus.COMPLETED,
                output_data={
                    "status": "timeout",
                    "message": "No requests received within timeout period",
                    "requests_processed": 0
                },
                metadata={
                    "endpoint_id": self.endpoint_id,
                    "duration_ms": duration_ms
                }
            )
            
        except Exception as e:
            logger.error(f"MCP Server Node {self.config.id} failed: {str(e)}")
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return NodeExecutionResult(
                node_instance_id=self.config.id,
                status=NodeStatus.FAILED,
                error=str(e),
                metadata={
                    "endpoint_id": self.endpoint_id,
                    "duration_ms": duration_ms
                }
            )
    
    async def _process_single_request(
        self,
        timeout: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a single MCP request."""
        # Subscribe to request events for this endpoint
        request_event = f"mcp.endpoint.request:{self.endpoint_id}"
        
        # Create a future to wait for request
        request_future = asyncio.Future()
        
        async def handle_request(event_data):
            """Handle incoming MCP request."""
            if not request_future.done():
                request_future.set_result(event_data)
        
        # Subscribe to request events
        await event_bus.subscribe(request_event, handle_request)
        
        try:
            # Wait for request with timeout
            request_data = await asyncio.wait_for(request_future, timeout=timeout)
            
            # Process the request
            response = await self._process_request(request_data, context)
            
            # Send response back via event
            await event_bus.publish(
                f"mcp.endpoint.response:{request_data.get('request_id')}",
                response
            )
            
            return {
                "status": "processed",
                "request": request_data,
                "response": response,
                "requests_processed": 1
            }
            
        finally:
            # Unsubscribe from events
            await event_bus.unsubscribe(request_event, handle_request)
    
    async def _process_continuous_requests(
        self,
        timeout: int,
        max_requests: int,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process multiple MCP requests continuously."""
        requests_processed = 0
        all_requests = []
        all_responses = []
        
        # Subscribe to request events
        request_event = f"mcp.endpoint.request:{self.endpoint_id}"
        
        async def handle_request(event_data):
            """Handle incoming MCP request."""
            await self._request_queue.put(event_data)
        
        await event_bus.subscribe(request_event, handle_request)
        
        try:
            # Process requests until timeout or max reached
            end_time = datetime.utcnow().timestamp() + timeout
            
            while requests_processed < max_requests and datetime.utcnow().timestamp() < end_time:
                try:
                    # Calculate remaining time
                    remaining = end_time - datetime.utcnow().timestamp()
                    if remaining <= 0:
                        break
                    
                    # Wait for request with remaining timeout
                    request_data = await asyncio.wait_for(
                        self._request_queue.get(),
                        timeout=remaining
                    )
                    
                    # Process the request
                    response = await self._process_request(request_data, context)
                    
                    # Send response back
                    await event_bus.publish(
                        f"mcp.endpoint.response:{request_data.get('request_id')}",
                        response
                    )
                    
                    all_requests.append(request_data)
                    all_responses.append(response)
                    requests_processed += 1
                    
                except asyncio.TimeoutError:
                    # No more requests within timeout
                    break
            
            return {
                "status": "completed",
                "requests_processed": requests_processed,
                "requests": all_requests,
                "responses": all_responses
            }
            
        finally:
            # Unsubscribe from events
            await event_bus.unsubscribe(request_event, handle_request)
    
    async def _register_webhook(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Register webhook for async processing."""
        webhook_url = self.config.parameters.get("webhook_url")
        if not webhook_url:
            # Generate internal webhook URL
            webhook_url = f"/api/v1/mcp-server/webhook/{self.endpoint_id}"
        
        # Store webhook configuration
        webhook_info = {
            "endpoint_id": self.endpoint_id,
            "webhook_url": webhook_url,
            "workflow_id": context.get("workflow_id"),
            "node_id": self.config.id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        cache_service = await get_cache()
        await cache_service.set(
            f"mcp:webhooks:{self.endpoint_id}",
            webhook_info,
            expire=86400  # 24 hour TTL
        )
        
        return {
            "status": "webhook_registered",
            "endpoint_id": self.endpoint_id,
            "webhook_url": webhook_url,
            "message": "Webhook registered. Requests will be processed asynchronously."
        }
    
    async def _process_request(
        self,
        request_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process an individual MCP request."""
        operation = request_data.get("operation", "execute")
        
        # Check if operation is allowed
        allowed_operations = self.config.parameters.get("allowed_operations", ["execute"])
        if operation not in allowed_operations:
            return {
                "status": "error",
                "error": f"Operation '{operation}' not allowed",
                "allowed_operations": allowed_operations
            }
        
        # Extract request payload
        payload = request_data.get("payload", {})
        
        # Merge with node input data
        merged_data = {
            **payload,
            "mcp_request_id": request_data.get("request_id"),
            "mcp_client_id": request_data.get("client_id"),
            "mcp_operation": operation
        }
        
        # Return processed data
        # In a real implementation, this would trigger workflow continuation
        return {
            "status": "success",
            "result": merged_data,
            "processed_at": datetime.utcnow().isoformat(),
            "endpoint_id": self.endpoint_id
        }
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        # Required parameters
        if not self.config.parameters.get("endpoint_name"):
            raise ValueError("endpoint_name is required")
        
        # Validate mode
        mode = self.config.parameters.get("mode", "single")
        valid_modes = ["single", "continuous", "webhook"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        
        # Validate timeout
        timeout = self.config.parameters.get("timeout", 300)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        
        # Validate max_requests for continuous mode
        if mode == "continuous":
            max_requests = self.config.parameters.get("max_requests", 10)
            if not isinstance(max_requests, int) or max_requests <= 0:
                raise ValueError("max_requests must be a positive integer")
        
        return True
    
    async def cleanup(self) -> None:
        """Cleanup MCP endpoint registration."""
        if self.endpoint_id and self._active:
            try:
                # Remove from cache
                cache_service = await get_cache()
                await cache_service.delete(f"mcp:endpoints:{self.endpoint_id}")
                if cache_service._redis_client:
                    await cache_service._redis_client.srem("mcp:endpoint:registry", self.endpoint_id)
                
                # Remove webhook if registered
                await cache_service.delete(f"mcp:webhooks:{self.endpoint_id}")
                
                # Publish cleanup event
                await event_bus.publish(
                    "mcp.endpoint.removed",
                    {
                        "endpoint_id": self.endpoint_id,
                        "node_id": self.config.id
                    }
                )
                
                logger.info(f"MCP Server Node cleaned up endpoint: {self.endpoint_id}")
                
            except Exception as e:
                logger.error(f"Error during MCP Server Node cleanup: {str(e)}")
        
        self._active = False