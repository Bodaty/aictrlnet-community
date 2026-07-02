"""WebSocket endpoints for real-time workflow updates."""

from fastapi import WebSocket, WebSocketDisconnect, Depends, Query
from typing import Optional, Dict, Set
import json
import asyncio
import logging
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncSession
from core.config import get_settings
from core.database import get_db
from core.security import verify_token
from core.tenant_context import get_current_tenant_id
from events.event_bus import event_bus
from schemas.workflow_node import WorkflowExecutionUpdate

logger = logging.getLogger(__name__)


class WorkflowConnectionManager:
    """Manage WebSocket connections for workflow updates."""
    
    def __init__(self):
        # Map of execution_id to set of websockets
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Map of websocket to user info
        self.connection_users: Dict[WebSocket, dict] = {}
        
    async def connect(self, websocket: WebSocket, execution_id: str, user: dict):
        """Accept new connection for workflow execution."""
        await websocket.accept()
        
        # Add to connections
        if execution_id not in self.active_connections:
            self.active_connections[execution_id] = set()
        self.active_connections[execution_id].add(websocket)
        self.connection_users[websocket] = user
        
        # Send initial connection message
        await self._send_message(websocket, {
            "type": "connection",
            "status": "connected",
            "execution_id": execution_id,
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
        
        logger.info(f"WebSocket connected for execution {execution_id} by user {user.get('email')}")
    
    def disconnect(self, websocket: WebSocket, execution_id: str):
        """Remove connection."""
        if execution_id in self.active_connections:
            self.active_connections[execution_id].discard(websocket)
            if not self.active_connections[execution_id]:
                del self.active_connections[execution_id]
        
        if websocket in self.connection_users:
            user = self.connection_users.pop(websocket)
            logger.info(f"WebSocket disconnected for execution {execution_id} by user {user.get('email')}")
    
    async def broadcast_to_execution(self, execution_id: str, message: dict):
        """Broadcast message to all connections watching an execution."""
        if execution_id in self.active_connections:
            disconnected = set()
            
            for websocket in self.active_connections[execution_id]:
                try:
                    await self._send_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error sending message: {e}")
                    disconnected.add(websocket)
            
            # Clean up disconnected sockets
            for websocket in disconnected:
                self.disconnect(websocket, execution_id)
    
    async def _send_message(self, websocket: WebSocket, message: dict):
        """Send JSON message to websocket."""
        await websocket.send_json(message)


# Global connection manager
manager = WorkflowConnectionManager()


# Event handlers for workflow events
async def handle_workflow_event(event_name: str, data: dict):
    """Handle workflow events and broadcast to connected clients."""
    execution_id = data.get("execution_id")
    if not execution_id:
        return
    
    # Map event types to WebSocket message types
    event_mapping = {
        "workflow.execution.started": "execution.started",
        "workflow.execution.completed": "execution.completed",
        "workflow.execution.failed": "execution.failed",
        "workflow.execution.paused": "execution.paused",
        "workflow.execution.resumed": "execution.resumed",
        "workflow.execution.cancelled": "execution.cancelled",
        "workflow.node.started": "node.started",
        "workflow.node.completed": "node.completed",
        "workflow.node.failed": "node.failed",
        "workflow.node.pending": "node.pending"
    }
    
    ws_type = event_mapping.get(event_name, "update")
    
    # Create WebSocket message
    message = {
        "type": ws_type,
        "execution_id": execution_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "data": data
    }
    
    # Broadcast to connected clients
    await manager.broadcast_to_execution(execution_id, message)


# Subscribe to workflow events
event_bus.subscribe("workflow.execution.*", handle_workflow_event)
event_bus.subscribe("workflow.node.*", handle_workflow_event)


async def workflow_execution_websocket(
    websocket: WebSocket,
    execution_id: str,
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """WebSocket endpoint for workflow execution updates.

    Connect with: ws://localhost:8001/ws/workflows/{execution_id}?token={jwt_token}
    """
    # Authenticate user — full DB verification (existence, is_active,
    # token_version, non-access-type rejection, gated dev token). The previous
    # bare jwt.decode ignored revocation/type and let deactivated users connect.
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return

    user_info = await verify_token(token, db=db)
    if not user_info:
        await websocket.close(code=4001, reason="Authentication failed")
        return
    user = {
        "email": user_info.get("email"),
        "user_id": user_info.get("id"),
        "tenant_id": user_info.get("tenant_id") or get_current_tenant_id(),
    }

    # Connect
    await manager.connect(websocket, execution_id, user)
    
    try:
        # Keep connection alive and handle incoming messages
        while True:
            # Wait for messages from client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                elif message.get("type") == "subscribe":
                    # Client can subscribe to specific node updates
                    node_id = message.get("node_id")
                    if node_id:
                        # This could be extended to filter updates
                        logger.info(f"Client subscribed to node {node_id}")
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "message": "Invalid JSON"
                })
                
    except WebSocketDisconnect:
        manager.disconnect(websocket, execution_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        manager.disconnect(websocket, execution_id)


async def workflow_catalog_websocket(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """WebSocket endpoint for workflow catalog updates.

    Notifies when new nodes/adapters/agents become available.
    Connect with: ws://localhost:8001/ws/workflows/catalog?token={jwt_token}
    """
    # Authenticate user — full DB verification via verify_token (see execution ws).
    if not token:
        await websocket.close(code=4001, reason="Authentication required")
        return

    user_info = await verify_token(token, db=db)
    if not user_info:
        await websocket.close(code=4001, reason="Authentication failed")
        return
    user = {
        "email": user_info.get("email"),
        "user_id": user_info.get("id"),
        "tenant_id": user_info.get("tenant_id") or get_current_tenant_id(),
    }

    await websocket.accept()
    
    # Send initial message
    await websocket.send_json({
        "type": "connection",
        "status": "connected",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })
    
    # Handler for catalog updates
    async def handle_catalog_update(event_name: str, data: dict):
        """Send catalog updates to client."""
        try:
            await websocket.send_json({
                "type": "catalog.update",
                "event": event_name,
                "data": data,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })
        except Exception as e:
            logger.error(f"Error sending catalog update: {e}")
    
    # Subscribe to relevant events
    event_bus.subscribe("adapter.registered", handle_catalog_update)
    event_bus.subscribe("agent.registered", handle_catalog_update)
    event_bus.subscribe("mcp.server.added", handle_catalog_update)
    
    try:
        # Keep connection alive
        while True:
            data = await websocket.receive_text()
            
            # Handle ping
            if data == "ping":
                await websocket.send_text("pong")
                
    except WebSocketDisconnect:
        # Unsubscribe
        event_bus.unsubscribe("adapter.registered", handle_catalog_update)
        event_bus.unsubscribe("agent.registered", handle_catalog_update)
        event_bus.unsubscribe("mcp.server.added", handle_catalog_update)
        logger.info("Catalog WebSocket disconnected")