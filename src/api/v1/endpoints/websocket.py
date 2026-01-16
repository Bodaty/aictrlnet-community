"""WebSocket endpoints for real-time updates."""

from typing import Dict, Set, Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import json
import asyncio
import logging
from datetime import datetime

from core.security import verify_token
from core.dependencies import get_edition

router = APIRouter()
logger = logging.getLogger(__name__)

# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections for real-time updates."""
    
    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict] = {}
    
    async def connect(self, websocket: WebSocket, user_id: str, metadata: Dict):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        
        self.active_connections[user_id].add(websocket)
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            **metadata
        }
        
        logger.info(f"WebSocket connected for user {user_id}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection."""
        metadata = self.connection_metadata.get(websocket, {})
        user_id = metadata.get("user_id")
        
        if user_id and user_id in self.active_connections:
            self.active_connections[user_id].discard(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        
        if websocket in self.connection_metadata:
            del self.connection_metadata[websocket]
        
        logger.info(f"WebSocket disconnected for user {user_id}")
    
    async def send_personal_message(self, message: str, user_id: str):
        """Send a message to all connections for a specific user."""
        if user_id in self.active_connections:
            disconnected = set()
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending message to user {user_id}: {e}")
                    disconnected.add(connection)
            
            # Clean up disconnected connections
            for conn in disconnected:
                self.disconnect(conn)
    
    async def broadcast(self, message: str, exclude_user: Optional[str] = None):
        """Broadcast a message to all connected users."""
        disconnected = set()
        
        for user_id, connections in self.active_connections.items():
            if user_id == exclude_user:
                continue
                
            for connection in connections:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error broadcasting to user {user_id}: {e}")
                    disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)
    
    async def send_to_edition(self, message: str, edition: str):
        """Send a message to all users of a specific edition."""
        disconnected = set()
        
        for connection, metadata in self.connection_metadata.items():
            if metadata.get("edition") == edition:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Error sending to edition {edition}: {e}")
                    disconnected.add(connection)
        
        # Clean up disconnected connections
        for conn in disconnected:
            self.disconnect(conn)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(..., description="Authentication token"),
    edition: str = Query("community", description="Edition type"),
):
    """Main WebSocket endpoint for real-time updates."""
    # Verify token
    try:
        user = verify_token(token)
        if not user:
            await websocket.close(code=1008, reason="Invalid token")
            return
    except Exception as e:
        logger.error(f"WebSocket auth error: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return
    
    # Connect the WebSocket
    await manager.connect(
        websocket,
        user["id"],
        {"edition": edition, "username": user.get("username")}
    )
    
    # Send welcome message
    await websocket.send_text(json.dumps({
        "type": "connected",
        "message": "Connected to AICtrlNet real-time updates",
        "user_id": user["id"],
        "edition": edition,
        "timestamp": datetime.utcnow().isoformat()
    }))
    
    try:
        while True:
            # Wait for messages from the client
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_client_message(websocket, user, message)
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }))
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # Notify others that user disconnected (optional)
        await manager.broadcast(
            json.dumps({
                "type": "user_disconnected",
                "user_id": user["id"],
                "timestamp": datetime.utcnow().isoformat()
            }),
            exclude_user=user["id"]
        )


async def handle_client_message(websocket: WebSocket, user: Dict, message: Dict):
    """Handle messages received from the client."""
    msg_type = message.get("type")
    
    if msg_type == "ping":
        # Respond to ping with pong
        await websocket.send_text(json.dumps({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    elif msg_type == "subscribe":
        # Subscribe to specific event types
        event_types = message.get("events", [])
        metadata = manager.connection_metadata.get(websocket, {})
        metadata["subscriptions"] = event_types
        
        await websocket.send_text(json.dumps({
            "type": "subscribed",
            "events": event_types,
            "timestamp": datetime.utcnow().isoformat()
        }))
    
    elif msg_type == "broadcast":
        # Allow users to broadcast messages (with permission checks)
        if user.get("role") in ["admin", "moderator"]:
            broadcast_msg = {
                "type": "broadcast",
                "from": user["id"],
                "message": message.get("message", ""),
                "timestamp": datetime.utcnow().isoformat()
            }
            await manager.broadcast(json.dumps(broadcast_msg), exclude_user=user["id"])
        else:
            await websocket.send_text(json.dumps({
                "type": "error",
                "message": "Insufficient permissions for broadcast",
                "timestamp": datetime.utcnow().isoformat()
            }))
    
    else:
        # Unknown message type
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Unknown message type: {msg_type}",
            "timestamp": datetime.utcnow().isoformat()
        }))


# Event notification functions (to be called from other parts of the application)
async def notify_task_created(task_id: str, user_id: str, task_data: Dict):
    """Notify users about a new task creation."""
    message = json.dumps({
        "type": "task_created",
        "task_id": task_id,
        "user_id": user_id,
        "data": task_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    await manager.send_personal_message(message, user_id)


async def notify_workflow_updated(workflow_id: str, edition: str, update_data: Dict):
    """Notify users about workflow updates."""
    message = json.dumps({
        "type": "workflow_updated",
        "workflow_id": workflow_id,
        "data": update_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    await manager.send_to_edition(message, edition)


async def notify_approval_required(approval_id: str, approver_ids: list, request_data: Dict):
    """Notify approvers about pending approval requests."""
    message = json.dumps({
        "type": "approval_required",
        "approval_id": approval_id,
        "data": request_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    for approver_id in approver_ids:
        await manager.send_personal_message(message, approver_id)


async def notify_system_event(event_type: str, event_data: Dict):
    """Broadcast system-wide events."""
    message = json.dumps({
        "type": "system_event",
        "event_type": event_type,
        "data": event_data,
        "timestamp": datetime.utcnow().isoformat()
    })
    await manager.broadcast(message)


# Export the connection manager for use in other modules
__all__ = [
    "router",
    "manager",
    "notify_task_created",
    "notify_workflow_updated",
    "notify_approval_required",
    "notify_system_event"
]