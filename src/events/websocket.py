"""WebSocket integration for real-time event delivery."""

import asyncio
import logging
from typing import Dict, Set, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime
import json

from .event_bus import event_bus
from .models import Event, EventSubscription


logger = logging.getLogger(__name__)


class WebSocketEventManager:
    """Manage WebSocket connections for event delivery."""
    
    def __init__(self):
        # Active connections by connection ID
        self._connections: Dict[str, WebSocket] = {}
        
        # Connection metadata
        self._connection_info: Dict[str, Dict[str, Any]] = {}
        
        # Subscriptions by connection
        self._connection_subscriptions: Dict[str, Set[str]] = {}
        
        # Lock for thread safety
        self._lock = asyncio.Lock()
    
    async def connect(
        self,
        websocket: WebSocket,
        connection_id: str,
        user_id: str,
        edition: str = "community"
    ):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        
        async with self._lock:
            self._connections[connection_id] = websocket
            self._connection_info[connection_id] = {
                "user_id": user_id,
                "edition": edition,
                "connected_at": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }
            self._connection_subscriptions[connection_id] = set()
        
        # Register with event bus
        await event_bus.register_websocket(connection_id, self)
        
        # Send welcome message
        await self.send_to_connection(connection_id, {
            "type": "welcome",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        logger.info(f"WebSocket connection established: {connection_id}")
    
    async def disconnect(self, connection_id: str):
        """Handle WebSocket disconnection."""
        async with self._lock:
            # Unsubscribe from all events
            subscription_ids = self._connection_subscriptions.get(connection_id, set())
            for sub_id in subscription_ids:
                await event_bus.unsubscribe(sub_id)
            
            # Clean up connection data
            self._connections.pop(connection_id, None)
            self._connection_info.pop(connection_id, None)
            self._connection_subscriptions.pop(connection_id, None)
        
        # Unregister from event bus
        await event_bus.unregister_websocket(connection_id)
        
        logger.info(f"WebSocket connection closed: {connection_id}")
    
    async def handle_message(
        self,
        connection_id: str,
        message: Dict[str, Any]
    ):
        """Handle incoming WebSocket message."""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            await self._handle_subscribe(connection_id, message)
        elif message_type == "unsubscribe":
            await self._handle_unsubscribe(connection_id, message)
        elif message_type == "publish":
            await self._handle_publish(connection_id, message)
        elif message_type == "ping":
            await self._handle_ping(connection_id)
        else:
            await self.send_to_connection(connection_id, {
                "type": "error",
                "error": f"Unknown message type: {message_type}"
            })
    
    async def _handle_subscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle subscription request."""
        event_types = message.get("event_types", [])
        filter_criteria = message.get("filter", {})
        
        if not event_types:
            await self.send_to_connection(connection_id, {
                "type": "error",
                "error": "No event types specified"
            })
            return
        
        # Get connection info
        conn_info = self._connection_info.get(connection_id, {})
        user_id = conn_info.get("user_id")
        
        # Create subscription
        subscription = await event_bus.subscribe(
            subscriber_id=user_id,
            event_types=event_types,
            websocket_connection_id=connection_id,
            filter_criteria=filter_criteria
        )
        
        # Track subscription
        async with self._lock:
            self._connection_subscriptions[connection_id].add(subscription.id)
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "subscribed",
            "subscription_id": subscription.id,
            "event_types": event_types
        })
    
    async def _handle_unsubscribe(self, connection_id: str, message: Dict[str, Any]):
        """Handle unsubscription request."""
        subscription_id = message.get("subscription_id")
        
        if not subscription_id:
            await self.send_to_connection(connection_id, {
                "type": "error",
                "error": "No subscription ID specified"
            })
            return
        
        # Verify ownership
        async with self._lock:
            if subscription_id not in self._connection_subscriptions.get(connection_id, set()):
                await self.send_to_connection(connection_id, {
                    "type": "error",
                    "error": "Invalid subscription ID"
                })
                return
            
            # Unsubscribe
            await event_bus.unsubscribe(subscription_id)
            self._connection_subscriptions[connection_id].discard(subscription_id)
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "unsubscribed",
            "subscription_id": subscription_id
        })
    
    async def _handle_publish(self, connection_id: str, message: Dict[str, Any]):
        """Handle event publication request."""
        event_type = message.get("event_type")
        data = message.get("data", {})
        
        if not event_type:
            await self.send_to_connection(connection_id, {
                "type": "error",
                "error": "No event type specified"
            })
            return
        
        # Get connection info
        conn_info = self._connection_info.get(connection_id, {})
        user_id = conn_info.get("user_id")
        edition = conn_info.get("edition")
        
        # Publish event
        event = await event_bus.publish(
            event_type=event_type,
            data=data,
            source_id=user_id,
            source_type="user",
            edition=edition
        )
        
        # Send confirmation
        await self.send_to_connection(connection_id, {
            "type": "published",
            "event_id": event.id,
            "event_type": event_type
        })
    
    async def _handle_ping(self, connection_id: str):
        """Handle ping message."""
        async with self._lock:
            if connection_id in self._connection_info:
                self._connection_info[connection_id]["last_activity"] = datetime.utcnow()
        
        await self.send_to_connection(connection_id, {
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        })
    
    async def send_to_connection(self, connection_id: str, data: Dict[str, Any]):
        """Send data to a specific connection."""
        websocket = self._connections.get(connection_id)
        if websocket:
            try:
                await websocket.send_json(data)
            except Exception as e:
                logger.error(f"Failed to send to connection {connection_id}: {str(e)}")
                await self.disconnect(connection_id)
    
    async def send_json(self, data: Dict[str, Any]):
        """Send data to all connections (for event bus integration)."""
        # Extract connection ID from event data if available
        connection_id = data.get("connection_id")
        if connection_id:
            await self.send_to_connection(connection_id, data)
        else:
            # Broadcast to all connections
            disconnected = []
            for conn_id, websocket in self._connections.items():
                try:
                    await websocket.send_json(data)
                except Exception:
                    disconnected.append(conn_id)
            
            # Clean up disconnected connections
            for conn_id in disconnected:
                await self.disconnect(conn_id)
    
    async def cleanup_inactive_connections(self, timeout_minutes: int = 30):
        """Clean up inactive connections."""
        now = datetime.utcnow()
        inactive = []
        
        async with self._lock:
            for conn_id, info in self._connection_info.items():
                last_activity = info.get("last_activity")
                if last_activity:
                    inactive_minutes = (now - last_activity).total_seconds() / 60
                    if inactive_minutes > timeout_minutes:
                        inactive.append(conn_id)
        
        # Disconnect inactive connections
        for conn_id in inactive:
            await self.disconnect(conn_id)
            logger.info(f"Cleaned up inactive connection: {conn_id}")
        
        return len(inactive)


# Global WebSocket manager
websocket_manager = WebSocketEventManager()


# WebSocket endpoint handler
async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: str,
    user_id: str,
    edition: str = "community"
):
    """Handle WebSocket connection for events."""
    await websocket_manager.connect(websocket, connection_id, user_id, edition)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            
            # Handle message
            await websocket_manager.handle_message(connection_id, data)
            
    except WebSocketDisconnect:
        await websocket_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await websocket_manager.disconnect(connection_id)