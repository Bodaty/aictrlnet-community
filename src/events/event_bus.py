"""Event bus for workflow and system events."""

import asyncio
import logging
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for publishing and subscribing to events."""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the event bus processor."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._process_events())
        logger.info("Event bus started")
    
    async def stop(self):
        """Stop the event bus processor."""
        if not self._running:
            return
        
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        logger.info("Event bus stopped")
    
    async def _process_events(self):
        """Process events from the queue."""
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=1.0
                )
                
                # Process event
                await self._dispatch_event(event["name"], event["data"])
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    async def _dispatch_event(self, event_name: str, data: Any):
        """Dispatch event to all subscribers."""
        # Get exact match subscribers
        subscribers = self._subscribers.get(event_name, [])
        
        # Get wildcard subscribers
        parts = event_name.split(".")
        for i in range(len(parts)):
            wildcard_pattern = ".".join(parts[:i]) + ".*"
            subscribers.extend(self._subscribers.get(wildcard_pattern, []))
        
        # Call all subscribers
        for subscriber in subscribers:
            try:
                if asyncio.iscoroutinefunction(subscriber):
                    await subscriber(event_name, data)
                else:
                    subscriber(event_name, data)
            except Exception as e:
                logger.error(f"Error in event subscriber: {e}")
    
    def subscribe(self, event_pattern: str, handler: Callable):
        """Subscribe to events matching a pattern.
        
        Args:
            event_pattern: Event name or pattern (supports wildcards like "workflow.*")
            handler: Callback function to handle the event
        """
        if event_pattern not in self._subscribers:
            self._subscribers[event_pattern] = []
        
        if handler not in self._subscribers[event_pattern]:
            self._subscribers[event_pattern].append(handler)
            logger.debug(f"Subscribed to {event_pattern}")
    
    def unsubscribe(self, event_pattern: str, handler: Callable):
        """Unsubscribe from events."""
        if event_pattern in self._subscribers:
            if handler in self._subscribers[event_pattern]:
                self._subscribers[event_pattern].remove(handler)
                logger.debug(f"Unsubscribed from {event_pattern}")
    
    async def publish(self, event_name: str, data: Any):
        """Publish an event.
        
        Args:
            event_name: Name of the event (e.g., "workflow.execution.started")
            data: Event data
        """
        event = {
            "name": event_name,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Add to queue for async processing
        await self._event_queue.put(event)
        
        # Log event
        logger.debug(f"Published event: {event_name}")
    
    def get_subscriber_count(self, event_pattern: Optional[str] = None) -> int:
        """Get count of subscribers for a pattern or all subscribers."""
        if event_pattern:
            return len(self._subscribers.get(event_pattern, []))
        return sum(len(subs) for subs in self._subscribers.values())


# Global event bus instance
event_bus = EventBus()


# Common event handlers
async def log_workflow_events(event_name: str, data: Any):
    """Log workflow events."""
    if event_name.startswith("workflow."):
        logger.info(f"Workflow event: {event_name} - {json.dumps(data)}")


async def track_execution_metrics(event_name: str, data: Any):
    """Track execution metrics."""
    if event_name == "workflow.execution.completed":
        duration = data.get("duration_ms", 0)
        workflow_id = data.get("workflow_id")
        logger.info(f"Workflow {workflow_id} completed in {duration}ms")
    elif event_name == "workflow.execution.failed":
        workflow_id = data.get("workflow_id")
        error = data.get("error", "Unknown")
        logger.error(f"Workflow {workflow_id} failed: {error}")


async def notify_channel_on_completion(event_name: str, data: Any):
    """When a conversation-triggered workflow completes, send notification via the originating channel.

    This handler looks up the conversation session that triggered the workflow
    and sends a completion message back through the channel adapter.
    """
    if event_name != "workflow.execution.completed":
        return

    triggered_by = data.get("triggered_by", "")
    if triggered_by not in ("conversation", "file_upload"):
        return

    trigger_meta = data.get("trigger_metadata", {})
    session_id = trigger_meta.get("session_id")
    if not session_id:
        return

    try:
        from core.database import async_session_factory
        from sqlalchemy import select as sa_select
        from models.conversation import ConversationSession

        async with async_session_factory() as db:
            result = await db.execute(
                sa_select(ConversationSession).filter(ConversationSession.id == session_id)
            )
            session = result.scalar_one_or_none()

            if not session or not session.channel_bindings:
                return

            channel = session.primary_channel or "web"
            if channel == "web":
                return  # Web users get notified via WebSocket, not adapter

            binding = session.channel_bindings.get(channel, {})
            sender_id = binding.get("sender_id")
            if not sender_id:
                return

            workflow_id = data.get("workflow_id", "unknown")
            duration_ms = data.get("duration_ms", 0)
            text = f"Workflow {workflow_id} completed in {duration_ms}ms."

            from adapters.registry import adapter_registry
            from adapters.models import AdapterRequest as AR

            adapter_class = adapter_registry.get_adapter_class(channel)
            if not adapter_class:
                return

            adapter = adapter_class({})
            await adapter.execute(AR(capability="send_message", parameters={
                "to": sender_id,
                "text": text,
                "chat_id": binding.get("chat_id", sender_id),
            }))
            logger.info(f"Sent workflow completion notification via {channel} to {sender_id}")
    except Exception as e:
        logger.warning(f"Failed to send channel completion notification: {e}")


# Subscribe default handlers
event_bus.subscribe("workflow.*", log_workflow_events)
event_bus.subscribe("workflow.execution.*", track_execution_metrics)
event_bus.subscribe("workflow.execution.completed", notify_channel_on_completion)