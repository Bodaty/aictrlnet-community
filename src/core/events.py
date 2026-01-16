"""Event bus for AICtrlNet platform monitoring"""
import asyncio
from typing import Dict, List, Any, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class EventBus:
    """Simple event bus for platform monitoring events"""
    
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    async def start(self):
        """Start the event bus processor"""
        self._running = True
        asyncio.create_task(self._process_events())
        
    async def stop(self):
        """Stop the event bus processor"""
        self._running = False
        
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """Publish an event to the bus"""
        await self._queue.put({
            "type": event_type,
            "data": data
        })
        
    def subscribe(self, event_type: str, handler: Callable):
        """Subscribe to an event type"""
        self._subscribers[event_type].append(handler)
        
    def unsubscribe(self, event_type: str, handler: Callable):
        """Unsubscribe from an event type"""
        if handler in self._subscribers[event_type]:
            self._subscribers[event_type].remove(handler)
            
    async def _process_events(self):
        """Process events from the queue"""
        while self._running:
            try:
                # Get event from queue with timeout
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                
                event_type = event["type"]
                event_data = event["data"]
                
                # Call all subscribers for this event type
                for handler in self._subscribers.get(event_type, []):
                    try:
                        if asyncio.iscoroutinefunction(handler):
                            await handler(event_data)
                        else:
                            handler(event_data)
                    except Exception as e:
                        logger.error(f"Error in event handler for {event_type}: {e}")
                        
            except asyncio.TimeoutError:
                # No events in queue, continue
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")


# Global event bus instance
event_bus = EventBus()