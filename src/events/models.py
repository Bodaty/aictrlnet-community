"""Event models for the event bus system."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Callable, List
from pydantic import BaseModel, Field
import uuid


class EventPriority(str, Enum):
    """Event priority levels."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class Event(BaseModel):
    """Base event model."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: str  # e.g., "component.registered", "workflow.completed"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    priority: EventPriority = EventPriority.NORMAL
    
    # Event source
    source_id: Optional[str] = None  # Component or service that generated the event
    source_type: Optional[str] = None  # Type of source (component, user, system)
    
    # Event data
    data: Dict[str, Any] = {}
    metadata: Dict[str, Any] = {}
    
    # Routing and filtering
    edition: Optional[str] = None  # Target edition (community, business, enterprise)
    tags: List[str] = []
    
    # Persistence and replay
    persistent: bool = True  # Whether to persist this event
    replay_safe: bool = True  # Whether this event can be replayed
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "component.registered",
                "source_id": "openai-adapter",
                "source_type": "component",
                "data": {
                    "component_id": "123",
                    "name": "openai-adapter",
                    "capabilities": ["chat_completion"]
                },
                "priority": "normal",
                "tags": ["adapter", "ai"]
            }
        }


class EventSubscription(BaseModel):
    """Subscription to event types."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    subscriber_id: str  # Component or service subscribing
    event_types: List[str]  # Event types to subscribe to (supports wildcards)
    filter_criteria: Dict[str, Any] = {}  # Additional filtering
    
    # Delivery options
    webhook_url: Optional[str] = None
    websocket_connection_id: Optional[str] = None
    callback_function: Optional[str] = None  # Name of callback function
    
    # Subscription metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    active: bool = True
    
    # Retry policy
    max_retries: int = 3
    retry_delay_seconds: int = 60


class EventHandler:
    """Handler for processing events."""
    
    def __init__(
        self,
        event_types: List[str],
        handler_func: Callable,
        priority: EventPriority = EventPriority.NORMAL,
        filter_func: Optional[Callable] = None,
        name: Optional[str] = None
    ):
        self.event_types = event_types
        self.handler_func = handler_func
        self.priority = priority
        self.filter_func = filter_func
        self.name = name or handler_func.__name__
        self.id = str(uuid.uuid4())
    
    async def can_handle(self, event: Event) -> bool:
        """Check if this handler can handle the event."""
        # Check event type match (supports wildcards)
        type_match = any(
            self._matches_pattern(event.type, pattern)
            for pattern in self.event_types
        )
        
        if not type_match:
            return False
        
        # Apply custom filter if provided
        if self.filter_func:
            try:
                return await self.filter_func(event)
            except Exception:
                return False
        
        return True
    
    async def handle(self, event: Event) -> Any:
        """Handle the event."""
        return await self.handler_func(event)
    
    def _matches_pattern(self, event_type: str, pattern: str) -> bool:
        """Check if event type matches pattern (supports wildcards)."""
        if pattern == "*":
            return True
        
        if pattern.endswith(".*"):
            prefix = pattern[:-2]
            return event_type.startswith(prefix)
        
        return event_type == pattern


class EventDeliveryStatus(BaseModel):
    """Status of event delivery to a subscriber."""
    event_id: str
    subscription_id: str
    status: str  # delivered, failed, pending, retrying
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None