"""Webhook models for event notifications."""

from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON, Integer, Text
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base


class Webhook(Base):
    """Webhook configuration for event notifications."""
    
    __tablename__ = "webhooks"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # User relationship
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    # Relationship defined in User model to avoid circular imports
    
    # Webhook configuration
    name = Column(String(255), nullable=False)
    description = Column(String(500))
    url = Column(String(2048), nullable=False)  # Webhook endpoint URL
    
    # Authentication
    secret = Column(String(255))  # Secret for HMAC signature verification
    auth_header = Column(String(255))  # Custom auth header name
    auth_value_encrypted = Column(Text)  # Encrypted auth header value
    
    # Event configuration
    events = Column(JSON, default=list)  # List of event types to subscribe to
    # Examples: ["task.created", "task.completed", "workflow.failed", "agent.error"]
    
    # Retry configuration
    max_retries = Column(Integer, default=3)
    retry_delay_seconds = Column(Integer, default=60)
    timeout_seconds = Column(Integer, default=30)
    
    # Status tracking
    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime)
    last_success_at = Column(DateTime)
    last_failure_at = Column(DateTime)
    consecutive_failures = Column(Integer, default=0)
    total_deliveries = Column(Integer, default=0)
    total_failures = Column(Integer, default=0)
    
    # Auto-disable after too many failures
    failure_threshold = Column(Integer, default=10)  # Disable after N consecutive failures
    disabled_reason = Column(String(500))
    
    # Metadata
    custom_headers = Column(JSON, default=dict)  # Additional headers to send
    webhook_metadata = Column('metadata', JSON, default=dict)  # User-defined metadata
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<Webhook {self.name} ({self.url})>"
    
    def should_deliver_event(self, event_type: str) -> bool:
        """Check if webhook should receive this event type."""
        if not self.is_active:
            return False
        if not self.events:  # Empty list means subscribe to all events
            return True
        
        # Check exact match or wildcard patterns
        for pattern in self.events:
            if pattern == event_type:
                return True
            if pattern.endswith(".*") and event_type.startswith(pattern[:-1]):
                return True
        
        return False
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "events": self.events or [],
            "is_active": self.is_active,
            "last_triggered_at": self.last_triggered_at.isoformat() if self.last_triggered_at else None,
            "last_success_at": self.last_success_at.isoformat() if self.last_success_at else None,
            "last_failure_at": self.last_failure_at.isoformat() if self.last_failure_at else None,
            "consecutive_failures": self.consecutive_failures,
            "total_deliveries": self.total_deliveries,
            "total_failures": self.total_failures,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class WebhookDelivery(Base):
    """Log of webhook delivery attempts."""
    
    __tablename__ = "webhook_deliveries"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    webhook_id = Column(String, ForeignKey("webhooks.id"), nullable=False)
    
    # Event details
    event_type = Column(String(100), nullable=False)
    event_id = Column(String)  # ID of the triggering event
    payload = Column(JSON, nullable=False)
    
    # Delivery details
    attempt_number = Column(Integer, default=1)
    status_code = Column(Integer)
    response_body = Column(Text)
    response_headers = Column(JSON)
    response_time_ms = Column(Integer)
    
    # Status
    is_success = Column(Boolean, default=False)
    error_message = Column(Text)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    delivered_at = Column(DateTime)
    next_retry_at = Column(DateTime)
    
    # Relationship to parent webhook
    # webhook = relationship("Webhook")  # Commented to avoid circular dependency