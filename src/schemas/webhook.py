"""Webhook schemas for request/response validation."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, HttpUrl, validator, ConfigDict


class WebhookCreate(BaseModel):
    """Schema for creating a new webhook."""
    name: str = Field(..., min_length=1, max_length=255, description="Name for the webhook")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: List[str] = Field(default_factory=list, description="Event types to subscribe to")
    secret: Optional[str] = Field(None, description="Secret for HMAC signature verification")
    custom_headers: Dict[str, str] = Field(default_factory=dict, description="Additional headers")
    max_retries: int = Field(3, ge=0, le=10, description="Maximum retry attempts")
    retry_delay_seconds: int = Field(60, ge=10, le=3600, description="Delay between retries")
    timeout_seconds: int = Field(30, ge=5, le=300, description="Request timeout")
    
    @validator('events')
    def validate_events(cls, v):
        """Validate event patterns."""
        valid_patterns = [
            # Task events
            "task.created", "task.updated", "task.completed", "task.failed",
            "task.*",  # Wildcard for all task events
            
            # Workflow events
            "workflow.started", "workflow.completed", "workflow.failed", "workflow.cancelled",
            "workflow.*",
            
            # Agent events
            "agent.created", "agent.updated", "agent.error", "agent.offline",
            "agent.*",
            
            # System events
            "system.error", "system.warning", "system.maintenance",
            "system.*",
            
            # All events
            "*"
        ]
        
        for event in v:
            # Check if it's a valid pattern or matches a wildcard pattern
            if event not in valid_patterns:
                # Check if it matches any wildcard pattern
                valid = False
                for pattern in valid_patterns:
                    if pattern.endswith('*'):
                        prefix = pattern[:-1]
                        if event.startswith(prefix):
                            valid = True
                            break
                if not valid:
                    raise ValueError(f"Invalid event pattern: {event}")
        return v
    
    @validator('custom_headers')
    def validate_headers(cls, v):
        """Validate custom headers."""
        # Prevent overriding certain headers
        reserved_headers = ['content-type', 'user-agent', 'x-webhook-signature']
        for header in v:
            if header.lower() in reserved_headers:
                raise ValueError(f"Cannot override reserved header: {header}")
        return v


class WebhookResponse(BaseModel):
    """Schema for webhook response."""
    id: str
    name: str
    description: Optional[str]
    url: str
    events: List[str]
    is_active: bool
    last_triggered_at: Optional[datetime]
    last_success_at: Optional[datetime]
    last_failure_at: Optional[datetime]
    consecutive_failures: int
    total_deliveries: int
    total_failures: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class WebhookCreateResponse(WebhookResponse):
    """Response when creating a webhook (includes secret)."""
    secret: Optional[str] = Field(None, description="Webhook secret for signature verification")


class WebhookUpdate(BaseModel):
    """Schema for updating a webhook."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    url: Optional[HttpUrl] = None
    events: Optional[List[str]] = None
    custom_headers: Optional[Dict[str, str]] = None
    max_retries: Optional[int] = Field(None, ge=0, le=10)
    retry_delay_seconds: Optional[int] = Field(None, ge=10, le=3600)
    timeout_seconds: Optional[int] = Field(None, ge=5, le=300)
    is_active: Optional[bool] = None
    
    @validator('events')
    def validate_events(cls, v):
        """Validate event patterns if provided."""
        if v is not None:
            # Reuse validation from WebhookCreate
            return WebhookCreate.validate_events(v)
        return v


class WebhookListResponse(BaseModel):
    """Response for listing webhooks."""
    webhooks: List[WebhookResponse]
    total: int


class WebhookTestRequest(BaseModel):
    """Request to test a webhook."""
    event_type: str = Field("test.ping", description="Event type for test")
    payload: Dict[str, Any] = Field(
        default_factory=lambda: {"message": "Test webhook delivery", "timestamp": datetime.utcnow().isoformat()},
        description="Test payload"
    )


class WebhookTestResponse(BaseModel):
    """Response for webhook test."""
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[int] = None
    error_message: Optional[str] = None
    response_body: Optional[str] = None


class WebhookDeliveryResponse(BaseModel):
    """Schema for webhook delivery log."""
    id: str
    webhook_id: str
    event_type: str
    event_id: Optional[str]
    attempt_number: int
    status_code: Optional[int]
    response_time_ms: Optional[int]
    is_success: bool
    error_message: Optional[str]
    created_at: datetime
    delivered_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class WebhookDeliveryListResponse(BaseModel):
    """Response for listing webhook deliveries."""
    deliveries: List[WebhookDeliveryResponse]
    total: int


class WebhookSignatureInfo(BaseModel):
    """Information about webhook signature verification."""
    algorithm: str = Field("hmac-sha256", description="Signature algorithm")
    header: str = Field("X-Webhook-Signature", description="Header containing signature")
    format: str = Field("hex", description="Signature encoding format")