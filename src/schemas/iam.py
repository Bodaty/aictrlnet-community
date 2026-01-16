"""Pydantic schemas for Internal Agent Messaging (IAM)."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime
from enum import Enum
import uuid


class IAMMessageType(str, Enum):
    """Types of messages in the IAM communication protocol."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    STATE_UPDATE = "state_update"
    CAPABILITY = "capability"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class IAMMessageStatus(str, Enum):
    """Status of IAM messages."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRYING = "retrying"


class IAMAgentType(str, Enum):
    """Types of internal agents."""
    WORKFLOW = "workflow"
    ADAPTER = "adapter"
    SERVICE = "service"
    MONITOR = "monitor"
    SCHEDULER = "scheduler"


class IAMAgentStatus(str, Enum):
    """Status of IAM agents."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


# Agent schemas
class IAMAgentBase(BaseModel):
    """Base schema for IAM agents."""
    name: str = Field(..., min_length=1, max_length=256)
    agent_type: IAMAgentType
    description: Optional[str] = None
    capabilities: Optional[List[str]] = []
    config: Optional[Dict[str, Any]] = {}


class IAMAgentCreate(IAMAgentBase):
    """Schema for creating an IAM agent."""
    pass


class IAMAgentUpdate(BaseModel):
    """Schema for updating an IAM agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    capabilities: Optional[List[str]] = None
    config: Optional[Dict[str, Any]] = None
    status: Optional[IAMAgentStatus] = None


class IAMAgentResponse(IAMAgentBase):
    """Schema for IAM agent responses."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: uuid.UUID
    status: IAMAgentStatus
    last_heartbeat: Optional[datetime] = None
    health_status: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: datetime


# Message schemas
class IAMMessageBase(BaseModel):
    """Base schema for IAM messages."""
    message_type: IAMMessageType
    content_type: str = "application/json"
    content: Dict[str, Any]
    priority: int = Field(default=0, ge=-100, le=100)
    ttl_seconds: Optional[int] = Field(None, gt=0)


class IAMMessageCreate(IAMMessageBase):
    """Schema for creating an IAM message."""
    recipient_id: Optional[uuid.UUID] = None  # None for broadcasts
    correlation_id: Optional[str] = None
    session_id: Optional[uuid.UUID] = None


class IAMMessageResponse(IAMMessageBase):
    """Schema for IAM message responses."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: uuid.UUID
    sender_id: uuid.UUID
    recipient_id: Optional[uuid.UUID] = None
    correlation_id: Optional[str] = None
    session_id: Optional[uuid.UUID] = None
    status: IAMMessageStatus
    delivered_at: Optional[datetime] = None
    error_message: Optional[str] = None
    retry_count: int
    created_at: datetime
    expires_at: Optional[datetime] = None


# Session schemas
class IAMSessionBase(BaseModel):
    """Base schema for IAM sessions."""
    session_type: str = Field(..., min_length=1, max_length=100)
    context: Dict[str, Any] = {}
    participants: List[str] = []


class IAMSessionCreate(IAMSessionBase):
    """Schema for creating an IAM session."""
    pass


class IAMSessionUpdate(BaseModel):
    """Schema for updating an IAM session."""
    context: Optional[Dict[str, Any]] = None
    participants: Optional[List[str]] = None
    is_active: Optional[bool] = None


class IAMSessionResponse(IAMSessionBase):
    """Schema for IAM session responses."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: uuid.UUID
    initiator_id: uuid.UUID
    is_active: bool
    created_at: datetime
    updated_at: datetime
    ended_at: Optional[datetime] = None


# Monitoring schemas
class IAMFlowData(BaseModel):
    """Schema for IAM flow visualization data."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metrics: Dict[str, Any] = {}


class IAMMessageFilter(BaseModel):
    """Schema for filtering IAM messages."""
    source_id: Optional[uuid.UUID] = None
    destination_id: Optional[uuid.UUID] = None
    message_type: Optional[IAMMessageType] = None
    status: Optional[IAMMessageStatus] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    correlation_id: Optional[str] = None
    session_id: Optional[uuid.UUID] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class IAMMessageHistory(BaseModel):
    """Schema for IAM message history response."""
    messages: List[IAMMessageResponse]
    total: int
    has_more: bool


class IAMSystemMetrics(BaseModel):
    """Schema for IAM system metrics."""
    total_messages: int
    messages_per_minute: float
    active_agents: int
    avg_message_size_bytes: float
    avg_response_time_ms: float
    error_rate: float
    uptime_seconds: int


class IAMAgentMetrics(BaseModel):
    """Schema for agent-specific metrics."""
    agent_id: uuid.UUID
    messages_sent: int
    messages_received: int
    avg_response_time_ms: float
    error_rate: float
    last_activity: datetime
    health_score: float = Field(ge=0, le=1)


class IAMCommunicationPattern(BaseModel):
    """Schema for communication pattern analysis."""
    pattern: str
    frequency: int
    percentage: float = Field(ge=0, le=1)


class IAMBusyRoute(BaseModel):
    """Schema for busy route information."""
    from_agent: str
    to_agent: str
    message_count: int


class IAMCommunicationPatterns(BaseModel):
    """Schema for communication patterns response."""
    most_common_patterns: List[IAMCommunicationPattern]
    busiest_routes: List[IAMBusyRoute]


class IAMErrorLog(BaseModel):
    """Schema for IAM error log entry."""
    id: uuid.UUID
    timestamp: datetime
    message_id: Optional[uuid.UUID] = None
    error_type: str
    error_message: str
    source_agent: Optional[str] = None
    destination_agent: Optional[str] = None
    severity: str = "error"