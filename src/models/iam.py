"""Internal Agent Messaging (IAM) models for Community Edition."""

from typing import Optional, Dict, Any, List
from sqlalchemy import String, Text, JSON, Boolean, Integer, ForeignKey, Enum, DateTime, Float, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from datetime import datetime
import enum
import uuid

from .base import Base


class IAMMessageType(str, enum.Enum):
    """Types of messages in the IAM communication protocol."""
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    BROADCAST = "broadcast"
    STATE_UPDATE = "state_update"
    CAPABILITY = "capability"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


class IAMMessageStatus(str, enum.Enum):
    """Status of IAM messages."""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    EXPIRED = "expired"
    RETRYING = "retrying"


class IAMAgentType(str, enum.Enum):
    """Types of internal agents."""
    WORKFLOW = "workflow"
    ADAPTER = "adapter"
    SERVICE = "service"
    MONITOR = "monitor"
    SCHEDULER = "scheduler"


class IAMAgentStatus(str, enum.Enum):
    """Status of IAM agents."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class IAMAgent(Base):
    """Internal agent in the IAM system."""
    
    __tablename__ = "iam_agents"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    agent_type: Mapped[IAMAgentType] = mapped_column(Enum(IAMAgentType, values_callable=lambda x: [e.value for e in x]), nullable=False)
    status: Mapped[IAMAgentStatus] = mapped_column(
        Enum(IAMAgentStatus, values_callable=lambda x: [e.value for e in x]), 
        default=IAMAgentStatus.ACTIVE,
        nullable=False
    )
    
    # Agent metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    capabilities: Mapped[Optional[List[str]]] = mapped_column(JSONB)
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Health and monitoring
    last_heartbeat: Mapped[Optional[datetime]] = mapped_column(DateTime)
    health_status: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    metrics: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sent_messages: Mapped[List["IAMMessage"]] = relationship(
        back_populates="sender",
        foreign_keys="IAMMessage.sender_id",
        lazy="selectin"
    )
    received_messages: Mapped[List["IAMMessage"]] = relationship(
        back_populates="recipient",
        foreign_keys="IAMMessage.recipient_id",
        lazy="selectin"
    )
    sessions: Mapped[List["IAMSession"]] = relationship(
        back_populates="initiator",
        lazy="selectin"
    )


class IAMMessage(Base):
    """Message in the IAM system."""
    
    __tablename__ = "iam_messages"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_type: Mapped[IAMMessageType] = mapped_column(Enum(IAMMessageType, values_callable=lambda x: [e.value for e in x]), nullable=False)
    
    # Sender and recipient
    sender_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_agents.id"), nullable=False)
    recipient_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_agents.id"))
    
    # Message content
    content_type: Mapped[str] = mapped_column(String(100), default="application/json")
    content: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    
    # Message metadata
    correlation_id: Mapped[Optional[str]] = mapped_column(String(256))
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_sessions.id"))
    priority: Mapped[int] = mapped_column(Integer, default=0)
    ttl_seconds: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Status tracking
    status: Mapped[IAMMessageStatus] = mapped_column(
        Enum(IAMMessageStatus, values_callable=lambda x: [e.value for e in x]),
        default=IAMMessageStatus.PENDING,
        nullable=False
    )
    delivered_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    sender: Mapped["IAMAgent"] = relationship(
        back_populates="sent_messages",
        foreign_keys=[sender_id],
        lazy="selectin"
    )
    recipient: Mapped[Optional["IAMAgent"]] = relationship(
        back_populates="received_messages",
        foreign_keys=[recipient_id],
        lazy="selectin"
    )
    session: Mapped[Optional["IAMSession"]] = relationship(
        back_populates="messages",
        lazy="selectin"
    )


class IAMSession(Base):
    """Session for stateful IAM conversations."""
    
    __tablename__ = "iam_sessions"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    initiator_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_agents.id"), nullable=False)
    
    # Session metadata
    session_type: Mapped[str] = mapped_column(String(100), nullable=False)
    context: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={})
    participants: Mapped[List[str]] = mapped_column(JSONB, default=[])
    
    # Status
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    ended_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    initiator: Mapped["IAMAgent"] = relationship(
        back_populates="sessions",
        lazy="selectin"
    )
    messages: Mapped[List["IAMMessage"]] = relationship(
        back_populates="session",
        lazy="selectin"
    )


class IAMEventLog(Base):
    """Event log for IAM system monitoring."""
    
    __tablename__ = "iam_event_logs"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    event_type: Mapped[str] = mapped_column(String(100), nullable=False)
    agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_agents.id"))
    message_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_messages.id"))
    session_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_sessions.id"))
    
    # Event data
    event_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    severity: Mapped[str] = mapped_column(String(20), default="info")
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Indexes
    __table_args__ = (
        # Index for time-based queries
        # Will be created by migration
    )


class IAMMetric(Base):
    """Performance metrics for IAM agents."""
    
    __tablename__ = "iam_metrics"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), ForeignKey("iam_agents.id"), nullable=False)
    
    # Metric data
    metric_type: Mapped[str] = mapped_column(String(100), nullable=False)
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_unit: Mapped[Optional[str]] = mapped_column(String(50))
    
    # Time window
    period_start: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    period_end: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    
    # Additional data
    metric_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metric_metadata", JSONB)
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint("agent_id", "metric_type", "period_start", name="uq_iam_metrics_period"),
    )