"""Workflow execution tracking models for Community Edition."""

from typing import Optional, Dict, Any, List
from sqlalchemy import String, Text, JSON, Boolean, Integer, ForeignKey, Enum, DateTime, Float, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from datetime import datetime
import enum
import uuid

from .base import Base


class WorkflowExecutionStatus(str, enum.Enum):
    """Status of workflow execution."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    RESUMING = "resuming"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class NodeExecutionStatus(str, enum.Enum):
    """Status of individual node execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"
    RETRY_REQUIRED = "retry_required"
    REQUIRES_APPROVAL = "requires_approval"


class WorkflowExecution(Base):
    """Tracks workflow execution instances."""
    
    __tablename__ = "workflow_executions"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflow_definitions.id"), nullable=False)
    instance_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("workflow_instances.id"))
    tenant_id: Mapped[Optional[str]] = mapped_column(String(255))
    user_id: Mapped[Optional[str]] = mapped_column(String(255))
    
    # Execution details
    status: Mapped[WorkflowExecutionStatus] = mapped_column(
        Enum(WorkflowExecutionStatus, name='workflow_execution_status', values_callable=lambda x: [e.value for e in x]),
        default=WorkflowExecutionStatus.PENDING,
        nullable=False
    )
    context: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={})
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    output_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Execution metadata
    execution_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column("metadata", JSONB)
    triggered_by: Mapped[Optional[str]] = mapped_column(String(100))  # manual, schedule, event, api
    trigger_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workflow: Mapped["WorkflowDefinition"] = relationship(back_populates="executions", lazy="selectin")
    instance: Mapped[Optional["WorkflowInstance"]] = relationship(lazy="selectin")
    node_executions: Mapped[List["NodeExecution"]] = relationship(
        back_populates="workflow_execution",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    checkpoints: Mapped[List["WorkflowCheckpoint"]] = relationship(
        back_populates="workflow_execution",
        cascade="all, delete-orphan",
        lazy="selectin"
    )


class NodeExecution(Base):
    """Tracks individual node execution within workflows."""
    
    __tablename__ = "node_executions"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), 
        ForeignKey("workflow_executions.id", ondelete="CASCADE"),
        nullable=False
    )
    node_id: Mapped[str] = mapped_column(String(255), nullable=False)
    node_type: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Execution details
    status: Mapped[NodeExecutionStatus] = mapped_column(
        Enum(NodeExecutionStatus, name='node_execution_status', values_callable=lambda x: [e.value for e in x]),
        default=NodeExecutionStatus.PENDING,
        nullable=False
    )
    inputs: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    outputs: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    error_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Execution context
    execution_context: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    retry_count: Mapped[int] = mapped_column(Integer, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, default=3)
    
    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)
    
    # Agent assignment (for distributed execution)
    assigned_agent_id: Mapped[Optional[uuid.UUID]] = mapped_column(PG_UUID(as_uuid=True))
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workflow_execution: Mapped["WorkflowExecution"] = relationship(back_populates="node_executions", lazy="selectin")


class WorkflowCheckpoint(Base):
    """Stores workflow execution checkpoints for pause/resume functionality."""
    
    __tablename__ = "workflow_checkpoints"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    execution_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("workflow_executions.id", ondelete="CASCADE"),
        nullable=False
    )
    
    # Checkpoint data
    checkpoint_type: Mapped[str] = mapped_column(String(50), nullable=False)  # manual, auto, error
    state_data: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    completed_nodes: Mapped[List[str]] = mapped_column(JSONB, default=[])
    pending_nodes: Mapped[List[str]] = mapped_column(JSONB, default=[])
    workflow_variables: Mapped[Dict[str, Any]] = mapped_column(JSONB, default={})
    
    # Metadata
    description: Mapped[Optional[str]] = mapped_column(Text)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    
    # Relationships
    workflow_execution: Mapped["WorkflowExecution"] = relationship(back_populates="checkpoints", lazy="selectin")


class WorkflowTrigger(Base):
    """Stores workflow triggers configuration."""
    
    __tablename__ = "workflow_triggers"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflow_definitions.id"), nullable=False)
    
    # Trigger configuration
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    trigger_type: Mapped[str] = mapped_column(String(50), nullable=False)  # webhook, event, schedule, condition
    config: Mapped[Dict[str, Any]] = mapped_column(JSONB, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Execution tracking
    last_triggered: Mapped[Optional[datetime]] = mapped_column(DateTime)
    trigger_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workflow: Mapped["WorkflowDefinition"] = relationship(lazy="selectin")


class WorkflowSchedule(Base):
    """Stores workflow scheduling configuration."""
    
    __tablename__ = "workflow_schedules"
    
    id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    workflow_id: Mapped[str] = mapped_column(String(36), ForeignKey("workflow_definitions.id"), nullable=False)
    
    # Schedule configuration
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    schedule_expression: Mapped[str] = mapped_column(String(255), nullable=False)  # Cron expression
    timezone: Mapped[str] = mapped_column(String(50), default="UTC")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Execution parameters
    input_parameters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    execution_config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONB)
    
    # Tracking
    next_run: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_run: Mapped[Optional[datetime]] = mapped_column(DateTime)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    workflow: Mapped["WorkflowDefinition"] = relationship(lazy="selectin")


# Update WorkflowDefinition to include executions relationship
# This would be added to the existing WorkflowDefinition model
# executions: Mapped[List["WorkflowExecution"]] = relationship(
#     back_populates="workflow",
#     cascade="all, delete-orphan",
#     lazy="selectin"
# )