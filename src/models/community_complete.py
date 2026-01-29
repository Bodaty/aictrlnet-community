"""Complete Community Edition models including MCP and IAM."""

from typing import Optional, Dict, Any, List, TYPE_CHECKING
from sqlalchemy import String, Text, JSON, Boolean, Integer, ForeignKey, Enum, DateTime, Date, Float, UniqueConstraint, ARRAY, text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from datetime import datetime
import enum
import uuid

from .base import Base, TimestampMixin, UUIDMixin, TenantMixin

if TYPE_CHECKING:
    from .workflow_execution import WorkflowExecution


class TaskStatus(str, enum.Enum):
    """Task status enumeration."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Task(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Task model."""
    
    __tablename__ = "tasks"
    
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    status: Mapped[TaskStatus] = mapped_column(
        Enum(TaskStatus),
        default=TaskStatus.PENDING,
        nullable=False,
    )
    task_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, name="metadata")
    
    # Relationships
    workflow_steps: Mapped[List["WorkflowStep"]] = relationship(
        back_populates="task",
        lazy="selectin",
    )


class WorkflowDefinition(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Workflow definition model."""
    
    __tablename__ = "workflow_definitions"
    
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    definition: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    workflow_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, name="metadata")
    
    # Relationships
    instances: Mapped[List["WorkflowInstance"]] = relationship(
        back_populates="definition",
        lazy="selectin",
    )
    executions: Mapped[List["WorkflowExecution"]] = relationship(
        back_populates="workflow",
        lazy="selectin",
    )


class WorkflowStatus(str, enum.Enum):
    """Workflow status enumeration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class WorkflowInstance(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Workflow instance model."""
    
    __tablename__ = "workflow_instances"
    
    definition_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_definitions.id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    status: Mapped[WorkflowStatus] = mapped_column(
        Enum(WorkflowStatus),
        default=WorkflowStatus.PENDING,
        nullable=False,
    )
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    output_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    instance_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, name="metadata")
    
    # Relationships
    definition: Mapped["WorkflowDefinition"] = relationship(
        back_populates="instances",
        lazy="selectin",
    )
    steps: Mapped[List["WorkflowStep"]] = relationship(
        back_populates="instance",
        lazy="selectin",
    )


class StepStatus(str, enum.Enum):
    """Workflow step status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class WorkflowStep(Base, UUIDMixin, TimestampMixin, TenantMixin):
    """Workflow step model."""
    
    __tablename__ = "workflow_steps"
    
    workflow_instance_id: Mapped[str] = mapped_column(
        ForeignKey("workflow_instances.id"),
        nullable=False,
    )
    task_id: Mapped[Optional[str]] = mapped_column(ForeignKey("tasks.id"))
    step_name: Mapped[str] = mapped_column(String(256), nullable=False)
    step_type: Mapped[str] = mapped_column(String(50), nullable=False)
    node_type: Mapped[Optional[str]] = mapped_column(String(50))
    status: Mapped[StepStatus] = mapped_column(
        Enum(StepStatus),
        default=StepStatus.PENDING,
        nullable=False,
    )
    input_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    output_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    
    # Relationships
    instance: Mapped["WorkflowInstance"] = relationship(
        back_populates="steps",
        lazy="selectin",
    )
    task: Mapped[Optional["Task"]] = relationship(
        back_populates="workflow_steps",
        lazy="selectin",
    )


# MCP Models

class MCPServer(Base, UUIDMixin):
    """MCP Server model conforming to MCP (Model Context Protocol) standard.

    MCP servers can be configured in two ways:
    1. stdio transport: Uses command + args + env_vars to spawn subprocess
    2. HTTP/SSE transport: Uses url + api_key for remote servers

    See: https://modelcontextprotocol.io/specification/2025-03-26
    """

    __tablename__ = "mcp_servers"

    name: Mapped[str] = mapped_column(String(128), nullable=False)

    # MCP Standard: Transport type - "stdio" (subprocess) or "http_sse" (HTTP with SSE)
    transport_type: Mapped[str] = mapped_column(String(20), nullable=False, server_default=text("'stdio'"))

    # MCP Standard: stdio transport configuration
    # command: The executable to run (e.g., "npx", "python", "node")
    command: Mapped[Optional[str]] = mapped_column(String(256))
    # args: JSON array of command arguments (e.g., ["-m", "mcp_server"])
    args: Mapped[Optional[str]] = mapped_column(Text)  # JSON array stored as text
    # env_vars: JSON object of environment variables for the subprocess
    env_vars: Mapped[Optional[str]] = mapped_column(Text)  # JSON object stored as text

    # HTTP/SSE transport configuration (for remote servers)
    url: Mapped[Optional[str]] = mapped_column(String(256), unique=True)  # Now optional, only for HTTP transport
    api_key: Mapped[Optional[str]] = mapped_column(String(256))  # Encrypted in production

    # Server metadata
    service_type: Mapped[str] = mapped_column(String(50), nullable=False, default="general")
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")  # pending, active, failed, connected
    last_checked: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: datetime.utcnow().timestamp())
    server_info: Mapped[Optional[str]] = mapped_column(Text)  # JSON string of server info from initialization

    # MCP Protocol info (populated after connection)
    protocol_version: Mapped[Optional[str]] = mapped_column(String(20))  # e.g., "2025-03-26"
    server_capabilities: Mapped[Optional[str]] = mapped_column(Text)  # JSON of MCP capabilities

    # OAuth2 provider for MCP server authentication (SEP-991)
    # No FK constraint - OAuth2Provider is in Business edition (accretive model)
    # This allows Community to store the reference, Business/Enterprise to use it
    oauth2_provider_id: Mapped[Optional[str]] = mapped_column(String(36))

    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: datetime.utcnow().timestamp())
    updated_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: datetime.utcnow().timestamp(), onupdate=lambda: datetime.utcnow().timestamp())
    
    # Relationships
    capabilities: Mapped[List["MCPServerCapability"]] = relationship(
        back_populates="server",
        lazy="selectin",
        cascade="all, delete-orphan"
    )
    tools: Mapped[List["MCPTool"]] = relationship(
        back_populates="server",
        lazy="selectin",
    )
    invocations: Mapped[List["MCPInvocation"]] = relationship(
        back_populates="server",
        lazy="selectin",
    )


class MCPServerCapability(Base, UUIDMixin):
    """MCP Server Capability model."""
    
    __tablename__ = "mcp_server_capabilities"
    
    server_id: Mapped[str] = mapped_column(
        ForeignKey("mcp_servers.id"),
        nullable=False,
    )
    capability: Mapped[str] = mapped_column(String(50), nullable=False)  # e.g., "messages", "embeddings"
    supported: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    details: Mapped[Optional[str]] = mapped_column(Text)  # JSON string with capability details
    created_at: Mapped[float] = mapped_column(Float, nullable=False, default=lambda: datetime.utcnow().timestamp())
    
    # Relationships
    server: Mapped["MCPServer"] = relationship(
        back_populates="capabilities",
        lazy="selectin",
    )


class TaskMCP(Base):
    """Extended Task model with MCP support."""
    
    __tablename__ = "tasks_mcp"
    
    # Use task_id as string to match Flask model
    task_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    source_id: Mapped[str] = mapped_column(String(36), nullable=False)
    destination: Mapped[str] = mapped_column(String(128), nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)  # JSON stored as text
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    quality_score: Mapped[Optional[float]] = mapped_column(Float)
    auto_escalated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # MCP-specific fields
    mcp_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    mcp_metadata: Mapped[Optional[str]] = mapped_column(Text)  # JSON stored as text
    mcp_context_id: Mapped[Optional[str]] = mapped_column(String(36))  # ID of the associated context
    input_tokens: Mapped[Optional[int]] = mapped_column(Integer)  # Token count for input
    output_tokens: Mapped[Optional[int]] = mapped_column(Integer)  # Token count for output
    
    # Relationships
    contexts: Mapped[List["MCPContextStorage"]] = relationship(
        back_populates="task",
        lazy="selectin",
    )


class MCPContextStorage(Base, UUIDMixin):
    """Storage for MCP context objects."""
    
    __tablename__ = "mcp_context_storage"
    
    task_id: Mapped[Optional[str]] = mapped_column(
        String(36),
        ForeignKey("tasks_mcp.task_id"),
    )
    context_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'message', 'history', 'response'
    content: Mapped[str] = mapped_column(Text, nullable=False)  # JSON stored as text
    role: Mapped[Optional[str]] = mapped_column(String(20))  # For message contexts
    context_metadata: Mapped[Optional[str]] = mapped_column(Text, name="context_metadata")  # JSON stored as text
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)
    
    # Relationships
    task: Mapped[Optional["TaskMCP"]] = relationship(
        back_populates="contexts",
        lazy="selectin",
    )


class MCPTool(Base, UUIDMixin, TimestampMixin):
    """MCP Tool model."""
    
    __tablename__ = "mcp_tools"
    
    server_id: Mapped[str] = mapped_column(
        ForeignKey("mcp_servers.id"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    tool_schema: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Relationships
    server: Mapped["MCPServer"] = relationship(
        back_populates="tools",
        lazy="selectin",
    )
    invocations: Mapped[List["MCPInvocation"]] = relationship(
        back_populates="tool",
        lazy="selectin",
    )


class MCPInvocation(Base, UUIDMixin, TimestampMixin):
    """MCP Invocation model."""

    __tablename__ = "mcp_invocations"

    tool_id: Mapped[str] = mapped_column(
        ForeignKey("mcp_tools.id"),
        nullable=False,
    )
    server_id: Mapped[str] = mapped_column(
        ForeignKey("mcp_servers.id"),
        nullable=False,
    )
    request_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    response_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    duration_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Relationships
    tool: Mapped["MCPTool"] = relationship(
        back_populates="invocations",
        lazy="selectin",
    )
    server: Mapped["MCPServer"] = relationship(
        back_populates="invocations",
        lazy="selectin",
    )


class MCPTaskState(str, enum.Enum):
    """MCP Task state per SEP-1686 specification.

    See: https://spec.modelcontextprotocol.io/specification/2025-11-25/server/tasks/
    """
    WORKING = "working"     # Task is actively processing
    COMPLETED = "completed"  # Task finished successfully
    FAILED = "failed"       # Task encountered an error
    CANCELLED = "cancelled"  # Task was cancelled by client


class MCPAsyncTask(Base, UUIDMixin, TimestampMixin):
    """MCP Async Task model per SEP-1686 (Tasks) specification.

    Represents a long-running operation tracked by MCP servers.
    Clients can poll for status updates using tasks/get.

    See: https://spec.modelcontextprotocol.io/specification/2025-11-25/server/tasks/
    """

    __tablename__ = "mcp_async_tasks"

    # Server that created this task
    server_id: Mapped[str] = mapped_column(
        ForeignKey("mcp_servers.id"),
        nullable=False,
    )

    # Tool that initiated this task (optional - could be a resource operation)
    tool_id: Mapped[Optional[str]] = mapped_column(
        ForeignKey("mcp_tools.id"),
    )

    # Task identification
    task_token: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    method: Mapped[str] = mapped_column(String(100), nullable=False)  # e.g., "tools/call", "resources/read"

    # Task state
    state: Mapped[MCPTaskState] = mapped_column(
        Enum(MCPTaskState),
        default=MCPTaskState.WORKING,
        nullable=False,
    )

    # Progress tracking (0-100)
    progress: Mapped[Optional[int]] = mapped_column(Integer)
    progress_message: Mapped[Optional[str]] = mapped_column(String(512))

    # Request/Response data
    request_params: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # Original request parameters
    result_content: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # Result when completed
    structured_result: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)  # For structured output

    # Error information
    error_code: Mapped[Optional[int]] = mapped_column(Integer)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    error_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
    last_activity_at: Mapped[Optional[datetime]] = mapped_column(DateTime, default=datetime.utcnow)

    # Timeout handling
    timeout_seconds: Mapped[Optional[int]] = mapped_column(Integer, default=300)

    # Client info
    client_id: Mapped[Optional[str]] = mapped_column(String(256))

    # Task metadata
    task_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, name="task_metadata")

    # Relationships
    server: Mapped["MCPServer"] = relationship(
        lazy="selectin",
    )
    tool: Mapped[Optional["MCPTool"]] = relationship(
        lazy="selectin",
    )


# Other Community models

class Adapter(Base, UUIDMixin, TimestampMixin):
    """Adapter model for third-party integrations."""
    
    __tablename__ = "adapters"
    
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    category: Mapped[str] = mapped_column(String(128), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text)
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    min_edition: Mapped[str] = mapped_column(String(20), default="community", nullable=False)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    adapter_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, name="metadata")
    config_schema: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    capabilities: Mapped[Optional[List[str]]] = mapped_column(JSON)
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON)
    available: Mapped[bool] = mapped_column(Boolean, default=True)
    installed: Mapped[bool] = mapped_column(Boolean, default=False)
    install_count: Mapped[int] = mapped_column(Integer, default=0)


class BridgeConnection(Base, UUIDMixin, TimestampMixin):
    """Bridge connection model for system integration."""
    
    __tablename__ = "bridge_connections"
    
    name: Mapped[str] = mapped_column(String(256), nullable=False)
    source_type: Mapped[str] = mapped_column(String(50), nullable=False)
    target_type: Mapped[str] = mapped_column(String(50), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="active")
    config: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    bridge_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, name="metadata")
    
    # Relationships
    syncs: Mapped[List["BridgeSync"]] = relationship(
        back_populates="connection",
        lazy="selectin",
    )


class BridgeSync(Base, UUIDMixin, TimestampMixin):
    """Bridge sync operation model."""
    
    __tablename__ = "bridge_syncs"
    
    connection_id: Mapped[str] = mapped_column(
        ForeignKey("bridge_connections.id"),
        nullable=False,
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    started_at: Mapped[Optional[datetime]] = mapped_column()
    completed_at: Mapped[Optional[datetime]] = mapped_column()
    items_processed: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    items_created: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    items_updated: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    items_failed: Mapped[Optional[int]] = mapped_column(Integer, default=0)
    error_message: Mapped[Optional[str]] = mapped_column(Text)
    sync_options: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON)
    
    # Relationships
    connection: Mapped["BridgeConnection"] = relationship(
        back_populates="syncs",
        lazy="selectin",
    )


class ResourcePoolConfig(Base, UUIDMixin, TimestampMixin):
    """Resource pool configuration model for basic resource pooling settings."""
    
    __tablename__ = "resource_pool_configs"
    
    name: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    resource_type: Mapped[str] = mapped_column(String(50), nullable=False)
    min_size: Mapped[int] = mapped_column(Integer, default=1)
    max_size: Mapped[int] = mapped_column(Integer, default=10)
    acquire_timeout: Mapped[float] = mapped_column(Float, default=30.0)
    idle_timeout: Mapped[float] = mapped_column(Float, default=300.0)
    max_lifetime: Mapped[float] = mapped_column(Float, default=3600.0)
    health_check_interval: Mapped[float] = mapped_column(Float, default=60.0)
    scale_up_threshold: Mapped[float] = mapped_column(Float, default=0.8)
    scale_down_threshold: Mapped[float] = mapped_column(Float, default=0.2)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    config_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, name="metadata")

# Import IAM models
from .iam import (
    IAMAgent, IAMMessage, IAMSession, IAMEventLog, IAMMetric,
    IAMMessageType, IAMMessageStatus, IAMAgentType, IAMAgentStatus
)
