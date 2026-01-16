"""Node system models."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uuid


class NodeType(str, Enum):
    """Types of nodes in workflows."""
    # Control flow
    START = "start"
    END = "end"
    DECISION = "decision"
    LOOP = "loop"
    PARALLEL = "parallel"
    JOIN = "join"
    
    # Processing
    TASK = "task"
    ADAPTER = "adapter"
    TRANSFORM = "transform"
    AGGREGATE = "aggregate"
    
    # Human interaction
    HUMAN_TASK = "human_task"
    APPROVAL = "approval"
    
    # Integration
    API_CALL = "api_call"
    DATABASE = "database"
    WEBHOOK = "webhook"
    
    # Messaging
    SEND_MESSAGE = "send_message"
    WAIT_MESSAGE = "wait_message"
    
    # Error handling
    ERROR_HANDLER = "error_handler"
    RETRY = "retry"
    COMPENSATE = "compensate"


class NodeStatus(str, Enum):
    """Node execution status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class NodeConfig(BaseModel):
    """Configuration for a node."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: NodeType
    description: Optional[str] = None
    
    # Node behavior
    parameters: Dict[str, Any] = {}
    inputs: List[str] = []  # Input variable names
    outputs: List[str] = []  # Output variable names
    
    # Execution settings
    timeout_seconds: int = 300
    retry_count: int = 0
    retry_delay_seconds: int = 60
    
    # Conditions
    condition: Optional[str] = None  # Expression to evaluate
    skip_condition: Optional[str] = None
    
    # Error handling
    error_handler_node_id: Optional[str] = None
    compensate_node_id: Optional[str] = None
    
    # Adapter integration
    adapter_id: Optional[str] = None
    adapter_capability: Optional[str] = None
    
    # Edition requirements
    required_edition: str = "community"


class NodeInstance(BaseModel):
    """Runtime instance of a node."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    node_config: NodeConfig
    workflow_instance_id: str
    
    # Status
    status: NodeStatus = NodeStatus.PENDING
    status_message: Optional[str] = None
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Retry tracking
    attempt_number: int = 0
    last_error: Optional[str] = None
    
    # Data
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    
    # Relationships
    previous_nodes: List[str] = []  # Node instance IDs
    next_nodes: List[str] = []  # Node instance IDs


class NodeConnection(BaseModel):
    """Connection between nodes in a workflow."""
    from_node_id: str
    to_node_id: str
    condition: Optional[str] = None  # Condition for following this path
    label: Optional[str] = None
    is_error_path: bool = False


class NodeExecutionRequest(BaseModel):
    """Request to execute a node."""
    node_instance_id: str
    input_data: Dict[str, Any] = {}
    context: Dict[str, Any] = {}
    force_execution: bool = False  # Skip condition checks


class NodeExecutionResult(BaseModel):
    """Result of node execution."""
    node_instance_id: str
    status: NodeStatus
    output_data: Dict[str, Any] = {}
    error: Optional[str] = None
    duration_ms: float
    
    # Next nodes to execute
    next_node_ids: List[str] = []
    
    # Side effects
    events_published: int = 0
    adapters_called: List[str] = []


class WorkflowTemplate(BaseModel):
    """Template for a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: Optional[str] = None
    version: str = "1.0.0"
    
    # Nodes
    nodes: List[NodeConfig] = []
    connections: List[NodeConnection] = []
    
    # Metadata
    tags: List[str] = []
    category: Optional[str] = None
    required_edition: str = "community"
    
    # Input/output schema
    input_schema: Dict[str, Any] = {}
    output_schema: Dict[str, Any] = {}
    
    # Settings
    timeout_seconds: int = 3600
    max_parallel_nodes: int = 10
    
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class WorkflowInstance(BaseModel):
    """Runtime instance of a workflow."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    template_id: str
    name: str
    
    # Status
    status: str = "pending"  # pending, running, completed, failed, cancelled
    status_message: Optional[str] = None
    
    # Execution tracking
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None
    
    # Node instances
    node_instances: Dict[str, NodeInstance] = {}  # node_id -> instance
    
    # Data
    input_data: Dict[str, Any] = {}
    output_data: Dict[str, Any] = {}
    variables: Dict[str, Any] = {}  # Workflow-level variables
    
    # Context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = {}