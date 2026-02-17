"""Schemas for workflow execution tracking."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict, model_validator
from datetime import datetime
import uuid

from models.workflow_execution import WorkflowExecutionStatus, NodeExecutionStatus


# Workflow Execution schemas
class WorkflowExecuteRequest(BaseModel):
    """Request body for executing a workflow via the REST endpoint."""
    input_data: Optional[Dict[str, Any]] = None
    dry_run: bool = False
    trigger_metadata: Optional[Dict[str, Any]] = None
    trigger_source: Optional[str] = None
    agent_id: Optional[str] = None


class WorkflowExecutionCreate(BaseModel):
    """Schema for creating a workflow execution."""
    workflow_id: uuid.UUID
    input_data: Optional[Dict[str, Any]] = Field(default_factory=dict)
    triggered_by: str = Field(default="manual", description="Trigger source: manual, schedule, event, api")
    trigger_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WorkflowExecutionUpdate(BaseModel):
    """Schema for updating a workflow execution."""
    status: Optional[WorkflowExecutionStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None


class WorkflowExecutionResponse(BaseModel):
    """Response schema for workflow execution."""
    id: uuid.UUID
    workflow_id: uuid.UUID
    instance_id: Optional[uuid.UUID]
    tenant_id: Optional[str]
    status: WorkflowExecutionStatus
    context: Dict[str, Any]
    input_data: Optional[Dict[str, Any]]
    output_data: Optional[Dict[str, Any]]
    error_details: Optional[Dict[str, Any]]
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    execution_metadata: Optional[Dict[str, Any]]
    triggered_by: Optional[str]
    trigger_metadata: Optional[Dict[str, Any]]
    dry_run: bool = Field(default=False, description="Whether this execution ran in dry-run mode")
    nodes_intercepted: Optional[int] = Field(default=None, description="Number of nodes intercepted during dry-run")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

    @model_validator(mode="after")
    def extract_dry_run_from_context(self):
        """Extract dry_run and nodes_intercepted from execution context/metadata."""
        if not self.dry_run:
            ctx = self.context or {}
            tm = self.trigger_metadata or {}
            if ctx.get("is_dry_run", False) or tm.get("is_dry_run", False):
                self.dry_run = True
        if self.nodes_intercepted is None:
            em = self.execution_metadata or {}
            if "nodes_intercepted" in em:
                self.nodes_intercepted = em["nodes_intercepted"]
        return self


# Node Execution schemas
class NodeExecutionUpdate(BaseModel):
    """Schema for updating node execution."""
    status: NodeExecutionStatus
    outputs: Optional[Dict[str, Any]] = None
    error_details: Optional[Dict[str, Any]] = None


class NodeExecutionResponse(BaseModel):
    """Response schema for node execution."""
    id: uuid.UUID
    execution_id: uuid.UUID
    node_id: str
    node_type: str
    status: NodeExecutionStatus
    inputs: Optional[Dict[str, Any]]
    outputs: Optional[Dict[str, Any]]
    error_details: Optional[Dict[str, Any]]
    execution_context: Optional[Dict[str, Any]]
    retry_count: int
    max_retries: int
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_ms: Optional[int]
    assigned_agent_id: Optional[uuid.UUID]
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Checkpoint schemas
class WorkflowCheckpointCreate(BaseModel):
    """Schema for creating a workflow checkpoint."""
    checkpoint_type: str = Field(..., description="manual, auto, error")
    state_data: Dict[str, Any]
    completed_nodes: List[str] = Field(default_factory=list)
    pending_nodes: List[str] = Field(default_factory=list)
    workflow_variables: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None
    expires_at: Optional[datetime] = None


class WorkflowCheckpointResponse(BaseModel):
    """Response schema for workflow checkpoint."""
    id: uuid.UUID
    execution_id: uuid.UUID
    checkpoint_type: str
    state_data: Dict[str, Any]
    completed_nodes: List[str]
    pending_nodes: List[str]
    workflow_variables: Dict[str, Any]
    description: Optional[str]
    is_active: bool
    created_at: datetime
    expires_at: Optional[datetime]
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Trigger schemas
class WorkflowTriggerCreate(BaseModel):
    """Schema for creating a workflow trigger."""
    trigger_type: str = Field(..., description="webhook, event, schedule, condition")
    config: Dict[str, Any]
    is_active: bool = True


class WorkflowTriggerUpdate(BaseModel):
    """Schema for updating a workflow trigger."""
    config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class WorkflowTriggerResponse(BaseModel):
    """Response schema for workflow trigger."""
    id: uuid.UUID
    workflow_id: uuid.UUID
    trigger_type: str
    config: Dict[str, Any]
    is_active: bool
    last_triggered: Optional[datetime]
    trigger_count: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Schedule schemas
class WorkflowScheduleCreate(BaseModel):
    """Schema for creating a workflow schedule."""
    name: str = Field(default="", description="Schedule name")
    schedule_expression: str = Field(..., description="Cron expression (e.g. '0 9 * * 5' for Friday 9am)")
    timezone: str = Field(default="UTC")
    is_active: bool = True
    input_parameters: Optional[Dict[str, Any]] = None
    execution_config: Optional[Dict[str, Any]] = None


class WorkflowScheduleUpdate(BaseModel):
    """Schema for updating a workflow schedule."""
    schedule_expression: Optional[str] = None
    timezone: Optional[str] = None
    is_active: Optional[bool] = None
    input_parameters: Optional[Dict[str, Any]] = None
    execution_config: Optional[Dict[str, Any]] = None


class WorkflowScheduleResponse(BaseModel):
    """Response schema for workflow schedule."""
    id: uuid.UUID
    workflow_id: uuid.UUID
    schedule_expression: str
    timezone: str
    is_active: bool
    input_parameters: Optional[Dict[str, Any]]
    execution_config: Optional[Dict[str, Any]]
    next_run: Optional[datetime]
    last_run: Optional[datetime]
    run_count: int
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Execution summary schemas
class ExecutionSummary(BaseModel):
    """Summary of workflow executions."""
    total_executions: int
    by_status: Dict[str, int]
    average_duration_ms: Optional[float]
    success_rate: float
    recent_executions: List[WorkflowExecutionResponse]


class NodeExecutionSummary(BaseModel):
    """Summary of node executions for a workflow execution."""
    total_nodes: int
    completed_nodes: int
    failed_nodes: int
    pending_nodes: int
    running_nodes: int
    average_duration_ms: Optional[float]
    slowest_nodes: List[Dict[str, Any]]