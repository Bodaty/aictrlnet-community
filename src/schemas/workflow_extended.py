"""Extended workflow schemas for missing models."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


# WorkflowDefinition Schemas
class WorkflowDefinitionBase(BaseModel):
    """Base schema for workflow definition."""
    name: str = Field(..., description="Workflow name")
    description: Optional[str] = Field(None, description="Workflow description")
    definition: Dict[str, Any] = Field(..., description="Workflow definition")
    version: str = Field("1.0.0", description="Workflow version")
    category: Optional[str] = Field(None, description="Workflow category")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")
    is_public: bool = Field(False, description="Is workflow public")
    created_by: str = Field(..., description="Creator user ID")


class WorkflowDefinitionCreate(WorkflowDefinitionBase):
    """Schema for creating workflow definition."""
    pass


class WorkflowDefinitionUpdate(BaseModel):
    """Schema for updating workflow definition."""
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[Dict[str, Any]] = None
    version: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None


class WorkflowDefinitionResponse(WorkflowDefinitionBase):
    """Schema for workflow definition response."""
    id: str = Field(..., description="Workflow definition ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# WorkflowTemplatePermission Schemas
class WorkflowTemplatePermissionBase(BaseModel):
    """Base schema for workflow template permission."""
    template_id: str = Field(..., description="Template ID")
    user_id: Optional[str] = Field(None, description="User ID")
    team_id: Optional[str] = Field(None, description="Team ID")
    organization_id: Optional[str] = Field(None, description="Organization ID")
    permission_type: str = Field(..., description="Permission type (read, write, execute, admin)")
    granted_by: str = Field(..., description="Granter user ID")


class WorkflowTemplatePermissionCreate(WorkflowTemplatePermissionBase):
    """Schema for creating workflow template permission."""
    pass


class WorkflowTemplatePermissionUpdate(BaseModel):
    """Schema for updating workflow template permission."""
    permission_type: Optional[str] = None


class WorkflowTemplatePermissionResponse(WorkflowTemplatePermissionBase):
    """Schema for workflow template permission response."""
    id: str = Field(..., description="Permission ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# WorkflowTemplateUsage Schemas
class WorkflowTemplateUsageBase(BaseModel):
    """Base schema for workflow template usage."""
    template_id: str = Field(..., description="Template ID")
    user_id: str = Field(..., description="User ID")
    workflow_id: str = Field(..., description="Created workflow ID")
    parameters_used: Dict[str, Any] = Field(default_factory=dict, description="Parameters used")
    execution_status: Optional[str] = Field(None, description="Execution status")
    execution_duration_ms: Optional[float] = Field(None, description="Execution duration")


class WorkflowTemplateUsageCreate(WorkflowTemplateUsageBase):
    """Schema for creating workflow template usage."""
    pass


class WorkflowTemplateUsageUpdate(BaseModel):
    """Schema for updating workflow template usage."""
    execution_status: Optional[str] = None
    execution_duration_ms: Optional[float] = None


class WorkflowTemplateUsageResponse(WorkflowTemplateUsageBase):
    """Schema for workflow template usage response."""
    id: str = Field(..., description="Usage ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# WorkflowTemplateReview Schemas
class WorkflowTemplateReviewBase(BaseModel):
    """Base schema for workflow template review."""
    template_id: str = Field(..., description="Template ID")
    reviewer_id: str = Field(..., description="Reviewer user ID")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    comment: Optional[str] = Field(None, description="Review comment")
    is_verified: bool = Field(False, description="Is verified review")


class WorkflowTemplateReviewCreate(WorkflowTemplateReviewBase):
    """Schema for creating workflow template review."""
    pass


class WorkflowTemplateReviewUpdate(BaseModel):
    """Schema for updating workflow template review."""
    rating: Optional[int] = Field(None, ge=1, le=5)
    comment: Optional[str] = None
    is_verified: Optional[bool] = None


class WorkflowTemplateReviewResponse(WorkflowTemplateReviewBase):
    """Schema for workflow template review response."""
    id: str = Field(..., description="Review ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# BridgeConnection Schemas
class BridgeConnectionBase(BaseModel):
    """Base schema for bridge connection."""
    source_system: str = Field(..., description="Source system name")
    target_system: str = Field(..., description="Target system name")
    connection_type: str = Field(..., description="Connection type")
    connection_config: Dict[str, Any] = Field(..., description="Connection configuration")
    is_active: bool = Field(True, description="Is connection active")
    last_sync_at: Optional[datetime] = Field(None, description="Last sync timestamp")
    sync_status: Optional[str] = Field(None, description="Sync status")


class BridgeConnectionCreate(BridgeConnectionBase):
    """Schema for creating bridge connection."""
    pass


class BridgeConnectionUpdate(BaseModel):
    """Schema for updating bridge connection."""
    connection_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    last_sync_at: Optional[datetime] = None
    sync_status: Optional[str] = None


class BridgeConnectionResponse(BridgeConnectionBase):
    """Schema for bridge connection response."""
    id: str = Field(..., description="Connection ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# BridgeSync Schemas
class BridgeSyncBase(BaseModel):
    """Base schema for bridge sync."""
    connection_id: str = Field(..., description="Bridge connection ID")
    sync_type: str = Field(..., description="Sync type (full, incremental, delta)")
    sync_direction: str = Field(..., description="Sync direction (push, pull, bidirectional)")
    records_synced: int = Field(0, description="Number of records synced")
    errors_count: int = Field(0, description="Number of errors")
    sync_status: str = Field("pending", description="Sync status")
    error_details: Optional[List[Dict[str, Any]]] = Field(None, description="Error details")


class BridgeSyncCreate(BridgeSyncBase):
    """Schema for creating bridge sync."""
    pass


class BridgeSyncUpdate(BaseModel):
    """Schema for updating bridge sync."""
    records_synced: Optional[int] = None
    errors_count: Optional[int] = None
    sync_status: Optional[str] = None
    error_details: Optional[List[Dict[str, Any]]] = None


class BridgeSyncResponse(BridgeSyncBase):
    """Schema for bridge sync response."""
    id: str = Field(..., description="Sync ID")
    started_at: datetime = Field(..., description="Start timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# QualityDimensionModel Schemas  
class QualityDimensionModelBase(BaseModel):
    """Base schema for quality dimension model."""
    name: str = Field(..., description="Dimension name")
    description: Optional[str] = Field(None, description="Dimension description")
    dimension_type: str = Field(..., description="Dimension type")
    evaluation_criteria: Dict[str, Any] = Field(..., description="Evaluation criteria")
    weight: float = Field(1.0, description="Dimension weight")
    is_active: bool = Field(True, description="Is dimension active")


class QualityDimensionModelCreate(QualityDimensionModelBase):
    """Schema for creating quality dimension model."""
    pass


class QualityDimensionModelUpdate(BaseModel):
    """Schema for updating quality dimension model."""
    name: Optional[str] = None
    description: Optional[str] = None
    evaluation_criteria: Optional[Dict[str, Any]] = None
    weight: Optional[float] = None
    is_active: Optional[bool] = None


class QualityDimensionModelResponse(QualityDimensionModelBase):
    """Schema for quality dimension model response."""
    id: str = Field(..., description="Dimension model ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    
    class Config:
        from_attributes = True