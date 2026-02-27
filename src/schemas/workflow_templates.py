"""Workflow template schemas for Community Edition."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
from datetime import datetime
from uuid import UUID
from enum import Enum


class TemplateComplexity(str, Enum):
    """Template complexity levels."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    ADVANCED = "advanced"


class TemplatePermission(str, Enum):
    """Template permission types."""
    VIEW = "view"
    USE = "use"
    EDIT = "edit"
    DELETE = "delete"


class CustomizationLevel(str, Enum):
    """Template customization levels."""
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"


class TemplateParameterType(str, Enum):
    """Types of template parameters."""
    TEXT = "text"
    NUMBER = "number"
    BOOLEAN = "boolean"
    SELECT = "select"
    DATE = "date"
    DATETIME = "datetime"


# Parameter schemas
class TemplateParameter(BaseModel):
    """Template parameter definition."""
    id: str
    name: str
    type: TemplateParameterType
    label: str
    description: Optional[str] = None
    required: bool = False
    default: Optional[Any] = None
    options: Optional[List[str]] = None  # For select type
    validation: Optional[str] = None  # Regex or rule
    depends_on: Optional[Dict[str, Any]] = None  # Conditional display


# Base schemas
class WorkflowTemplateBase(BaseModel):
    """Base schema for workflow templates."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=50)
    tags: List[str] = Field(default_factory=list)
    edition: str = Field("community", max_length=20)
    is_public: bool = False
    complexity: Optional[TemplateComplexity] = None
    estimated_duration: Optional[str] = Field(None, max_length=50)
    required_adapters: List[str] = Field(default_factory=list)
    required_capabilities: List[str] = Field(default_factory=list)


# Create/Update schemas
class WorkflowTemplateCreate(WorkflowTemplateBase):
    """Schema for creating a workflow template."""
    definition_path: str = Field(..., max_length=500)
    thumbnail_path: Optional[str] = Field(None, max_length=500)
    parent_template_id: Optional[UUID] = None
    
    @validator('definition_path')
    def validate_definition_path(cls, v):
        if not v.endswith('.json'):
            raise ValueError('Definition path must point to a JSON file')
        return v


class WorkflowTemplateUpdate(BaseModel):
    """Schema for updating a workflow template."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    category: Optional[str] = Field(None, max_length=50)
    tags: Optional[List[str]] = None
    is_public: Optional[bool] = None
    complexity: Optional[TemplateComplexity] = None
    estimated_duration: Optional[str] = Field(None, max_length=50)
    required_adapters: Optional[List[str]] = None
    required_capabilities: Optional[List[str]] = None
    thumbnail_path: Optional[str] = Field(None, max_length=500)


# Response schemas
class WorkflowTemplateResponse(WorkflowTemplateBase):
    """Schema for workflow template responses."""
    id: UUID
    owner_id: Optional[str] = None
    is_system: bool = False
    parent_template_id: Optional[UUID] = None
    version: int = 1
    definition_path: str
    thumbnail_path: Optional[str] = None
    usage_count: int = 0
    rating: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    published_at: Optional[datetime] = None
    
    # Preview data for efficient list display
    preview: Optional[Dict[str, Any]] = None  # Contains first few nodes for preview
    node_count: Optional[int] = None
    edge_count: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class WorkflowTemplateDetail(WorkflowTemplateResponse):
    """Detailed workflow template with full content."""
    workflow_definition: Optional[Dict[str, Any]] = None
    parameters: List[TemplateParameter] = Field(default_factory=list)
    preview_available: bool = True
    owner_name: Optional[str] = None
    can_edit: bool = False
    can_delete: bool = False
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Permission schemas
class TemplatePermissionCreate(BaseModel):
    """Schema for creating template permissions."""
    template_id: UUID
    user_id: str
    permission: TemplatePermission


class TemplatePermissionResponse(BaseModel):
    """Schema for template permission responses."""
    id: UUID
    template_id: UUID
    user_id: Optional[str] = None
    permission: TemplatePermission
    granted_at: datetime
    granted_by: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Usage schemas
class TemplateUsageCreate(BaseModel):
    """Schema for recording template usage."""
    template_id: UUID
    workflow_id: UUID
    customization_level: CustomizationLevel = CustomizationLevel.NONE


class TemplateUsageResponse(BaseModel):
    """Schema for template usage responses."""
    id: UUID
    template_id: UUID
    user_id: str
    workflow_id: Optional[UUID] = None
    instantiated_at: datetime
    customization_level: Optional[CustomizationLevel] = None
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Review schemas
class TemplateReviewCreate(BaseModel):
    """Schema for creating template reviews."""
    rating: int = Field(..., ge=1, le=5)
    review: Optional[str] = None


class TemplateReviewUpdate(BaseModel):
    """Schema for updating template reviews."""
    rating: Optional[int] = Field(None, ge=1, le=5)
    review: Optional[str] = None


class TemplateReviewResponse(BaseModel):
    """Schema for template review responses."""
    id: UUID
    template_id: UUID
    user_id: str
    rating: int
    review: Optional[str] = None
    created_at: datetime
    user_name: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Template instantiation schemas
class InstantiateTemplateRequest(BaseModel):
    """Request to create a workflow from a template."""
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    project_id: Optional[UUID] = None
    # Enhancement options for unified pipeline
    enhancements: Optional[Dict[str, bool]] = Field(None, description="Enhancement options to apply")
    enhancement_sub_options: Optional[Dict[str, Any]] = Field(None, description="Sub-options for enhancements")


class InstantiateTemplateResponse(BaseModel):
    """Response after instantiating a template."""
    workflow_id: UUID
    workflow_name: str
    template_id: UUID
    template_name: str
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class WorkflowFromTemplate(BaseModel):
    """Schema for creating a workflow from a template (legacy compatibility)."""
    template_id: str = Field(..., description="ID of the template to use")
    name: str = Field(..., min_length=1, max_length=255, description="Name for the new workflow")
    description: Optional[str] = Field(None, description="Description for the new workflow")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Parameters to customize the template")


# Fork template schemas
class ForkTemplateRequest(BaseModel):
    """Request to fork a template."""
    new_name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    make_public: bool = False


# List/Filter schemas
class TemplateListRequest(BaseModel):
    """Request for listing templates with filters."""
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    edition: Optional[str] = None
    complexity: Optional[TemplateComplexity] = None
    include_public: bool = True
    include_system: bool = True
    include_private: bool = True
    search: Optional[str] = None
    sort_by: str = "created_at"  # created_at, updated_at, usage_count, rating, name
    sort_desc: bool = True
    skip: int = 0
    limit: int = 100


class TemplateListResponse(BaseModel):
    """Response for template listing."""
    templates: List[WorkflowTemplateResponse]
    total: int
    skip: int
    limit: int


# Analytics schemas
class TemplateAnalyticsResponse(BaseModel):
    """Template usage analytics."""
    template_id: UUID
    template_name: str
    total_uses: int
    unique_users: int
    average_rating: Optional[float] = None
    total_reviews: int
    usage_by_month: Dict[str, int] = Field(default_factory=dict)
    customization_stats: Dict[CustomizationLevel, int] = Field(default_factory=dict)
    most_common_parameters: Dict[str, Any] = Field(default_factory=dict)


# Common response schemas
class SuccessResponse(BaseModel):
    """Generic success response."""
    success: bool = True
    message: str