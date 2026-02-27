"""Workflow-related Pydantic schemas."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from datetime import datetime

from .common import TimestampSchema, PaginationResponse


class NodeSchema(BaseModel):
    """Workflow node schema."""
    id: str
    type: str
    name: Optional[str] = None
    position: Optional[Dict[str, float]] = None
    data: Optional[Dict[str, Any]] = None
    parameters: Optional[Dict[str, Any]] = None  # Node execution parameters (custom_node_type, etc.)
    agent: Optional[str] = None  # Descriptive agent name for AI agent nodes
    agent_id: Optional[str] = None  # UUID of the assigned AI agent
    agent_name: Optional[str] = None  # Display name of the AI agent (with "AI " prefix)
    metadata: Optional[Dict[str, Any]] = None  # Rich metadata for the node


class EdgeSchema(BaseModel):
    """Workflow edge schema."""
    id: Optional[str] = None
    source: str = Field(..., alias="from")
    target: str = Field(..., alias="to")
    label: Optional[str] = None
    condition: Optional[str] = None
    
    class Config:
        populate_by_name = True


class WorkflowDefinitionSchema(BaseModel):
    """Workflow definition schema."""
    nodes: List[NodeSchema] = Field(default_factory=list)
    edges: List[EdgeSchema] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None


class WorkflowBase(BaseModel):
    """Base workflow schema."""
    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None


class WorkflowCreate(WorkflowBase):
    """Workflow creation schema."""
    definition: Optional[WorkflowDefinitionSchema] = None
    template_id: Optional[str] = None
    is_template: bool = False
    status: str = "active"  # draft, active, archived - default to active


class WorkflowUpdate(BaseModel):
    """Workflow update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    definition: Optional[WorkflowDefinitionSchema] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    status: Optional[str] = None  # draft, active, archived


class WorkflowResponse(WorkflowBase, TimestampSchema):
    """Workflow response schema."""
    id: str
    definition: WorkflowDefinitionSchema
    version: int
    is_template: bool = False
    template_id: Optional[str] = None
    status: str = "draft"  # draft, active, archived
    tenant_id: str
    
    @model_validator(mode='before')
    @classmethod
    def extract_metadata_fields(cls, data: Any) -> Any:
        """Extract fields from workflow_metadata if needed."""
        if isinstance(data, dict):
            return data
            
        # Handle SQLAlchemy model
        if hasattr(data, '__dict__'):
            result = {}
            for key in ['id', 'name', 'description', 'tags', 'version', 'tenant_id', 'created_at', 'updated_at']:
                if hasattr(data, key):
                    result[key] = getattr(data, key)
            
            # Extract definition
            if hasattr(data, 'definition'):
                result['definition'] = data.definition
            
            # Extract from metadata
            metadata = getattr(data, 'workflow_metadata', {}) or {}
            result['category'] = metadata.get('category')
            result['is_template'] = metadata.get('is_template', False)
            result['template_id'] = metadata.get('template_id')
            result['status'] = metadata.get('status', 'draft')
            
            return result
        
        return data
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class WorkflowListResponse(PaginationResponse[WorkflowResponse]):
    """Workflow list response schema."""
    pass


class TemplateMatch(BaseModel):
    """Template match information."""
    template_id: str
    template_name: str
    confidence: float
    category: str
    tags: List[str]


class ExtractedParameter(BaseModel):
    """Extracted parameter from NLP."""
    name: str
    value: Any
    confidence: float
    source: str  # e.g., "explicit", "inferred", "default"


class EditionRequirement(BaseModel):
    """Edition requirement for a feature."""
    feature: str
    required_edition: str  # "community", "business", "enterprise"
    reason: str
    current_availability: bool


class UpgradeSuggestion(BaseModel):
    """Upgrade suggestion based on user intent."""
    target_edition: str
    features_unlocked: List[str]
    reason: str
    priority: str  # "high", "medium", "low"


class TemplatePreview(BaseModel):
    """Template preview information."""
    id: str
    name: str
    description: str
    category: str
    tags: List[str]
    node_count: int
    complexity: str  # "simple", "moderate", "complex"
    edition_required: str
    preview_available: bool
    example_use_cases: List[str]


class NLPWorkflowResponse(BaseModel):
    """Enhanced NLP workflow response with full transparency."""
    # Standard workflow fields
    workflow: WorkflowResponse
    
    # Transparency fields
    generation_method: str  # "ai_generated", "template_based", "hybrid"
    templates_used: List[TemplateMatch]
    extracted_parameters: List[ExtractedParameter]
    intent_analysis: Dict[str, Any]
    confidence_score: float
    
    # Edition and upgrade info
    edition_requirements: List[EditionRequirement]
    current_edition: str
    upgrade_suggestions: List[UpgradeSuggestion]
    
    # Available templates for customization
    related_templates: List[TemplatePreview]
    alternative_templates: List[TemplatePreview]
    
    # Debug info (optional)
    processing_steps: Optional[List[Dict[str, Any]]] = None
    ai_model_used: Optional[str] = None


class WorkflowStepResponse(TimestampSchema):
    """Workflow step response schema."""
    id: str
    workflow_instance_id: str
    step_name: str
    step_type: str
    node_type: Optional[str] = None
    status: str
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    tenant_id: str
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class WorkflowInstanceResponse(TimestampSchema):
    """Workflow instance response schema."""
    id: str
    workflow_definition_id: str
    status: str
    context: Optional[Dict[str, Any]] = None
    steps: List[WorkflowStepResponse] = []
    tenant_id: str
    
    class Config:
        from_attributes = True