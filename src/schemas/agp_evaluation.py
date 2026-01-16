"""AGP (AI Governance Policy) Evaluation schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums

class PolicyType(str, Enum):
    """Policy type."""
    INPUT_VALIDATION = "input_validation"
    OUTPUT_FILTERING = "output_filtering"
    MODEL_SELECTION = "model_selection"
    TOKEN_LIMIT = "token_limit"
    QUALITY = "quality"
    COMPLIANCE = "compliance"
    CONTENT_FILTERING = "content_filtering"
    PROMPT_INJECTION = "prompt_injection"
    BIAS_DETECTION = "bias_detection"
    DATA_PRIVACY = "data_privacy"
    RATE_LIMITING = "rate_limiting"


class PolicySeverity(str, Enum):
    """Policy severity."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# Policy Assignment schemas

class PolicyAssignmentCreate(BaseModel):
    """Policy assignment creation request."""
    policy_id: str = Field(..., description="Policy to assign")
    resource_type: str = Field(..., description="Resource type (agent, workflow, etc)")
    resource_id: str = Field(..., description="Resource ID")
    assigned_by: str = Field(..., description="User assigning the policy")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Assignment metadata")


class PolicyAssignmentResponse(BaseModel):
    """Policy assignment response."""
    id: str
    policy_id: str
    policy_name: str
    resource_type: str
    resource_id: str
    assigned_by: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Policy Template schemas

class PolicyTemplateResponse(BaseModel):
    """Policy template response."""
    id: str
    name: str
    description: Optional[str] = None
    category: str
    template_data: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Policy Evaluation Request schemas

class PolicyEvaluationRequest(BaseModel):
    """Base policy evaluation request."""
    context: Dict[str, Any] = Field(..., description="Evaluation context")


class InputEvaluationRequest(PolicyEvaluationRequest):
    """Input evaluation request."""
    input_data: Dict[str, Any] = Field(..., description="Input data to evaluate")


class OutputEvaluationRequest(PolicyEvaluationRequest):
    """Output evaluation request."""
    output_data: Dict[str, Any] = Field(..., description="Output data to evaluate")


class ModelEvaluationRequest(PolicyEvaluationRequest):
    """Model evaluation request."""
    ai_model_id: str = Field(..., description="Model ID to evaluate")
    ai_model_provider: str = Field(..., description="Model provider")
    ai_model_capabilities: List[str] = Field(default_factory=list, description="Model capabilities")


class TokenEvaluationRequest(PolicyEvaluationRequest):
    """Token evaluation request."""
    input_tokens: int = Field(..., description="Number of input tokens")
    output_tokens: int = Field(..., description="Number of output tokens")
    estimated_cost: Optional[float] = Field(None, description="Estimated cost in USD")


# Policy Evaluation Response schemas

class PolicyViolation(BaseModel):
    """Policy violation details."""
    policy_id: str
    policy_name: str
    violation: str
    severity: str


class PolicyEvaluationResponse(BaseModel):
    """Policy evaluation response."""
    compliant: bool
    violations: List[PolicyViolation]
    evaluated_policies: int
    recommendations: List[str]


# Policy Log schemas

class PolicyLogResponse(BaseModel):
    """Policy log response."""
    id: str
    evaluation_type: str
    resource_id: Optional[str] = None
    compliant: bool
    violations: List[Dict[str, Any]]
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Policy CRUD schemas

class PolicyCreate(BaseModel):
    """Policy creation request."""
    name: str = Field(..., description="Policy name")
    description: Optional[str] = Field(None, description="Policy description")
    policy_type: PolicyType = Field(..., description="Policy type")
    severity: PolicySeverity = Field(PolicySeverity.MEDIUM, description="Policy severity")
    rules: List[Dict[str, Any]] = Field(..., description="Policy rules")
    enabled: bool = Field(True, description="Is policy enabled")
    config: Optional[Dict[str, Any]] = Field(None, description="Policy configuration")
    policy_metadata: Optional[Dict[str, Any]] = Field(None, description="Policy metadata")


class PolicyUpdate(BaseModel):
    """Policy update request."""
    name: Optional[str] = None
    description: Optional[str] = None
    severity: Optional[str] = None
    rules: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None
    policy_metadata: Optional[Dict[str, Any]] = None


class PolicyResponse(BaseModel):
    """Policy response."""
    id: str
    name: str
    description: Optional[str] = None
    policy_type: str
    severity: str
    rules: List[Dict[str, Any]]
    enabled: bool
    tenant_id: str
    config: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    policy_metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True