"""Security Management schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums

class RateLimitScope(str, Enum):
    """Rate limit scope."""
    GLOBAL = "global"
    USER = "user"
    IP = "ip"
    API_KEY = "api_key"


class ValidationRuleType(str, Enum):
    """Validation rule type."""
    REGEX = "regex"
    LENGTH = "length"
    RANGE = "range"
    TYPE = "type"
    CUSTOM = "custom"


class AlertSeverity(str, Enum):
    """Alert severity."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


# Rate Limit schemas

class RateLimitCreate(BaseModel):
    """Rate limit creation request."""
    resource: str = Field(..., description="Resource path or pattern")
    method: str = Field("*", description="HTTP method or * for all")
    rate: int = Field(..., description="Number of requests allowed")
    per_seconds: int = Field(60, description="Time window in seconds")
    burst: Optional[int] = Field(None, description="Burst allowance")
    scope: RateLimitScope = Field(RateLimitScope.GLOBAL, description="Limit scope")
    enabled: bool = Field(True, description="Is limit enabled")
    description: Optional[str] = Field(None, description="Limit description")


class RateLimitUpdate(BaseModel):
    """Rate limit update request."""
    rate: Optional[int] = None
    per_seconds: Optional[int] = None
    burst: Optional[int] = None
    enabled: Optional[bool] = None
    description: Optional[str] = None


class RateLimitResponse(BaseModel):
    """Rate limit response."""
    id: str
    resource: str
    method: str
    rate: int
    per_seconds: int
    burst: Optional[int] = None
    scope: str
    enabled: bool
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Validation Rule schemas

class ValidationRuleCreate(BaseModel):
    """Validation rule creation request."""
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    rule_type: ValidationRuleType = Field(..., description="Rule type")
    field: str = Field(..., description="Field to validate")
    pattern: Optional[str] = Field(None, description="Regex pattern for regex type")
    min_length: Optional[int] = Field(None, description="Minimum length")
    max_length: Optional[int] = Field(None, description="Maximum length")
    required: bool = Field(False, description="Is field required")
    error_message: Optional[str] = Field(None, description="Custom error message")
    enabled: bool = Field(True, description="Is rule enabled")


class ValidationRuleResponse(BaseModel):
    """Validation rule response."""
    id: str
    name: str
    description: Optional[str] = None
    rule_type: str
    field: str
    pattern: Optional[str] = None
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    required: bool
    error_message: Optional[str] = None
    enabled: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ValidationRequest(BaseModel):
    """Validation request."""
    field: str = Field(..., description="Field name")
    value: Any = Field(..., description="Value to validate")
    rule_type: Optional[str] = Field(None, description="Specific rule type")


class ValidationResponse(BaseModel):
    """Validation response."""
    field: str
    value: Any
    valid: bool
    errors: List[str]


# Blocked IP schemas

class BlockedIPCreate(BaseModel):
    """Blocked IP creation request."""
    ip_address: str = Field(..., description="IP address to block")
    reason: str = Field(..., description="Reason for blocking")
    blocked_by: str = Field(..., description="User who blocked the IP")
    expires_at: Optional[datetime] = Field(None, description="When the block expires")


class BlockedIPResponse(BaseModel):
    """Blocked IP response."""
    id: str
    ip_address: str
    reason: str
    blocked_by: str
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Password Validation schemas

class PasswordValidationRequest(BaseModel):
    """Password validation request."""
    password: str = Field(..., description="Password to validate")
    username: Optional[str] = Field(None, description="Username for similarity check")


class PasswordValidationResponse(BaseModel):
    """Password validation response."""
    valid: bool
    strength: str  # weak, medium, strong
    score: int  # 0-100
    errors: List[str]
    suggestions: List[str]


# Security Alert schemas

class SecurityAlertResponse(BaseModel):
    """Security alert response."""
    id: str
    alert_type: str
    severity: str
    title: str
    description: str
    source: str
    metadata: Optional[Dict[str, Any]] = None
    resolved: bool
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    resolution_notes: Optional[str] = None
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ResolveAlertRequest(BaseModel):
    """Resolve alert request."""
    resolved_by: str = Field(..., description="User resolving the alert")
    resolution_notes: Optional[str] = Field(None, description="Resolution notes")


# Security Health schemas

class SecurityHealthResponse(BaseModel):
    """Security health response."""
    status: str  # healthy, warning, critical
    health_score: int  # 0-100
    metrics: Dict[str, int]
    recent_issues: List[Dict[str, Any]]
    recommendations: List[str]