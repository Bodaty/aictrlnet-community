"""Control Plane schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums

class ComponentType(str, Enum):
    """Component types."""
    ADAPTER = "adapter"
    PROCESSOR = "processor" 
    ANALYZER = "analyzer"
    MONITOR = "monitor"
    VALIDATOR = "validator"
    TRANSFORMER = "transformer"
    GATEWAY = "gateway"
    ORCHESTRATOR = "orchestrator"


class ComponentStatus(str, Enum):
    """Component status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEGRADED = "degraded"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class ReviewStatus(str, Enum):
    """Review status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


# Component schemas

class ComponentRegisterRequest(BaseModel):
    """Component registration request."""
    name: str = Field(..., description="Component name")
    type: ComponentType = Field(..., description="Component type")
    description: Optional[str] = Field(None, description="Component description")
    endpoint: str = Field(..., description="Component endpoint URL")
    capabilities: List[str] = Field(default_factory=list, description="Component capabilities")
    health_check_endpoint: Optional[str] = Field(None, description="Health check endpoint")
    version: Optional[str] = Field("1.0.0", description="Component version")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ComponentResponse(BaseModel):
    """Component response."""
    id: str
    name: str
    type: str
    description: Optional[str] = None
    endpoint: str
    capabilities: List[str]
    status: str
    health_check_endpoint: Optional[str] = None
    version: Optional[str] = None
    last_health_check: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ComponentUpdate(BaseModel):
    """Component update request."""
    name: Optional[str] = None
    description: Optional[str] = None
    endpoint: Optional[str] = None
    capabilities: Optional[List[str]] = None
    health_check_endpoint: Optional[str] = None
    version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# Discovery schemas

class ComponentDiscoveryResponse(BaseModel):
    """Component discovery response."""
    total_components: int
    components_by_type: Dict[str, List[Dict[str, Any]]]
    available_capabilities: List[str]
    types: List[str]


# Health check schemas

class HealthCheckResponse(BaseModel):
    """Health check response."""
    component_id: str
    status: str
    latency_ms: int
    last_check: str
    details: Dict[str, Any]


# Metrics schemas

class ComponentMetric(BaseModel):
    """Component metric."""
    id: str
    type: str
    value: float
    metadata: Dict[str, Any]
    timestamp: str