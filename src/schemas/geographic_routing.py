"""Geographic Routing schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


class RegionStatus(str, Enum):
    """Region status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"


# Service Region schemas

class ServiceRegionCreate(BaseModel):
    """Service region creation request."""
    server_id: str = Field(..., description="MCP server ID")
    region: str = Field(..., description="Region code (e.g., us-east-1)")
    url: str = Field(..., description="Service URL for this region")
    status: str = Field("active", description="Region status")
    priority: int = Field(1, description="Priority (lower is higher priority)")


class ServiceRegionUpdate(BaseModel):
    """Service region update request."""
    url: Optional[str] = None
    status: Optional[str] = None
    priority: Optional[int] = None


class ServiceRegionResponse(BaseModel):
    """Service region response."""
    id: str
    server_id: str
    region: str
    url: str
    status: str
    priority: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Geographic Routing schemas

class GeographicRoutingRequest(BaseModel):
    """Geographic routing request."""
    server_id: str = Field(..., description="MCP server ID")
    client_region: str = Field(..., description="Client's region")
    request_type: Optional[str] = Field(None, description="Type of request")


class GeographicRoutingResponse(BaseModel):
    """Geographic routing response."""
    server_id: str
    selected_region: str
    selected_url: str
    priority: int
    alternatives: List[Dict[str, Any]]
    routing_metadata: Dict[str, Any]


# Health Status schemas

class RegionHealthStatus(BaseModel):
    """Region health status."""
    region_id: str
    server_id: str
    region: str
    url: str
    status: str
    healthy: bool
    latency_ms: Optional[int] = None
    last_check: datetime
    error: Optional[str] = None


class RegionPriorityUpdate(BaseModel):
    """Region priority update request."""
    priority_map: Dict[str, int] = Field(..., description="Map of region to priority")