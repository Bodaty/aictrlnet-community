"""Geographic routing and region management schemas."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class RegionStatus(str, Enum):
    """Geographic region status."""
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    MAINTENANCE = "MAINTENANCE"
    DEGRADED = "DEGRADED"


class RoutingPolicy(str, Enum):
    """Routing policy types."""
    NEAREST = "NEAREST"
    LOAD_BALANCED = "LOAD_BALANCED"
    FAILOVER = "FAILOVER"
    COMPLIANCE = "COMPLIANCE"
    CUSTOM = "CUSTOM"


# Region schemas
class GeographicRegionBase(BaseModel):
    """Base schema for geographic regions."""
    name: str = Field(..., description="Region identifier (e.g., us-east-1)")
    display_name: str = Field(..., description="Display name (e.g., US East (Virginia))")
    country_code: str = Field(..., description="ISO 3166-1 alpha-2 country code")
    continent: str = Field(..., description="Continent name")
    city: Optional[str] = Field(None, description="City name")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    timezone: str = Field(..., description="Timezone (e.g., America/New_York)")
    provider: Optional[str] = Field(None, description="Cloud provider (aws, azure, gcp)")
    max_capacity: int = Field(1000, description="Maximum concurrent workflows")
    services_enabled: List[str] = Field(default_factory=list, description="Available services")
    compliance_certifications: List[str] = Field(default_factory=list, description="Compliance certifications")
    description: Optional[str] = Field(None, description="Region description")


class GeographicRegionCreate(GeographicRegionBase):
    """Schema for creating a geographic region."""
    pass


class GeographicRegionUpdate(BaseModel):
    """Schema for updating a geographic region."""
    display_name: Optional[str] = None
    city: Optional[str] = None
    provider: Optional[str] = None
    max_capacity: Optional[int] = None
    services_enabled: Optional[List[str]] = None
    compliance_certifications: Optional[List[str]] = None
    description: Optional[str] = None
    status: Optional[RegionStatus] = None


class GeographicRegionResponse(GeographicRegionBase):
    """Response schema for geographic region."""
    id: str
    status: RegionStatus
    health_score: Optional[float] = None
    current_load: Optional[int] = None
    utilization: float = 0.0
    average_latency_ms: Optional[float] = None
    last_health_check: Optional[datetime] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class GeographicRegionList(BaseModel):
    """List of geographic regions."""
    regions: List[GeographicRegionResponse]


# Routing rule schemas
class RoutingRuleBase(BaseModel):
    """Base schema for routing rules."""
    name: str = Field(..., description="Rule name")
    description: Optional[str] = Field(None, description="Rule description")
    priority: int = Field(100, description="Lower number = higher priority")
    is_active: bool = Field(True, description="Whether rule is active")
    policy_type: RoutingPolicy = Field(RoutingPolicy.NEAREST, description="Routing policy")
    conditions: Dict[str, Any] = Field(default_factory=dict, description="Rule conditions")
    user_countries: List[str] = Field(default_factory=list, description="Countries this rule applies to")
    compliance_requirements: List[str] = Field(default_factory=list, description="Required compliance")
    primary_regions: List[str] = Field(default_factory=list, description="Primary region IDs")
    fallback_regions: List[str] = Field(default_factory=list, description="Fallback region IDs")
    excluded_regions: List[str] = Field(default_factory=list, description="Excluded region IDs")
    region_weights: Dict[str, float] = Field(default_factory=dict, description="Region weights for load balancing")


class RoutingRuleCreate(RoutingRuleBase):
    """Schema for creating a routing rule."""
    pass


class RoutingRuleUpdate(BaseModel):
    """Schema for updating a routing rule."""
    name: Optional[str] = None
    description: Optional[str] = None
    priority: Optional[int] = None
    is_active: Optional[bool] = None
    policy_type: Optional[RoutingPolicy] = None
    conditions: Optional[Dict[str, Any]] = None
    user_countries: Optional[List[str]] = None
    compliance_requirements: Optional[List[str]] = None
    primary_regions: Optional[List[str]] = None
    fallback_regions: Optional[List[str]] = None
    excluded_regions: Optional[List[str]] = None
    region_weights: Optional[Dict[str, float]] = None


class RoutingRuleResponse(RoutingRuleBase):
    """Response schema for routing rule."""
    id: str
    tenant_id: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0


class RoutingRuleList(BaseModel):
    """List of routing rules."""
    routing_rules: List[RoutingRuleResponse]


# Health metric schemas
class RegionHealthMetricBase(BaseModel):
    """Base schema for region health metrics."""
    cpu_usage_percent: Optional[float] = None
    memory_usage_percent: Optional[float] = None
    disk_usage_percent: Optional[float] = None
    network_latency_ms: Optional[float] = None
    requests_per_second: Optional[float] = None
    error_rate_percent: Optional[float] = None
    average_response_time_ms: Optional[float] = None
    active_workflows: Optional[int] = None
    utilization_percent: Optional[float] = None
    status: Optional[str] = None
    alerts: List[str] = Field(default_factory=list)


class RegionHealthMetricCreate(RegionHealthMetricBase):
    """Schema for creating health metrics."""
    region_id: str


class RegionHealthMetricResponse(RegionHealthMetricBase):
    """Response schema for health metrics."""
    id: str
    region_id: str
    health_score: Optional[float] = None
    timestamp: datetime


class RegionHealthList(BaseModel):
    """List of region health metrics."""
    region_id: Optional[str] = None
    health_metrics: List[RegionHealthMetricResponse]


# Traffic route schemas
class TrafficRouteBase(BaseModel):
    """Base schema for traffic routes."""
    request_id: str
    selected_region: str
    user_ip: Optional[str] = None
    user_country: Optional[str] = None
    routing_rule_id: Optional[str] = None
    routing_policy: Optional[str] = None
    decision_reason: Optional[str] = None
    candidate_regions: List[str] = Field(default_factory=list)
    routing_latency_ms: Optional[float] = None


class TrafficRouteCreate(TrafficRouteBase):
    """Schema for creating a traffic route."""
    pass


class TrafficRouteResponse(TrafficRouteBase):
    """Response schema for traffic route."""
    id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: datetime


class TrafficSummary(BaseModel):
    """Traffic routing summary statistics."""
    period_hours: int
    total_requests: int
    success_rate: float
    average_routing_latency_ms: float
    requests_by_region: Dict[str, int]
    requests_by_policy: Dict[str, int]


# Routing request/response schemas
class RouteRequest(BaseModel):
    """Request schema for finding optimal region."""
    user_latitude: Optional[float] = None
    user_longitude: Optional[float] = None
    user_country: Optional[str] = None
    compliance_requirements: Optional[List[str]] = None
    service_requirements: Optional[List[str]] = None
    policy_override: Optional[RoutingPolicy] = None


class RouteResponse(BaseModel):
    """Response schema for optimal region selection."""
    selected_region: Optional[GeographicRegionResponse] = None
    reason: str
    candidates: List[str]
    policy: str
    rule_id: Optional[str] = None


# Configuration schemas
class RegionConfigurationBase(BaseModel):
    """Base schema for region configuration."""
    region_id: str
    config_type: str = Field(..., description="Configuration type (service, feature, security)")
    config_key: str = Field(..., description="Configuration key")
    config_value: Dict[str, Any] = Field(..., description="Configuration value")
    description: Optional[str] = None
    is_sensitive: bool = Field(False, description="Whether config contains secrets")
    requires_restart: bool = Field(False, description="Whether change requires restart")


class RegionConfigurationCreate(RegionConfigurationBase):
    """Schema for creating region configuration."""
    pass


class RegionConfigurationUpdate(BaseModel):
    """Schema for updating region configuration."""
    config_value: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    is_sensitive: Optional[bool] = None
    requires_restart: Optional[bool] = None
    is_active: Optional[bool] = None


class RegionConfigurationResponse(RegionConfigurationBase):
    """Response schema for region configuration."""
    id: str
    tenant_id: Optional[str] = None
    version: int = 1
    previous_value: Optional[Dict[str, Any]] = None
    is_active: bool = True
    applied_at: Optional[datetime] = None
    created_by: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


# Utility schemas
class DistanceResponse(BaseModel):
    """Response schema for distance calculation."""
    distance_km: float
    distance_miles: float


class PolicyInfo(BaseModel):
    """Information about a routing policy."""
    value: str
    label: str
    description: str


class StatusInfo(BaseModel):
    """Information about a region status."""
    value: str
    label: str
    description: str