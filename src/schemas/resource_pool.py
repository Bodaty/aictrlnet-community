"""Resource pool schemas."""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class ResourcePoolConfigBase(BaseModel):
    """Base schema for resource pool configuration."""
    name: str = Field(..., description="Unique name for the resource pool")
    resource_type: str = Field(..., description="Type of resource (compute, storage, etc.)")
    min_size: int = Field(1, ge=1, description="Minimum pool size")
    max_size: int = Field(10, ge=1, description="Maximum pool size")
    acquire_timeout: float = Field(30.0, description="Timeout for acquiring resources")
    idle_timeout: float = Field(300.0, description="Idle timeout before resource cleanup")
    max_lifetime: float = Field(3600.0, description="Maximum lifetime of a resource")
    health_check_interval: float = Field(60.0, description="Health check interval")
    scale_up_threshold: float = Field(0.8, description="Utilization threshold for scaling up")
    scale_down_threshold: float = Field(0.2, description="Utilization threshold for scaling down")
    enabled: bool = Field(True, description="Whether the pool is enabled")
    config_metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional configuration")


class ResourcePoolConfigCreate(ResourcePoolConfigBase):
    """Schema for creating a resource pool configuration."""
    pass


class ResourcePoolConfigUpdate(BaseModel):
    """Schema for updating a resource pool configuration."""
    name: Optional[str] = None
    resource_type: Optional[str] = None
    min_size: Optional[int] = Field(None, ge=1)
    max_size: Optional[int] = Field(None, ge=1)
    acquire_timeout: Optional[float] = None
    idle_timeout: Optional[float] = None
    max_lifetime: Optional[float] = None
    health_check_interval: Optional[float] = None
    scale_up_threshold: Optional[float] = None
    scale_down_threshold: Optional[float] = None
    enabled: Optional[bool] = None
    config_metadata: Optional[Dict[str, Any]] = None


class ResourcePoolConfigResponse(ResourcePoolConfigBase):
    """Schema for resource pool configuration response."""
    id: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Business Edition schemas
class ResourcePoolBase(BaseModel):
    """Base schema for enhanced resource pool (Business Edition)."""
    name: str
    resource_type: str
    description: Optional[str] = None
    pool_size: int = 10
    max_pool_size: int = 100
    min_pool_size: int = 1
    configuration: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    is_active: bool = True
    priority: int = 5
    auto_scale: bool = False
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.2
    cost_per_hour: float = 0.0


class ResourcePoolCreate(ResourcePoolBase):
    """Schema for creating a resource pool."""
    pass


class ResourcePoolUpdate(BaseModel):
    """Schema for updating a resource pool."""
    name: Optional[str] = None
    description: Optional[str] = None
    pool_size: Optional[int] = None
    max_pool_size: Optional[int] = None
    min_pool_size: Optional[int] = None
    configuration: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    is_active: Optional[bool] = None
    priority: Optional[int] = None
    auto_scale: Optional[bool] = None
    scale_up_threshold: Optional[float] = None
    scale_down_threshold: Optional[float] = None
    cost_per_hour: Optional[float] = None


class ResourcePoolResponse(ResourcePoolBase):
    """Schema for resource pool response."""
    id: str
    current_size: int
    available_count: int
    reserved_count: int
    total_cost: float
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ResourceAllocationCreate(BaseModel):
    """Schema for creating a resource allocation."""
    resource_pool_id: str
    allocated_to: str
    allocation_type: str = "task"


class ResourceAllocationResponse(BaseModel):
    """Schema for resource allocation response."""
    id: str
    resource_pool_id: str
    resource_id: str
    allocated_to: Optional[str]
    allocation_type: str
    allocated_at: datetime
    released_at: Optional[datetime]
    duration_seconds: Optional[int]
    cost: float
    status: str

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ResourcePoolMetricResponse(BaseModel):
    """Schema for resource pool metric response."""
    id: str
    resource_pool_id: str
    metric_name: str
    metric_value: float
    metric_unit: Optional[str]
    timestamp: datetime

    class Config:
        from_attributes = True