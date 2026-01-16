"""Adapter-related Pydantic schemas."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from datetime import datetime

from .common import TimestampSchema


class AdapterBase(BaseModel):
    """Base adapter schema."""
    model_config = ConfigDict(populate_by_name=True,
        protected_namespaces=()
    )
    
    name: str = Field(..., min_length=1, max_length=256)
    category: str = Field(..., min_length=1, max_length=128)
    description: Optional[str] = None
    version: str = Field(..., pattern=r"^\d+\.\d+\.\d+$")
    min_edition: str = Field(..., pattern="^(community|business|enterprise)$")
    enabled: bool = True
    adapter_metadata: Optional[Dict[str, Any]] = None
    config_schema: Optional[Dict[str, Any]] = None
    capabilities: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class AdapterCreate(AdapterBase):
    """Adapter creation schema."""
    pass


class AdapterUpdate(BaseModel):
    """Adapter update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    enabled: Optional[bool] = None
    adapter_metadata: Optional[Dict[str, Any]] = None
    config_schema: Optional[Dict[str, Any]] = None
    capabilities: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class AdapterResponse(AdapterBase, TimestampSchema):
    """Adapter response schema."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )
    
    id: str
    available: bool = True
    installed: bool = False
    install_count: int = 0
    
    @field_validator('capabilities', 'tags', mode='before')
    @classmethod
    def ensure_list(cls, v):
        """Convert None to empty list for capabilities and tags."""
        if v is None:
            return []
        return v


class AdapterDiscoverResponse(BaseModel):
    """Adapter discovery response."""
    adapters: List[AdapterResponse]
    total: int
    edition: str
    categories: List[str]


class AdapterCategoryResponse(BaseModel):
    """Adapter category with count."""
    category: str
    count: int
    description: Optional[str] = None


class AdapterAvailabilityRequest(BaseModel):
    """Request to check adapter availability."""
    adapter_ids: List[str] = Field(..., min_items=1)


class AdapterAvailabilityResponse(BaseModel):
    """Response for adapter availability check."""
    available: List[str]
    unavailable: List[str]
    edition: str