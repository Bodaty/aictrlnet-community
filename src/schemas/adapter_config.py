"""Pydantic schemas for user adapter configuration.

These schemas are for the UserAdapterConfig database model that stores
user-specific adapter configurations with encrypted credentials.

NOTE: These are NOT for the AdapterConfig Pydantic model in adapters/models.py
which is used for adapter registry metadata.
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from pydantic import BaseModel, Field, UUID4, validator, ConfigDict
from enum import Enum


class TestStatus(str, Enum):
    """Test status for adapter configurations."""
    SUCCESS = "success"
    FAILED = "failed"
    UNTESTED = "untested"
    TESTING = "testing"


class AdapterConfigBase(BaseModel):
    """Base schema for user adapter configuration (UserAdapterConfig model)."""
    adapter_type: str = Field(..., description="Type of adapter (from registry)")
    name: Optional[str] = Field(None, description="User's name for this configuration")
    display_name: Optional[str] = Field(None, description="Display name for UI")
    credentials: Optional[Dict[str, Any]] = Field(None, description="Adapter credentials")
    settings: Optional[Dict[str, Any]] = Field(None, description="Custom settings")
    enabled: bool = Field(True, description="Whether configuration is enabled")


class AdapterConfigCreate(AdapterConfigBase):
    """Schema for creating a user adapter configuration (UserAdapterConfig)."""
    pass


class AdapterConfigUpdate(BaseModel):
    """Schema for updating a user adapter configuration (UserAdapterConfig)."""
    name: Optional[str] = None
    display_name: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None
    settings: Optional[Dict[str, Any]] = None
    enabled: Optional[bool] = None


class AdapterConfigResponse(AdapterConfigBase):
    """Schema for user adapter configuration responses (UserAdapterConfig)."""
    id: UUID4
    user_id: str  # users.id is String(36), not UUID
    test_status: Optional[TestStatus] = None
    test_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_tested_at: Optional[datetime] = None
    version: Optional[str] = None
    metadata_field: Dict[str, Any] = Field(default_factory=dict, alias="metadata")

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class AdapterConfigTestRequest(BaseModel):
    """Request to test an adapter configuration."""
    timeout: int = Field(30, description="Timeout in seconds", ge=1, le=300)


class AdapterConfigTestResponse(BaseModel):
    """Response from testing an adapter configuration."""
    status: TestStatus
    message: str
    details: Optional[Dict[str, Any]] = None
    tested_at: datetime
    duration_ms: Optional[int] = None


class AdapterConfigListResponse(BaseModel):
    """Response for listing adapter configurations."""
    configs: List[AdapterConfigResponse]
    total: int
    
    
class AdapterConfigBulkTestRequest(BaseModel):
    """Request to test multiple adapter configurations."""
    config_ids: List[UUID4] = Field(..., description="List of configuration IDs to test")
    parallel: bool = Field(True, description="Test in parallel or sequentially")
    timeout: int = Field(30, description="Timeout per test in seconds", ge=1, le=300)


class AdapterConfigBulkTestResponse(BaseModel):
    """Response from bulk testing adapter configurations."""
    results: Dict[str, AdapterConfigTestResponse] = Field(..., description="Test results by config ID")
    total_tested: int
    successful: int
    failed: int
    duration_ms: int


class AdapterConfigWithCapabilities(AdapterConfigResponse):
    """Adapter configuration with runtime capabilities from registry."""
    capabilities: List[Dict[str, Any]] = Field(default_factory=list)
    is_registered: bool = Field(False, description="Whether adapter type exists in registry")
    category: Optional[str] = None
    description: Optional[str] = None