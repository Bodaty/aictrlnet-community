"""Schemas for basic usage metrics in Community edition."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class UsageMetricResponse(BaseModel):
    """Current usage metrics response."""
    workflow_count: int = Field(..., description="Current number of workflows")
    adapter_count: int = Field(..., description="Current number of adapters")
    user_count: int = Field(..., description="Current number of users")
    api_calls_month: int = Field(..., description="API calls this month")
    storage_bytes: int = Field(..., description="Current storage usage in bytes")
    period_start: datetime = Field(..., description="Start of tracking period")
    period_end: datetime = Field(..., description="End of tracking period")
    last_updated: datetime = Field(..., description="Last update timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class UsageLimitResponse(BaseModel):
    """Usage limits for current edition."""
    edition: str = Field(..., description="Edition name")
    max_workflows: int = Field(..., description="Maximum allowed workflows")
    max_adapters: int = Field(..., description="Maximum allowed adapters")
    max_users: int = Field(..., description="Maximum allowed users")
    max_api_calls_month: int = Field(..., description="Maximum API calls per month")
    max_storage_bytes: int = Field(..., description="Maximum storage in bytes")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class UsageStatusResponse(BaseModel):
    """Combined usage status with current usage and limits."""
    current_usage: UsageMetricResponse
    limits: UsageLimitResponse
    
    # Computed fields to show percentage used
    workflows_percent: float = Field(..., description="Percentage of workflow limit used")
    adapters_percent: float = Field(..., description="Percentage of adapter limit used")
    users_percent: float = Field(..., description="Percentage of user limit used")
    api_calls_percent: float = Field(..., description="Percentage of API call limit used")
    storage_percent: float = Field(..., description="Percentage of storage limit used")
    
    # Upgrade prompts
    needs_upgrade: bool = Field(..., description="Whether any limit is approaching")
    upgrade_reasons: list[str] = Field(default_factory=list, description="Reasons to upgrade")
    
    class Config:
        from_attributes = True