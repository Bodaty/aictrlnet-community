"""Task-related Pydantic schemas."""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime

from .common import TimestampSchema, PaginationResponse
from core.tenant_context import DEFAULT_TENANT_ID


class TaskBase(BaseModel):
    """Base task schema."""
    model_config = ConfigDict(
        populate_by_name=True,
        protected_namespaces=()
    )

    name: str = Field(..., min_length=1, max_length=256)
    description: Optional[str] = None
    task_metadata: Optional[Dict[str, Any]] = Field(None, alias="metadata")


class TaskCreate(TaskBase):
    """Task creation schema."""
    pass


class TaskUpdate(BaseModel):
    """Task update schema."""
    name: Optional[str] = Field(None, min_length=1, max_length=256)
    description: Optional[str] = None
    status: Optional[str] = Field(None, pattern="^(pending|in_progress|completed|failed|cancelled)$")
    task_metadata: Optional[Dict[str, Any]] = Field(None, alias="metadata")


class TaskResponse(TaskBase, TimestampSchema):
    """Task response schema."""
    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "id": "123e4567-e89b-12d3-a456-426614174000",
                "name": "Example Task",
                "description": "This is an example task",
                "status": "pending",
                "metadata": {"key": "value"},
                "tenant_id": DEFAULT_TENANT_ID,
                "created_at": "2023-01-01T00:00:00Z",
                "updated_at": "2023-01-01T00:00:00Z"
            }
        },
        protected_namespaces=()
    )
    
    id: str
    status: str
    tenant_id: str


class TaskListResponse(PaginationResponse[TaskResponse]):
    """Task list response schema."""
    pass