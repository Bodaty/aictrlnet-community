"""Common Pydantic schemas."""

from typing import Optional, Dict, Any, List, Generic, TypeVar
from pydantic import BaseModel, Field, ConfigDict
from datetime import datetime


T = TypeVar("T")


class TimestampSchema(BaseModel):
    """Schema with timestamps."""
    model_config = ConfigDict(protected_namespaces=())

    created_at: datetime
    updated_at: datetime


class PaginationParams(BaseModel):
    """Pagination parameters."""
    model_config = ConfigDict(protected_namespaces=())

    page: int = Field(default=1, ge=1)
    per_page: int = Field(default=20, le=100)
    sort_by: Optional[str] = None
    sort_order: str = Field(default="asc", pattern="^(asc|desc)$")


class PaginationResponse(BaseModel, Generic[T]):
    """Pagination response wrapper."""
    model_config = ConfigDict(
        from_attributes=True,
        protected_namespaces=()
    )

    items: List[T]
    total: int
    page: int
    per_page: int
    pages: int


class ErrorResponse(BaseModel):
    """Error response schema."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class SuccessResponse(BaseModel, Generic[T]):
    """Success response wrapper."""
    success: bool = True
    data: T
    message: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "ok"
    edition: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    services: Dict[str, str] = Field(default_factory=dict)


class EditionInfo(BaseModel):
    """Edition information."""
    edition: str
    features: Dict[str, Any]
    limits: Dict[str, int]