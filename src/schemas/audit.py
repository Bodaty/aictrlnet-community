"""Audit log schemas for security and compliance tracking."""

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum


class AuditActionType(str, Enum):
    """Types of audit actions."""
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    READ = "READ"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    ACCESS_DENIED = "ACCESS_DENIED"
    PERMISSION_CHANGE = "PERMISSION_CHANGE"
    CONFIG_CHANGE = "CONFIG_CHANGE"
    SYSTEM_EVENT = "SYSTEM_EVENT"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


# Audit log schemas
class AuditLogBase(BaseModel):
    """Base schema for audit logs."""
    action: str = Field(..., description="Action performed")
    resource_type: str = Field(..., description="Type of resource")
    description: str = Field(..., description="Description of the action")
    resource_id: Optional[str] = Field(None, description="ID of the resource")
    resource_name: Optional[str] = Field(None, description="Name of the resource")
    severity: AuditSeverity = Field(AuditSeverity.LOW, description="Severity level")
    success: bool = Field(True, description="Whether action was successful")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    old_values: Optional[Dict[str, Any]] = Field(None, description="Previous values")
    new_values: Optional[Dict[str, Any]] = Field(None, description="New values")
    custom_fields: Optional[Dict[str, Any]] = Field(None, description="Custom fields")
    compliance_tags: Optional[List[str]] = Field(None, description="Compliance tags")


class AuditLogCreate(AuditLogBase):
    """Schema for creating an audit log."""
    pass


class AuditLogResponse(AuditLogBase):
    """Response schema for audit log."""
    id: str
    tenant_id: Optional[str] = None
    user_id: Optional[str] = None
    user_email: Optional[str] = None
    user_roles: Optional[List[str]] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    timestamp: datetime
    created_at: datetime


class AuditLogQuery(BaseModel):
    """Query parameters for audit logs."""
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    action: Optional[str] = None
    severity: Optional[str] = None
    success: Optional[bool] = None
    user_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)
    order_by: str = Field("timestamp", pattern="^(timestamp|action|severity|resource_type)$")
    order_direction: str = Field("desc", pattern="^(asc|desc)$")


class AuditLogList(BaseModel):
    """List of audit logs."""
    logs: List[AuditLogResponse]
    total: int
    limit: int
    offset: int


# Audit filter schemas
class AuditFilterBase(BaseModel):
    """Base schema for audit filters."""
    name: str = Field(..., description="Filter name")
    filter_criteria: Dict[str, Any] = Field(..., description="Filter criteria")
    description: Optional[str] = Field(None, description="Filter description")
    is_default: bool = Field(False, description="Whether this is a default filter")
    is_shared: bool = Field(False, description="Whether this filter is shared")


class AuditFilterCreate(AuditFilterBase):
    """Schema for creating an audit filter."""
    pass


class AuditFilterUpdate(BaseModel):
    """Schema for updating an audit filter."""
    name: Optional[str] = None
    filter_criteria: Optional[Dict[str, Any]] = None
    description: Optional[str] = None
    is_default: Optional[bool] = None
    is_shared: Optional[bool] = None


class AuditFilterResponse(AuditFilterBase):
    """Response schema for audit filter."""
    id: str
    user_id: str
    tenant_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None


class AuditFilterList(BaseModel):
    """List of audit filters."""
    filters: List[AuditFilterResponse]


# Audit summary schemas
class AuditSummary(BaseModel):
    """Audit log summary statistics."""
    period_days: int
    total_events: int
    success_rate: float
    events_by_action: Dict[str, int]
    events_by_severity: Dict[str, int]
    events_by_resource_type: Dict[str, int]
    top_users: List[Dict[str, Any]]
    failed_events: int
    critical_events: int


# Audit export schemas
class AuditExportBase(BaseModel):
    """Base schema for audit export."""
    name: str = Field(..., description="Export name")
    filter_criteria: Dict[str, Any] = Field(..., description="Filter criteria for export")
    format: str = Field("csv", description="Export format (csv, json, pdf)")
    include_fields: Optional[List[str]] = Field(None, description="Fields to include")
    exclude_fields: Optional[List[str]] = Field(None, description="Fields to exclude")


class AuditExportCreate(AuditExportBase):
    """Schema for creating an audit export."""
    pass


class AuditExportResponse(AuditExportBase):
    """Response schema for audit export."""
    id: str
    user_id: str
    tenant_id: Optional[str] = None
    status: str = Field(..., description="Export status (pending, processing, completed, failed)")
    file_path: Optional[str] = None
    file_size: Optional[int] = None
    record_count: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


# Utility schemas
class AuditActionInfo(BaseModel):
    """Information about an audit action."""
    value: str
    label: str


class AuditSeverityInfo(BaseModel):
    """Information about an audit severity level."""
    value: str
    label: str


class AuditResourceTypeList(BaseModel):
    """List of resource types."""
    resource_types: List[str]


class AuditCleanupRequest(BaseModel):
    """Request for cleaning up old audit logs."""
    retention_days: int = Field(2555, ge=1, le=3650, description="Delete logs older than this many days")
    dry_run: bool = Field(True, description="Preview what would be deleted without actually deleting")


class AuditCleanupResponse(BaseModel):
    """Response for audit cleanup operation."""
    dry_run: bool
    retention_days: int
    cutoff_date: Optional[datetime] = None
    logs_to_delete: Optional[int] = None
    deleted_count: Optional[int] = None


class AuditHealthResponse(BaseModel):
    """Health check response for audit service."""
    service: str = "audit_logging"
    status: str = "healthy"
    timestamp: datetime
    features: List[str]