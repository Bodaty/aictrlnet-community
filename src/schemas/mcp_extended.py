"""Extended MCP schemas for missing models."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


# MCPServerCapability Schemas
class MCPServerCapabilityBase(BaseModel):
    """Base schema for MCP server capability."""
    server_id: str = Field(..., description="MCP server ID")
    capability_type: str = Field(..., description="Capability type")
    capability_name: str = Field(..., description="Capability name")
    capability_config: Dict[str, Any] = Field(default_factory=dict, description="Capability configuration")
    is_enabled: bool = Field(True, description="Is capability enabled")


class MCPServerCapabilityCreate(MCPServerCapabilityBase):
    """Schema for creating MCP server capability."""
    pass


class MCPServerCapabilityUpdate(BaseModel):
    """Schema for updating MCP server capability."""
    capability_config: Optional[Dict[str, Any]] = None
    is_enabled: Optional[bool] = None


class MCPServerCapabilityResponse(MCPServerCapabilityBase):
    """Schema for MCP server capability response."""
    id: str = Field(..., description="Capability ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# TaskMCP Schemas
class TaskMCPBase(BaseModel):
    """Base schema for task MCP."""
    task_id: str = Field(..., description="Task ID")
    mcp_server_id: str = Field(..., description="MCP server ID")
    invocation_id: Optional[str] = Field(None, description="MCP invocation ID")
    status: str = Field("pending", description="MCP task status")
    request_data: Dict[str, Any] = Field(default_factory=dict, description="Request data")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    error_message: Optional[str] = Field(None, description="Error message")


class TaskMCPCreate(TaskMCPBase):
    """Schema for creating task MCP."""
    pass


class TaskMCPUpdate(BaseModel):
    """Schema for updating task MCP."""
    status: Optional[str] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


class TaskMCPResponse(TaskMCPBase):
    """Schema for task MCP response."""
    id: str = Field(..., description="Task MCP ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# MCPContextStorage Schemas
class MCPContextStorageBase(BaseModel):
    """Base schema for MCP context storage."""
    server_id: str = Field(..., description="MCP server ID")
    context_key: str = Field(..., description="Context key")
    context_data: Dict[str, Any] = Field(..., description="Context data")
    expires_at: Optional[datetime] = Field(None, description="Context expiration")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    user_id: Optional[str] = Field(None, description="User ID")


class MCPContextStorageCreate(MCPContextStorageBase):
    """Schema for creating MCP context storage."""
    pass


class MCPContextStorageUpdate(BaseModel):
    """Schema for updating MCP context storage."""
    context_data: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None


class MCPContextStorageResponse(MCPContextStorageBase):
    """Schema for MCP context storage response."""
    id: str = Field(..., description="Context storage ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Update timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# MCPTool Schemas
class MCPToolBase(BaseModel):
    """Base schema for MCP tool."""
    server_id: str = Field(..., description="MCP server ID")
    tool_name: str = Field(..., description="Tool name")
    tool_description: Optional[str] = Field(None, description="Tool description")
    input_schema: Dict[str, Any] = Field(default_factory=dict, description="Input schema")
    output_schema: Dict[str, Any] = Field(default_factory=dict, description="Output schema")
    is_enabled: bool = Field(True, description="Is tool enabled")
    rate_limit: Optional[int] = Field(None, description="Rate limit per minute")


class MCPToolCreate(MCPToolBase):
    """Schema for creating MCP tool."""
    pass


class MCPToolUpdate(BaseModel):
    """Schema for updating MCP tool."""
    tool_description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    is_enabled: Optional[bool] = None
    rate_limit: Optional[int] = None


class MCPToolResponse(MCPToolBase):
    """Schema for MCP tool response."""
    id: str = Field(..., description="Tool ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# MCPInvocation Schemas
class MCPInvocationBase(BaseModel):
    """Base schema for MCP invocation."""
    server_id: str = Field(..., description="MCP server ID")
    tool_id: Optional[str] = Field(None, description="Tool ID")
    user_id: str = Field(..., description="User ID")
    tenant_id: Optional[str] = Field(None, description="Tenant ID")
    invocation_type: str = Field(..., description="Invocation type")
    request_data: Dict[str, Any] = Field(..., description="Request data")
    response_data: Optional[Dict[str, Any]] = Field(None, description="Response data")
    status: str = Field("pending", description="Invocation status")
    error_message: Optional[str] = Field(None, description="Error message")
    duration_ms: Optional[float] = Field(None, description="Duration in milliseconds")


class MCPInvocationCreate(MCPInvocationBase):
    """Schema for creating MCP invocation."""
    pass


class MCPInvocationUpdate(BaseModel):
    """Schema for updating MCP invocation."""
    response_data: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    error_message: Optional[str] = None
    duration_ms: Optional[float] = None


class MCPInvocationResponse(MCPInvocationBase):
    """Schema for MCP invocation response."""
    id: str = Field(..., description="Invocation ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    completed_at: Optional[datetime] = Field(None, description="Completion timestamp")
    
    class Config:
        from_attributes = True