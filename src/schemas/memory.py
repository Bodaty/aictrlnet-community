"""Memory schemas."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Memory store schemas

class MemoryStoreResponse(BaseModel):
    """Memory store response."""
    stores: List[Dict[str, Any]]
    total: int


# Memory context schemas

class MemoryContextResponse(BaseModel):
    """Memory context response."""
    context_id: str
    entries: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]


# Memory entry schemas

class MemoryEntryCreate(BaseModel):
    """Memory entry creation request."""
    key: str = Field(..., description="Memory key")
    value: Any = Field(..., description="Value to store")
    scope: Optional[str] = Field("global", description="Scope (global, user, workflow, component)")
    scope_id: Optional[str] = Field(None, description="Scope ID")
    expiration: Optional[int] = Field(None, description="Expiration time in seconds")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class MemoryEntryUpdate(BaseModel):
    """Memory entry update request."""
    value: Optional[Any] = Field(None, description="New value")
    scope: Optional[str] = Field(None, description="New scope")
    scope_id: Optional[str] = Field(None, description="New scope ID")
    ttl_seconds: Optional[int] = Field(None, description="New TTL in seconds")


class MemoryEntryResponse(BaseModel):
    """Memory entry response."""
    key: str
    value: Any
    version: int
    owner: str
    scope: str
    scope_id: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    expires_at: Optional[str] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Bulk operations schemas

class BulkMemoryEntry(BaseModel):
    """Bulk memory entry."""
    key: str = Field(..., description="Entry key")
    value: Any = Field(..., description="Entry value")
    scope: Optional[str] = Field("global", description="Entry scope")
    scope_id: Optional[str] = Field(None, description="Scope ID")


class BulkMemoryRequest(BaseModel):
    """Bulk memory operation request."""
    entries: List[BulkMemoryEntry]


class BulkMemoryResponse(BaseModel):
    """Bulk memory operation response."""
    entries: List[MemoryEntryResponse]
    success_count: int
    error_count: int


# Search schemas

class MemorySearchRequest(BaseModel):
    """Memory search request."""
    pattern: Optional[str] = Field(None, description="Key pattern to search")
    scope: Optional[str] = Field(None, description="Filter by scope")
    owner: Optional[str] = Field(None, description="Filter by owner")
    limit: int = Field(100, description="Maximum results", ge=1, le=1000)


class MemorySearchResponse(BaseModel):
    """Memory search response."""
    entries: List[MemoryEntryResponse]
    total: int


# Memory statistics

class MemoryStatsResponse(BaseModel):
    """Memory statistics response."""
    total_entries: int
    total_size_bytes: int
    entries_by_scope: Dict[str, int]
    entries_by_owner: Dict[str, int]
    expired_entries: int