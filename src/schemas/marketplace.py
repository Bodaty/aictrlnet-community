"""Community Marketplace schemas for browse, search, install, and review."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ── Item Schemas ─────────────────────────────────────────────────────────────


class MarketplaceItemCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Item display name")
    description: Optional[str] = Field(None, description="Full description (Markdown supported)")
    short_description: Optional[str] = Field(None, max_length=500, description="One-line summary")
    category: str = Field(..., description="Category: workflow, template, adapter, agent")
    item_type: Optional[str] = Field(None, max_length=100, description="Sub-type within category")
    version: str = Field(default="1.0.0", max_length=50, description="Semantic version")
    tags: List[str] = Field(default_factory=list, description="Searchable tags")
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema for configuration")
    visibility: str = Field(default="public", description="Visibility: public, private, org")
    resource_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        allowed = {"workflow", "template", "adapter", "agent"}
        if v not in allowed:
            raise ValueError(f"category must be one of {allowed}")
        return v

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, v: str) -> str:
        allowed = {"public", "private", "org"}
        if v not in allowed:
            raise ValueError(f"visibility must be one of {allowed}")
        return v


class MarketplaceItemUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    short_description: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = None
    item_type: Optional[str] = None
    version: Optional[str] = None
    tags: Optional[List[str]] = None
    config_schema: Optional[Dict[str, Any]] = None
    visibility: Optional[str] = None
    status: Optional[str] = None
    resource_metadata: Optional[Dict[str, Any]] = None

    @field_validator("category")
    @classmethod
    def validate_category(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = {"workflow", "template", "adapter", "agent"}
            if v not in allowed:
                raise ValueError(f"category must be one of {allowed}")
        return v

    @field_validator("visibility")
    @classmethod
    def validate_visibility(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = {"public", "private", "org"}
            if v not in allowed:
                raise ValueError(f"visibility must be one of {allowed}")
        return v

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            allowed = {"draft", "published", "archived"}
            if v not in allowed:
                raise ValueError(f"status must be one of {allowed}")
        return v


class MarketplaceItemResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    short_description: Optional[str] = None
    author_id: str
    author_name: str
    category: str
    item_type: Optional[str] = None
    version: str
    tags: List[str] = Field(default_factory=list)
    config_schema: Dict[str, Any] = Field(default_factory=dict)
    install_count: int = 0
    rating_avg: Optional[float] = None
    rating_count: int = 0
    status: str
    visibility: str
    resource_metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MarketplaceItemListResponse(BaseModel):
    items: List[MarketplaceItemResponse]
    total: int
    limit: int
    offset: int


# ── Search ───────────────────────────────────────────────────────────────────


class MarketplaceSearchRequest(BaseModel):
    query: Optional[str] = Field(None, description="Free-text search query")
    category: Optional[str] = Field(None, description="Filter by category")
    tags: Optional[List[str]] = Field(None, description="Filter by tags (AND)")
    sort_by: str = Field(default="rating_avg", description="Sort field: rating_avg, install_count, created_at, name")
    sort_order: str = Field(default="desc", description="Sort order: asc, desc")
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


# ── Review Schemas ───────────────────────────────────────────────────────────


class MarketplaceReviewCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating from 1 to 5")
    comment: Optional[str] = Field(None, description="Optional review comment")


class MarketplaceReviewResponse(BaseModel):
    id: str
    item_id: str
    user_id: str
    rating: int
    comment: Optional[str] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MarketplaceReviewListResponse(BaseModel):
    reviews: List[MarketplaceReviewResponse]
    total: int


# ── Installation Schemas ─────────────────────────────────────────────────────


class MarketplaceInstallRequest(BaseModel):
    organization_id: Optional[str] = Field(None, description="Install for an organization (optional)")
    config: Dict[str, Any] = Field(default_factory=dict, description="Installation configuration")


class MarketplaceInstallResponse(BaseModel):
    id: str
    item_id: str
    user_id: str
    organization_id: Optional[str] = None
    version: str
    status: str
    installed_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class MarketplaceInstallationListResponse(BaseModel):
    installations: List[MarketplaceInstallResponse]
    total: int


class MarketplaceUninstallResponse(BaseModel):
    id: str
    item_id: str
    status: str = "uninstalled"
    uninstalled_at: Optional[datetime] = None
