"""Pydantic schemas for Knowledge Service."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


# Knowledge Item Schemas

class KnowledgeItemBase(BaseModel):
    """Base schema for knowledge items."""
    item_type: str = Field(..., description="Type of knowledge item (template, agent, adapter, feature, endpoint)")
    category: str = Field(..., description="Category of the item")
    name: str = Field(..., description="Name of the item")
    description: str = Field(..., description="Description of the item")
    content: Dict[str, Any] = Field(..., description="Full structured content")
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")
    keywords: List[str] = Field(default_factory=list, description="Keywords for search")
    edition_required: str = Field(default="community", description="Minimum edition required")


class KnowledgeItemCreate(KnowledgeItemBase):
    """Schema for creating a knowledge item."""
    source_file: Optional[str] = None
    source_version: Optional[str] = None
    related_items: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list)


class KnowledgeItemUpdate(BaseModel):
    """Schema for updating a knowledge item."""
    description: Optional[str] = None
    content: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    relevance_score: Optional[float] = None
    is_active: Optional[bool] = None
    is_deprecated: Optional[bool] = None


class KnowledgeItemResponse(KnowledgeItemBase):
    """Response schema for a knowledge item."""
    id: UUID
    usage_count: int
    relevance_score: float
    success_rate: Optional[float]
    created_at: datetime
    updated_at: datetime
    last_accessed: Optional[datetime]
    is_active: bool
    is_deprecated: bool
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Knowledge Query Schemas

class KnowledgeQueryCreate(BaseModel):
    """Schema for creating a knowledge query."""
    query_text: str = Field(..., description="The query text")
    query_type: str = Field(..., description="Type of query (search, retrieve, suggest)")
    context: Dict[str, Any] = Field(default_factory=dict, description="Query context")
    session_id: Optional[UUID] = None


class KnowledgeQueryResponse(BaseModel):
    """Response schema for a knowledge query."""
    id: UUID
    query_text: str
    query_type: str
    context: Dict[str, Any]
    results_returned: List[UUID]
    result_count: int
    query_time_ms: Optional[int]
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# System Manifest Schemas

class SystemManifestResponse(BaseModel):
    """Response schema for system manifest."""
    id: UUID
    manifest_version: str
    manifest_type: str
    manifest_data: Dict[str, Any]
    statistics: Dict[str, Any]
    feature_count: int
    endpoint_count: int
    template_count: int
    agent_count: int
    adapter_count: int
    generated_at: datetime
    is_current: bool
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Knowledge Retrieval Schemas

class KnowledgeRetrievalRequest(BaseModel):
    """Request schema for knowledge retrieval."""
    query: str = Field(..., description="User query to search for")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    limit: int = Field(default=10, ge=1, le=50, description="Maximum number of results")
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional filters")


class KnowledgeRetrievalResponse(BaseModel):
    """Response schema for knowledge retrieval."""
    query: str
    items: List[KnowledgeItemResponse]
    total: int
    context_used: bool
    retrieval_time_ms: Optional[int] = None


# Knowledge Search Schemas

class KnowledgeSearchRequest(BaseModel):
    """Request schema for knowledge search."""
    query: str = Field(..., description="Search query")
    types: Optional[List[str]] = Field(None, description="Filter by types")
    categories: Optional[List[str]] = Field(None, description="Filter by categories")
    limit: int = Field(default=20, ge=1, le=100)


class KnowledgeSearchResponse(BaseModel):
    """Response schema for knowledge search."""
    query: str
    results: Dict[str, List[Dict[str, Any]]]  # Grouped by type
    total: int
    types_searched: List[str]
    ml_enhanced: Optional[bool] = False  # Business/Enterprise editions use ML


# Suggestion Schemas

class SuggestionRequest(BaseModel):
    """Request schema for getting suggestions."""
    current_action: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class SuggestionItem(BaseModel):
    """Individual suggestion item."""
    action: str
    description: str
    confidence: float
    category: str
    related_items: List[str] = Field(default_factory=list)


class SuggestionResponse(BaseModel):
    """Response schema for suggestions."""
    current_action: Optional[str]
    suggestions: List[SuggestionItem]
    total: int


# Learning Pattern Schemas

class LearnedPatternResponse(BaseModel):
    """Response schema for learned patterns."""
    id: UUID
    pattern_type: str
    pattern_signature: str
    pattern_data: Dict[str, Any]
    occurrence_count: int
    success_count: int
    confidence_score: float
    is_active: bool
    first_observed: datetime
    last_observed: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

# Capability Summary Schema

class CapabilitySummaryResponse(BaseModel):
    """Response schema for system capabilities summary."""
    templates: int = Field(..., description="Number of available templates")
    agents: int = Field(..., description="Number of available agents")
    adapters: int = Field(..., description="Number of available adapters")
    features: int = Field(..., description="Number of available features")
    endpoints: int = Field(..., description="Number of available endpoints")
    automation_coverage: float = Field(..., description="Percentage of tasks that can be automated")
    learning_status: str = Field(..., description="Current learning status")
    active_automations: int = Field(..., description="Number of currently active automations")
    user_edition: Optional[str] = Field(None, description="User's edition")
    personalized: bool = Field(default=False, description="Whether response is personalized")


# Feature Detail Schema

class FeatureDetailResponse(BaseModel):
    """Response schema for detailed feature information."""
    feature: Dict[str, Any]
    endpoints: List[Dict[str, str]]
    ui_locations: List[str]
    related_templates: List[Dict[str, Any]]
    examples: List[Dict[str, Any]]
    access_restricted: bool = False
    upgrade_required: Optional[str] = None


# Knowledge Stats Schema

class KnowledgeStatsResponse(BaseModel):
    """Response schema for knowledge statistics."""
    indexed_items: Dict[str, int]
    last_indexed: Optional[datetime]
    total_queries: int = 0
    popular_queries: List[str] = Field(default_factory=list)
    average_query_time_ms: Optional[float] = None
    cache_hit_rate: Optional[float] = None
