"""Cache Management schemas for Community edition."""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime

class CacheStatsResponse(BaseModel):
    """Cache statistics response."""
    total_keys: int = Field(description="Total number of cache keys")
    total_size: int = Field(description="Total cache size in bytes")
    hit_rate: float = Field(description="Cache hit rate percentage")
    avg_age: int = Field(description="Average key age in milliseconds")
    memory_limit: int = Field(description="Memory limit in bytes")
    eviction_count: int = Field(description="Number of evicted keys")
    uptime: int = Field(description="Cache service uptime in seconds")

class CacheKeyResponse(BaseModel):
    """Cache key information response."""
    key: str = Field(description="Cache key name")
    type: str = Field(description="Key type category")
    size: int = Field(description="Key size in bytes")
    created_at: int = Field(description="Creation timestamp in milliseconds")
    hit_count: int = Field(description="Number of times key was accessed")
    expires_soon: bool = Field(description="Whether key expires soon")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")

class CacheKeyDetailsResponse(BaseModel):
    """Detailed cache key information response."""
    key: str = Field(description="Cache key name")
    value: str = Field(description="Key value (truncated if large)")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    created_at: int = Field(description="Creation timestamp in milliseconds")
    last_accessed: int = Field(description="Last access timestamp in milliseconds")
    access_count: int = Field(description="Total access count")
    size: int = Field(description="Key size in bytes")
    type: str = Field(description="Redis data type")

class CacheOperationResponse(BaseModel):
    """Cache operation result response."""
    success: bool = Field(description="Whether operation succeeded")
    message: str = Field(description="Operation result message")
    affected_keys: Optional[int] = Field(None, description="Number of keys affected")

class UpgradePromptResponse(BaseModel):
    """Upgrade prompt for Enterprise features."""
    error: str = Field(description="Error code")
    message: str = Field(description="Upgrade message")
    upgrade_url: str = Field(description="URL to upgrade page")
    benefits: List[str] = Field(description="List of Enterprise benefits")

class CacheSearchRequest(BaseModel):
    """Cache key search request."""
    pattern: Optional[str] = Field(None, description="Key pattern to search")
    limit: int = Field(100, description="Maximum results to return")
    offset: int = Field(0, description="Results offset for pagination")

class CacheKeyPattern(BaseModel):
    """Cache key pattern for bulk operations."""
    pattern: str = Field(description="Key pattern (supports wildcards)")
    count: int = Field(description="Number of matching keys")

class CacheBulkOperationRequest(BaseModel):
    """Bulk cache operation request."""
    patterns: List[str] = Field(description="List of key patterns")
    confirm: bool = Field(False, description="Confirmation flag")
    dry_run: bool = Field(False, description="Dry run mode")

class CacheBulkOperationResponse(BaseModel):
    """Bulk cache operation response."""
    success: bool = Field(description="Whether operation succeeded")
    message: str = Field(description="Operation result message")
    affected_keys: int = Field(description="Number of keys affected")
    patterns_processed: int = Field(description="Number of patterns processed")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")

class CacheExportRequest(BaseModel):
    """Cache export request."""
    patterns: List[str] = Field(description="Key patterns to export")
    format: str = Field("json", description="Export format (json, csv)")
    include_values: bool = Field(False, description="Include key values")
    compress: bool = Field(False, description="Compress export")

class CacheExportResponse(BaseModel):
    """Cache export response."""
    success: bool = Field(description="Whether export succeeded")
    download_url: str = Field(description="URL to download export")
    filename: str = Field(description="Export filename")
    size: int = Field(description="Export size in bytes")
    key_count: int = Field(description="Number of keys exported")
    expires_at: datetime = Field(description="Download URL expiration")

class CacheMetricsResponse(BaseModel):
    """Cache metrics response."""
    timestamp: datetime = Field(description="Metrics timestamp")
    total_keys: int = Field(description="Total keys at timestamp")
    memory_usage: int = Field(description="Memory usage in bytes")
    hit_rate: float = Field(description="Hit rate percentage")
    operations_per_second: float = Field(description="Operations per second")
    connections: int = Field(description="Active connections")
    
class CacheAnalyticsRequest(BaseModel):
    """Cache analytics request."""
    start_time: datetime = Field(description="Start time for analysis")
    end_time: datetime = Field(description="End time for analysis")
    granularity: str = Field("hour", description="Time granularity (minute, hour, day)")
    metrics: List[str] = Field(description="Metrics to include")

class CacheAnalyticsResponse(BaseModel):
    """Cache analytics response."""
    period: str = Field(description="Analysis period")
    metrics: List[CacheMetricsResponse] = Field(description="Time-series metrics")
    summary: Dict[str, Any] = Field(description="Summary statistics")
    recommendations: List[str] = Field(description="Optimization recommendations")