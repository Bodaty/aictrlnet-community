"""Agent Performance schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums

class IssueSeverity(str, Enum):
    """Issue severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class IssueType(str, Enum):
    """Performance issue types."""
    HIGH_MEMORY = "high_memory"
    HIGH_CPU = "high_cpu"
    SLOW_RESPONSE = "slow_response"
    HIGH_ERROR_RATE = "high_error_rate"
    CONNECTION_FAILURE = "connection_failure"
    RESOURCE_EXHAUSTED = "resource_exhausted"


# Performance metrics schemas

class PerformanceMetrics(BaseModel):
    """Agent performance metrics."""
    execution_count: int
    avg_execution_time_ms: float
    success_rate: float
    error_rate: float
    throughput_per_minute: float
    active_time_percentage: float
    memory_usage_mb: float
    cpu_usage_percentage: float
    last_active: str
    uptime_hours: float


class AgentPerformanceItem(BaseModel):
    """Individual agent performance data."""
    id: str
    name: str
    type: str
    metrics: PerformanceMetrics
    status: str


class SystemMetrics(BaseModel):
    """System-wide performance metrics."""
    total_executions: int
    avg_execution_time_ms: float
    overall_success_rate: float
    total_errors: int
    peak_throughput_per_minute: float
    avg_throughput_per_minute: float
    total_memory_usage_mb: float
    total_cpu_usage_percentage: float
    active_agents: int
    total_agents: int


class AgentPerformanceResponse(BaseModel):
    """Agent performance response."""
    agents: List[AgentPerformanceItem]
    system_metrics: SystemMetrics


# Resource usage schemas

class ResourceMetrics(BaseModel):
    """Resource usage metrics."""
    cpu: float
    memory: float
    network: float
    disk: Optional[float] = None


class ResourceUsagePoint(BaseModel):
    """Resource usage data point."""
    timestamp: str
    # Dynamic agent IDs as keys with ResourceMetrics as values
    # This is handled in the actual response

    class Config:
        extra = "allow"  # Allow dynamic fields


class ResourceUsageResponse(BaseModel):
    """Resource usage response."""
    time_range: str
    data_points: List[Dict[str, Any]]  # List of ResourceUsagePoint-like dicts
    agents: List[str]


# Performance issue schemas

class PerformanceIssueResponse(BaseModel):
    """Performance issue response."""
    id: str
    agent_id: str
    agent_name: str
    type: str
    severity: str
    description: str
    timestamp: str
    metrics: Dict[str, Any]
    resolved: Optional[bool] = False


class PerformanceIssuesSummary(BaseModel):
    """Performance issues summary."""
    critical: int
    warning: int
    info: int


class PerformanceIssuesResponse(BaseModel):
    """Performance issues list response."""
    issues: List[PerformanceIssueResponse]
    total: int
    time_range: str
    summary: PerformanceIssuesSummary


# Performance comparison schemas

class PerformanceSummary(BaseModel):
    """Agent performance summary."""
    avg_response_time: float
    total_throughput: int
    error_rate: float
    avg_cpu_usage: float
    avg_memory_usage: float
    execution_count: int


class AgentComparisonItem(BaseModel):
    """Agent comparison data."""
    id: str
    name: str
    type: str
    performance_summary: PerformanceSummary


class MetricsComparison(BaseModel):
    """Metrics comparison data."""
    response_time: List[Dict[str, Any]]
    throughput: List[Dict[str, Any]]
    error_rate: List[Dict[str, Any]]
    cpu_usage: List[Dict[str, Any]]
    memory_usage: List[Dict[str, Any]]


class PerformanceComparisonRequest(BaseModel):
    """Performance comparison request."""
    agent_ids: List[str] = Field(..., min_items=2, max_items=5)
    time_range: str = Field("24h", description="Time range: 1h, 24h, 7d")


class PerformanceComparisonResponse(BaseModel):
    """Performance comparison response."""
    agents: List[AgentComparisonItem]
    metrics: Dict[str, List[Dict[str, Any]]]
    time_range: str


# Benchmark schemas

class BenchmarkCreate(BaseModel):
    """Benchmark creation request."""
    name: str = Field(..., description="Benchmark name")
    description: Optional[str] = Field(None, description="Benchmark description")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Benchmark metrics")


class BenchmarkResponse(BaseModel):
    """Benchmark response."""
    id: str
    name: str
    description: Optional[str] = None
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class BenchmarkComparison(BaseModel):
    """Benchmark comparison result."""
    agent_id: str
    benchmark_id: str
    benchmark_name: str
    comparison: Dict[str, Dict[str, Any]]


# Tracking schemas

class ExecutionTrackRequest(BaseModel):
    """Execution tracking request."""
    agent_id: str
    execution_time_ms: int
    success: bool
    error_code: Optional[str] = None


class ResourceTrackRequest(BaseModel):
    """Resource tracking request."""
    agent_id: str
    cpu_usage: float
    memory_usage: float
    network_usage: float
    disk_usage: Optional[float] = None