"""Analytics schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums

class AnalyticsTimeRange(str, Enum):
    """Analytics time range."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    CUSTOM = "custom"


class MetricType(str, Enum):
    """Metric type."""
    TASK_COMPLETION = "task_completion"
    WORKFLOW_EXECUTION = "workflow_execution"
    AGENT_PERFORMANCE = "agent_performance"
    RESOURCE_USAGE = "resource_usage"
    API_CALLS = "api_calls"
    ERROR_RATE = "error_rate"
    COST = "cost"
    CUSTOM = "custom"


class AggregationType(str, Enum):
    """Aggregation type."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    PERCENTILE = "percentile"
    NONE = "none"


# Query schemas

class AnalyticsQuery(BaseModel):
    """Analytics query request."""
    metric_type: MetricType = Field(..., description="Type of metric to query")
    time_range: AnalyticsTimeRange = Field(..., description="Time range for the query")
    start_date: Optional[datetime] = Field(None, description="Start date for custom range")
    end_date: Optional[datetime] = Field(None, description="End date for custom range")
    filters: Optional[Dict[str, Any]] = Field(None, description="Additional filters")
    group_by: Optional[List[str]] = Field(None, description="Fields to group by")
    aggregation: AggregationType = Field(AggregationType.NONE, description="Aggregation type")
    limit: Optional[int] = Field(100, description="Limit results")


class AnalyticsResponse(BaseModel):
    """Analytics query response."""
    query: AnalyticsQuery
    data: List[Dict[str, Any]]
    metadata: Dict[str, Any]


# Dashboard schemas

class DashboardMetrics(BaseModel):
    """Dashboard metrics summary."""
    tasks_total: int
    tasks_completed: int
    tasks_failed: int
    tasks_success_rate: float
    workflows_total: int
    workflows_active: int
    agents_total: int
    agents_active: int
    avg_task_duration_seconds: float
    avg_agent_success_rate: float
    avg_execution_time_ms: float
    last_updated: datetime


# Trend schemas

class TrendData(BaseModel):
    """Trend data point."""
    timestamp: datetime
    value: float
    metadata: Optional[Dict[str, Any]] = None


class TrendAnalysis(BaseModel):
    """Trend analysis response."""
    metric_type: MetricType
    time_range: AnalyticsTimeRange
    data_points: List[TrendData]
    trend_direction: str  # up, down, stable
    change_percentage: float
    forecast: Optional[List[TrendData]] = None


# Performance schemas

class PerformanceMetrics(BaseModel):
    """Performance metrics."""
    total_executions: int
    success_rate: float
    error_rate: float
    avg_execution_time_ms: float
    p95_execution_time_ms: float
    p99_execution_time_ms: float
    throughput_per_minute: float
    resource_efficiency: float
    uptime_percentage: float


# Usage schemas

class UsageMetrics(BaseModel):
    """Usage metrics."""
    users_count: int
    workflows_count: int
    tasks_count: int
    storage_gb: float
    api_calls_count: int
    compute_hours: float
    bandwidth_gb: float
    period_start: datetime
    period_end: datetime


# Cost schemas

class CostAnalytics(BaseModel):
    """Cost analytics."""
    total_cost: float
    compute_cost: float
    storage_cost: float
    network_cost: float
    api_cost: float
    cost_breakdown: Dict[str, float]
    cost_trends: List[TrendData]
    projected_monthly_cost: float
    cost_optimization_suggestions: List[str]


# Report schemas

class ReportConfig(BaseModel):
    """Report configuration."""
    name: str = Field(..., description="Report name")
    description: Optional[str] = Field(None, description="Report description")
    metrics: List[MetricType] = Field(..., description="Metrics to include")
    time_range: AnalyticsTimeRange = Field(..., description="Time range")
    schedule: Optional[str] = Field(None, description="Cron schedule")
    recipients: Optional[List[str]] = Field(None, description="Email recipients")
    format: str = Field("pdf", description="Report format (pdf, csv, json)")


class GeneratedReport(BaseModel):
    """Generated report."""
    id: str
    config: ReportConfig
    generated_at: datetime
    file_url: str
    file_size_bytes: int
    status: str  # completed, failed, processing


# Missing schemas for Enterprise endpoints

class MetricQuery(BaseModel):
    """Metric query request."""
    metric_type: MetricType
    time_range: AnalyticsTimeRange
    filters: Optional[Dict[str, Any]] = None
    group_by: Optional[List[str]] = None
    aggregation: AggregationType = AggregationType.NONE


class MetricResponse(BaseModel):
    """Metric response."""
    metric_name: str
    metric_type: str
    value: float
    tags: Optional[Dict[str, Any]] = None
    timestamp: datetime
    aggregation: Optional[str] = None


class InsightResponse(BaseModel):
    """Analytics insight response."""
    id: str
    title: str
    description: str
    severity: str  # high, medium, low
    metric_type: MetricType
    value: float
    change_percentage: float
    recommendation: Optional[str] = None
    created_at: datetime


class ReportRequest(BaseModel):
    """Report generation request."""
    report_type: str
    metrics: List[MetricType]
    time_range: AnalyticsTimeRange
    format: str = "pdf"
    filters: Optional[Dict[str, Any]] = None


class ReportResponse(BaseModel):
    """Report response."""
    id: str
    report_type: str
    status: str
    file_url: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None


class DashboardResponse(BaseModel):
    """Dashboard response."""
    total_tasks: int
    total_workflows: int
    active_tasks: int
    failed_tasks: int
    recent_activity: Dict[str, Any]
    top_adapters: List[Dict[str, Any]]
    system_health: Dict[str, Any]


class DashboardCreate(BaseModel):
    """Dashboard creation request."""
    name: str
    description: Optional[str] = None
    widgets: List[Dict[str, Any]]


class DashboardUpdate(BaseModel):
    """Dashboard update request."""
    name: Optional[str] = None
    description: Optional[str] = None
    widgets: Optional[List[Dict[str, Any]]] = None


class AlertResponse(BaseModel):
    """Alert response."""
    id: str
    name: str
    description: Optional[str] = None
    metric_type: MetricType
    condition: str
    threshold: float
    enabled: bool
    created_at: datetime
    updated_at: datetime


class AlertCreate(BaseModel):
    """Alert creation request."""
    name: str
    description: Optional[str] = None
    metric_type: MetricType
    condition: str  # gt, lt, eq, gte, lte
    threshold: float
    enabled: bool = True


class AlertUpdate(BaseModel):
    """Alert update request."""
    name: Optional[str] = None
    description: Optional[str] = None
    condition: Optional[str] = None
    threshold: Optional[float] = None
    enabled: Optional[bool] = None


# Alias for backward compatibility
MetricAggregation = AggregationType


class TimeSeriesData(BaseModel):
    """Time series data point."""
    timestamp: datetime
    value: float
    tags: Optional[Dict[str, str]] = None