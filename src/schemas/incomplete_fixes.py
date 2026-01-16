"""Complete schemas for Community models with incomplete sets."""

from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict


# SubscriptionPlan Complete Schemas
class SubscriptionPlanBase(BaseModel):
    """Base schema for subscription plan."""
    name: str = Field(..., description="Plan name")
    edition: str = Field(..., description="Plan edition")
    price_monthly: float = Field(..., description="Monthly price")
    price_yearly: float = Field(..., description="Yearly price")
    features: Dict[str, Any] = Field(default_factory=dict, description="Plan features")
    limits: Dict[str, Any] = Field(default_factory=dict, description="Plan limits")
    is_active: bool = Field(True, description="Is plan active")
    trial_days: int = Field(0, description="Trial period in days")


class SubscriptionPlanCreate(SubscriptionPlanBase):
    """Schema for creating subscription plan."""
    pass


class SubscriptionPlanUpdate(BaseModel):
    """Schema for updating subscription plan."""
    name: Optional[str] = None
    price_monthly: Optional[float] = None
    price_yearly: Optional[float] = None
    features: Optional[Dict[str, Any]] = None
    limits: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    trial_days: Optional[int] = None


# Subscription Complete Base Schema
class SubscriptionBase(BaseModel):
    """Base schema for subscription."""
    user_id: str = Field(..., description="User ID")
    plan_id: str = Field(..., description="Plan ID")
    status: str = Field("active", description="Subscription status")
    payment_method_id: Optional[str] = Field(None, description="Payment method ID")
    start_date: datetime = Field(..., description="Start date")
    end_date: Optional[datetime] = Field(None, description="End date")
    auto_renew: bool = Field(True, description="Auto-renew enabled")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Subscription metadata")


# IAMMessage Update Schema
class IAMMessageUpdate(BaseModel):
    """Schema for updating IAM message."""
    status: Optional[str] = None
    processed_at: Optional[datetime] = None
    response_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# IAMEventLog Update Schema
class IAMEventLogUpdate(BaseModel):
    """Schema for updating IAM event log."""
    event_data: Optional[Dict[str, Any]] = None
    processed: Optional[bool] = None
    processing_result: Optional[Dict[str, Any]] = None


# IAMMetric Update Schema
class IAMMetricUpdate(BaseModel):
    """Schema for updating IAM metric."""
    metric_value: Optional[float] = None
    dimensions: Optional[Dict[str, Any]] = None
    aggregated: Optional[bool] = None


# TenantLimitOverride Update Schema  
class TenantLimitOverrideUpdate(BaseModel):
    """Schema for updating tenant limit override."""
    limit_value: Optional[float] = None
    reason: Optional[str] = None
    expires_at: Optional[datetime] = None
    is_active: Optional[bool] = None


# FeatureTrial Update Schema
class FeatureTrialUpdate(BaseModel):
    """Schema for updating feature trial."""
    end_date: Optional[datetime] = None
    is_active: Optional[bool] = None
    converted: Optional[bool] = None
    conversion_date: Optional[datetime] = None


# LicenseCache Update Schema
class LicenseCacheUpdate(BaseModel):
    """Schema for updating license cache."""
    license_data: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None
    is_valid: Optional[bool] = None


# PlatformWebhookDelivery Update Schema
class PlatformWebhookDeliveryUpdate(BaseModel):
    """Schema for updating platform webhook delivery."""
    status: Optional[str] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    delivered_at: Optional[datetime] = None
    retry_count: Optional[int] = None
    next_retry_at: Optional[datetime] = None


# APIKeyLog Update Schema
class APIKeyLogUpdate(BaseModel):
    """Schema for updating API key log."""
    response_status: Optional[int] = None
    response_time_ms: Optional[float] = None
    error_message: Optional[str] = None


# WorkflowDefinition Update Schema
class WorkflowDefinitionUpdate(BaseModel):
    """Schema for updating workflow definition."""
    name: Optional[str] = None
    description: Optional[str] = None
    template_data: Optional[Dict[str, Any]] = None
    version: Optional[int] = None
    is_active: Optional[bool] = None
    updated_by: Optional[str] = None


# MCPServerCapability Update Schema
class MCPServerCapabilityUpdate(BaseModel):
    """Schema for updating MCP server capability."""
    name: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None
    parameters_schema: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


# MCPInvocation Update Schema
class MCPInvocationUpdate(BaseModel):
    """Schema for updating MCP invocation."""
    status: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[float] = None


# MCPServer Complete Schemas
class MCPServerBase(BaseModel):
    """Base schema for MCP server."""
    name: str = Field(..., description="Server name")
    url: str = Field(..., description="Server URL")
    description: Optional[str] = Field(None, description="Server description")
    is_active: bool = Field(True, description="Is server active")
    capabilities: List[str] = Field(default_factory=list, description="Server capabilities")
    auth_config: Dict[str, Any] = Field(default_factory=dict, description="Authentication config")


class MCPServerCreate(MCPServerBase):
    """Schema for creating MCP server."""
    pass


class MCPServerUpdate(BaseModel):
    """Schema for updating MCP server."""
    name: Optional[str] = None
    url: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    capabilities: Optional[List[str]] = None
    auth_config: Optional[Dict[str, Any]] = None


# MCPTool Complete Schemas
class MCPToolBase(BaseModel):
    """Base schema for MCP tool."""
    server_id: str = Field(..., description="Server ID")
    name: str = Field(..., description="Tool name")
    description: Optional[str] = Field(None, description="Tool description")
    input_schema: Dict[str, Any] = Field(..., description="Input schema")
    output_schema: Dict[str, Any] = Field(..., description="Output schema")
    is_active: bool = Field(True, description="Is tool active")


class MCPToolCreate(MCPToolBase):
    """Schema for creating MCP tool."""
    pass


class MCPToolUpdate(BaseModel):
    """Schema for updating MCP tool."""
    name: Optional[str] = None
    description: Optional[str] = None
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


# MCPContextStorage Update Schema
class MCPContextStorageUpdate(BaseModel):
    """Schema for updating MCP context storage."""
    context_data: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None
    accessed_count: Optional[int] = None
    last_accessed_at: Optional[datetime] = None


# BridgeConnection Complete Schemas
class BridgeConnectionBase(BaseModel):
    """Base schema for bridge connection."""
    name: str = Field(..., description="Connection name")
    bridge_type: str = Field(..., description="Bridge type")
    source_config: Dict[str, Any] = Field(..., description="Source configuration")
    target_config: Dict[str, Any] = Field(..., description="Target configuration")
    is_active: bool = Field(True, description="Is connection active")
    sync_config: Dict[str, Any] = Field(default_factory=dict, description="Sync configuration")


class BridgeConnectionCreate(BridgeConnectionBase):
    """Schema for creating bridge connection."""
    pass


class BridgeConnectionUpdate(BaseModel):
    """Schema for updating bridge connection."""
    name: Optional[str] = None
    source_config: Optional[Dict[str, Any]] = None
    target_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None
    sync_config: Optional[Dict[str, Any]] = None


# BridgeSync Update Schema
class BridgeSyncUpdate(BaseModel):
    """Schema for updating bridge sync."""
    status: Optional[str] = None
    records_synced: Optional[int] = None
    error_message: Optional[str] = None
    completed_at: Optional[datetime] = None


# WorkflowTemplateUsage Update Schema
class WorkflowTemplateUsageUpdate(BaseModel):
    """Schema for updating workflow template usage."""
    execution_count: Optional[int] = None
    last_executed_at: Optional[datetime] = None
    success_count: Optional[int] = None
    failure_count: Optional[int] = None


# WorkflowTemplatePermission Update Schema
class WorkflowTemplatePermissionUpdate(BaseModel):
    """Schema for updating workflow template permission."""
    permission_type: Optional[str] = None
    granted_by: Optional[str] = None
    expires_at: Optional[datetime] = None


# WorkflowTemplateReview Update Schema
class WorkflowTemplateReviewUpdate(BaseModel):
    """Schema for updating workflow template review."""
    rating: Optional[int] = None
    comment: Optional[str] = None
    is_approved: Optional[bool] = None


# UsageTracking Update Schema
class UsageTrackingUpdate(BaseModel):
    """Schema for updating usage tracking."""
    usage_count: Optional[int] = None
    last_used_at: Optional[datetime] = None
    quota_remaining: Optional[float] = None


# PaymentMethod Update Schema
class PaymentMethodUpdate(BaseModel):
    """Schema for updating payment method."""
    is_default: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None
    expires_at: Optional[datetime] = None


# BillingHistory Update Schema  
class BillingHistoryUpdate(BaseModel):
    """Schema for updating billing history."""
    status: Optional[str] = None
    paid_at: Optional[datetime] = None
    payment_method_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


# BillingEvent Update Schema
class BillingEventUpdate(BaseModel):
    """Schema for updating billing event."""
    processed: Optional[bool] = None
    processed_at: Optional[datetime] = None
    billing_history_id: Optional[str] = None
    error_message: Optional[str] = None


# Webhook Complete Schemas
class WebhookBase(BaseModel):
    """Base schema for webhook."""
    name: str = Field(..., description="Webhook name")
    url: str = Field(..., description="Webhook URL")
    events: List[str] = Field(..., description="Events to trigger webhook")
    is_active: bool = Field(True, description="Is webhook active")
    secret: Optional[str] = Field(None, description="Webhook secret for verification")
    headers: Dict[str, str] = Field(default_factory=dict, description="Custom headers")
    retry_config: Dict[str, Any] = Field(default_factory=dict, description="Retry configuration")


# WebhookDelivery Complete Schemas  
class WebhookDeliveryBase(BaseModel):
    """Base schema for webhook delivery."""
    webhook_id: str = Field(..., description="Webhook ID")
    event_type: str = Field(..., description="Event type")
    payload: Dict[str, Any] = Field(..., description="Delivery payload")
    status: str = Field("pending", description="Delivery status")
    attempt_count: int = Field(0, description="Delivery attempt count")
    response_code: Optional[int] = Field(None, description="HTTP response code")
    response_body: Optional[str] = Field(None, description="Response body")
    error_message: Optional[str] = Field(None, description="Error message")


class WebhookDeliveryCreate(WebhookDeliveryBase):
    """Schema for creating webhook delivery."""
    pass


class WebhookDeliveryUpdate(BaseModel):
    """Schema for updating webhook delivery."""
    status: Optional[str] = None
    attempt_count: Optional[int] = None
    response_code: Optional[int] = None
    response_body: Optional[str] = None
    error_message: Optional[str] = None
    delivered_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None


# UsageMetric Complete Schemas
class UsageMetricBase(BaseModel):
    """Base schema for usage metric."""
    user_id: str = Field(..., description="User ID")
    metric_type: str = Field(..., description="Metric type")
    metric_value: float = Field(..., description="Metric value")
    period: str = Field(..., description="Usage period")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metric metadata")


class UsageMetricCreate(UsageMetricBase):
    """Schema for creating usage metric."""
    pass


class UsageMetricUpdate(BaseModel):
    """Schema for updating usage metric."""
    metric_value: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


# UpgradePrompt Complete Schemas
class UpgradePromptBase(BaseModel):
    """Base schema for upgrade prompt."""
    user_id: str = Field(..., description="User ID")
    prompt_type: str = Field(..., description="Prompt type")
    feature_name: str = Field(..., description="Feature that triggered prompt")
    target_plan: str = Field(..., description="Target plan for upgrade")
    shown_at: datetime = Field(..., description="When prompt was shown")
    dismissed: bool = Field(False, description="Was prompt dismissed")
    converted: bool = Field(False, description="Did user convert")


class UpgradePromptCreate(UpgradePromptBase):
    """Schema for creating upgrade prompt."""
    pass


class UpgradePromptUpdate(BaseModel):
    """Schema for updating upgrade prompt."""
    dismissed: Optional[bool] = None
    converted: Optional[bool] = None
    conversion_date: Optional[datetime] = None


# UsageSummary Complete Schemas
class UsageSummaryBase(BaseModel):
    """Base schema for usage summary."""
    user_id: str = Field(..., description="User ID")
    period: str = Field(..., description="Summary period")
    start_date: datetime = Field(..., description="Period start date")
    end_date: datetime = Field(..., description="Period end date")
    metrics: Dict[str, float] = Field(default_factory=dict, description="Usage metrics")
    cost: Optional[float] = Field(None, description="Total cost")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Summary metadata")


class UsageSummaryCreate(UsageSummaryBase):
    """Schema for creating usage summary."""
    pass


class UsageSummaryUpdate(BaseModel):
    """Schema for updating usage summary."""
    metrics: Optional[Dict[str, float]] = None
    cost: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


# WorkflowExecution Complete Schemas
class WorkflowExecutionBase(BaseModel):
    """Base schema for workflow execution."""
    workflow_id: str = Field(..., description="Workflow ID")
    status: str = Field("pending", description="Execution status")
    started_at: datetime = Field(..., description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")
    error_message: Optional[str] = Field(None, description="Error message")


# NodeExecution Complete Schemas
class NodeExecutionBase(BaseModel):
    """Base schema for node execution."""
    workflow_execution_id: str = Field(..., description="Workflow execution ID")
    node_id: str = Field(..., description="Node ID")
    node_type: str = Field(..., description="Node type")
    status: str = Field("pending", description="Node status")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data")


class NodeExecutionCreate(NodeExecutionBase):
    """Schema for creating node execution."""
    pass


# WorkflowCheckpoint Complete Schemas
class WorkflowCheckpointBase(BaseModel):
    """Base schema for workflow checkpoint."""
    workflow_execution_id: str = Field(..., description="Workflow execution ID")
    checkpoint_name: str = Field(..., description="Checkpoint name")
    checkpoint_data: Dict[str, Any] = Field(..., description="Checkpoint data")
    created_at: datetime = Field(..., description="Creation time")


class WorkflowCheckpointUpdate(BaseModel):
    """Schema for updating workflow checkpoint."""
    checkpoint_data: Optional[Dict[str, Any]] = None
    restored: Optional[bool] = None
    restored_at: Optional[datetime] = None


# WorkflowTrigger Base Schema
class WorkflowTriggerBase(BaseModel):
    """Base schema for workflow trigger."""
    workflow_id: str = Field(..., description="Workflow ID")
    trigger_type: str = Field(..., description="Trigger type")
    trigger_config: Dict[str, Any] = Field(..., description="Trigger configuration")
    is_active: bool = Field(True, description="Is trigger active")
    last_triggered: Optional[datetime] = Field(None, description="Last trigger time")


# WorkflowSchedule Base Schema
class WorkflowScheduleBase(BaseModel):
    """Base schema for workflow schedule."""
    workflow_id: str = Field(..., description="Workflow ID")
    schedule_name: str = Field(..., description="Schedule name")
    cron_expression: str = Field(..., description="Cron expression")
    timezone: str = Field("UTC", description="Timezone")
    is_active: bool = Field(True, description="Is schedule active")
    next_run: Optional[datetime] = Field(None, description="Next scheduled run")