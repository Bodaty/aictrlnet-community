"""Platform Integration schemas for Community Edition"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class PlatformType(str, Enum):
    """Supported automation platforms"""
    N8N = "n8n"
    ZAPIER = "zapier"
    MAKE = "make"
    POWER_AUTOMATE = "power_automate"
    IFTTT = "ifttt"
    CUSTOM = "custom"


class AuthMethod(str, Enum):
    """Platform authentication methods"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    TOKEN = "token"
    CUSTOM = "custom"


class WorkflowStatus(str, Enum):
    """Platform workflow status"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DRAFT = "draft"
    ERROR = "error"
    SUSPENDED = "suspended"


class ExecutionStatus(str, Enum):
    """Platform execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


# Request/Response schemas for Platform Credentials
class PlatformCredentialBase(BaseModel):
    """Base platform credential schema"""
    name: str = Field(..., max_length=255, description="Credential name")
    platform: PlatformType = Field(..., description="Platform type")
    auth_method: AuthMethod = Field(..., description="Authentication method")
    is_shared: bool = Field(False, description="Share with team members")
    config_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional configuration")


class PlatformCredentialCreate(PlatformCredentialBase):
    """Create platform credential request"""
    credentials: Dict[str, Any] = Field(..., description="Credential data (will be encrypted)")


class PlatformCredentialUpdate(BaseModel):
    """Update platform credential request"""
    name: Optional[str] = Field(None, max_length=255)
    is_shared: Optional[bool] = None
    credentials: Optional[Dict[str, Any]] = Field(None, description="New credential data")
    config_metadata: Optional[Dict[str, Any]] = None


class PlatformCredentialResponse(PlatformCredentialBase):
    """Platform credential response"""
    id: int
    user_id: str
    created_at: datetime
    updated_at: datetime
    last_used_at: Optional[datetime] = None
    execution_count: int = 0
    last_error: Optional[str] = None
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


# Request/Response schemas for Platform Executions
class PlatformExecutionCreate(BaseModel):
    """Create platform execution request"""
    workflow_id: str = Field(..., description="AICtrlNet workflow ID")
    node_id: str = Field(..., description="Node ID in workflow")
    platform: PlatformType = Field(..., description="Target platform")
    external_workflow_id: str = Field(..., description="Workflow ID on external platform")
    credential_id: int = Field(..., description="Credential to use")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for platform")
    execution_metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class PlatformExecutionUpdate(BaseModel):
    """Update platform execution"""
    external_execution_id: Optional[str] = None
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: Optional[str] = None
    estimated_cost: Optional[int] = None


class PlatformExecutionResponse(BaseModel):
    """Platform execution response"""
    id: int
    workflow_id: str
    node_id: str
    execution_id: str
    platform: PlatformType
    external_workflow_id: str
    external_execution_id: Optional[str] = None
    credential_id: Optional[int] = None
    input_data: Dict[str, Any]
    output_data: Optional[Dict[str, Any]] = None
    error_data: Optional[Dict[str, Any]] = None
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: str
    estimated_cost: int = 0
    execution_metadata: Dict[str, Any]
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


# Request/Response schemas for Platform Adapters
class PlatformAdapterCreate(BaseModel):
    """Create platform adapter"""
    platform: PlatformType
    adapter_class: str = Field(..., description="Python class path")
    version: str = Field(..., max_length=50)
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    supported_auth_methods: List[AuthMethod] = Field(default_factory=list)
    is_active: bool = True
    is_beta: bool = False
    config_schema: Dict[str, Any] = Field(default_factory=dict, description="JSON Schema")
    documentation_url: Optional[str] = None
    icon_url: Optional[str] = None


class PlatformAdapterResponse(BaseModel):
    """Platform adapter response"""
    id: int
    platform: PlatformType
    adapter_class: str
    version: str
    capabilities: Dict[str, Any]
    supported_auth_methods: List[str]
    is_active: bool
    is_beta: bool
    config_schema: Dict[str, Any]
    documentation_url: Optional[str] = None
    icon_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


# Request/Response schemas for Platform Health
class PlatformHealthUpdate(BaseModel):
    """Update platform health status"""
    is_healthy: bool
    response_time_ms: Optional[int] = None
    last_error: Optional[str] = None
    health_metadata: Dict[str, Any] = Field(default_factory=dict)


class PlatformHealthResponse(BaseModel):
    """Platform health response"""
    id: int
    platform: PlatformType
    is_healthy: bool
    last_check_at: datetime
    response_time_ms: Optional[int] = None
    consecutive_failures: int = 0
    last_error: Optional[str] = None
    uptime_percentage: int = 100
    total_checks: int = 0
    failed_checks: int = 0
    avg_response_time_ms: Optional[int] = None
    p95_response_time_ms: Optional[int] = None
    health_metadata: Dict[str, Any]
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


# Workflow Node Configuration schemas
class PlatformNodeConfig(BaseModel):
    """Configuration for platform integration node"""
    platform: PlatformType
    workflow_id: str = Field(..., description="External workflow ID")
    workflow_name: Optional[str] = Field(None, description="Human readable name")
    credential_id: int = Field(..., description="Platform credential ID")
    input_mapping: Dict[str, Any] = Field(default_factory=dict, description="Map inputs to platform")
    output_mapping: Dict[str, Any] = Field(default_factory=dict, description="Map outputs from platform")
    timeout: int = Field(300, description="Execution timeout in seconds", ge=1, le=3600)
    retry_config: Dict[str, Any] = Field(
        default_factory=lambda: {"max_attempts": 3, "backoff_multiplier": 2},
        description="Retry configuration"
    )


# Discovery and capabilities
class PlatformCapabilities(BaseModel):
    """Platform capabilities information"""
    platform: PlatformType
    supports_webhooks: bool = False
    supports_scheduling: bool = False
    supports_versioning: bool = False
    supports_rollback: bool = False
    supports_monitoring: bool = True
    max_execution_time: int = 300  # seconds
    rate_limits: Dict[str, Any] = Field(default_factory=dict)
    available_triggers: List[str] = Field(default_factory=list)
    available_actions: List[str] = Field(default_factory=list)


class PlatformWorkflowInfo(BaseModel):
    """Information about a workflow on external platform"""
    workflow_id: str
    name: str
    description: Optional[str] = None
    status: WorkflowStatus = WorkflowStatus.ACTIVE
    trigger_type: str = "unknown"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


# UI-specific schemas for workflow discovery and configuration
class WorkflowBrowserRequest(BaseModel):
    """Request for browsing platform workflows"""
    platform: PlatformType
    credential_id: int
    search_term: Optional[str] = None
    page: int = Field(1, ge=1)
    page_size: int = Field(20, ge=1, le=100)
    status_filter: Optional[WorkflowStatus] = None
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


class WorkflowBrowserResponse(BaseModel):
    """Response for workflow browser"""
    workflows: List[PlatformWorkflowInfo]
    total: int
    page: int
    page_size: int
    has_more: bool
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


class PlatformNodeUIConfig(BaseModel):
    """UI configuration for platform node"""
    platform: PlatformType
    workflow_id: str
    workflow_name: str
    credential_id: int
    credential_name: str
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    icon_url: Optional[str] = None
    supports_test: bool = True
    test_data: Optional[Dict[str, Any]] = None
    input_fields: Optional[List[Dict[str, Any]]] = None
    output_fields: Optional[List[Dict[str, Any]]] = None
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


class TestExecutionRequest(BaseModel):
    """Request to test a platform workflow"""
    platform: PlatformType
    workflow_id: str
    credential_id: int
    test_inputs: Dict[str, Any] = Field(default_factory=dict)
    timeout: int = Field(30, ge=1, le=300)
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


class TestExecutionResponse(BaseModel):
    """Response from test execution"""
    success: bool
    execution_time_ms: int
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    logs: List[str] = Field(default_factory=list)
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


# Webhook schemas
class PlatformWebhookBase(BaseModel):
    """Base platform webhook schema"""
    webhook_key: str = Field(..., max_length=255, description="Unique webhook identifier")
    platform_type: PlatformType = Field(..., description="Platform type")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    webhook_config: Dict[str, Any] = Field(default_factory=dict, description="Webhook configuration")
    is_active: bool = Field(True, description="Webhook active status")


class PlatformWebhookCreate(PlatformWebhookBase):
    """Create platform webhook request"""
    pass


class PlatformWebhookUpdate(BaseModel):
    """Update platform webhook request"""
    workflow_id: Optional[str] = None
    webhook_config: Optional[Dict[str, Any]] = None
    is_active: Optional[bool] = None


class PlatformWebhookResponse(PlatformWebhookBase):
    """Platform webhook response"""
    id: int
    created_at: datetime
    updated_at: datetime
    last_triggered_at: Optional[datetime] = None
    trigger_count: int = 0
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    webhook_metadata: Dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )


class WebhookEventData(BaseModel):
    """Webhook event data"""
    webhook_key: str
    platform: PlatformType
    event_type: str
    event_data: Dict[str, Any]
    headers: Dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class WebhookEventType(str, Enum):
    """Types of webhook events"""
    EXECUTION_STARTED = "execution.started"
    EXECUTION_COMPLETED = "execution.completed"
    EXECUTION_FAILED = "execution.failed"
    EXECUTION_CANCELLED = "execution.cancelled"
    EXECUTION_WAITING = "execution.waiting"
    EXECUTION_WARNING = "execution.warning"
    WORKFLOW_CREATED = "workflow.created"
    WORKFLOW_UPDATED = "workflow.updated"
    WORKFLOW_DELETED = "workflow.deleted"
    RATE_LIMITED = "rate_limited"
    ERROR = "error"
    OTHER = "other"


class WebhookDeliveryStatus(str, Enum):
    """Status of webhook delivery"""
    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class PlatformWebhookCreate(BaseModel):
    """Create a new webhook endpoint"""
    platform: PlatformType
    webhook_url: str = Field(..., pattern=r"^https?://")
    events: List[WebhookEventType]
    secret: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "platform": "n8n",
                "webhook_url": "https://api.example.com/webhooks/n8n",
                "events": ["execution.completed", "execution.failed"],
                "secret": "optional-signing-secret",
                "metadata": {"description": "Production webhook"}
            }
        }


class PlatformWebhookUpdate(BaseModel):
    """Update webhook configuration"""
    webhook_url: Optional[str] = Field(None, pattern=r"^https?://")
    events: Optional[List[WebhookEventType]] = None
    is_active: Optional[bool] = None
    metadata: Optional[Dict[str, Any]] = None


class PlatformWebhookResponse(BaseModel):
    """Webhook response"""
    id: int
    platform: str
    webhook_url: str
    events: List[str]
    is_active: bool
    verified: bool
    user_id: Optional[str] = None
    
    # Stats
    last_triggered_at: Optional[datetime] = None
    total_deliveries: int = 0
    successful_deliveries: int = 0
    failed_deliveries: int = 0
    consecutive_failures: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, alias="webhook_metadata")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class WebhookDeliveryResponse(BaseModel):
    """Webhook delivery details"""
    id: int
    webhook_id: int
    event_type: str
    status: str
    attempts: int
    
    response_status: Optional[int] = None
    response_headers: Optional[Dict[str, str]] = None
    response_body: Optional[str] = None
    
    created_at: datetime
    delivered_at: Optional[datetime] = None
    next_retry_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())


class WebhookEventRequest(BaseModel):
    """Incoming webhook event from platform"""
    platform: PlatformType
    headers: Dict[str, str]
    body: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "platform": "n8n",
                "headers": {
                    "x-n8n-signature": "abc123",
                    "content-type": "application/json"
                },
                "body": {
                    "workflowId": "123",
                    "executionId": "456",
                    "status": "success"
                }
            }
        }


class WebhookVerificationRequest(BaseModel):
    """Request to verify webhook endpoint"""
    test_payload: Optional[Dict[str, Any]] = None