"""Database models for AICtrlNet FastAPI."""

from .user import User
from .conversation import (
    ConversationSession,
    ConversationMessage,
    ConversationAction,
    ConversationIntent,
    ConversationPattern,
)
from .knowledge import (
    KnowledgeItem,
    KnowledgeIndex,
    KnowledgeQuery,
    SystemManifest,
    LearnedPattern,
)
# Import directly from community_complete to avoid duplicate imports
from .community_complete import (
    Task,
    WorkflowDefinition,
    WorkflowInstance,
    WorkflowStep,
    MCPServer,
    MCPServerCapability,
    MCPTool,
    MCPInvocation,
    TaskMCP,
    MCPContextStorage,
    Adapter,
    BridgeConnection,
    BridgeSync,
    ResourcePoolConfig,
)

# Business and Enterprise models removed - they belong in their respective editions

from .enforcement import (
    UsageMetric as EnforcementUsageMetric,
    TenantLimitOverride,
    FeatureTrial,
    UpgradePrompt,
    LicenseCache,
    BillingEvent,
    UsageSummary,
)

from .subscription import (
    SubscriptionPlan,
    Subscription,
    UsageTracking,
    PaymentMethod,
    BillingHistory,
    BillingPeriod,
    SubscriptionStatus,
    PaymentStatus,
)

from .iam import (
    IAMAgent,
    IAMMessage,
    IAMSession,
    IAMEventLog,
    IAMMetric,
)

from .api_key import (
    APIKey,
    APIKeyLog,
)

from .webhook import (
    Webhook,
    WebhookDelivery,
)

from .workflow_execution import (
    WorkflowExecution,
    NodeExecution,
    WorkflowCheckpoint,
    WorkflowTrigger,
    WorkflowSchedule,
    WorkflowExecutionStatus,
    NodeExecutionStatus,
)

from .usage_metrics import (
    UsageMetric as BasicUsageMetric,
    UsageLimit,
)

from .platform_integration import (
    PlatformCredential,
    PlatformExecution,
    PlatformAdapter,
    PlatformHealth,
    PlatformWebhook,
    PlatformType,
    AuthMethod,
)

from .workflow_templates import (
    WorkflowTemplate,
    WorkflowTemplatePermission,
    WorkflowTemplateUsage,
    WorkflowTemplateReview,
)

from .adapter_config import UserAdapterConfig
from .basic_agent import BasicAgent
from .tenant import Tenant, TenantStatus

__all__ = [
    # Tenant model (multi-tenancy infrastructure)
    "Tenant",
    "TenantStatus",
    # User model
    "User",
    # User adapter configuration
    "UserAdapterConfig",
    # Basic Agent model (Community Edition)
    "BasicAgent",
    # Conversation models
    "ConversationSession",
    "ConversationMessage",
    "ConversationAction",
    "ConversationIntent",
    "ConversationPattern",
    # Knowledge models
    "KnowledgeItem",
    "KnowledgeIndex",
    "KnowledgeQuery",
    "SystemManifest",
    "LearnedPattern",
    # Community models
    "Task",
    "WorkflowDefinition",
    "WorkflowInstance", 
    "WorkflowStep",
    "MCPServer",
    "MCPServerCapability",
    "MCPTool",
    "MCPInvocation",
    "TaskMCP",
    "MCPContextStorage",
    "Adapter",
    "BridgeConnection",
    "BridgeSync",
    "ResourcePoolConfig",
    # Enforcement models
    "EnforcementUsageMetric",
    "TenantLimitOverride",
    "FeatureTrial",
    "UpgradePrompt",
    "LicenseCache",
    "BillingEvent",
    "UsageSummary",
    # Subscription models
    "SubscriptionPlan",
    "Subscription",
    "UsageTracking",
    "PaymentMethod",
    "BillingHistory",
    "BillingPeriod",
    "SubscriptionStatus",
    "PaymentStatus",
    # IAM models
    "IAMAgent",
    "IAMMessage",
    "IAMSession",
    "IAMEventLog",
    "IAMMetric",
    # API Key models
    "APIKey",
    "APIKeyLog",
    # Webhook models
    "Webhook",
    "WebhookDelivery",
    # Workflow execution models
    "WorkflowExecution",
    "NodeExecution",
    "WorkflowCheckpoint",
    "WorkflowTrigger",
    "WorkflowSchedule",
    "WorkflowExecutionStatus",
    "NodeExecutionStatus",
    # Basic usage tracking for Community
    "BasicUsageMetric",
    "UsageLimit",
    # Platform Integration models
    "PlatformCredential",
    "PlatformExecution",
    "PlatformAdapter",
    "PlatformHealth",
    "PlatformWebhook",
    "PlatformType",
    "AuthMethod",
    # Workflow Template models
    "WorkflowTemplate",
    "WorkflowTemplatePermission",
    "WorkflowTemplateUsage",
    "WorkflowTemplateReview",
]