"""Simplified license enforcement for Community Edition."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.cache import get_cache

logger = logging.getLogger(__name__)


class EnforcementMode(str, Enum):
    """Enforcement modes."""
    NONE = "none"       # No enforcement
    SOFT = "soft"       # Log warnings only
    STRICT = "strict"   # Block operations


class LimitType(str, Enum):
    """Types of limits that can be enforced."""
    WORKFLOWS = "workflows"
    ADAPTERS = "adapters"
    USERS = "users"
    API_CALLS = "api_calls"
    STORAGE = "storage"
    EXECUTIONS = "executions"
    SESSIONS = "sessions"
    AGENTS = "agents"


class Edition(str, Enum):
    """Available editions."""
    COMMUNITY = "community"
    TEAM = "team"
    BUSINESS_STARTER = "business_starter"
    BUSINESS_GROWTH = "business_growth"
    BUSINESS_SCALE = "business_scale"
    ENTERPRISE = "enterprise"


class LimitExceededException(Exception):
    """Raised when a limit is exceeded in strict mode."""

    def __init__(self, limit_type: LimitType, current: int, limit: int,
                 edition: str = "community"):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.status_code = 402  # Payment Required
        self.detail = {
            "error": "limit_exceeded",
            "limit_type": limit_type.value,
            "current": current,
            "limit": limit,
            "edition": edition,
            "message": f"You have reached the limit of {limit} {limit_type.value}",
            "upgrade_url": UPGRADE_URL,
        }
        super().__init__(self.detail["message"])


# Mapping of features to their minimum required edition
FEATURE_REQUIRED_EDITION = {
    # Team features
    "team_collaboration": Edition.TEAM,
    "shared_workspaces": Edition.TEAM,
    "team_management": Edition.TEAM,
    # Business Starter features
    "business_adapters": Edition.BUSINESS_STARTER,
    "approval_workflows": Edition.BUSINESS_STARTER,
    "rbac": Edition.BUSINESS_STARTER,
    "email_support": Edition.BUSINESS_STARTER,
    "sla_99_9": Edition.BUSINESS_STARTER,
    "ai_governance": Edition.BUSINESS_STARTER,
    "a2a_protocol": Edition.BUSINESS_STARTER,
    "sla_monitoring": Edition.BUSINESS_STARTER,
    "organization_management": Edition.BUSINESS_STARTER,
    "template_discovery": Edition.BUSINESS_STARTER,
    "subscription_licensing": Edition.BUSINESS_STARTER,
    # Business Growth features
    "custom_branding": Edition.BUSINESS_GROWTH,
    "advanced_analytics": Edition.BUSINESS_GROWTH,
    "priority_support": Edition.BUSINESS_GROWTH,
    "ml_enhanced_features": Edition.BUSINESS_GROWTH,
    "learning_loops": Edition.BUSINESS_GROWTH,
    "agent_performance_analytics": Edition.BUSINESS_GROWTH,
    "oauth2_oidc": Edition.BUSINESS_GROWTH,
    "phone_support": Edition.BUSINESS_GROWTH,
    # Business Scale features
    "api_access": Edition.BUSINESS_SCALE,
    "dedicated_support": Edition.BUSINESS_SCALE,
    "sla_99_95": Edition.BUSINESS_SCALE,
    "dedicated_account_manager": Edition.BUSINESS_SCALE,
    # Enterprise features
    "enterprise_adapters": Edition.ENTERPRISE,
    "multi_tenant": Edition.ENTERPRISE,
    "federation": Edition.ENTERPRISE,
    "compliance": Edition.ENTERPRISE,
    "custom_contracts": Edition.ENTERPRISE,
    "24_7_support": Edition.ENTERPRISE,
    "sla_99_99": Edition.ENTERPRISE,
    "audit_logging": Edition.ENTERPRISE,
    "saml_sso": Edition.ENTERPRISE,
    "geographic_routing": Edition.ENTERPRISE,
    "cross_tenant_workflows": Edition.ENTERPRISE,
    "dedicated_infrastructure": Edition.ENTERPRISE,
}

# Human-readable names for features
FEATURE_DISPLAY_NAMES = {
    # Team
    "team_collaboration": "Team Collaboration",
    "shared_workspaces": "Shared Workspaces",
    # Business Starter
    "approval_workflows": "Approval Workflows",
    "business_adapters": "Business Adapters",
    "rbac": "Role-Based Access Control",
    "ai_governance": "AI Governance & Risk Assessment",
    "a2a_protocol": "Google A2A Protocol",
    "sla_monitoring": "SLA Monitoring & Enforcement",
    "organization_management": "Organization Management",
    "team_management": "Team Management",
    "template_discovery": "ML Template Discovery",
    "subscription_licensing": "Subscription & Licensing",
    # Business Growth
    "advanced_analytics": "Advanced Analytics",
    "custom_branding": "Custom Branding",
    "ml_enhanced_features": "ML-Enhanced Features",
    "learning_loops": "Learning Loops",
    "agent_performance_analytics": "Agent Performance Analytics",
    "oauth2_oidc": "OAuth2/OIDC Integration",
    "phone_support": "Phone Support",
    # Business Scale
    "api_access": "API Access",
    "dedicated_account_manager": "Dedicated Account Manager",
    # Enterprise
    "multi_tenant": "Multi-Tenant Support",
    "federation": "Federation",
    "compliance": "Compliance Suite",
    "enterprise_adapters": "Enterprise Adapters",
    "audit_logging": "Audit Logging",
    "saml_sso": "SAML 2.0 SSO",
    "geographic_routing": "Geographic Routing",
    "cross_tenant_workflows": "Cross-Tenant Workflows",
    "dedicated_infrastructure": "Dedicated Infrastructure",
}

# Human-readable names for editions
EDITION_DISPLAY_NAMES = {
    Edition.COMMUNITY: "Community",
    Edition.TEAM: "Team",
    Edition.BUSINESS_STARTER: "Business Starter",
    Edition.BUSINESS_GROWTH: "Business Growth",
    Edition.BUSINESS_SCALE: "Business Scale",
    Edition.ENTERPRISE: "Enterprise",
}

UPGRADE_URL = "https://aictrlnet.com/pricing"


def get_feature_upgrade_info(feature: str, current_edition: Edition) -> Dict[str, Any]:
    """Build an enhanced 403 response body for a feature gate."""
    required_edition = FEATURE_REQUIRED_EDITION.get(feature, Edition.BUSINESS_STARTER)
    display_name = FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())
    required_display = EDITION_DISPLAY_NAMES.get(required_edition, required_edition.value)

    return {
        "error": "feature_not_available",
        "feature": feature,
        "message": f"{display_name} is available in {required_display} Edition",
        "edition": current_edition.value,
        "required_edition": required_edition.value,
        "upgrade_url": UPGRADE_URL,
    }


class LicenseEnforcer:
    """Simplified license enforcement for Community Edition."""

    # Edition limits
    EDITION_LIMITS = {
        Edition.COMMUNITY: {
            LimitType.WORKFLOWS: 10,
            LimitType.ADAPTERS: 5,
            LimitType.USERS: 1,
            LimitType.API_CALLS: 10000,
            LimitType.STORAGE: 1024,  # 1GB in MB
            LimitType.EXECUTIONS: 1000,
            LimitType.SESSIONS: 5,
            LimitType.AGENTS: 2,
        },
        Edition.TEAM: {
            LimitType.WORKFLOWS: 30,
            LimitType.ADAPTERS: 10,
            LimitType.USERS: 3,
            LimitType.API_CALLS: 100000,
            LimitType.STORAGE: 10240,  # 10GB
            LimitType.EXECUTIONS: 10000,
            LimitType.SESSIONS: 20,
            LimitType.AGENTS: 5,
        },
        Edition.BUSINESS_STARTER: {
            LimitType.WORKFLOWS: 100,
            LimitType.ADAPTERS: 20,
            LimitType.USERS: 5,
            LimitType.API_CALLS: 1000000,
            LimitType.STORAGE: 10240,  # 10GB
            LimitType.EXECUTIONS: 10000,
            LimitType.SESSIONS: 50,
            LimitType.AGENTS: 10,
        },
        Edition.BUSINESS_GROWTH: {
            LimitType.WORKFLOWS: 500,
            LimitType.ADAPTERS: 50,
            LimitType.USERS: 20,
            LimitType.API_CALLS: 5000000,
            LimitType.STORAGE: 204800,  # 200GB
            LimitType.EXECUTIONS: 50000,
            LimitType.SESSIONS: 200,
            LimitType.AGENTS: 25,
        },
        Edition.BUSINESS_SCALE: {
            LimitType.WORKFLOWS: 1000,
            LimitType.ADAPTERS: 999999,  # Unlimited
            LimitType.USERS: 50,
            LimitType.API_CALLS: 20000000,
            LimitType.STORAGE: 512000,  # 500GB
            LimitType.EXECUTIONS: 100000,
            LimitType.SESSIONS: 500,
            LimitType.AGENTS: 50,
        },
        Edition.ENTERPRISE: {
            LimitType.WORKFLOWS: 999999,  # Unlimited
            LimitType.ADAPTERS: 999999,
            LimitType.USERS: 999999,
            LimitType.API_CALLS: 999999999,
            LimitType.STORAGE: 999999999,
            LimitType.EXECUTIONS: 999999999,
            LimitType.SESSIONS: 999999,
            LimitType.AGENTS: 999999,
        }
    }
    
    # Edition features (accretive — each tier includes all features from lower tiers)
    EDITION_FEATURES = {
        Edition.COMMUNITY: [
            "basic_workflows",
            "core_adapters",
            "single_user",
            "community_support",
        ],
        Edition.TEAM: [
            "basic_workflows",
            "core_adapters",
            "community_support",
            "team_collaboration",
            "shared_workspaces",
            "team_management",
        ],
        Edition.BUSINESS_STARTER: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9",
            "ai_governance",
            "a2a_protocol",
            "sla_monitoring",
            "organization_management",
            "team_management",
            "template_discovery",
            "subscription_licensing",
        ],
        Edition.BUSINESS_GROWTH: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9",
            "ai_governance",
            "a2a_protocol",
            "sla_monitoring",
            "organization_management",
            "team_management",
            "template_discovery",
            "subscription_licensing",
            "custom_branding",
            "advanced_analytics",
            "priority_support",
            "ml_enhanced_features",
            "learning_loops",
            "agent_performance_analytics",
            "oauth2_oidc",
            "phone_support",
        ],
        Edition.BUSINESS_SCALE: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9",
            "ai_governance",
            "a2a_protocol",
            "sla_monitoring",
            "organization_management",
            "team_management",
            "template_discovery",
            "subscription_licensing",
            "custom_branding",
            "advanced_analytics",
            "priority_support",
            "ml_enhanced_features",
            "learning_loops",
            "agent_performance_analytics",
            "oauth2_oidc",
            "phone_support",
            "api_access",
            "dedicated_support",
            "sla_99_95",
            "dedicated_account_manager",
        ],
        Edition.ENTERPRISE: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "enterprise_adapters",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9",
            "ai_governance",
            "a2a_protocol",
            "sla_monitoring",
            "organization_management",
            "team_management",
            "template_discovery",
            "subscription_licensing",
            "custom_branding",
            "advanced_analytics",
            "priority_support",
            "ml_enhanced_features",
            "learning_loops",
            "agent_performance_analytics",
            "oauth2_oidc",
            "phone_support",
            "api_access",
            "dedicated_support",
            "sla_99_95",
            "dedicated_account_manager",
            "multi_tenant",
            "federation",
            "compliance",
            "custom_contracts",
            "24_7_support",
            "sla_99_99",
            "audit_logging",
            "saml_sso",
            "geographic_routing",
            "cross_tenant_workflows",
            "dedicated_infrastructure",
        ],
    }
    
    def __init__(
        self,
        db: AsyncSession,
        mode: Optional[EnforcementMode] = None
    ):
        self.db = db
        self.settings = get_settings()
        
        # Determine enforcement mode
        if mode:
            self.mode = mode
        else:
            # Default to NONE for community edition
            self.mode = EnforcementMode.NONE
        
        self.cache = None  # Will be initialized on first use
        logger.info(f"License enforcer initialized in {self.mode.value} mode")
    
    async def check_limit(
        self,
        tenant_id: str,
        limit_type: LimitType,
        increment: int = 0
    ) -> Dict[str, Any]:
        """Check if operation would exceed limit."""
        
        # In NONE mode, always allow
        if self.mode == EnforcementMode.NONE:
            return {
                "allowed": True,
                "current": 0,
                "limit": 999999,
                "percentage": 0.0,
                "warning": None
            }
        
        # Get tenant info and limits
        tenant_info = await self._get_tenant_info(tenant_id)
        edition = Edition(tenant_info["edition"])
        limit = self.EDITION_LIMITS[edition].get(limit_type, 999999)
        
        # Get current usage (mock data for community edition)
        current = await self._get_current_usage(tenant_id, limit_type)
        projected = current + increment
        percentage = (projected / limit * 100) if limit > 0 else 0
        
        # Check if limit would be exceeded
        would_exceed = projected > limit
        
        result = {
            "allowed": not would_exceed,
            "current": current,
            "limit": limit,
            "percentage": round(percentage, 1),
            "warning": None
        }
        
        # Add warnings if approaching limit
        if percentage >= 80 and not would_exceed:
            result["warning"] = f"Approaching limit: {current}/{limit} {limit_type.value} used"
        elif would_exceed:
            result["warning"] = f"Limit exceeded: {projected} would exceed limit of {limit}"
        
        # In STRICT mode, raise exception if limit exceeded
        if self.mode == EnforcementMode.STRICT and would_exceed:
            raise LimitExceededException(limit_type, current, limit)
        
        return result
    
    async def check_feature(self, tenant_id: str, feature: str) -> bool:
        """Check if tenant has access to a feature."""
        
        # In NONE mode, all features available
        if self.mode == EnforcementMode.NONE:
            return True
        
        tenant_info = await self._get_tenant_info(tenant_id)
        edition = Edition(tenant_info["edition"])
        
        return feature in self.EDITION_FEATURES.get(edition, [])
    
    async def get_usage_summary(self, tenant_id: str) -> Dict[str, Any]:
        """Get usage summary for tenant."""
        
        tenant_info = await self._get_tenant_info(tenant_id)
        edition = Edition(tenant_info["edition"])
        limits = self.EDITION_LIMITS[edition]
        
        summary = {}
        for limit_type in LimitType:
            if limit_type in limits:
                current = await self._get_current_usage(tenant_id, limit_type)
                limit = limits[limit_type]
                summary[limit_type.value] = {
                    "current": current,
                    "limit": limit,
                    "percentage": round((current / limit * 100) if limit > 0 else 0, 1)
                }
        
        return {
            "tenant_id": tenant_id,
            "edition": edition.value,
            "usage": summary,
            "features": self.EDITION_FEATURES[edition]
        }
    
    async def _get_tenant_info(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant information - uses actual edition from settings."""
        
        # Get the actual edition from settings
        edition_str = self.settings.EDITION.lower()
        
        # Map to proper enum value
        edition_map = {
            "community": Edition.COMMUNITY,
            "team": Edition.TEAM,
            "business": Edition.BUSINESS_STARTER,  # Default business tier
            "enterprise": Edition.ENTERPRISE
        }
        
        edition = edition_map.get(edition_str, Edition.COMMUNITY)
        
        return {
            "id": tenant_id,
            "edition": edition.value,
            "created_at": datetime.utcnow().isoformat()
        }
    
    async def _get_current_usage(self, tenant_id: str, limit_type: LimitType) -> int:
        """Get current usage - returns mock data for community edition."""
        
        # Mock usage data
        mock_usage = {
            LimitType.WORKFLOWS: 5,
            LimitType.ADAPTERS: 3,
            LimitType.USERS: 1,
            LimitType.API_CALLS: 1000,
            LimitType.STORAGE: 512,
            LimitType.EXECUTIONS: 100,
            LimitType.SESSIONS: 2,
            LimitType.AGENTS: 1,
        }
        
        return mock_usage.get(limit_type, 0)
    
    def _get_upgrade_path(self, current_edition: str) -> Dict[str, Any]:
        """Get upgrade options from current edition."""
        
        upgrade_options = []
        
        if current_edition == "community":
            upgrade_options = [
                {
                    "edition": "team",
                    "price": "$149/month",
                    "highlights": [
                        "30 workflows",
                        "10 adapters",
                        "3 users",
                        "Team collaboration",
                        "Shared workspaces"
                    ]
                },
                {
                    "edition": "business_starter",
                    "price": "$599/month",
                    "highlights": [
                        "100 workflows",
                        "20 adapters",
                        "5 users",
                        "Email support",
                        "99.9% uptime SLA",
                        "2 hrs/mo expert assistance"
                    ]
                },
                {
                    "edition": "business_growth",
                    "price": "$1,499/month",
                    "highlights": [
                        "500 workflows",
                        "50 adapters",
                        "20 users",
                        "Priority support",
                        "Advanced analytics",
                        "5 hrs/mo expert assistance"
                    ]
                }
            ]
        elif current_edition in ["business_starter", "business_growth"]:
            upgrade_options = [
                {
                    "edition": "business_scale",
                    "price": "$2,499/month",
                    "highlights": [
                        "1,000 workflows",
                        "Unlimited adapters",
                        "50 users",
                        "Dedicated support",
                        "API access",
                        "8 hrs/mo expert assistance"
                    ]
                },
                {
                    "edition": "enterprise",
                    "price": "Contact Sales",
                    "highlights": [
                        "Unlimited everything",
                        "Multi-tenant",
                        "24/7 support",
                        "Custom contracts"
                    ]
                }
            ]
        elif current_edition == "business_scale":
            upgrade_options = [
                {
                    "edition": "enterprise",
                    "price": "Contact Sales",
                    "highlights": [
                        "Unlimited everything",
                        "Multi-tenant support",
                        "24/7 phone support",
                        "Custom contracts",
                        "Professional services"
                    ]
                }
            ]
        
        return {
            "current_edition": current_edition,
            "upgrade_options": upgrade_options,
            "contact_sales": current_edition in ["business_scale", "enterprise"]
        }