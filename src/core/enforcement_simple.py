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
    BUSINESS_STARTER = "business_starter"
    BUSINESS_GROWTH = "business_growth"
    BUSINESS_SCALE = "business_scale"
    ENTERPRISE = "enterprise"


class LimitExceededException(Exception):
    """Raised when a limit is exceeded in strict mode."""
    
    def __init__(self, limit_type: LimitType, current: int, limit: int):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.status_code = 402  # Payment Required
        self.detail = {
            "error": "limit_exceeded",
            "limit_type": limit_type.value,
            "current": current,
            "limit": limit,
            "message": f"You have reached the limit of {limit} {limit_type.value}",
            "upgrade_url": "/api/v1/upgrade/options"
        }
        super().__init__(self.detail["message"])


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
            LimitType.STORAGE: 51200,  # 50GB
            LimitType.EXECUTIONS: 50000,
            LimitType.SESSIONS: 200,
            LimitType.AGENTS: 25,
        },
        Edition.BUSINESS_SCALE: {
            LimitType.WORKFLOWS: 1000,
            LimitType.ADAPTERS: 999999,  # Unlimited
            LimitType.USERS: 50,
            LimitType.API_CALLS: 10000000,
            LimitType.STORAGE: 102400,  # 100GB
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
    
    # Edition features
    EDITION_FEATURES = {
        Edition.COMMUNITY: [
            "basic_workflows",
            "core_adapters",
            "single_user",
            "community_support"
        ],
        Edition.BUSINESS_STARTER: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9"
        ],
        Edition.BUSINESS_GROWTH: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "approval_workflows",
            "rbac",
            "custom_branding",
            "advanced_analytics",
            "priority_support",
            "sla_99_9"
        ],
        Edition.BUSINESS_SCALE: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "approval_workflows",
            "rbac",
            "custom_branding",
            "advanced_analytics",
            "api_access",
            "dedicated_support",
            "sla_99_95"
        ],
        Edition.ENTERPRISE: [
            "basic_workflows",
            "core_adapters",
            "business_adapters",
            "enterprise_adapters",
            "approval_workflows",
            "rbac",
            "custom_branding",
            "advanced_analytics",
            "api_access",
            "multi_tenant",
            "federation",
            "compliance",
            "custom_contracts",
            "24_7_support",
            "sla_99_99"
        ]
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
                    "edition": "business_starter",
                    "price": "$599/month",
                    "highlights": [
                        "100 workflows",
                        "20 adapters",
                        "5 users",
                        "Email support",
                        "99.9% uptime SLA",
                        "2 hrs/mo expert services"
                    ]
                },
                {
                    "edition": "business_growth",
                    "price": "$1,199/month",
                    "highlights": [
                        "500 workflows",
                        "50 adapters",
                        "20 users",
                        "Priority support",
                        "Advanced analytics",
                        "4 hrs/mo expert services"
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
                        "8 hrs/mo expert services"
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