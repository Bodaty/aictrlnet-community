"""Value Ladder License Enforcement System."""

from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_
from fastapi import HTTPException, status
import json
import logging

from core.config import get_settings
from core.cache import get_cache

logger = logging.getLogger(__name__)


class EnforcementMode(Enum):
    """Enforcement modes for different deployment types."""
    NONE = "none"           # Self-hosted open source - no enforcement
    SOFT = "soft"           # Log and warn only - for testing
    STRICT = "strict"       # Hard enforcement - for cloud deployments


class LimitType(Enum):
    """Types of limits that can be enforced."""
    WORKFLOWS = "workflows"
    ADAPTERS = "adapters" 
    USERS = "users"
    API_CALLS = "api_calls"
    EXECUTIONS = "executions"
    STORAGE_GB = "storage_gb"
    SESSIONS = "sessions"
    AGENTS = "agents"
    AI_AGENT_REQUESTS = "ai_agent_requests"  # AI agent framework requests


class Edition(Enum):
    """Product editions with different feature sets."""
    COMMUNITY = "community"
    BUSINESS_STARTER = "business_starter"
    BUSINESS_GROWTH = "business_growth"
    BUSINESS_SCALE = "business_scale"
    ENTERPRISE = "enterprise"


class LimitExceededException(HTTPException):
    """Exception raised when a limit is exceeded in strict mode."""
    
    def __init__(
        self,
        limit_type: LimitType,
        current: int,
        limit: int,
        upgrade_path: Dict[str, Any]
    ):
        self.limit_type = limit_type
        self.current = current
        self.limit = limit
        self.upgrade_path = upgrade_path
        
        super().__init__(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail={
                "error": "limit_exceeded",
                "limit_type": limit_type.value,
                "current_usage": current,
                "limit": limit,
                "message": f"You have reached the limit of {limit} {limit_type.value}. Please upgrade to continue.",
                "upgrade_options": upgrade_path
            }
        )


class LicenseEnforcer:
    """Central enforcement system for all limits."""
    
    # Default limits by edition
    EDITION_LIMITS = {
        Edition.COMMUNITY: {
            LimitType.WORKFLOWS: 10,
            LimitType.ADAPTERS: 5,
            LimitType.USERS: 1,
            LimitType.API_CALLS: 10000,      # per month
            LimitType.EXECUTIONS: 1000,       # per month
            LimitType.STORAGE_GB: 1,
            LimitType.SESSIONS: 0,            # No sessions in community
            LimitType.AGENTS: 1,              # Only self agent
            LimitType.AI_AGENT_REQUESTS: 100,   # 100 AI agent requests per day
        },
        Edition.BUSINESS_STARTER: {
            LimitType.WORKFLOWS: 100,
            LimitType.ADAPTERS: 20,
            LimitType.USERS: 5,
            LimitType.API_CALLS: 1000000,     # 1M per month
            LimitType.EXECUTIONS: 100000,     # 100k per month
            LimitType.STORAGE_GB: 50,
            LimitType.SESSIONS: 10,
            LimitType.AGENTS: 5,
            LimitType.AI_AGENT_REQUESTS: 999999,  # Unlimited for Business
        },
        Edition.BUSINESS_GROWTH: {
            LimitType.WORKFLOWS: 500,
            LimitType.ADAPTERS: 50,
            LimitType.USERS: 20,
            LimitType.API_CALLS: 5000000,     # 5M per month
            LimitType.EXECUTIONS: 500000,     # 500k per month
            LimitType.STORAGE_GB: 200,
            LimitType.SESSIONS: 50,
            LimitType.AGENTS: 20,
            LimitType.AI_AGENT_REQUESTS: 999999,  # Unlimited for Business
        },
        Edition.BUSINESS_SCALE: {
            LimitType.WORKFLOWS: 1000,
            LimitType.ADAPTERS: 999999,       # Effectively unlimited
            LimitType.USERS: 50,
            LimitType.API_CALLS: 20000000,    # 20M per month
            LimitType.EXECUTIONS: 2000000,    # 2M per month
            LimitType.STORAGE_GB: 500,
            LimitType.SESSIONS: 200,
            LimitType.AGENTS: 50,
            LimitType.AI_AGENT_REQUESTS: 999999,  # Unlimited for Business
        },
        Edition.ENTERPRISE: {
            # Unlimited for all
            LimitType.WORKFLOWS: 999999,
            LimitType.ADAPTERS: 999999,
            LimitType.USERS: 999999,
            LimitType.API_CALLS: 999999999,
            LimitType.EXECUTIONS: 999999999,
            LimitType.STORAGE_GB: 999999,
            LimitType.SESSIONS: 999999,
            LimitType.AGENTS: 999999,
            LimitType.AI_AGENT_REQUESTS: 999999,  # Unlimited for Enterprise
        }
    }
    
    # Features available by edition
    EDITION_FEATURES = {
        Edition.COMMUNITY: {
            "basic_workflows",
            "basic_adapters", 
            "rest_api",
            "community_support",
        },
        Edition.BUSINESS_STARTER: {
            "basic_workflows",
            "basic_adapters",
            "business_adapters",
            "rest_api",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9",
        },
        Edition.BUSINESS_GROWTH: {
            # Includes all from starter plus:
            "basic_workflows",
            "basic_adapters", 
            "business_adapters",
            "rest_api",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9",
            "advanced_analytics",
            "custom_branding",
            "priority_support",
        },
        Edition.BUSINESS_SCALE: {
            # Includes all from growth plus:
            "basic_workflows",
            "basic_adapters",
            "business_adapters",
            "rest_api",
            "approval_workflows",
            "rbac",
            "email_support",
            "sla_99_9", 
            "advanced_analytics",
            "custom_branding",
            "priority_support",
            "phone_support",
            "dedicated_account_manager",
        },
        Edition.ENTERPRISE: {
            # Everything
            "basic_workflows",
            "basic_adapters",
            "business_adapters",
            "enterprise_adapters",
            "rest_api",
            "approval_workflows",
            "rbac",
            "multi_tenant",
            "federation",
            "compliance",
            "audit_logging",
            "geographic_routing",
            "email_support",
            "phone_support",
            "sla_99_99",
            "dedicated_infrastructure",
            "custom_contracts",
            "24_7_support",
        }
    }
    
    def __init__(
        self, 
        db: AsyncSession,
        mode: Optional[EnforcementMode] = None
    ):
        self.db = db
        settings = get_settings()
        
        # Determine enforcement mode
        if mode:
            self.mode = mode
        else:
            # Default modes based on deployment
            deployment_type = getattr(settings, "DEPLOYMENT_TYPE", "self-hosted")
            if deployment_type == "cloud":
                self.mode = EnforcementMode.STRICT
            elif deployment_type == "test":
                self.mode = EnforcementMode.SOFT
            else:
                self.mode = EnforcementMode.NONE
        
        self.cache = None  # Will be initialized on first use
        logger.info(f"License enforcer initialized in {self.mode.value} mode")
    
    async def check_limit(
        self,
        tenant_id: str,
        limit_type: LimitType,
        current_value: Optional[int] = None,
        increment: int = 1
    ) -> Dict[str, Any]:
        """Check if a limit would be exceeded."""
        
        # No enforcement in NONE mode
        if self.mode == EnforcementMode.NONE:
            return {
                "allowed": True,
                "warning": None,
                "current": current_value or 0,
                "limit": 999999,
                "percentage": 0
            }
        
        # Get tenant's edition and limits
        tenant_info = await self._get_tenant_info(tenant_id)
        limit = await self._get_limit_for_tenant(tenant_id, tenant_info["edition"], limit_type)
        
        # Get current usage if not provided
        if current_value is None:
            current_value = await self._get_current_usage(tenant_id, limit_type)
        
        # Check if increment would exceed limit
        new_value = current_value + increment
        percentage = (current_value / limit * 100) if limit > 0 else 0
        
        result = {
            "allowed": True,
            "warning": None,
            "current": current_value,
            "limit": limit,
            "percentage": percentage,
            "upgrade_suggested": False
        }
        
        # Check if limit would be exceeded
        if new_value > limit:
            if self.mode == EnforcementMode.STRICT:
                raise LimitExceededException(
                    limit_type=limit_type,
                    current=current_value,
                    limit=limit,
                    upgrade_path=self._get_upgrade_path(tenant_info["edition"])
                )
            else:  # SOFT mode
                result["allowed"] = True
                result["warning"] = f"Limit exceeded: {new_value}/{limit} {limit_type.value}"
                result["upgrade_suggested"] = True
                logger.warning(f"Soft limit exceeded for tenant {tenant_id}: {limit_type.value}")
        
        # Check if approaching limit (80% threshold)
        elif percentage >= 80:
            result["warning"] = f"Approaching limit: {current_value}/{limit} {limit_type.value} ({percentage:.0f}% used)"
            result["upgrade_suggested"] = True
        
        return result
    
    async def check_feature(
        self,
        tenant_id: str,
        feature: str
    ) -> bool:
        """Check if a tenant has access to a specific feature."""
        
        # All features available in NONE mode
        if self.mode == EnforcementMode.NONE:
            return True
        
        tenant_info = await self._get_tenant_info(tenant_id)
        edition = Edition(tenant_info["edition"])
        
        # Check if in trial
        trial_features = await self._get_trial_features(tenant_id)
        if feature in trial_features:
            return True
        
        # Check edition features
        return feature in self.EDITION_FEATURES.get(edition, set())
    
    async def get_usage_summary(
        self,
        tenant_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive usage summary for a tenant."""
        
        tenant_info = await self._get_tenant_info(tenant_id)
        edition = Edition(tenant_info["edition"])
        
        summary = {
            "tenant_id": tenant_id,
            "edition": edition.value,
            "enforcement_mode": self.mode.value,
            "limits": {},
            "features": list(self.EDITION_FEATURES.get(edition, set())),
            "trial_features": await self._get_trial_features(tenant_id)
        }
        
        # Get usage for each limit type
        for limit_type in LimitType:
            limit = await self._get_limit_for_tenant(tenant_id, edition.value, limit_type)
            current = await self._get_current_usage(tenant_id, limit_type)
            percentage = (current / limit * 100) if limit > 0 else 0
            
            summary["limits"][limit_type.value] = {
                "current": current,
                "limit": limit,
                "percentage": percentage,
                "status": self._get_limit_status(percentage)
            }
        
        return summary
    
    async def _get_tenant_info(self, tenant_id: str) -> Dict[str, Any]:
        """Get tenant information from cache or database."""
        
        # Check cache first
        if not self.cache:
            self.cache = await get_cache()
        cache_key = f"tenant_info:{tenant_id}"
        cached = await self.cache.get(cache_key)
        if cached:
            return json.loads(cached)
        
        # For community edition, we don't have tenant tables
        # Just return a default tenant info
        info = {
            "id": tenant_id,
            "edition": Edition.COMMUNITY.value,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Cache for 5 minutes
        await self.cache.set(cache_key, json.dumps(info), 300)
        
        return info
    
    async def _get_limit_for_tenant(
        self,
        tenant_id: str,
        edition: str,
        limit_type: LimitType
    ) -> int:
        """Get limit value considering overrides."""
        
        # Check for custom override first
        from models.enforcement import TenantLimitOverride
        result = await self.db.execute(
            select(TenantLimitOverride).where(
                and_(
                    TenantLimitOverride.tenant_id == tenant_id,
                    TenantLimitOverride.limit_type == limit_type.value,
                    or_(
                        TenantLimitOverride.expires_at.is_(None),
                        TenantLimitOverride.expires_at > datetime.utcnow()
                    )
                )
            )
        )
        override = result.scalar_one_or_none()
        
        if override:
            return override.limit_value
        
        # Return default for edition
        edition_enum = Edition(edition)
        return self.EDITION_LIMITS.get(edition_enum, {}).get(limit_type, 0)
    
    async def _get_current_usage(
        self,
        tenant_id: str,
        limit_type: LimitType
    ) -> int:
        """Get current usage for a limit type."""
        
        # For some limits, we need to query specific tables
        if limit_type == LimitType.WORKFLOWS:
            from models.community import WorkflowDefinition as Workflow
            result = await self.db.execute(
                select(func.count(Workflow.id)).where(
                    Workflow.tenant_id == tenant_id
                )
            )
            return result.scalar() or 0
            
        elif limit_type == LimitType.USERS:
            from models.community import User
            result = await self.db.execute(
                select(func.count(User.id)).where(
                    User.tenant_id == tenant_id
                )
            )
            return result.scalar() or 0
            
        elif limit_type in [LimitType.API_CALLS, LimitType.EXECUTIONS]:
            # These are tracked in usage_metrics
            from models.community import UsageMetric
            
            # Current month
            start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            result = await self.db.execute(
                select(func.sum(UsageMetric.value)).where(
                    and_(
                        UsageMetric.tenant_id == tenant_id,
                        UsageMetric.metric_type == limit_type.value,
                        UsageMetric.timestamp >= start_of_month
                    )
                )
            )
            return int(result.scalar() or 0)
        
        # Default to 0 for unimplemented types
        return 0
    
    async def _get_trial_features(self, tenant_id: str) -> List[str]:
        """Get list of features currently in trial."""
        
        from models.enforcement import FeatureTrial
        
        result = await self.db.execute(
            select(FeatureTrial).where(
                and_(
                    FeatureTrial.tenant_id == tenant_id,
                    FeatureTrial.expires_at > datetime.utcnow(),
                    FeatureTrial.converted == False
                )
            )
        )
        trials = result.scalars().all()
        
        features = []
        for trial in trials:
            # Get features for the trial edition
            trial_edition = Edition(trial.edition_required)
            features.extend(self.EDITION_FEATURES.get(trial_edition, set()))
        
        return list(set(features))
    
    def _get_upgrade_path(self, current_edition: str) -> Dict[str, Any]:
        """Get upgrade options from current edition."""
        
        current = Edition(current_edition)
        options = []
        
        # Define upgrade paths
        if current == Edition.COMMUNITY:
            options = [
                {
                    "edition": Edition.BUSINESS_STARTER.value,
                    "price": "$500/month",
                    "highlights": ["100 workflows", "20 adapters", "5 users", "Email support"]
                },
                {
                    "edition": Edition.BUSINESS_GROWTH.value,
                    "price": "$1,000/month", 
                    "highlights": ["500 workflows", "50 adapters", "20 users", "Priority support"]
                }
            ]
        elif current == Edition.BUSINESS_STARTER:
            options = [
                {
                    "edition": Edition.BUSINESS_GROWTH.value,
                    "price": "$1,000/month",
                    "highlights": ["5x more workflows", "2.5x more adapters", "4x more users"]
                },
                {
                    "edition": Edition.BUSINESS_SCALE.value,
                    "price": "$2,000/month",
                    "highlights": ["10x more workflows", "Unlimited adapters", "Phone support"]
                }
            ]
        elif current in [Edition.BUSINESS_GROWTH, Edition.BUSINESS_SCALE]:
            options = [
                {
                    "edition": Edition.ENTERPRISE.value,
                    "price": "Contact sales",
                    "highlights": ["Unlimited everything", "24/7 support", "Custom contracts"]
                }
            ]
        
        return {
            "current_edition": current.value,
            "upgrade_options": options,
            "contact_sales": current in [Edition.BUSINESS_SCALE, Edition.ENTERPRISE]
        }
    
    def _get_limit_status(self, percentage: float) -> str:
        """Get status based on usage percentage."""
        
        if percentage >= 100:
            return "exceeded"
        elif percentage >= 90:
            return "critical"
        elif percentage >= 80:
            return "warning"
        elif percentage >= 50:
            return "moderate"
        else:
            return "healthy"


# Helper function for dependency injection
async def get_enforcer(db: AsyncSession) -> LicenseEnforcer:
    """Get license enforcer instance."""
    return LicenseEnforcer(db)


# Import for backward compatibility
from sqlalchemy import func