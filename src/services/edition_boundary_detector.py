"""Edition Boundary Detector for Conversational Edition Escalation.

Detects when a user's request requires features from a higher edition tier
and provides conversational feedback with upgrade prompts.

Part of the Intelligent Assistant v3 Edition Escalation Framework.
"""

from typing import Dict, List, Optional, Tuple
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EditionTier(str, Enum):
    """Edition tier levels."""
    COMMUNITY = "community"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class BoundaryType(str, Enum):
    """Types of edition boundaries."""
    FEATURE_LOCKED = "feature_locked"  # Feature doesn't exist in current edition
    QUOTA_EXCEEDED = "quota_exceeded"  # Usage limit reached
    CAPABILITY_MISSING = "capability_missing"  # Required capability not available
    ADAPTER_UNAVAILABLE = "adapter_unavailable"  # Adapter restricted to higher tier
    WORKFLOW_RESTRICTED = "workflow_restricted"  # Workflow template restricted
    AGENT_TIER_MISMATCH = "agent_tier_mismatch"  # AI agent requires higher tier


class EditionBoundary:
    """Represents a detected edition boundary."""

    def __init__(
        self,
        boundary_type: BoundaryType,
        current_edition: EditionTier,
        required_edition: EditionTier,
        feature_name: str,
        user_intent: str,
        context: Optional[Dict] = None
    ):
        self.boundary_type = boundary_type
        self.current_edition = current_edition
        self.required_edition = required_edition
        self.feature_name = feature_name
        self.user_intent = user_intent
        self.context = context or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'boundary_type': self.boundary_type.value,
            'current_edition': self.current_edition.value,
            'required_edition': self.required_edition.value,
            'feature_name': self.feature_name,
            'user_intent': self.user_intent,
            'context': self.context
        }


class EditionBoundaryDetector:
    """Detects when user requests cross edition boundaries."""

    # Feature to edition mapping
    FEATURE_REQUIREMENTS = {
        # AI Governance features (Business+)
        'ai_governance': EditionTier.BUSINESS,
        'model_validation': EditionTier.BUSINESS,
        'bias_detection': EditionTier.BUSINESS,
        'quality_monitoring': EditionTier.BUSINESS,
        'risk_assessment': EditionTier.BUSINESS,

        # Advanced ML features (Business+)
        'semantic_search': EditionTier.BUSINESS,
        'pattern_learning': EditionTier.BUSINESS,
        'intelligent_routing': EditionTier.BUSINESS,
        'pod_swarm_intelligence': EditionTier.BUSINESS,

        # Resource management (Business+)
        'resource_pools': EditionTier.BUSINESS,
        'sla_management': EditionTier.BUSINESS,
        'approval_workflows': EditionTier.BUSINESS,

        # Multi-tenancy features (Enterprise only)
        'multi_tenancy': EditionTier.ENTERPRISE,
        'organization_management': EditionTier.ENTERPRISE,
        'federation': EditionTier.ENTERPRISE,
        'cross_org_workflows': EditionTier.ENTERPRISE,
        'white_labeling': EditionTier.ENTERPRISE,

        # Compliance & Security (Enterprise)
        'advanced_compliance': EditionTier.ENTERPRISE,
        'audit_logging': EditionTier.ENTERPRISE,
        'role_based_access': EditionTier.ENTERPRISE,
        'sso_integration': EditionTier.ENTERPRISE,

        # Advanced automation (Enterprise)
        'company_automation': EditionTier.ENTERPRISE,
        'strategic_planning': EditionTier.ENTERPRISE,
        'cross_departmental': EditionTier.ENTERPRISE,
    }

    # Adapter tier requirements
    ADAPTER_REQUIREMENTS = {
        # Community adapters
        'ollama': EditionTier.COMMUNITY,
        'claude': EditionTier.COMMUNITY,
        'openai': EditionTier.COMMUNITY,
        'slack': EditionTier.COMMUNITY,
        'email': EditionTier.COMMUNITY,

        # Business adapters
        'azure_openai': EditionTier.BUSINESS,
        'google_gemini': EditionTier.BUSINESS,
        'salesforce': EditionTier.BUSINESS,
        'hubspot': EditionTier.BUSINESS,
        'jira': EditionTier.BUSINESS,

        # Enterprise adapters
        'sap': EditionTier.ENTERPRISE,
        'oracle': EditionTier.ENTERPRISE,
        'workday': EditionTier.ENTERPRISE,
        'servicenow': EditionTier.ENTERPRISE,
    }

    # Usage quotas by edition
    EDITION_QUOTAS = {
        EditionTier.COMMUNITY: {
            'workflows_per_month': 100,
            'agents_max': 5,
            'adapters_max': 10,
            'api_calls_per_day': 1000,
            'storage_gb': 1,
        },
        EditionTier.BUSINESS: {
            'workflows_per_month': 10000,
            'agents_max': 50,
            'adapters_max': 50,
            'api_calls_per_day': 100000,
            'storage_gb': 100,
        },
        EditionTier.ENTERPRISE: {
            'workflows_per_month': -1,  # Unlimited
            'agents_max': -1,
            'adapters_max': -1,
            'api_calls_per_day': -1,
            'storage_gb': -1,
        },
    }

    def __init__(self, user_edition: EditionTier):
        """Initialize detector with user's current edition.

        Args:
            user_edition: Current edition tier of the user
        """
        self.user_edition = user_edition

    def detect_boundary(
        self,
        intent: str,
        requested_features: List[str],
        requested_adapters: List[str] = None,
        current_usage: Dict[str, int] = None,
        context: Dict = None
    ) -> Optional[EditionBoundary]:
        """Detect if the request crosses an edition boundary.

        Args:
            intent: User's primary intent
            requested_features: List of features needed for the request
            requested_adapters: List of adapters needed
            current_usage: Current usage metrics
            context: Additional context

        Returns:
            EditionBoundary if boundary detected, None otherwise
        """
        # Check feature requirements
        for feature in requested_features:
            required_edition = self.FEATURE_REQUIREMENTS.get(feature)
            if required_edition and self._is_higher_tier(required_edition):
                return EditionBoundary(
                    boundary_type=BoundaryType.FEATURE_LOCKED,
                    current_edition=self.user_edition,
                    required_edition=required_edition,
                    feature_name=feature,
                    user_intent=intent,
                    context=context or {}
                )

        # Check adapter requirements
        if requested_adapters:
            for adapter in requested_adapters:
                required_edition = self.ADAPTER_REQUIREMENTS.get(adapter.lower())
                if required_edition and self._is_higher_tier(required_edition):
                    return EditionBoundary(
                        boundary_type=BoundaryType.ADAPTER_UNAVAILABLE,
                        current_edition=self.user_edition,
                        required_edition=required_edition,
                        feature_name=adapter,
                        user_intent=intent,
                        context=context or {}
                    )

        # Check usage quotas
        if current_usage:
            quota_boundary = self._check_quotas(current_usage, intent, context)
            if quota_boundary:
                return quota_boundary

        return None

    def _is_higher_tier(self, required_edition: EditionTier) -> bool:
        """Check if required edition is higher than current.

        Args:
            required_edition: Required edition tier

        Returns:
            True if required edition is higher than current
        """
        tier_order = {
            EditionTier.COMMUNITY: 1,
            EditionTier.BUSINESS: 2,
            EditionTier.ENTERPRISE: 3,
        }
        return tier_order[required_edition] > tier_order[self.user_edition]

    def _check_quotas(
        self,
        current_usage: Dict[str, int],
        intent: str,
        context: Optional[Dict]
    ) -> Optional[EditionBoundary]:
        """Check if current usage exceeds edition quotas.

        Args:
            current_usage: Current usage metrics
            intent: User intent
            context: Additional context

        Returns:
            EditionBoundary if quota exceeded, None otherwise
        """
        quotas = self.EDITION_QUOTAS[self.user_edition]

        for metric, limit in quotas.items():
            if limit == -1:  # Unlimited
                continue

            current = current_usage.get(metric, 0)
            if current >= limit:
                # Determine next tier that has higher quota
                next_tier = self._get_next_tier()
                return EditionBoundary(
                    boundary_type=BoundaryType.QUOTA_EXCEEDED,
                    current_edition=self.user_edition,
                    required_edition=next_tier,
                    feature_name=metric,
                    user_intent=intent,
                    context={
                        **(context or {}),
                        'current_usage': current,
                        'limit': limit,
                        'metric': metric,
                    }
                )

        return None

    def _get_next_tier(self) -> EditionTier:
        """Get the next higher edition tier.

        Returns:
            Next edition tier
        """
        if self.user_edition == EditionTier.COMMUNITY:
            return EditionTier.BUSINESS
        elif self.user_edition == EditionTier.BUSINESS:
            return EditionTier.ENTERPRISE
        else:
            return EditionTier.ENTERPRISE  # Already at top

    def get_upgrade_benefits(self, target_edition: EditionTier) -> List[str]:
        """Get list of benefits for upgrading to target edition.

        Args:
            target_edition: Target edition tier

        Returns:
            List of key benefits
        """
        if target_edition == EditionTier.BUSINESS:
            return [
                "AI Governance & Quality Monitoring",
                "Advanced ML Features (semantic search, pattern learning)",
                "Pod & Swarm Intelligence for team coordination",
                "50 AI agents and 50 adapters",
                "100GB storage",
                "SLA management and approval workflows",
                "Priority support",
            ]
        elif target_edition == EditionTier.ENTERPRISE:
            return [
                "Everything in Business, plus:",
                "Multi-tenant organization management",
                "Cross-organization workflows and federation",
                "Advanced compliance and audit logging",
                "SSO & role-based access control",
                "White-labeling capabilities",
                "Company-wide automation and strategic planning",
                "Unlimited agents, adapters, and workflows",
                "Dedicated support and custom integrations",
            ]
        return []

    def analyze_intent_for_features(self, intent: str, context: Dict) -> List[str]:
        """Analyze user intent to determine required features.

        Args:
            intent: User's stated intent
            context: Conversation context

        Returns:
            List of feature names required
        """
        features = []
        intent_lower = intent.lower()

        # AI Governance keywords
        if any(kw in intent_lower for kw in ['bias', 'fairness', 'quality', 'governance', 'compliance', 'validate']):
            features.append('ai_governance')

        # Pattern learning keywords
        if any(kw in intent_lower for kw in ['learn', 'pattern', 'optimize', 'improve']):
            features.append('pattern_learning')

        # Pod/Swarm keywords
        if any(kw in intent_lower for kw in ['team', 'collaborate', 'swarm', 'pod', 'group']):
            features.append('pod_swarm_intelligence')

        # Multi-tenancy keywords
        if any(kw in intent_lower for kw in ['organization', 'department', 'multi-tenant', 'cross-org']):
            features.append('organization_management')

        # Company automation keywords
        if any(kw in intent_lower for kw in ['company', 'enterprise', 'automate everything', 'strategic']):
            features.append('company_automation')

        # Semantic search keywords
        if any(kw in intent_lower for kw in ['semantic', 'similar', 'find related', 'search by meaning']):
            features.append('semantic_search')

        return features

    def can_suggest_workaround(self, boundary: EditionBoundary) -> Tuple[bool, Optional[str]]:
        """Check if there's a workaround in the current edition.

        Args:
            boundary: Detected edition boundary

        Returns:
            Tuple of (can_workaround, workaround_suggestion)
        """
        workarounds = {
            'pod_swarm_intelligence': (
                True,
                "You can manually coordinate multiple agents in Community edition by creating "
                "separate workflows for each agent and managing their execution order."
            ),
            'semantic_search': (
                True,
                "You can use keyword-based search in Community edition. For more advanced "
                "matching, try using specific tags and categories in your workflows."
            ),
            'pattern_learning': (
                False,
                "Pattern learning requires Business edition. However, you can manually save "
                "successful workflows as templates for reuse."
            ),
            'organization_management': (
                False,
                "Multi-organization features require Enterprise edition and cannot be "
                "replicated in lower tiers."
            ),
        }

        return workarounds.get(
            boundary.feature_name,
            (False, None)
        )
