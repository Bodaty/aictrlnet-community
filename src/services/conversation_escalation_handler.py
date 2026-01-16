"""Conversation Escalation Handler for Edition Boundary Communication.

Generates natural, helpful conversational responses when users encounter
edition boundaries, with upgrade prompts and workarounds when available.

Part of the Intelligent Assistant v3 Edition Escalation Framework.
"""

from typing import Dict, List, Optional, Any
import logging

from services.edition_boundary_detector import (
    EditionBoundary,
    BoundaryType,
    EditionTier,
    EditionBoundaryDetector
)

logger = logging.getLogger(__name__)


class ConversationEscalationHandler:
    """Generates conversational responses for edition boundaries."""

    def __init__(self):
        """Initialize the escalation handler."""
        self.tone = "helpful"  # friendly, professional, helpful

    def generate_escalation_message(
        self,
        boundary: EditionBoundary,
        include_workaround: bool = True,
        include_upgrade_link: bool = True
    ) -> Dict[str, Any]:
        """Generate a conversational escalation message.

        Args:
            boundary: Detected edition boundary
            include_workaround: Whether to include workaround suggestions
            include_upgrade_link: Whether to include upgrade CTA

        Returns:
            Dictionary with message components
        """
        # Generate appropriate message based on boundary type
        if boundary.boundary_type == BoundaryType.FEATURE_LOCKED:
            message = self._generate_feature_locked_message(boundary)
        elif boundary.boundary_type == BoundaryType.QUOTA_EXCEEDED:
            message = self._generate_quota_exceeded_message(boundary)
        elif boundary.boundary_type == BoundaryType.ADAPTER_UNAVAILABLE:
            message = self._generate_adapter_unavailable_message(boundary)
        elif boundary.boundary_type == BoundaryType.WORKFLOW_RESTRICTED:
            message = self._generate_workflow_restricted_message(boundary)
        else:
            message = self._generate_generic_message(boundary)

        # Add workaround if available
        workaround = None
        if include_workaround:
            detector = EditionBoundaryDetector(boundary.current_edition)
            can_workaround, workaround_text = detector.can_suggest_workaround(boundary)
            if can_workaround and workaround_text:
                workaround = {
                    'available': True,
                    'message': workaround_text
                }

        # Add upgrade benefits
        upgrade_benefits = None
        if include_upgrade_link:
            detector = EditionBoundaryDetector(boundary.current_edition)
            benefits = detector.get_upgrade_benefits(boundary.required_edition)
            upgrade_benefits = {
                'target_edition': boundary.required_edition.value,
                'benefits': benefits,
                'cta_text': self._generate_cta_text(boundary.required_edition)
            }

        return {
            'type': 'edition_escalation',
            'boundary_type': boundary.boundary_type.value,
            'current_edition': boundary.current_edition.value,
            'required_edition': boundary.required_edition.value,
            'feature_name': boundary.feature_name,
            'message': message,
            'workaround': workaround,
            'upgrade': upgrade_benefits,
            'tone': 'helpful',
            'action_required': 'upgrade' if not workaround else 'consider_upgrade'
        }

    def _generate_feature_locked_message(self, boundary: EditionBoundary) -> str:
        """Generate message for feature-locked boundary.

        Args:
            boundary: Edition boundary details

        Returns:
            Conversational message string
        """
        feature_display = self._humanize_feature_name(boundary.feature_name)
        edition_display = boundary.required_edition.value.title()

        messages = {
            'ai_governance': (
                f"I'd love to help you with {boundary.user_intent}, but AI Governance features "
                f"like {feature_display} are available in {edition_display} edition. "
                f"These powerful tools help ensure your AI systems are fair, accurate, and compliant."
            ),
            'pod_swarm_intelligence': (
                f"Great idea! Pod and Swarm Intelligence for coordinating multiple AI agents "
                f"is a {edition_display} feature. This lets teams of AI agents work together "
                f"intelligently to solve complex problems."
            ),
            'semantic_search': (
                f"Semantic search is an amazing feature in {edition_display} edition! "
                f"It uses advanced ML to understand the meaning behind your queries, "
                f"not just keyword matching."
            ),
            'organization_management': (
                f"Multi-organization management is an {edition_display} capability that "
                f"lets you manage multiple teams, departments, or organizations in one place "
                f"with proper isolation and governance."
            ),
            'company_automation': (
                f"Company-wide automation and strategic planning features are designed for "
                f"{edition_display} edition. These help automate entire business processes "
                f"across departments and functions."
            ),
        }

        return messages.get(
            boundary.feature_name,
            f"The feature you're looking for ({feature_display}) is available in "
            f"{edition_display} edition. This would be perfect for {boundary.user_intent}!"
        )

    def _generate_quota_exceeded_message(self, boundary: EditionBoundary) -> str:
        """Generate message for quota exceeded boundary.

        Args:
            boundary: Edition boundary details

        Returns:
            Conversational message string
        """
        metric = boundary.context.get('metric', 'resource')
        current = boundary.context.get('current_usage', 0)
        limit = boundary.context.get('limit', 0)
        metric_display = self._humanize_metric_name(metric)

        return (
            f"You've hit your {boundary.current_edition.value.title()} edition limit for "
            f"{metric_display} ({current}/{limit}). You're doing great work! "
            f"To continue, consider upgrading to {boundary.required_edition.value.title()} "
            f"edition for higher limits or unlimited usage."
        )

    def _generate_adapter_unavailable_message(self, boundary: EditionBoundary) -> str:
        """Generate message for adapter unavailable boundary.

        Args:
            boundary: Edition boundary details

        Returns:
            Conversational message string
        """
        adapter_name = boundary.feature_name
        edition_display = boundary.required_edition.value.title()

        return (
            f"The {adapter_name} adapter is available in {edition_display} edition. "
            f"This integration would be perfect for {boundary.user_intent}. "
            f"{edition_display} edition includes access to premium integrations "
            f"with enterprise platforms."
        )

    def _generate_workflow_restricted_message(self, boundary: EditionBoundary) -> str:
        """Generate message for workflow restricted boundary.

        Args:
            boundary: Edition boundary details

        Returns:
            Conversational message string
        """
        edition_display = boundary.required_edition.value.title()

        return (
            f"This workflow template requires {edition_display} edition. "
            f"It includes advanced capabilities specifically designed for "
            f"{boundary.user_intent}. {edition_display} edition unlocks "
            f"sophisticated workflow templates for complex automation scenarios."
        )

    def _generate_generic_message(self, boundary: EditionBoundary) -> str:
        """Generate generic escalation message.

        Args:
            boundary: Edition boundary details

        Returns:
            Conversational message string
        """
        edition_display = boundary.required_edition.value.title()

        return (
            f"To accomplish {boundary.user_intent}, you'll need {edition_display} edition. "
            f"This unlocks powerful features designed for more advanced use cases."
        )

    def _generate_cta_text(self, target_edition: EditionTier) -> str:
        """Generate call-to-action text for upgrade.

        Args:
            target_edition: Target edition tier

        Returns:
            CTA text
        """
        if target_edition == EditionTier.BUSINESS:
            return "Upgrade to Business Edition"
        elif target_edition == EditionTier.ENTERPRISE:
            return "Explore Enterprise Edition"
        return "Learn More About Upgrades"

    def _humanize_feature_name(self, feature_name: str) -> str:
        """Convert feature name to human-readable format.

        Args:
            feature_name: Technical feature name

        Returns:
            Human-readable name
        """
        humanized = {
            'ai_governance': 'AI Governance',
            'model_validation': 'Model Validation',
            'bias_detection': 'Bias Detection',
            'quality_monitoring': 'Quality Monitoring',
            'risk_assessment': 'Risk Assessment',
            'semantic_search': 'Semantic Search',
            'pattern_learning': 'Pattern Learning',
            'intelligent_routing': 'Intelligent Routing',
            'pod_swarm_intelligence': 'Pod & Swarm Intelligence',
            'resource_pools': 'Resource Pools',
            'sla_management': 'SLA Management',
            'approval_workflows': 'Approval Workflows',
            'multi_tenancy': 'Multi-Tenancy',
            'organization_management': 'Organization Management',
            'federation': 'Federation',
            'cross_org_workflows': 'Cross-Organization Workflows',
            'white_labeling': 'White Labeling',
            'advanced_compliance': 'Advanced Compliance',
            'audit_logging': 'Audit Logging',
            'role_based_access': 'Role-Based Access Control',
            'sso_integration': 'SSO Integration',
            'company_automation': 'Company-Wide Automation',
            'strategic_planning': 'Strategic Planning',
            'cross_departmental': 'Cross-Departmental Automation',
        }
        return humanized.get(feature_name, feature_name.replace('_', ' ').title())

    def _humanize_metric_name(self, metric: str) -> str:
        """Convert metric name to human-readable format.

        Args:
            metric: Technical metric name

        Returns:
            Human-readable name
        """
        humanized = {
            'workflows_per_month': 'monthly workflows',
            'agents_max': 'AI agents',
            'adapters_max': 'adapters',
            'api_calls_per_day': 'daily API calls',
            'storage_gb': 'storage',
        }
        return humanized.get(metric, metric.replace('_', ' '))

    def generate_follow_up_suggestions(
        self,
        boundary: EditionBoundary,
        conversation_context: Dict
    ) -> List[str]:
        """Generate follow-up suggestions after escalation message.

        Args:
            boundary: Edition boundary details
            conversation_context: Current conversation context

        Returns:
            List of follow-up suggestion strings
        """
        suggestions = []

        # Check for workaround availability
        detector = EditionBoundaryDetector(boundary.current_edition)
        can_workaround, _ = detector.can_suggest_workaround(boundary)

        if can_workaround:
            suggestions.append("Show me the workaround for this")
            suggestions.append("What can I do in my current edition?")

        # Always offer upgrade path
        suggestions.append(f"Tell me more about {boundary.required_edition.value.title()} edition")
        suggestions.append("What are the pricing options?")

        # Offer alternative actions based on context
        if 'alternative_intents' in conversation_context:
            suggestions.append("Suggest something else I can do instead")

        return suggestions[:3]  # Limit to 3 suggestions

    def format_for_chat_ui(self, escalation_response: Dict) -> Dict:
        """Format escalation response for chat UI display.

        Args:
            escalation_response: Response from generate_escalation_message

        Returns:
            UI-formatted response
        """
        # Handle None input
        if escalation_response is None:
            return {}

        # Main message card
        message_card = {
            'type': 'edition_boundary',
            'title': f"{escalation_response['required_edition'].title()} Feature",
            'message': escalation_response['message'],
            'icon': '🔒' if escalation_response['boundary_type'] == 'feature_locked' else '📊',
            'color': 'blue',  # Informational, not error
        }

        # Workaround card (if available)
        workaround_card = None
        workaround = escalation_response.get('workaround') or {}
        if workaround.get('available'):
            workaround_card = {
                'type': 'workaround',
                'title': 'Alternative Approach',
                'message': workaround['message'],
                'icon': '💡',
                'color': 'yellow',
            }

        # Upgrade card
        upgrade_card = None
        if escalation_response.get('upgrade'):
            upgrade_info = escalation_response['upgrade']
            upgrade_card = {
                'type': 'upgrade_prompt',
                'title': upgrade_info['cta_text'],
                'benefits': upgrade_info['benefits'],
                'action_url': f"/upgrade?target={upgrade_info['target_edition']}",
                'icon': '⬆️',
                'color': 'green',
            }

        return {
            'message_card': message_card,
            'workaround_card': workaround_card,
            'upgrade_card': upgrade_card,
            'type': 'escalation_response',
        }
