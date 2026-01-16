"""Tests for conversational edition escalation system.

Tests cover:
- Edition boundary detection (features, adapters, quotas)
- Conversational escalation message generation
- Workaround suggestions
- UI formatting for chat interface
- Integration with action planner
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from services.edition_boundary_detector import (
    EditionBoundaryDetector,
    EditionBoundary,
    BoundaryType,
    EditionTier
)
from services.conversation_escalation_handler import ConversationEscalationHandler


class TestEditionBoundaryDetector:
    """Test edition boundary detection."""

    def test_detect_feature_boundary_business(self):
        """Test detecting Business feature from Community edition."""
        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        boundary = detector.detect_boundary(
            intent="I want to detect bias in my AI model",
            requested_features=["bias_detection"]
        )

        assert boundary is not None
        assert boundary.boundary_type == BoundaryType.FEATURE_LOCKED
        assert boundary.current_edition == EditionTier.COMMUNITY
        assert boundary.required_edition == EditionTier.BUSINESS
        assert boundary.feature_name == "bias_detection"

    def test_detect_feature_boundary_enterprise(self):
        """Test detecting Enterprise feature from Business edition."""
        detector = EditionBoundaryDetector(EditionTier.BUSINESS)

        boundary = detector.detect_boundary(
            intent="Set up multi-tenant organization management",
            requested_features=["organization_management"]
        )

        assert boundary is not None
        assert boundary.boundary_type == BoundaryType.FEATURE_LOCKED
        assert boundary.current_edition == EditionTier.BUSINESS
        assert boundary.required_edition == EditionTier.ENTERPRISE
        assert boundary.feature_name == "organization_management"

    def test_no_boundary_for_available_feature(self):
        """Test that no boundary is detected for available features."""
        detector = EditionBoundaryDetector(EditionTier.BUSINESS)

        # Bias detection is available in Business
        boundary = detector.detect_boundary(
            intent="I want to detect bias",
            requested_features=["bias_detection"]
        )

        assert boundary is None

    def test_detect_adapter_boundary(self):
        """Test detecting adapter tier requirements."""
        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        boundary = detector.detect_boundary(
            intent="Connect to Salesforce",
            requested_features=[],
            requested_adapters=["salesforce"]
        )

        assert boundary is not None
        assert boundary.boundary_type == BoundaryType.ADAPTER_UNAVAILABLE
        assert boundary.current_edition == EditionTier.COMMUNITY
        assert boundary.required_edition == EditionTier.BUSINESS
        assert boundary.feature_name == "salesforce"

    def test_community_adapters_available(self):
        """Test that Community adapters are available in Community edition."""
        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        # Ollama is a Community adapter
        boundary = detector.detect_boundary(
            intent="Use Ollama for AI",
            requested_features=[],
            requested_adapters=["ollama"]
        )

        assert boundary is None

    def test_detect_quota_exceeded(self):
        """Test detecting quota exceeded boundary."""
        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        current_usage = {
            "workflows_per_month": 100,  # At limit
            "agents_max": 3,
            "adapters_max": 8
        }

        boundary = detector.detect_boundary(
            intent="Create another workflow",
            requested_features=[],
            current_usage=current_usage
        )

        assert boundary is not None
        assert boundary.boundary_type == BoundaryType.QUOTA_EXCEEDED
        assert boundary.current_edition == EditionTier.COMMUNITY
        assert boundary.required_edition == EditionTier.BUSINESS
        assert boundary.context["metric"] == "workflows_per_month"
        assert boundary.context["current_usage"] == 100
        assert boundary.context["limit"] == 100

    def test_enterprise_unlimited_quotas(self):
        """Test that Enterprise has unlimited quotas."""
        detector = EditionBoundaryDetector(EditionTier.ENTERPRISE)

        current_usage = {
            "workflows_per_month": 1000000,
            "agents_max": 5000
        }

        boundary = detector.detect_boundary(
            intent="Create workflow",
            requested_features=[],
            current_usage=current_usage
        )

        # Should have no quota boundaries in Enterprise
        assert boundary is None

    def test_analyze_intent_for_features(self):
        """Test automatic feature detection from intent."""
        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        # Test AI Governance detection
        features = detector.analyze_intent_for_features(
            "I need to check for bias in my model",
            {}
        )
        assert "ai_governance" in features

        # Test pattern learning detection
        features = detector.analyze_intent_for_features(
            "Can the system learn from my workflows?",
            {}
        )
        assert "pattern_learning" in features

        # Test pod/swarm detection
        features = detector.analyze_intent_for_features(
            "I want multiple AI agents to collaborate as a team",
            {}
        )
        assert "pod_swarm_intelligence" in features

        # Test organization management detection
        features = detector.analyze_intent_for_features(
            "Set up multi-tenant organization structure",
            {}
        )
        assert "organization_management" in features

        # Test company automation detection
        features = detector.analyze_intent_for_features(
            "Automate my entire company operations",
            {}
        )
        assert "company_automation" in features

    def test_get_upgrade_benefits_business(self):
        """Test getting upgrade benefits for Business edition."""
        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        benefits = detector.get_upgrade_benefits(EditionTier.BUSINESS)

        assert len(benefits) > 0
        assert any("AI Governance" in b for b in benefits)
        assert any("Pod & Swarm" in b for b in benefits)
        assert any("50 AI agents" in b for b in benefits)

    def test_get_upgrade_benefits_enterprise(self):
        """Test getting upgrade benefits for Enterprise edition."""
        detector = EditionBoundaryDetector(EditionTier.BUSINESS)

        benefits = detector.get_upgrade_benefits(EditionTier.ENTERPRISE)

        assert len(benefits) > 0
        assert any("Multi-tenant" in b for b in benefits)
        assert any("Unlimited" in b for b in benefits)
        assert any("SSO" in b for b in benefits)

    def test_workaround_suggestions(self):
        """Test workaround suggestions for some features."""
        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        boundary = EditionBoundary(
            boundary_type=BoundaryType.FEATURE_LOCKED,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="pod_swarm_intelligence",
            user_intent="coordinate multiple agents"
        )

        can_workaround, suggestion = detector.can_suggest_workaround(boundary)

        assert can_workaround is True
        assert "manually coordinate" in suggestion.lower()
        assert "separate workflows" in suggestion.lower()

    def test_no_workaround_for_enterprise_features(self):
        """Test that some features have no workarounds."""
        detector = EditionBoundaryDetector(EditionTier.BUSINESS)

        boundary = EditionBoundary(
            boundary_type=BoundaryType.FEATURE_LOCKED,
            current_edition=EditionTier.BUSINESS,
            required_edition=EditionTier.ENTERPRISE,
            feature_name="organization_management",
            user_intent="multi-tenant setup"
        )

        can_workaround, suggestion = detector.can_suggest_workaround(boundary)

        assert can_workaround is False


class TestConversationEscalationHandler:
    """Test conversational escalation message generation."""

    def test_generate_feature_locked_message_ai_governance(self):
        """Test generating message for AI Governance feature lock."""
        handler = ConversationEscalationHandler()

        boundary = EditionBoundary(
            boundary_type=BoundaryType.FEATURE_LOCKED,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="ai_governance",
            user_intent="detect bias in my AI model"
        )

        response = handler.generate_escalation_message(boundary)

        assert response['type'] == 'edition_escalation'
        assert response['boundary_type'] == 'feature_locked'
        assert response['current_edition'] == 'community'
        assert response['required_edition'] == 'business'
        assert 'AI Governance' in response['message']
        assert 'Business' in response['message']

    def test_generate_quota_exceeded_message(self):
        """Test generating message for quota exceeded."""
        handler = ConversationEscalationHandler()

        boundary = EditionBoundary(
            boundary_type=BoundaryType.QUOTA_EXCEEDED,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="workflows_per_month",
            user_intent="create another workflow",
            context={
                "metric": "workflows_per_month",
                "current_usage": 100,
                "limit": 100
            }
        )

        response = handler.generate_escalation_message(boundary)

        assert response['boundary_type'] == 'quota_exceeded'
        assert 'limit' in response['message'].lower()
        assert '100/100' in response['message']

    def test_generate_adapter_unavailable_message(self):
        """Test generating message for unavailable adapter."""
        handler = ConversationEscalationHandler()

        boundary = EditionBoundary(
            boundary_type=BoundaryType.ADAPTER_UNAVAILABLE,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="salesforce",
            user_intent="connect to Salesforce CRM"
        )

        response = handler.generate_escalation_message(boundary)

        assert response['boundary_type'] == 'adapter_unavailable'
        assert 'salesforce' in response['message'].lower()
        assert 'Business' in response['message']

    def test_include_workaround_in_response(self):
        """Test including workaround suggestions."""
        handler = ConversationEscalationHandler()

        boundary = EditionBoundary(
            boundary_type=BoundaryType.FEATURE_LOCKED,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="pod_swarm_intelligence",
            user_intent="coordinate agents"
        )

        response = handler.generate_escalation_message(
            boundary,
            include_workaround=True
        )

        assert response['workaround'] is not None
        assert response['workaround']['available'] is True
        assert len(response['workaround']['message']) > 0

    def test_include_upgrade_benefits(self):
        """Test including upgrade benefits in response."""
        handler = ConversationEscalationHandler()

        boundary = EditionBoundary(
            boundary_type=BoundaryType.FEATURE_LOCKED,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="ai_governance",
            user_intent="ensure AI quality"
        )

        response = handler.generate_escalation_message(
            boundary,
            include_upgrade_link=True
        )

        assert response['upgrade'] is not None
        assert response['upgrade']['target_edition'] == 'business'
        assert len(response['upgrade']['benefits']) > 0
        assert response['upgrade']['cta_text'] == "Upgrade to Business Edition"

    def test_generate_follow_up_suggestions(self):
        """Test generating follow-up suggestions."""
        handler = ConversationEscalationHandler()

        boundary = EditionBoundary(
            boundary_type=BoundaryType.FEATURE_LOCKED,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="semantic_search",
            user_intent="find similar workflows"
        )

        suggestions = handler.generate_follow_up_suggestions(
            boundary,
            conversation_context={}
        )

        assert len(suggestions) > 0
        assert any("Business edition" in s for s in suggestions)
        assert any("pricing" in s.lower() for s in suggestions)

    def test_format_for_chat_ui(self):
        """Test UI formatting for chat interface."""
        handler = ConversationEscalationHandler()

        boundary = EditionBoundary(
            boundary_type=BoundaryType.FEATURE_LOCKED,
            current_edition=EditionTier.COMMUNITY,
            required_edition=EditionTier.BUSINESS,
            feature_name="ai_governance",
            user_intent="validate model quality"
        )

        escalation_response = handler.generate_escalation_message(boundary)
        ui_format = handler.format_for_chat_ui(escalation_response)

        # Check message card
        assert ui_format['message_card']['type'] == 'edition_boundary'
        assert ui_format['message_card']['title'] == 'Business Feature'
        assert ui_format['message_card']['icon'] == '🔒'
        assert ui_format['message_card']['color'] == 'blue'

        # Check upgrade card
        assert ui_format['upgrade_card'] is not None
        assert ui_format['upgrade_card']['type'] == 'upgrade_prompt'
        assert '/upgrade?target=business' in ui_format['upgrade_card']['action_url']

    def test_humanize_feature_names(self):
        """Test feature name humanization."""
        handler = ConversationEscalationHandler()

        assert handler._humanize_feature_name("ai_governance") == "AI Governance"
        assert handler._humanize_feature_name("pod_swarm_intelligence") == "Pod & Swarm Intelligence"
        assert handler._humanize_feature_name("organization_management") == "Organization Management"
        assert handler._humanize_feature_name("company_automation") == "Company-Wide Automation"

    def test_humanize_metric_names(self):
        """Test metric name humanization."""
        handler = ConversationEscalationHandler()

        assert handler._humanize_metric_name("workflows_per_month") == "monthly workflows"
        assert handler._humanize_metric_name("agents_max") == "AI agents"
        assert handler._humanize_metric_name("adapters_max") == "adapters"


class TestEditionEscalationIntegration:
    """Integration tests for edition escalation with action planner."""

    @pytest.mark.asyncio
    async def test_action_planner_edition_check(self):
        """Test action planner checks edition boundaries."""
        from services.action_planner import ActionPlanner

        mock_db = AsyncMock()
        planner = ActionPlanner(db=mock_db, user_edition="community")

        # Test detecting Business feature boundary
        boundary_response = await planner.check_edition_boundary(
            intent="I want to detect bias in my AI model",
            context={},
            requested_adapters=[]
        )

        assert boundary_response is not None
        assert boundary_response['boundary_type'] == 'feature_locked'
        assert boundary_response['required_edition'] == 'business'
        assert 'ui_format' in boundary_response

    @pytest.mark.asyncio
    async def test_action_planner_no_boundary(self):
        """Test action planner allows available features."""
        from services.action_planner import ActionPlanner

        mock_db = AsyncMock()
        planner = ActionPlanner(db=mock_db, user_edition="business")

        # Bias detection is available in Business
        boundary_response = await planner.check_edition_boundary(
            intent="I want to detect bias in my AI model",
            context={},
            requested_adapters=[]
        )

        # Should return None since feature is available
        assert boundary_response is None

    @pytest.mark.asyncio
    async def test_complete_escalation_workflow(self):
        """Test complete escalation workflow from intent to UI-ready response."""
        from services.action_planner import ActionPlanner

        mock_db = AsyncMock()
        planner = ActionPlanner(db=mock_db, user_edition="community")

        # User requests Enterprise feature
        intent = "Set up multi-organization management for my company"

        boundary_response = await planner.check_edition_boundary(
            intent=intent,
            context={},
            requested_adapters=[]
        )

        # Should detect Enterprise requirement
        assert boundary_response is not None
        assert boundary_response['required_edition'] == 'enterprise'

        # Should have conversational message
        assert len(boundary_response['message']) > 0
        assert 'Enterprise' in boundary_response['message']

        # Should have upgrade information
        assert boundary_response['upgrade'] is not None
        assert len(boundary_response['upgrade']['benefits']) > 0

        # Should have UI formatting
        assert boundary_response['ui_format'] is not None
        assert boundary_response['ui_format']['message_card'] is not None
        assert boundary_response['ui_format']['upgrade_card'] is not None

    @pytest.mark.asyncio
    async def test_escalation_with_adapter_request(self):
        """Test escalation when requesting unavailable adapter."""
        from services.action_planner import ActionPlanner

        mock_db = AsyncMock()
        planner = ActionPlanner(db=mock_db, user_edition="community")

        boundary_response = await planner.check_edition_boundary(
            intent="Connect to Salesforce to sync contacts",
            context={},
            requested_adapters=["salesforce"]
        )

        assert boundary_response is not None
        assert boundary_response['boundary_type'] == 'adapter_unavailable'
        assert boundary_response['feature_name'] == 'salesforce'
        assert 'Salesforce' in boundary_response['message']
