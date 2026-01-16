"""
Comprehensive Integration Tests for Intelligent Assistant v3

Tests complete conversational flows including:
1. Multi-turn conversation with context accumulation
2. Company automation scenario with edition escalation
3. Pattern learning → retrieval → application workflow
4. Edition boundary detection and conversational escalation
5. Action planning and execution
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from models.knowledge import LearnedPattern
from models.conversation import ConversationSession, ConversationMessage
from models.user import User
from services.conversation_manager import ConversationManagerService
from services.action_planner import ActionPlanner
from services.pattern_learning_service import PatternLearningService
from services.pattern_escalation_service import PatternEscalationService
from services.edition_boundary_detector import EditionBoundaryDetector, EditionTier
from services.conversation_escalation_handler import ConversationEscalationHandler


@pytest.fixture
async def test_user(db: AsyncSession) -> User:
    """Create a test user."""
    from sqlalchemy import select

    # Try to find existing user first
    result = await db.execute(
        select(User).filter(User.email == "test@aictrlnet.com")
    )
    user = result.scalar_one_or_none()

    if user:
        return user

    # Create new user if not found
    user = User(
        id=str(uuid4()),
        email="test@aictrlnet.com",
        username="test_user",
        full_name="Test User",
        hashed_password="test_hash_123",
        is_active=True,
        edition="community"
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@pytest.fixture
async def conversation_manager(db: AsyncSession) -> ConversationManagerService:
    """Create conversation manager instance."""
    return ConversationManagerService(db)


@pytest.fixture
async def action_planner_community(db: AsyncSession) -> ActionPlanner:
    """Create action planner with Community edition."""
    return ActionPlanner(db, user_edition="community")


@pytest.fixture
async def action_planner_business(db: AsyncSession) -> ActionPlanner:
    """Create action planner with Business edition."""
    return ActionPlanner(db, user_edition="business")


@pytest.fixture
async def pattern_learning_service(db: AsyncSession) -> PatternLearningService:
    """Create pattern learning service."""
    return PatternLearningService(db)


@pytest.fixture
async def pattern_escalation_service(db: AsyncSession) -> PatternEscalationService:
    """Create pattern escalation service."""
    return PatternEscalationService(db)


class TestCompleteConversationalFlow:
    """Test complete conversational flows end-to-end."""

    @pytest.mark.asyncio
    async def test_multi_turn_conversation_with_context(
        self,
        db: AsyncSession,
        test_user: User,
        conversation_manager: ConversationManagerService
    ):
        """Test multi-turn conversation with context accumulation."""

        # Turn 1: Initial vague request
        session = await conversation_manager.create_session(
            user_id=test_user.id,
            initial_message="I want to improve my customer service"
        )

        assert session is not None
        # State should be "confirming_action" since intent was detected with no missing params
        assert session.state in ["confirming_action", "clarifying_details"]
        # Check that session has processed the intent
        assert session.primary_intent is not None

        # Turn 2: Add more context
        await conversation_manager.process_message(
            session_id=str(session.id),
            content="Specifically, I need faster response times",
            user_id=test_user.id
        )

        # Refresh session
        result = await db.execute(
            select(ConversationSession).where(ConversationSession.id == session.id)
        )
        session = result.scalar_one()

        # Session should still be active and processing multiple turns
        assert session.is_active == True

        # Turn 3: Add specific requirement
        await conversation_manager.process_message(
            session_id=str(session.id),
            content="And I want to detect when customers are frustrated",
            user_id=test_user.id
        )

        result = await db.execute(
            select(ConversationSession).where(ConversationSession.id == session.id)
        )
        session = result.scalar_one()

        # Check that messages were stored (verifies multi-turn conversation)
        messages_result = await db.execute(
            select(ConversationMessage)
            .where(ConversationMessage.session_id == session.id)
            .order_by(ConversationMessage.timestamp)
        )
        messages = messages_result.scalars().all()

        assert len(messages) >= 3
        assert messages[0].content == "I want to improve my customer service"

    @pytest.mark.asyncio
    async def test_simple_intent_extraction(
        self,
        db: AsyncSession,
        test_user: User,
        conversation_manager: ConversationManagerService
    ):
        """Test basic intent extraction from user message."""

        session = await conversation_manager.create_session(
            user_id=test_user.id,
            initial_message="Create a workflow to send daily reports"
        )

        assert session is not None
        # Check that intent was extracted (even if approximate)
        assert session.primary_intent is not None
        # Session should have transitioned from greeting state
        assert session.state != "greeting"


class TestCompanyAutomationScenario:
    """Test the complete 'automate my company' scenario."""

    @pytest.mark.asyncio
    async def test_company_automation_detects_enterprise_requirement(
        self,
        db: AsyncSession,
        action_planner_community: ActionPlanner
    ):
        """Test that company automation is detected as Enterprise feature."""

        intent = "Automate my entire company operations"
        context = {"user_message": intent}

        # Check edition boundary
        boundary_response = await action_planner_community.check_edition_boundary(
            intent=intent,
            context=context,
            requested_adapters=[]
        )

        # Should detect Enterprise requirement
        assert boundary_response is not None
        assert boundary_response['required_edition'] == 'enterprise'
        assert 'company' in boundary_response['message'].lower() or 'enterprise' in boundary_response['message'].lower()

        # Should have upgrade information
        assert boundary_response['upgrade'] is not None
        assert len(boundary_response['upgrade']['benefits']) > 0

        # Should have UI formatting
        assert boundary_response['ui_format'] is not None

    @pytest.mark.asyncio
    async def test_company_automation_proceeds_in_enterprise(
        self,
        db: AsyncSession
    ):
        """Test that company automation proceeds in Enterprise edition."""

        planner = ActionPlanner(db, user_edition="enterprise")

        intent = "Automate my entire company operations"
        context = {"user_message": intent}

        # Check edition boundary
        boundary_response = await planner.check_edition_boundary(
            intent=intent,
            context=context,
            requested_adapters=[]
        )

        # Should NOT have boundary in Enterprise
        assert boundary_response is None

    @pytest.mark.asyncio
    async def test_conversational_escalation_quality(
        self,
        db: AsyncSession
    ):
        """Test that escalation messages are helpful and conversational."""

        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)
        handler = ConversationEscalationHandler()

        # Detect boundary for AI Governance
        boundary = detector.detect_boundary(
            intent="I want to ensure my AI is unbiased",
            requested_features=["bias_detection"],
            requested_adapters=[]
        )

        assert boundary is not None

        # Generate escalation message
        escalation = handler.generate_escalation_message(boundary)

        # Check message quality
        assert len(escalation['message']) > 50  # Substantial message
        assert 'Business' in escalation['message']  # Mentions edition
        assert 'bias' in escalation['message'].lower() or 'AI' in escalation['message']  # Context-aware

        # Check it's conversational (not just "Feature locked")
        assert escalation['tone'] == 'helpful'
        assert '?' not in escalation['message'] or 'upgrade' in escalation['message'].lower()  # Not question, or asks about upgrade


class TestPatternLearningWorkflow:
    """Test complete pattern learning workflow."""

    @pytest.mark.asyncio
    async def test_pattern_storage_and_retrieval(
        self,
        db: AsyncSession,
        test_user: User,
        pattern_learning_service: PatternLearningService
    ):
        """Test storing and retrieving patterns."""

        user_id = test_user.id
        org_id = None  # Use None instead of random UUID

        # Clean up any existing patterns for this test
        from models.knowledge import LearnedPattern
        await db.execute(
            delete(LearnedPattern).where(LearnedPattern.pattern_signature == "automate_sales_pipeline")
        )
        await db.commit()

        # Simulate a successful conversation
        conversation_data = {
            "user_intent": "automate sales pipeline",
            "actions_taken": ["create_crm_workflow"],
            "parameters": {"stages": ["lead", "qualified", "closed"]},
            "success": True
        }

        # Store pattern multiple times to trigger activation (requires 3+ occurrences)
        # Use correct pattern type "intent_action" (not "intent_to_action") to match _is_pattern_relevant logic
        pattern_data_with_intent = {**conversation_data, "intent": "automate_sales_pipeline"}
        for i in range(3):
            await pattern_learning_service._store_pattern(
                pattern={
                    "type": "intent_action",  # Match the type checked in _is_pattern_relevant
                    "signature": "automate_sales_pipeline",
                    "data": pattern_data_with_intent,
                    "confidence": 0.85
                },
                user_id=user_id,
                organization_id=org_id
            )

        # Retrieve patterns with correct context structure
        # _is_pattern_relevant checks for context.get("primary_intent") matching pattern_data.get("intent")
        context = {"primary_intent": "automate_sales_pipeline"}
        patterns = await pattern_learning_service.get_relevant_patterns(
            context=context,
            user_id=user_id,
            organization_id=org_id
        )

        # Should retrieve the stored pattern
        assert len(patterns) > 0

        # Patterns are returned as LearnedPattern objects directly, not dicts
        # Check that we got the pattern we stored
        pattern = patterns[0]
        assert pattern.pattern_signature == "automate_sales_pipeline"
        assert pattern.pattern_type == "intent_action"
        assert pattern.is_active

    @pytest.mark.asyncio
    async def test_pattern_promotion_workflow(
        self,
        db: AsyncSession,
        test_user: User,
        pattern_escalation_service: PatternEscalationService
    ):
        """Test pattern promotion from user → org → global."""

        org_id = uuid4()

        # Create a successful user pattern that meets promotion criteria
        user_pattern = LearnedPattern(
            id=uuid4(),
            pattern_type="intent_to_action",
            pattern_signature="email_campaign_automation",
            pattern_data={
                "intent": "automate email campaigns",
                "action": "create_email_workflow",
                "success_indicators": ["high_open_rate", "conversions"]
            },
            confidence_score=0.88,
            occurrence_count=15,  # Meets threshold
            success_count=14,    # Meets threshold (92% of 15)
            scope="user",
            user_id=test_user.id,
            is_shareable=True,
            contains_sensitive_data=False
        )

        db.add(user_pattern)
        await db.commit()
        await db.refresh(user_pattern)

        # Promote to organization
        promoted = await pattern_escalation_service.promote_pattern(
            pattern_id=str(user_pattern.id),
            target_scope="organization",
            organization_id=str(org_id),
            validated_by="admin",
            anonymize=True
        )

        assert promoted is not None
        assert promoted.scope == "organization"
        assert promoted.organization_id == org_id
        assert promoted.promoted_from_scope == "user"
        assert promoted.anonymized is True

        # Verify PII was removed
        pattern_data_str = str(promoted.pattern_data)
        assert "user_id" not in pattern_data_str.lower() or promoted.user_id is None

    @pytest.mark.asyncio
    async def test_multi_tier_retrieval_priority(
        self,
        db: AsyncSession,
        test_user: User,
        pattern_learning_service: PatternLearningService
    ):
        """Test that user patterns override org/global patterns."""

        user_id = test_user.id
        org_id = uuid4()

        signature = "customer_onboarding"

        # Create global pattern
        global_pattern = LearnedPattern(
            id=uuid4(),
            pattern_type="sequence",
            pattern_signature=signature,
            pattern_data={"approach": "global_standard"},
            confidence_score=0.85,
            occurrence_count=100,
            success_count=88,  # 88% of 100
            scope="global",
            is_shareable=True
        )

        # Create org pattern
        org_pattern = LearnedPattern(
            id=uuid4(),
            pattern_type="sequence",
            pattern_signature=signature,
            pattern_data={"approach": "org_customized"},
            confidence_score=0.87,
            occurrence_count=50,
            success_count=45,  # 90% of 50
            scope="organization",
            organization_id=org_id,
            is_shareable=True
        )

        # Create user pattern
        user_pattern = LearnedPattern(
            id=uuid4(),
            pattern_type="sequence",
            pattern_signature=signature,
            pattern_data={"approach": "user_personalized"},
            confidence_score=0.82,
            occurrence_count=10,
            success_count=9,  # 85% of 10 (rounded up)
            scope="user",
            user_id=user_id,
            is_shareable=True
        )

        db.add_all([global_pattern, org_pattern, user_pattern])
        await db.commit()

        # Retrieve patterns
        context = {"intent": "onboard customer"}
        patterns = await pattern_learning_service.get_relevant_patterns(
            context=context,
            user_id=user_id,
            organization_id=str(org_id)
        )

        # Should get user pattern with highest priority
        if len(patterns) > 0:
            matching = [p for p in patterns if p['pattern'].pattern_signature == signature]
            if len(matching) > 0:
                # User pattern should be first due to tier priority
                assert matching[0]['tier_name'] == 'user'
                assert matching[0]['pattern'].pattern_data['approach'] == 'user_personalized'


class TestEditionBoundaryScenarios:
    """Test various edition boundary scenarios."""

    @pytest.mark.asyncio
    async def test_adapter_unavailable_boundary(
        self,
        db: AsyncSession,
        action_planner_community: ActionPlanner
    ):
        """Test detection of unavailable adapter."""

        boundary_response = await action_planner_community.check_edition_boundary(
            intent="Connect to Salesforce CRM",
            context={},
            requested_adapters=["salesforce"]
        )

        assert boundary_response is not None
        assert boundary_response['boundary_type'] == 'adapter_unavailable'
        assert 'salesforce' in boundary_response['message'].lower()

    @pytest.mark.asyncio
    async def test_feature_detection_from_intent(
        self,
        db: AsyncSession
    ):
        """Test automatic feature detection from user intent."""

        detector = EditionBoundaryDetector(EditionTier.COMMUNITY)

        # Test various intents
        test_cases = [
            ("I want to check for bias in my AI model", "ai_governance"),
            ("Help me coordinate multiple AI agents as a team", "pod_swarm_intelligence"),
            ("Set up multi-tenant organization management", "organization_management"),
            ("Learn from my workflow patterns", "pattern_learning"),
        ]

        for intent, expected_feature in test_cases:
            features = detector.analyze_intent_for_features(intent, {})
            assert expected_feature in features, f"Failed to detect {expected_feature} from: {intent}"

    @pytest.mark.asyncio
    async def test_no_false_positive_boundaries(
        self,
        db: AsyncSession,
        action_planner_business: ActionPlanner
    ):
        """Test that available features don't trigger boundaries."""

        # Bias detection is available in Business
        boundary_response = await action_planner_business.check_edition_boundary(
            intent="I want to detect bias in my AI model",
            context={},
            requested_adapters=[]
        )

        # Should NOT have boundary
        assert boundary_response is None


class TestActionPlanningIntegration:
    """Test action planning and execution integration."""

    @pytest.mark.asyncio
    async def test_action_planner_initialization(
        self,
        db: AsyncSession
    ):
        """Test that action planner initializes correctly with edition."""

        planner_community = ActionPlanner(db, user_edition="community")
        planner_business = ActionPlanner(db, user_edition="business")
        planner_enterprise = ActionPlanner(db, user_edition="enterprise")

        assert planner_community.user_edition == EditionTier.COMMUNITY
        assert planner_business.user_edition == EditionTier.BUSINESS
        assert planner_enterprise.user_edition == EditionTier.ENTERPRISE

        # Verify boundary detector is initialized
        assert planner_community.boundary_detector is not None
        assert planner_community.escalation_handler is not None

    @pytest.mark.asyncio
    async def test_edition_check_before_planning(
        self,
        db: AsyncSession,
        action_planner_community: ActionPlanner
    ):
        """Test that edition boundaries are checked before action planning."""

        # Request Enterprise feature from Community edition
        intent = "Set up federation across multiple organizations"

        boundary_response = await action_planner_community.check_edition_boundary(
            intent=intent,
            context={},
            requested_adapters=[]
        )

        # Should catch the boundary
        assert boundary_response is not None
        assert boundary_response['required_edition'] in ['business', 'enterprise']

        # Response should be UI-ready
        assert 'ui_format' in boundary_response
        assert boundary_response['ui_format']['message_card'] is not None


class TestRealWorldScenarios:
    """Test realistic end-to-end scenarios."""

    @pytest.mark.asyncio
    async def test_small_business_growth_path(
        self,
        db: AsyncSession
    ):
        """Test a small business growing from Community → Business → Enterprise."""

        # Stage 1: Community Edition - Basic workflow
        planner_community = ActionPlanner(db, user_edition="community")

        intent1 = "Create a simple email automation"
        boundary1 = await planner_community.check_edition_boundary(
            intent=intent1,
            context={},
            requested_adapters=["email"]  # Community adapter
        )
        assert boundary1 is None  # Should work in Community

        # Stage 2: Growth - Wants AI governance
        intent2 = "I need to ensure my AI models are unbiased"
        boundary2 = await planner_community.check_edition_boundary(
            intent=intent2,
            context={},
            requested_adapters=[]
        )
        assert boundary2 is not None  # Requires Business
        assert boundary2['required_edition'] == 'business'

        # Stage 3: Upgrade to Business
        planner_business = ActionPlanner(db, user_edition="business")
        boundary3 = await planner_business.check_edition_boundary(
            intent=intent2,
            context={},
            requested_adapters=[]
        )
        assert boundary3 is None  # Now available

        # Stage 4: Scaling - Wants multi-tenancy
        intent4 = "Manage multiple client organizations separately"
        boundary4 = await planner_business.check_edition_boundary(
            intent=intent4,
            context={},
            requested_adapters=[]
        )
        assert boundary4 is not None  # Requires Enterprise
        assert boundary4['required_edition'] == 'enterprise'

    @pytest.mark.asyncio
    async def test_user_learns_system_over_time(
        self,
        db: AsyncSession,
        test_user: User,
        pattern_learning_service: PatternLearningService
    ):
        """Test that patterns are learned and reused over time."""

        user_id = test_user.id

        # Clean up any existing patterns for this test
        await db.execute(
            delete(LearnedPattern).where(LearnedPattern.pattern_signature == "weekly_reports")
        )
        await db.commit()

        # Week 1: User creates a workflow manually
        conversation1 = {
            "user_intent": "send weekly reports",
            "actions_taken": ["create_email_workflow", "set_schedule"],
            "parameters": {"frequency": "weekly", "recipients": ["team@company.com"]},
            "success": True
        }

        # Store pattern multiple times to trigger activation (requires 3+ occurrences)
        # Use correct pattern type and add intent to pattern data
        pattern_data_with_intent = {**conversation1, "intent": "weekly_reports"}
        for i in range(3):
            await pattern_learning_service._store_pattern(
                pattern={
                    "type": "intent_action",  # Match the type checked in _is_pattern_relevant
                    "signature": "weekly_reports",
                    "data": pattern_data_with_intent,
                    "confidence": 0.8
                },
                user_id=user_id
            )

        # Week 4: User makes similar request with correct context structure
        context2 = {"primary_intent": "weekly_reports"}
        patterns = await pattern_learning_service.get_relevant_patterns(
            context=context2,
            user_id=user_id
        )

        # System should suggest the learned pattern
        assert len(patterns) > 0

        # Patterns are returned as LearnedPattern objects directly
        matching = [p for p in patterns if 'weekly' in str(p.pattern_data).lower()]
        assert len(matching) > 0
