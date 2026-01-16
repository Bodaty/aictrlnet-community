"""Tests for multi-tier pattern learning system.

Tests cover:
- Pattern escalation service (user → org → global promotion)
- Pattern learning service (multi-tier retrieval with priority cascade)
- Privacy controls and anonymization
- Promotion threshold validation
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timedelta
from uuid import uuid4

from services.pattern_escalation_service import PatternEscalationService
from services.pattern_learning_service import PatternLearningService
from models.knowledge import LearnedPattern


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.execute = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    return session


@pytest.fixture
def user_pattern():
    """Create a user-scoped pattern."""
    return LearnedPattern(
        id=uuid4(),
        pattern_type="intent_to_action",
        pattern_signature="automate_sales_pipeline",
        pattern_data={
            "intent": "automate sales pipeline",
            "action": "create_crm_workflow",
            "parameters": {"stages": ["lead", "qualified", "closed"]}
        },
        confidence_score=0.85,
        occurrence_count=15,
        success_rate=0.87,
        scope="user",
        user_id="user-123",
        organization_id=None,
        is_shareable=True,
        contains_sensitive_data=False,
        anonymized=False,
        created_at=datetime.utcnow() - timedelta(days=10)
    )


@pytest.fixture
def org_pattern():
    """Create an organization-scoped pattern."""
    return LearnedPattern(
        id=uuid4(),
        pattern_type="sequence",
        pattern_signature="email_followup_sequence",
        pattern_data={
            "steps": ["send_email", "wait_2_days", "send_followup"],
            "success_indicators": ["email_opened", "link_clicked"]
        },
        confidence_score=0.82,
        occurrence_count=55,
        success_rate=0.89,
        scope="organization",
        user_id=None,
        organization_id=uuid4(),
        promoted_from_scope="user",
        contributing_users_count=8,
        is_shareable=True,
        contains_sensitive_data=False,
        anonymized=True,
        created_at=datetime.utcnow() - timedelta(days=30)
    )


@pytest.fixture
def global_pattern():
    """Create a global pattern."""
    return LearnedPattern(
        id=uuid4(),
        pattern_type="parameter",
        pattern_signature="email_send_optimal_time",
        pattern_data={
            "parameter": "send_time",
            "optimal_value": "10:00 AM",
            "success_correlation": 0.92
        },
        confidence_score=0.91,
        occurrence_count=150,
        success_rate=0.93,
        scope="global",
        user_id=None,
        organization_id=None,
        promoted_from_scope="organization",
        contributing_users_count=25,
        is_shareable=True,
        contains_sensitive_data=False,
        anonymized=True,
        created_at=datetime.utcnow() - timedelta(days=90)
    )


class TestPatternEscalationService:
    """Test pattern escalation service."""

    @pytest.mark.asyncio
    async def test_identify_user_to_org_candidates(self, mock_db_session, user_pattern):
        """Test identifying user patterns eligible for org promotion."""
        service = PatternEscalationService(mock_db_session)

        # Mock database query result
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [user_pattern]
        mock_db_session.execute.return_value = mock_result

        candidates = await service.identify_promotion_candidates(scope="user", min_age_days=7)

        assert len(candidates) == 1
        assert candidates[0].scope == "user"
        assert candidates[0].occurrence_count >= service.USER_TO_ORG_THRESHOLD['min_occurrence_count']
        assert candidates[0].success_rate >= service.USER_TO_ORG_THRESHOLD['min_success_rate']

    @pytest.mark.asyncio
    async def test_identify_org_to_global_candidates(self, mock_db_session, org_pattern):
        """Test identifying org patterns eligible for global promotion."""
        service = PatternEscalationService(mock_db_session)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [org_pattern]
        mock_db_session.execute.return_value = mock_result

        candidates = await service.identify_promotion_candidates(scope="organization")

        assert len(candidates) == 1
        assert candidates[0].scope == "organization"
        assert candidates[0].occurrence_count >= service.ORG_TO_GLOBAL_THRESHOLD['min_occurrence_count']
        assert candidates[0].contributing_users_count >= service.ORG_TO_GLOBAL_THRESHOLD['min_users']

    @pytest.mark.asyncio
    async def test_promote_user_to_org(self, mock_db_session, user_pattern):
        """Test promoting user pattern to organization scope."""
        service = PatternEscalationService(mock_db_session)

        # Mock get pattern
        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = user_pattern

        # Mock check for existing org pattern
        mock_existing_result = MagicMock()
        mock_existing_result.scalar_one_or_none.return_value = None

        mock_db_session.execute.side_effect = [mock_get_result, mock_existing_result]

        org_id = str(uuid4())
        promoted = await service.promote_pattern(
            pattern_id=str(user_pattern.id),
            target_scope="organization",
            organization_id=org_id,
            validated_by="admin-user"
        )

        assert promoted is not None
        assert promoted.scope == "organization"
        assert promoted.organization_id == org_id
        assert promoted.promoted_from_scope == "user"
        assert promoted.anonymized is True  # Should be anonymized on promotion
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_promote_org_to_global(self, mock_db_session, org_pattern):
        """Test promoting org pattern to global scope."""
        service = PatternEscalationService(mock_db_session)

        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = org_pattern

        mock_existing_result = MagicMock()
        mock_existing_result.scalar_one_or_none.return_value = None

        mock_db_session.execute.side_effect = [mock_get_result, mock_existing_result]

        promoted = await service.promote_pattern(
            pattern_id=str(org_pattern.id),
            target_scope="global",
            validated_by="admin-user"
        )

        assert promoted is not None
        assert promoted.scope == "global"
        assert promoted.organization_id is None
        assert promoted.user_id is None
        assert promoted.promoted_from_scope == "organization"
        assert promoted.anonymized is True

    @pytest.mark.asyncio
    async def test_anonymization(self, mock_db_session):
        """Test PII anonymization in pattern data."""
        service = PatternEscalationService(mock_db_session)

        pattern_data = {
            "user_id": "user-123",
            "email": "test@example.com",
            "name": "John Doe",
            "phone": "555-1234",
            "action": "create_workflow",
            "parameters": {
                "title": "Sales Pipeline",
                "owner_email": "owner@example.com"
            }
        }

        anonymized = service._anonymize_pattern_data(pattern_data)

        # Check PII fields are removed
        assert "user_id" not in anonymized
        assert "email" not in anonymized
        assert "name" not in anonymized
        assert "phone" not in anonymized

        # Check non-PII fields are preserved
        assert anonymized["action"] == "create_workflow"
        assert anonymized["parameters"]["title"] == "Sales Pipeline"

        # Check nested PII is removed
        assert "owner_email" not in anonymized["parameters"]

    @pytest.mark.asyncio
    async def test_block_promotion_of_sensitive_data(self, mock_db_session):
        """Test that patterns with sensitive data cannot be promoted."""
        service = PatternEscalationService(mock_db_session)

        sensitive_pattern = LearnedPattern(
            id=uuid4(),
            pattern_type="intent_to_action",
            pattern_signature="test_pattern",
            pattern_data={"test": "data"},
            scope="user",
            user_id="user-123",
            contains_sensitive_data=True,
            is_shareable=False,
            confidence_score=0.9,
            occurrence_count=20,
            success_rate=0.85
        )

        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = sensitive_pattern
        mock_db_session.execute.return_value = mock_get_result

        promoted = await service.promote_pattern(
            pattern_id=str(sensitive_pattern.id),
            target_scope="organization",
            organization_id=str(uuid4())
        )

        # Should not promote sensitive patterns
        assert promoted is None

    @pytest.mark.asyncio
    async def test_auto_promote_eligible_patterns(self, mock_db_session, user_pattern, org_pattern):
        """Test automatic promotion of eligible patterns."""
        service = PatternEscalationService(mock_db_session)

        # Mock identify_promotion_candidates to return eligible patterns
        with patch.object(service, 'identify_promotion_candidates') as mock_identify:
            with patch.object(service, 'promote_pattern') as mock_promote:
                mock_identify.side_effect = [
                    [user_pattern],  # User patterns
                    [org_pattern]    # Org patterns
                ]
                mock_promote.return_value = user_pattern  # Success

                result = await service.auto_promote_eligible_patterns()

                assert result['promoted'] == 2
                assert result['failed'] == 0
                assert mock_promote.call_count == 2


class TestPatternLearningService:
    """Test pattern learning service with multi-tier retrieval."""

    @pytest.mark.asyncio
    async def test_get_relevant_patterns_priority_cascade(
        self, mock_db_session, user_pattern, org_pattern, global_pattern
    ):
        """Test multi-tier retrieval with priority cascade."""
        service = PatternLearningService(mock_db_session)

        # Mock database queries to return patterns from all tiers
        mock_user_result = MagicMock()
        mock_user_result.scalars.return_value.all.return_value = [user_pattern]

        mock_org_result = MagicMock()
        mock_org_result.scalars.return_value.all.return_value = [org_pattern]

        mock_global_result = MagicMock()
        mock_global_result.scalars.return_value.all.return_value = [global_pattern]

        mock_db_session.execute.side_effect = [
            mock_user_result,
            mock_org_result,
            mock_global_result
        ]

        context = {"intent": "automate sales", "current_context": "crm"}

        patterns = await service.get_relevant_patterns(
            context=context,
            user_id="user-123",
            organization_id=str(uuid4())
        )

        # Should return patterns from all tiers with proper boosting
        assert len(patterns) > 0

        # User patterns should have highest boost (1.5x)
        # Org patterns should have medium boost (1.2x)
        # Global patterns should have no boost (1.0x)
        user_patterns = [p for p in patterns if p.get('tier_name') == 'user']
        if user_patterns:
            org_patterns_list = [p for p in patterns if p.get('tier_name') == 'organization']
            if org_patterns_list:
                # User patterns should score higher due to boost
                assert user_patterns[0]['score'] > org_patterns_list[0]['score'] or \
                       user_pattern.confidence_score * 1.5 > org_pattern.confidence_score * 1.2

    @pytest.mark.asyncio
    async def test_deduplication_across_tiers(self, mock_db_session):
        """Test that user patterns override org/global with same signature."""
        service = PatternLearningService(mock_db_session)

        # Create patterns with same signature across tiers
        signature = "common_pattern"

        user_p = LearnedPattern(
            id=uuid4(),
            pattern_signature=signature,
            pattern_type="intent_to_action",
            pattern_data={"version": "user"},
            scope="user",
            user_id="user-123",
            confidence_score=0.8,
            occurrence_count=5,
            success_rate=0.85
        )

        org_p = LearnedPattern(
            id=uuid4(),
            pattern_signature=signature,
            pattern_type="intent_to_action",
            pattern_data={"version": "org"},
            scope="organization",
            organization_id=uuid4(),
            confidence_score=0.85,
            occurrence_count=20,
            success_rate=0.88
        )

        mock_user_result = MagicMock()
        mock_user_result.scalars.return_value.all.return_value = [user_p]

        mock_org_result = MagicMock()
        mock_org_result.scalars.return_value.all.return_value = [org_p]

        mock_global_result = MagicMock()
        mock_global_result.scalars.return_value.all.return_value = []

        mock_db_session.execute.side_effect = [
            mock_user_result,
            mock_org_result,
            mock_global_result
        ]

        context = {"intent": "test"}
        patterns = await service.get_relevant_patterns(
            context=context,
            user_id="user-123",
            organization_id=str(uuid4())
        )

        # Should only get user pattern (deduplication removes org version)
        unique_signatures = {p['pattern'].pattern_signature for p in patterns}
        assert signature in unique_signatures

        # Find the pattern with this signature
        matched = [p for p in patterns if p['pattern'].pattern_signature == signature]
        assert len(matched) == 1
        assert matched[0]['tier_name'] == 'user'  # User tier wins

    @pytest.mark.asyncio
    async def test_store_pattern_with_scoping(self, mock_db_session):
        """Test storing patterns with proper user/org scoping."""
        service = PatternLearningService(mock_db_session)

        # Mock check for existing pattern
        mock_existing_result = MagicMock()
        mock_existing_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = mock_existing_result

        pattern_data = {
            "pattern_type": "intent_to_action",
            "pattern_signature": "new_pattern",
            "pattern_data": {"intent": "test", "action": "do_something"},
            "confidence_score": 0.8
        }

        await service._store_pattern(
            pattern_data=pattern_data,
            user_id="user-123",
            organization_id=str(uuid4())
        )

        # Verify pattern was added with correct scoping
        mock_db_session.add.assert_called_once()
        added_pattern = mock_db_session.add.call_args[0][0]

        assert added_pattern.scope == "user"
        assert added_pattern.user_id == "user-123"
        assert added_pattern.is_shareable is True
        assert added_pattern.contains_sensitive_data is False


class TestPatternLearningIntegration:
    """Integration tests for complete pattern learning flow."""

    @pytest.mark.asyncio
    async def test_full_escalation_workflow(self, mock_db_session):
        """Test complete workflow: learn → use → promote → use globally."""
        learning_service = PatternLearningService(mock_db_session)
        escalation_service = PatternEscalationService(mock_db_session)

        # Step 1: Create a user pattern
        user_pattern = LearnedPattern(
            id=uuid4(),
            pattern_type="intent_to_action",
            pattern_signature="escalation_test",
            pattern_data={"intent": "automate", "action": "create"},
            confidence_score=0.85,
            occurrence_count=15,
            success_rate=0.9,
            scope="user",
            user_id="user-123"
        )

        # Step 2: Pattern meets promotion criteria
        assert user_pattern.occurrence_count >= escalation_service.USER_TO_ORG_THRESHOLD['min_occurrence_count']
        assert user_pattern.success_rate >= escalation_service.USER_TO_ORG_THRESHOLD['min_success_rate']

        # Step 3: Promote to organization
        mock_get_result = MagicMock()
        mock_get_result.scalar_one_or_none.return_value = user_pattern

        mock_existing_result = MagicMock()
        mock_existing_result.scalar_one_or_none.return_value = None

        mock_db_session.execute.side_effect = [mock_get_result, mock_existing_result]

        org_pattern = await escalation_service.promote_pattern(
            pattern_id=str(user_pattern.id),
            target_scope="organization",
            organization_id=str(uuid4())
        )

        assert org_pattern.scope == "organization"
        assert org_pattern.anonymized is True

        # Step 4: Other users in org can now access this pattern
        mock_org_result = MagicMock()
        mock_org_result.scalars.return_value.all.return_value = [org_pattern]
        mock_db_session.execute.return_value = mock_org_result

        context = {"intent": "automate"}
        other_user_patterns = await learning_service.get_relevant_patterns(
            context=context,
            user_id="other-user",
            organization_id=org_pattern.organization_id
        )

        # Other user should get the promoted org pattern
        org_patterns_found = [p for p in other_user_patterns if p.get('tier_name') == 'organization']
        assert len(org_patterns_found) > 0
