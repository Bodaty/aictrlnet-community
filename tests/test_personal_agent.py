"""Tests for Personal Agent Hub service (OPENCLAW Cross-cutting: PAH)."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, MagicMock, patch
import uuid

from services.personal_agent_service import PersonalAgentService
from models.personal_agent import COMMUNITY_MAX_WORKFLOWS, ALLOWED_MEMORY_TYPES
from schemas.personal_agent import (
    PersonalAgentConfigUpdate,
    PersonalAgentAskRequest,
)


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = AsyncMock()
    session.add = Mock()
    session.commit = AsyncMock()
    session.refresh = AsyncMock()
    session.rollback = AsyncMock()

    # Create a proper mock result for execute
    # This fixes the AsyncMock chaining issue
    mock_result = MagicMock()
    session.execute = AsyncMock(return_value=mock_result)

    return session


@pytest.fixture
def personal_agent_service(mock_db_session):
    """Create PersonalAgentService instance."""
    return PersonalAgentService(mock_db_session)


@pytest.fixture
def mock_agent_config():
    """Create a mock PersonalAgentConfig."""
    config = MagicMock()
    config.id = str(uuid.uuid4())
    config.user_id = "user-123"
    config.agent_name = "My Assistant"
    config.personality = {"tone": "friendly", "style": "concise"}
    config.preferences = {"notifications": {"enabled": True}}
    config.active_workflows = ["workflow-1", "workflow-2"]
    config.max_workflows = COMMUNITY_MAX_WORKFLOWS
    config.status = "active"
    config.created_at = datetime.utcnow()
    config.updated_at = datetime.utcnow()
    return config


@pytest.fixture
def mock_agent_memory():
    """Create a mock PersonalAgentMemory."""
    memory = MagicMock()
    memory.id = str(uuid.uuid4())
    memory.config_id = str(uuid.uuid4())
    memory.memory_type = "interaction"
    memory.content = {"question": "Hello", "answer": "Hi there!"}
    memory.importance_score = 0.5
    memory.created_at = datetime.utcnow()
    return memory


class TestPersonalAgentServiceConfig:
    """Tests for personal agent configuration operations."""

    @pytest.mark.asyncio
    async def test_get_or_create_config_existing(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test getting existing config."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config

        result = await personal_agent_service.get_or_create_config("user-123")

        assert result.user_id == "user-123"
        assert result.agent_name == "My Assistant"
        mock_db_session.add.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_create_config_new(self, personal_agent_service, mock_db_session):
        """Test creating new config when none exists."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = None

        # Create a mock for the new config
        new_config = MagicMock()
        new_config.id = str(uuid.uuid4())
        new_config.user_id = "user-new"
        new_config.agent_name = "My Assistant"
        new_config.personality = {"tone": "friendly", "style": "concise", "expertise_areas": []}
        new_config.preferences = {"notifications": {"enabled": True, "frequency": "daily"}, "auto_actions": {"enabled": False, "require_confirmation": True}}
        new_config.active_workflows = []
        new_config.max_workflows = COMMUNITY_MAX_WORKFLOWS
        new_config.status = "active"
        new_config.created_at = datetime.utcnow()
        new_config.updated_at = datetime.utcnow()

        # Mock refresh to set the values
        async def mock_refresh(obj):
            for key, value in vars(new_config).items():
                if not key.startswith('_'):
                    setattr(obj, key, value)

        mock_db_session.refresh = mock_refresh

        result = await personal_agent_service.get_or_create_config("user-new")

        mock_db_session.add.assert_called()
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_config(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test updating agent config."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        updates = PersonalAgentConfigUpdate(
            agent_name="Updated Assistant",
            status="paused",
        )

        result = await personal_agent_service.update_config("user-123", updates)

        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_config_personality(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test updating personality settings."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        updates = PersonalAgentConfigUpdate(
            personality={"tone": "formal", "style": "detailed"},  # 'formal' is valid, not 'professional'
        )

        result = await personal_agent_service.update_config("user-123", updates)

        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_update_config_creates_if_missing(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test update creates config if missing."""
        # First call returns None, subsequent calls return the config
        mock_db_session.execute.return_value.scalar_one_or_none.side_effect = [None, mock_agent_config]
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        updates = PersonalAgentConfigUpdate(agent_name="New Name")

        await personal_agent_service.update_config("user-new", updates)

        # Should have created a config first
        assert mock_db_session.add.called or mock_db_session.commit.called


class TestPersonalAgentServiceAsk:
    """Tests for the ask (NLP routing) operations."""

    @pytest.mark.asyncio
    async def test_ask_routes_to_nlp(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test ask routes question to NLP service."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config

        request = PersonalAgentAskRequest(
            message="What is the weather?",
            context={"location": "NYC"},
        )

        with patch('services.nlp.NLPService') as mock_nlp_class:
            mock_nlp = MagicMock()
            mock_nlp.generate_from_prompt = AsyncMock(return_value={
                "response": "I cannot check the weather directly.",
                "model_used": "llama3.2:1b",
            })
            mock_nlp_class.return_value = mock_nlp

            result = await personal_agent_service.ask("user-123", request)

        assert result.message == "What is the weather?"
        assert result.response is not None
        assert result.memory_id is not None
        mock_db_session.add.assert_called()  # Memory was saved

    @pytest.mark.asyncio
    async def test_ask_stores_interaction_as_memory(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test ask stores the interaction in memory."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config

        request = PersonalAgentAskRequest(message="Hello!")

        with patch('services.nlp.NLPService') as mock_nlp_class:
            mock_nlp = MagicMock()
            mock_nlp.generate_from_prompt = AsyncMock(return_value={
                "response": "Hello! How can I help?",
            })
            mock_nlp_class.return_value = mock_nlp

            result = await personal_agent_service.ask("user-123", request)

        # Verify memory was added
        add_calls = mock_db_session.add.call_args_list
        assert len(add_calls) >= 1

    @pytest.mark.asyncio
    async def test_ask_handles_nlp_failure(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test ask handles NLP service failure gracefully."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config

        request = PersonalAgentAskRequest(message="Test question")

        with patch('services.nlp.NLPService') as mock_nlp_class:
            mock_nlp = MagicMock()
            mock_nlp.generate_from_prompt = AsyncMock(side_effect=Exception("NLP unavailable"))
            mock_nlp_class.return_value = mock_nlp

            result = await personal_agent_service.ask("user-123", request)

        assert "unavailable" in result.response.lower()
        assert result.model_used == "none"


class TestPersonalAgentServiceActivityFeed:
    """Tests for activity feed operations."""

    @pytest.mark.asyncio
    async def test_get_activity_feed(self, personal_agent_service, mock_db_session, mock_agent_config, mock_agent_memory):
        """Test getting activity feed."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = [mock_agent_memory]
        mock_db_session.execute.return_value.scalar.return_value = 1

        result = await personal_agent_service.get_activity_feed("user-123", limit=20)

        assert result.total >= 0
        assert result.limit == 20

    @pytest.mark.asyncio
    async def test_get_activity_feed_empty(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test getting empty activity feed."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalars.return_value.all.return_value = []
        mock_db_session.execute.return_value.scalar.return_value = 0

        result = await personal_agent_service.get_activity_feed("user-123")

        assert result.total == 0
        assert len(result.items) == 0


class TestPersonalAgentServiceWorkflows:
    """Tests for personal workflow management."""

    @pytest.mark.asyncio
    async def test_add_workflow(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test adding a workflow to personal list."""
        mock_agent_config.active_workflows = ["workflow-1"]
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        result = await personal_agent_service.add_workflow("user-123", "workflow-new")

        assert "workflow-new" in result.active_workflows
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_add_workflow_already_exists(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test adding workflow that already exists."""
        mock_agent_config.active_workflows = ["workflow-1", "workflow-2"]
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        result = await personal_agent_service.add_workflow("user-123", "workflow-1")

        assert "already in your personal list" in result.message.lower()

    @pytest.mark.asyncio
    async def test_add_workflow_exceeds_limit(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test adding workflow when at limit."""
        # Fill up to max
        mock_agent_config.active_workflows = [f"workflow-{i}" for i in range(COMMUNITY_MAX_WORKFLOWS)]
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        with pytest.raises(ValueError) as exc_info:
            await personal_agent_service.add_workflow("user-123", "workflow-new")

        assert "limit" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_remove_workflow(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test removing a workflow from personal list."""
        mock_agent_config.active_workflows = ["workflow-1", "workflow-2"]
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        result = await personal_agent_service.remove_workflow("user-123", "workflow-1")

        assert "workflow-1" not in result.active_workflows
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_remove_workflow_not_found(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test removing workflow that doesn't exist."""
        mock_agent_config.active_workflows = ["workflow-1"]
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        with pytest.raises(ValueError) as exc_info:
            await personal_agent_service.remove_workflow("user-123", "nonexistent")

        assert "not in your personal list" in str(exc_info.value).lower()


class TestCommunityLimits:
    """Tests for Community Edition limits."""

    def test_max_workflows_limit(self):
        """Test Community max workflows limit."""
        assert COMMUNITY_MAX_WORKFLOWS == 5

    def test_allowed_memory_types(self):
        """Test allowed memory types."""
        expected = ["interaction", "preference", "context", "learning"]
        for mem_type in expected:
            assert mem_type in ALLOWED_MEMORY_TYPES


class TestPersonalAgentServiceMemory:
    """Tests for memory management."""

    @pytest.mark.asyncio
    async def test_memory_stored_on_ask(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test that asking stores memory."""
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config

        request = PersonalAgentAskRequest(message="Remember this")

        with patch('services.nlp.NLPService') as mock_nlp_class:
            mock_nlp = MagicMock()
            mock_nlp.generate_from_prompt = AsyncMock(return_value={"response": "OK"})
            mock_nlp_class.return_value = mock_nlp

            await personal_agent_service.ask("user-123", request)

        # Check that memory was added
        assert mock_db_session.add.called

    @pytest.mark.asyncio
    async def test_memory_stored_on_workflow_add(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test that adding workflow stores preference memory."""
        mock_agent_config.active_workflows = []
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        await personal_agent_service.add_workflow("user-123", "new-workflow")

        # Should add config and memory
        assert mock_db_session.add.call_count >= 1

    @pytest.mark.asyncio
    async def test_memory_stored_on_workflow_remove(self, personal_agent_service, mock_db_session, mock_agent_config):
        """Test that removing workflow stores preference memory."""
        mock_agent_config.active_workflows = ["workflow-1"]
        mock_db_session.execute.return_value.scalar_one_or_none.return_value = mock_agent_config
        mock_db_session.execute.return_value.scalar_one.return_value = mock_agent_config

        await personal_agent_service.remove_workflow("user-123", "workflow-1")

        # Should store memory
        assert mock_db_session.add.called
