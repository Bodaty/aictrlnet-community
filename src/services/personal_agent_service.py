"""Service for managing Personal Agent Hub in Community Edition."""

import logging
import uuid
from typing import Optional, List, Dict, Any
from datetime import datetime

from sqlalchemy import select, func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import IntegrityError

from models.personal_agent import (
    PersonalAgentConfig,
    PersonalAgentMemory,
    COMMUNITY_MAX_WORKFLOWS,
    ALLOWED_MEMORY_TYPES,
)
from models.workflow_templates import WorkflowTemplate
from schemas.personal_agent import (
    PersonalAgentConfigResponse,
    PersonalAgentConfigUpdate,
    PersonalAgentAskRequest,
    PersonalAgentAskResponse,
    ActivityFeedItem,
    ActivityFeedResponse,
    WorkflowAddResponse,
    WorkflowRemoveResponse,
)

logger = logging.getLogger(__name__)


class PersonalAgentService:
    """Service for the Personal Agent Hub (Community Edition).

    Provides per-user agent configuration, NLP-backed ask, activity
    feed, and personal workflow management (max 5 in Community).
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def _enrich_with_workflow_details(
        self, response: PersonalAgentConfigResponse
    ) -> PersonalAgentConfigResponse:
        """Resolve workflow IDs to names and attach as workflow_details."""
        workflow_ids = response.active_workflows or []
        if not workflow_ids:
            response.workflow_details = []
            return response
        result = await self.db.execute(
            select(WorkflowTemplate.id, WorkflowTemplate.name)
            .where(WorkflowTemplate.id.in_(workflow_ids))
        )
        wf_map = {str(row.id): row.name for row in result.all()}
        response.workflow_details = [
            {"id": wid, "name": wf_map.get(wid, wid)} for wid in workflow_ids
        ]
        return response

    # ------------------------------------------------------------------
    # Config management
    # ------------------------------------------------------------------

    async def get_or_create_config(self, user_id: str) -> PersonalAgentConfigResponse:
        """Return existing config for *user_id*, or create a default one."""
        result = await self.db.execute(
            select(PersonalAgentConfig).where(
                PersonalAgentConfig.user_id == user_id
            )
        )
        config = result.scalar_one_or_none()

        if config is not None:
            resp = PersonalAgentConfigResponse.model_validate(config)
            return await self._enrich_with_workflow_details(resp)

        # Create default config
        config = PersonalAgentConfig(
            id=str(uuid.uuid4()),
            user_id=user_id,
            agent_name="My Assistant",
            personality={
                "tone": "friendly",
                "style": "concise",
                "expertise_areas": [],
            },
            preferences={
                "notifications": {"enabled": True, "frequency": "daily"},
                "auto_actions": {"enabled": False, "require_confirmation": True},
            },
            active_workflows=[],
            max_workflows=COMMUNITY_MAX_WORKFLOWS,
            status="active",
        )
        self.db.add(config)
        try:
            await self.db.commit()
            await self.db.refresh(config)
            logger.info("Created personal agent config %s for user %s", config.id, user_id)
        except IntegrityError:
            await self.db.rollback()
            # Race condition — another request already created the row
            result = await self.db.execute(
                select(PersonalAgentConfig).where(
                    PersonalAgentConfig.user_id == user_id
                )
            )
            config = result.scalar_one()

        resp = PersonalAgentConfigResponse.model_validate(config)
        return await self._enrich_with_workflow_details(resp)

    async def update_config(
        self, user_id: str, updates: PersonalAgentConfigUpdate
    ) -> PersonalAgentConfigResponse:
        """Update the personal agent config for *user_id*."""
        result = await self.db.execute(
            select(PersonalAgentConfig).where(
                PersonalAgentConfig.user_id == user_id
            )
        )
        config = result.scalar_one_or_none()
        if config is None:
            # Auto-create then apply updates
            await self.get_or_create_config(user_id)
            result = await self.db.execute(
                select(PersonalAgentConfig).where(
                    PersonalAgentConfig.user_id == user_id
                )
            )
            config = result.scalar_one()

        update_dict = updates.model_dump(exclude_unset=True)

        # Serialise nested Pydantic models to plain dicts for JSON columns
        if "personality" in update_dict and update_dict["personality"] is not None:
            val = update_dict["personality"]
            update_dict["personality"] = val if isinstance(val, dict) else val.model_dump()
        if "preferences" in update_dict and update_dict["preferences"] is not None:
            val = update_dict["preferences"]
            update_dict["preferences"] = val if isinstance(val, dict) else val.model_dump()

        for key, value in update_dict.items():
            setattr(config, key, value)

        config.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(config)
        logger.info("Updated personal agent config for user %s", user_id)
        resp = PersonalAgentConfigResponse.model_validate(config)
        return await self._enrich_with_workflow_details(resp)

    # ------------------------------------------------------------------
    # Ask — DEPRECATED (use conversation system instead)
    # ------------------------------------------------------------------

    async def ask(
        self,
        user_id: str,
        request: PersonalAgentAskRequest,
    ) -> PersonalAgentAskResponse:
        """DEPRECATED: Use the conversation system instead.

        The personal agent chat has been unified with the main conversation
        pipeline.  PersonalAgent personality now feeds into
        SystemPromptAssembler, and all conversations route through
        EnhancedConversationService with full tool access.

        This method is kept only for backward compatibility and raises
        a deprecation warning.  The /ask endpoint returns 410 Gone.
        """
        import warnings
        warnings.warn(
            "PersonalAgentService.ask() is deprecated. "
            "Use the conversation system (POST /api/v1/conversation/message) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return PersonalAgentAskResponse(
            message=request.message,
            response=(
                "This endpoint has been deprecated. Please use the main conversation "
                "interface, which now includes your personality settings automatically."
            ),
            model_used="none",
            memory_id="",
            processing_time=0.0,
            timestamp=datetime.utcnow(),
        )

    # ------------------------------------------------------------------
    # Activity feed
    # ------------------------------------------------------------------

    async def get_activity_feed(
        self, user_id: str, limit: int = 20
    ) -> ActivityFeedResponse:
        """Return the most recent interactions for *user_id*."""
        config_resp = await self.get_or_create_config(user_id)

        result = await self.db.execute(
            select(PersonalAgentMemory)
            .where(PersonalAgentMemory.config_id == config_resp.id)
            .order_by(PersonalAgentMemory.created_at.desc())
            .limit(limit)
        )
        memories = result.scalars().all()

        # Total count
        count_result = await self.db.execute(
            select(func.count(PersonalAgentMemory.id)).where(
                PersonalAgentMemory.config_id == config_resp.id
            )
        )
        total = count_result.scalar() or 0

        items = [
            ActivityFeedItem(
                id=m.id,
                memory_type=m.memory_type,
                content=m.content or {},
                importance_score=m.importance_score,
                created_at=m.created_at,
            )
            for m in memories
        ]

        return ActivityFeedResponse(items=items, total=total, limit=limit)

    # ------------------------------------------------------------------
    # Personal workflow management
    # ------------------------------------------------------------------

    async def add_workflow(
        self, user_id: str, workflow_id: str
    ) -> WorkflowAddResponse:
        """Add a personal workflow (max 5 in Community)."""
        config_resp = await self.get_or_create_config(user_id)

        # Re-fetch the ORM object for mutation
        result = await self.db.execute(
            select(PersonalAgentConfig).where(
                PersonalAgentConfig.user_id == user_id
            )
        )
        config = result.scalar_one()

        workflows: List[str] = list(config.active_workflows or [])

        if workflow_id in workflows:
            return WorkflowAddResponse(
                workflow_id=workflow_id,
                active_workflows=workflows,
                current_count=len(workflows),
                max_allowed=config.max_workflows,
                message="Workflow is already in your personal list.",
            )

        if len(workflows) >= config.max_workflows:
            raise ValueError(
                f"You've reached the Community Edition limit of {config.max_workflows} "
                "personal workflows. Upgrade to Business Edition for unlimited workflows."
            )

        workflows.append(workflow_id)
        config.active_workflows = workflows
        config.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(config)

        # Record as memory
        memory = PersonalAgentMemory(
            id=str(uuid.uuid4()),
            config_id=config.id,
            memory_type="preference",
            content={"action": "add_workflow", "workflow_id": workflow_id},
            importance_score=0.3,
        )
        self.db.add(memory)
        await self.db.commit()

        logger.info("User %s added personal workflow %s", user_id, workflow_id)

        return WorkflowAddResponse(
            workflow_id=workflow_id,
            active_workflows=workflows,
            current_count=len(workflows),
            max_allowed=config.max_workflows,
            message=f"Workflow {workflow_id} added to your personal agent.",
        )

    async def remove_workflow(
        self, user_id: str, workflow_id: str
    ) -> WorkflowRemoveResponse:
        """Remove a personal workflow."""
        config_resp = await self.get_or_create_config(user_id)

        result = await self.db.execute(
            select(PersonalAgentConfig).where(
                PersonalAgentConfig.user_id == user_id
            )
        )
        config = result.scalar_one()

        workflows: List[str] = list(config.active_workflows or [])

        if workflow_id not in workflows:
            raise ValueError(f"Workflow {workflow_id} is not in your personal list.")

        workflows.remove(workflow_id)
        config.active_workflows = workflows
        config.updated_at = datetime.utcnow()

        await self.db.commit()
        await self.db.refresh(config)

        # Record as memory
        memory = PersonalAgentMemory(
            id=str(uuid.uuid4()),
            config_id=config.id,
            memory_type="preference",
            content={"action": "remove_workflow", "workflow_id": workflow_id},
            importance_score=0.2,
        )
        self.db.add(memory)
        await self.db.commit()

        logger.info("User %s removed personal workflow %s", user_id, workflow_id)

        return WorkflowRemoveResponse(
            workflow_id=workflow_id,
            active_workflows=workflows,
            current_count=len(workflows),
            message=f"Workflow {workflow_id} removed from your personal agent.",
        )
