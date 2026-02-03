"""Personal Agent Hub endpoints for Community Edition."""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.dependencies import get_current_user_safe
from schemas.personal_agent import (
    PersonalAgentConfigResponse,
    PersonalAgentConfigUpdate,
    PersonalAgentAskRequest,
    PersonalAgentAskResponse,
    ActivityFeedResponse,
    WorkflowAddResponse,
    WorkflowRemoveResponse,
)
from services.personal_agent_service import PersonalAgentService

router = APIRouter()


# ---------------------------------------------------------------------------
# Config endpoints
# ---------------------------------------------------------------------------

@router.get("/config", response_model=PersonalAgentConfigResponse)
async def get_personal_agent_config(
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """
    Get the personal agent configuration for the current user.

    Creates a default configuration if one does not exist yet.
    """
    service = PersonalAgentService(db)
    return await service.get_or_create_config(current_user["id"])


@router.put("/config", response_model=PersonalAgentConfigResponse)
async def update_personal_agent_config(
    updates: PersonalAgentConfigUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """
    Update the personal agent configuration for the current user.

    Supports partial updates — only fields included in the request body
    will be changed.
    """
    service = PersonalAgentService(db)
    return await service.update_config(current_user["id"], updates)


# ---------------------------------------------------------------------------
# Ask endpoint
# ---------------------------------------------------------------------------

@router.post("/ask", response_model=PersonalAgentAskResponse)
async def ask_personal_agent(
    request: PersonalAgentAskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """
    Ask the personal agent a question.

    Routes the question through the NLP service, taking the user's
    personality and preference configuration into account. The
    interaction is stored as a memory entry for future context.
    """
    service = PersonalAgentService(db)
    return await service.ask(current_user["id"], request)


# ---------------------------------------------------------------------------
# Activity feed
# ---------------------------------------------------------------------------

@router.get("/activity", response_model=ActivityFeedResponse)
async def get_activity_feed(
    limit: int = Query(20, ge=1, le=100, description="Number of items to return"),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """
    Get the recent activity feed for the personal agent.

    Returns the most recent memory entries (interactions, preferences,
    context, and learnings) ordered by creation time descending.
    """
    service = PersonalAgentService(db)
    return await service.get_activity_feed(current_user["id"], limit=limit)


# ---------------------------------------------------------------------------
# Workflow management
# ---------------------------------------------------------------------------

@router.post("/workflows/{workflow_id}", response_model=WorkflowAddResponse)
async def add_personal_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """
    Add a workflow to the personal agent.

    Community Edition supports up to 5 personal workflows. Upgrade to
    Business Edition for unlimited personal workflows.
    """
    service = PersonalAgentService(db)
    try:
        return await service.add_workflow(current_user["id"], workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.delete("/workflows/{workflow_id}", response_model=WorkflowRemoveResponse)
async def remove_personal_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """
    Remove a workflow from the personal agent.
    """
    service = PersonalAgentService(db)
    try:
        return await service.remove_workflow(current_user["id"], workflow_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
