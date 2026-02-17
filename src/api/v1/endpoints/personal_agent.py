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
    ActivityFeedResponse,
    WorkflowAddResponse,
    WorkflowRemoveResponse,
    OnboardingAnswerRequest,
    OnboardingSkipRequest,
    OnboardingStateResponse,
    OnboardingStartResponse,
    OnboardingAnswerResponse,
)
from services.personal_agent_service import PersonalAgentService
from services.onboarding_service import OnboardingService

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

@router.post("/ask")
async def ask_personal_agent(
    request: PersonalAgentAskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """
    DEPRECATED: Use the conversation system instead.

    The personal agent chat has been unified with the main conversation
    pipeline, which provides full tool access (89-171 tools), streaming
    SSE responses, and multi-turn context.  PersonalAgent personality
    configuration now feeds into the conversation system prompt.

    Use POST /api/v1/conversation/message to chat with personality applied.
    Use PUT /api/v1/personal-agent/config to update personality settings.
    """
    raise HTTPException(
        status_code=410,
        detail={
            "message": "The /personal-agent/ask endpoint has been deprecated. "
                       "All conversations now go through the main conversation system, "
                       "which includes your personality settings automatically.",
            "use_instead": "/api/v1/conversation/message",
            "configure_personality": "/api/v1/personal-agent/config",
        },
    )


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


# ---------------------------------------------------------------------------
# Onboarding Interview
# ---------------------------------------------------------------------------

@router.get("/onboarding")
async def get_onboarding_state(
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Get the current onboarding interview state."""
    try:
        service = OnboardingService(db)
        return await service.get_state(current_user["id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to get onboarding state: {exc}")


@router.post("/onboarding/start")
async def start_onboarding(
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Start or resume the onboarding interview.

    Returns the current/next question with options.
    If already completed, returns the summary.
    """
    try:
        service = OnboardingService(db)
        return await service.start_or_resume(current_user["id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start onboarding: {exc}")


@router.post("/onboarding/answer")
async def submit_onboarding_answer(
    request: OnboardingAnswerRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Submit an answer for an onboarding question.

    Applies the answer to the personal agent config immediately
    and returns the next question or completion summary.
    """
    try:
        service = OnboardingService(db)
        result = await service.process_answer(
            current_user["id"],
            request.chapter,
            request.question,
            request.answer,
        )
        if not result.get("applied", True):
            raise HTTPException(status_code=400, detail=result.get("message", "Invalid answer"))
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to submit answer: {exc}")


@router.post("/onboarding/skip")
async def skip_onboarding(
    request: OnboardingSkipRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Skip a question or the entire onboarding interview."""
    try:
        service = OnboardingService(db)
        return await service.skip(
            current_user["id"],
            skip_all=request.skip_all,
            chapter=request.chapter,
            question=request.question,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to skip: {exc}")


@router.post("/onboarding/reset")
async def reset_onboarding(
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Reset the onboarding interview to start over."""
    try:
        service = OnboardingService(db)
        return await service.reset(current_user["id"])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to reset onboarding: {exc}")
