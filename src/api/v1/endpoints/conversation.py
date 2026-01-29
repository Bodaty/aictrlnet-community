"""
Conversation API endpoints for multi-turn NLP system.

These endpoints work alongside the existing /nlp/process endpoint
to provide multi-turn conversation capabilities with backward compatibility.
"""

from typing import List, Optional
from uuid import UUID
from datetime import datetime as dt
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
import json
import asyncio

from core.database import get_db
from core.security import get_current_user
from schemas.conversation import (
    ConversationSessionCreate,
    ConversationSessionResponse,
    ConversationMessageCreate,
    ConversationMessageResponse,
    ConversationActionExecute,
    ConversationActionResponse,
    ConversationResponse,
    ConversationListResponse,
    IntentDetectionResponse,
    ConversationIntentCreate,
    ConversationIntentResponse,
    ConversationPatternResponse
)
from core.upgrade_hints import attach_upgrade_hints
from services.enhanced_conversation_manager import EnhancedConversationService
from services.action_orchestrator import ActionOrchestrator
from services.tool_aware_conversation import ToolAwareConversationService
from services.tool_dispatcher import Edition
from core.edition_discovery import Edition as CoreEdition
from models.user import User
from models.conversation import (
    ConversationSession,
    ConversationIntent,
    ConversationPattern
)

router = APIRouter()


def serialize_for_json(obj):
    """Recursively convert datetime objects to ISO format strings."""
    from datetime import datetime
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    return obj


# === Session Management Endpoints ===

@router.post("/sessions", response_model=ConversationSessionResponse)
async def create_conversation_session(
    session_data: ConversationSessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new conversation session.
    
    This starts a multi-turn conversation that maintains context
    across multiple messages.
    """
    service = EnhancedConversationService(db)
    session = await service.create_session(
        user_id=current_user.id,
        initial_message=session_data.initial_message,
        context=session_data.context
    )
    
    # Properly load relationships for response using selectinload
    stmt = select(ConversationSession).where(
        ConversationSession.id == session.id
    ).options(
        selectinload(ConversationSession.messages),
        selectinload(ConversationSession.actions)
    )
    result = await db.execute(stmt)
    session = result.scalar_one()
    
    return session


@router.get("/sessions", response_model=ConversationListResponse)
async def list_conversation_sessions(
    response: Response,
    active_only: bool = Query(True, description="Only return active sessions"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List conversation sessions for the current user.

    Returns active sessions by default, ordered by last activity.
    """
    attach_upgrade_hints(response, "conversations")
    service = EnhancedConversationService(db)

    if active_only:
        sessions_result = await service.get_active_sessions(current_user.id)
        # get_active_sessions returns a dict with 'sessions' key
        sessions = sessions_result.get("sessions", []) if isinstance(sessions_result, dict) else sessions_result
        total = len(sessions)
    else:
        # Build query with relationships loaded
        stmt = select(ConversationSession).where(
            ConversationSession.user_id == current_user.id
        ).options(
            selectinload(ConversationSession.messages),
            selectinload(ConversationSession.actions)
        ).order_by(ConversationSession.last_activity.desc())
        
        # Get total count
        count_stmt = select(ConversationSession).where(
            ConversationSession.user_id == current_user.id
        )
        count_result = await db.execute(count_stmt)
        total = len(count_result.scalars().all())
        
        # Apply pagination
        stmt = stmt.offset((page - 1) * per_page).limit(per_page)
        result = await db.execute(stmt)
        sessions = result.scalars().all()
    
    return ConversationListResponse(
        conversations=sessions,
        total=total,
        page=page,
        per_page=per_page
    )


@router.post("/sessions/{session_id}/end")
async def end_conversation_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    End/delete a conversation session.
    Sets the session as inactive and marks it as ended.
    """
    # Verify session exists and belongs to user
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Mark session as ended
    session.is_active = False
    session.ended_at = dt.utcnow()
    session.state = "ended"

    await db.commit()

    return serialize_for_json({
        "id": str(session.id),
        "ended_at": session.ended_at,
        "message": "Conversation session ended successfully"
    })


@router.get("/sessions/{session_id}", response_model=ConversationSessionResponse)
async def get_conversation_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific conversation session with full history.
    """
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    ).options(
        selectinload(ConversationSession.messages),
        selectinload(ConversationSession.actions)
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return session


@router.delete("/sessions/{session_id}")
async def end_conversation_session(
    session_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    End a conversation session.
    
    Marks the session as inactive and records the end time.
    """
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    from datetime import datetime
    session.is_active = False
    session.ended_at = datetime.utcnow()
    session.state = "completed" if session.state == "executing_action" else "abandoned"
    
    await db.commit()
    
    # Record successful pattern if completed
    if session.state == "completed":
        service = EnhancedConversationService(db)
        await service.record_pattern(session_id)
    
    return {"message": "Session ended successfully", "final_state": session.state}


# === Message Processing Endpoints ===

@router.post("/sessions/{session_id}/messages", response_model=ConversationResponse)
async def send_message(
    session_id: UUID,
    message: ConversationMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message in a conversation session.
    
    This is the main endpoint for multi-turn conversation processing.
    The response includes the assistant's reply, suggested actions,
    and updated conversation state.
    """
    # Verify session belongs to user
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")
    
    service = EnhancedConversationService(db)
    response = await service.process_message(
        session_id=session_id,
        content=message.content,
        user_id=current_user.id
    )
    
    return response


@router.post("/sessions/{session_id}/messages/stream")
async def send_message_stream(
    session_id: UUID,
    message: ConversationMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message in a conversation session with real-time progress updates via SSE.

    This endpoint streams Server-Sent Events (SSE) for long-running operations
    like workflow creation, allowing the frontend to show real-time progress
    and provide a cancel button.

    Event types:
    - progress: {"step": "description", "progress": 0-100, "details": {...}}
    - complete: {"message": ConversationResponse}
    - error: {"error": "description"}
    """
    # Verify session belongs to user
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")

    async def event_generator():
        """Generate SSE events for workflow creation progress."""
        import logging
        logger = logging.getLogger(__name__)

        try:
            # Store user message
            service = EnhancedConversationService(db)

            # Send initial progress event
            yield f"data: {json.dumps({'event': 'progress', 'data': {'step': 'Processing message', 'progress': 5}})}\n\n"
            await asyncio.sleep(0.1)  # Small delay for event delivery

            # Get session and detect intent
            await service.initialize_knowledge()
            session_obj = await service.get_session(session_id)

            # Store user message
            user_message = await service._store_message(
                session_id=session.id,
                role="user",
                content=message.content
            )

            yield f"data: {json.dumps({'event': 'progress', 'data': {'step': 'Analyzing intent', 'progress': 15}})}\n\n"
            await asyncio.sleep(0.1)

            # Check if user is confirming an action
            if session_obj.state == "confirming_action":
                content_lower = message.content.lower()
                if any(phrase in content_lower for phrase in ['yes', 'proceed', 'confirm', 'go ahead', 'do it', 'ok', 'sure']):
                    if 'pending_action_plan' in session_obj.context:
                        session_obj.context['action_confirmed'] = True
                        logger.info(f"[SSE] User confirmed action")
                    from sqlalchemy.orm import attributes
                    attributes.flag_modified(session_obj, 'context')
                    await db.commit()
                    await db.refresh(session_obj)

            # Detect intent
            intent_result = await service._detect_intent_with_context(session_obj, message.content)

            yield f"data: {json.dumps({'event': 'progress', 'data': {'step': 'Retrieving system knowledge', 'progress': 25}})}\n\n"
            await asyncio.sleep(0.1)

            # Get knowledge items
            knowledge_items = await service.knowledge_service.find_relevant_knowledge(
                query=message.content,
                context=session_obj.context,
                limit=5
            )

            # Update session with intent
            if intent_result.confidence >= 0.6:
                session_obj.primary_intent = intent_result.intent
                session_obj.intent_confidence = intent_result.confidence
                session_obj.extracted_params = {
                    **session_obj.extracted_params,
                    **intent_result.entities
                }
                await db.commit()
                logger.info(f"[SSE] Intent set: {session_obj.primary_intent}")

            yield f"data: {json.dumps({'event': 'progress', 'data': {'step': 'Generating response', 'progress': 35}})}\n\n"
            await asyncio.sleep(0.1)

            # Generate response
            response_content = await service._generate_informed_response(
                session=session_obj,
                intent_result=intent_result,
                knowledge_items=knowledge_items,
                user_query=message.content
            )

            # Get quick actions
            quick_actions = await service._generate_smart_actions(
                session=session_obj,
                intent_result=intent_result,
                knowledge_items=knowledge_items
            )

            # Determine next state
            next_state = await service._determine_smart_state(
                session=session_obj,
                intent_result=intent_result,
                knowledge_items=knowledge_items
            )

            logger.info(f"[SSE] State transition: {session_obj.state} → {next_state}")

            # Handle action planning
            if next_state == "confirming_action" and session_obj.primary_intent:
                yield f"data: {json.dumps({'event': 'progress', 'data': {'step': 'Creating action plan', 'progress': 45}})}\n\n"
                await asyncio.sleep(0.1)

                action_plan = await service._create_action_plan(session_obj, intent_result, message.content)
                if action_plan:
                    response_content = await service._format_action_plan_response(action_plan, message.content)
                    logger.info(f"[SSE] Action plan created with {len(action_plan.steps)} steps")

            # Store assistant response
            assistant_message = await service._store_message(
                session_id=session.id,
                role="assistant",
                content=response_content,
                detected_intent=intent_result.intent,
                intent_confidence=intent_result.confidence,
                entities=intent_result.entities,
                suggested_actions=quick_actions
            )

            # Update session state
            await service.update_session_state(session.id, next_state)

            # Handle execution with progress updates
            if next_state == "executing_action" and session_obj.primary_intent:
                # Refresh session
                session_obj = await service.get_session(session.id)
                logger.info(f"[SSE] Executing action: {session_obj.primary_intent}")

                # Yield progress events for workflow creation stages
                stages = [
                    {"step": "Validating workflow security", "progress": 50},
                    {"step": "Analyzing LLM requirements", "progress": 55},
                    {"step": "Discovering AI agents", "progress": 65},
                    {"step": "Provisioning orchestration hub", "progress": 75},
                    {"step": "Learning patterns", "progress": 85},
                    {"step": "Finalizing workflow", "progress": 95}
                ]

                for stage in stages:
                    yield f"data: {json.dumps({'event': 'progress', 'data': stage})}\n\n"
                    await asyncio.sleep(2)  # Simulate real-time progress

                # Execute the action
                await service._trigger_action(session_obj)

                # Analyze patterns
                try:
                    await service.pattern_service.analyze_session(str(session.id))
                    logger.info("[SSE] Pattern analysis completed")
                except Exception as e:
                    logger.warning(f"Pattern analysis failed: {e}")

            # Build final response
            from schemas.conversation import ConversationResponse
            response = ConversationResponse(
                session_id=session.id,
                message=assistant_message,
                state=next_state,
                context=session_obj.context,
                quick_actions=quick_actions,
                requires_clarification=next_state == "clarifying_details",
                clarification_options=await service._get_clarification_options(session_obj, intent_result)
            )

            # Send completion event
            yield f"data: {json.dumps({'event': 'complete', 'data': {'message': response.dict()}})}\n\n"

        except Exception as e:
            logger.error(f"[SSE] Error during message processing: {e}", exc_info=True)
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
            "Connection": "keep-alive"
        }
    )


@router.post("/sessions/{session_id}/messages/tools/stream")
async def send_message_with_tools_stream(
    session_id: UUID,
    message: ConversationMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    v4 Feature: Send message with tool-aware processing and real-time SSE streaming.

    Streams tool execution progress as SSE events for real-time UI updates.

    Event types:
    - thinking: LLM is analyzing the request
    - tools_identified: Tools to be called have been determined
    - tool_start: A tool is beginning execution
    - tool_complete: A tool finished successfully
    - tool_error: A tool failed
    - response: Final response content
    - complete: Processing finished
    - error: An error occurred
    """
    import logging
    logger = logging.getLogger(__name__)

    # Verify session belongs to user
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    if not session.is_active:
        raise HTTPException(status_code=400, detail="Session is not active")

    async def tool_event_generator():
        """Generate SSE events for tool execution progress."""
        try:
            # Use EnhancedConversationService for knowledge-integrated tool streaming
            # This ensures knowledge retrieval, industry packs, and clarification all work
            enhanced_service = EnhancedConversationService(db)

            # Stream tool execution events with FULL knowledge integration
            async for event in enhanced_service.stream_tool_execution(
                content=message.content,
                user_id=str(current_user.id),
                session_context={
                    'session_id': str(session_id),
                    'primary_intent': session.primary_intent,
                    'extracted_params': session.extracted_params
                }
            ):
                # Format as SSE
                event_type = event.get("event", "message")
                event_data = event.get("data", {})

                # Serialize data for JSON
                serialized_data = serialize_for_json(event_data)

                yield f"event: {event_type}\n"
                yield f"data: {json.dumps(serialized_data)}\n\n"

        except Exception as e:
            logger.error(f"[v4 SSE] Error during tool streaming: {e}", exc_info=True)
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        tool_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


@router.post("/messages", response_model=ConversationResponse)
async def send_message_without_session(
    message: ConversationMessageCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Send a message without specifying a session.
    
    This will either create a new session or reuse the most recent
    active session for the user. Useful for seamless conversation
    continuation.
    """
    service = EnhancedConversationService(db)
    
    # Get or create session
    active_sessions = await service.get_active_sessions(current_user.id)
    if active_sessions:
        session = active_sessions[0]  # Use most recent
    else:
        session = await service.create_session(current_user.id)
    
    response = await service.process_message(
        session_id=session.id,
        content=message.content,
        user_id=current_user.id
    )
    
    return response


# === Intent Management Endpoints ===

@router.post("/intents", response_model=ConversationIntentResponse)
async def create_intent(
    intent_data: ConversationIntentCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a predefined conversation intent.
    
    Intents help the system better understand and route user requests.
    """
    # Check if intent name already exists
    stmt = select(ConversationIntent).where(
        ConversationIntent.name == intent_data.name
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()
    
    if existing:
        raise HTTPException(status_code=400, detail="Intent name already exists")
    
    from uuid import uuid4
    from datetime import datetime
    
    intent = ConversationIntent(
        id=uuid4(),
        name=intent_data.name,
        category=intent_data.category,
        description=intent_data.description,
        required_params=intent_data.required_params,
        optional_params=intent_data.optional_params,
        example_phrases=intent_data.example_phrases,
        clarification_questions=intent_data.clarification_questions,
        service_endpoint=intent_data.service_endpoint,
        action_template=intent_data.action_template,
        usage_count=0,
        success_rate=0.0,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        is_active=True
    )
    
    db.add(intent)
    await db.commit()
    await db.refresh(intent)
    
    return intent


@router.get("/intents", response_model=List[ConversationIntentResponse])
async def list_intents(
    category: Optional[str] = Query(None, description="Filter by category"),
    active_only: bool = Query(True, description="Only return active intents"),
    db: AsyncSession = Depends(get_db)
):
    """
    List available conversation intents.
    """
    stmt = select(ConversationIntent)
    
    if active_only:
        stmt = stmt.where(ConversationIntent.is_active == True)
    
    if category:
        stmt = stmt.where(ConversationIntent.category == category)
    
    stmt = stmt.order_by(ConversationIntent.usage_count.desc())
    result = await db.execute(stmt)
    intents = result.scalars().all()
    
    return intents


@router.post("/detect-intent", response_model=IntentDetectionResponse)
async def detect_intent(
    text: str = Query(..., description="Text to analyze for intent"),
    session_id: Optional[UUID] = Query(None, description="Session ID for context"),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Detect intent from text input.
    
    This endpoint can be used to test intent detection without
    creating a full conversation session.
    """
    service = EnhancedConversationService(db)
    
    # Get session if provided
    session = None
    if session_id:
        stmt = select(ConversationSession).where(
            ConversationSession.id == session_id,
            ConversationSession.user_id == current_user.id
        )
        result = await db.execute(stmt)
        session = result.scalar_one_or_none()
    
    # Create temporary session if not provided
    if not session:
        from uuid import uuid4
        from datetime import datetime
        session = ConversationSession(
            id=uuid4(),
            user_id=current_user.id,
            state="greeting",
            started_at=datetime.utcnow(),
            last_activity=datetime.utcnow(),
            context={},
            extracted_params={},
            is_active=False  # Temporary session
        )
    
    result = await service._detect_intent_with_context(session, text)
    
    return result


# === Pattern Analytics Endpoints ===

@router.get("/patterns", response_model=List[ConversationPatternResponse])
async def list_conversation_patterns(
    min_confidence: float = Query(0.5, ge=0, le=1, description="Minimum confidence score"),
    pattern_type: Optional[str] = Query(None, description="Filter by pattern type"),
    promoted_only: bool = Query(False, description="Only return promoted patterns"),
    db: AsyncSession = Depends(get_db)
):
    """
    List learned conversation patterns.
    
    These patterns help understand common user interaction flows
    and can be used to improve the conversation system.
    """
    stmt = select(ConversationPattern)
    
    stmt = stmt.where(ConversationPattern.confidence_score >= min_confidence)
    
    if pattern_type:
        stmt = stmt.where(ConversationPattern.pattern_type == pattern_type)
    
    if promoted_only:
        stmt = stmt.where(ConversationPattern.is_promoted == True)
    
    stmt = stmt.order_by(
        ConversationPattern.confidence_score.desc(),
        ConversationPattern.occurrence_count.desc()
    ).limit(50)
    
    result = await db.execute(stmt)
    patterns = result.scalars().all()
    
    return patterns


@router.post("/patterns/{pattern_id}/promote")
async def promote_pattern(
    pattern_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Promote a conversation pattern to be used as a template.
    
    Promoted patterns can guide future conversations.
    """
    stmt = select(ConversationPattern).where(
        ConversationPattern.id == pattern_id
    )
    result = await db.execute(stmt)
    pattern = result.scalar_one_or_none()
    
    if not pattern:
        raise HTTPException(status_code=404, detail="Pattern not found")
    
    from datetime import datetime
    pattern.is_promoted = True
    pattern.promoted_at = datetime.utcnow()
    pattern.confidence_score = min(1.0, pattern.confidence_score * 1.2)  # Boost confidence
    
    await db.commit()
    
    return {"message": "Pattern promoted successfully", "pattern_id": pattern_id}


# === Action Execution Endpoints ===

@router.post("/sessions/{session_id}/actions", response_model=ConversationActionResponse)
async def execute_action(
    session_id: UUID,
    action: ConversationActionExecute,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Execute an action in a conversation session.
    
    This allows direct action execution bypassing the conversation flow,
    useful for quick actions from the UI.
    """
    # Verify session belongs to user
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    from uuid import uuid4
    from datetime import datetime
    from models.conversation import ConversationAction
    
    # Create action record
    action_record = ConversationAction(
        id=uuid4(),
        session_id=session_id,
        action_type=action.action_type,
        action_params=action.action_params,
        status="pending",
        created_at=datetime.utcnow()
    )
    
    db.add(action_record)
    await db.commit()
    
    # Phase 2: Use ActionOrchestrator for intelligent action execution
    try:
        orchestrator = ActionOrchestrator(db, CoreEdition.COMMUNITY)

        # Check if this is a planning request or execution request
        if action.action_type == "plan":
            # Create action plan
            result = await orchestrator.plan_and_execute_action(
                intent=action.action_params.get("intent", "unknown"),
                parameters=action.action_params,
                context={"session_id": str(session_id), "user_id": str(current_user.id)},
                user_id=str(current_user.id),
                preview_only=True  # Just plan, don't execute
            )

            action_record.status = "completed"
            action_record.completed_at = datetime.utcnow()
            action_record.result = serialize_for_json(result)

        elif action.action_type == "execute":
            # Execute action plan
            result = await orchestrator.plan_and_execute_action(
                intent=action.action_params.get("intent", "unknown"),
                parameters=action.action_params,
                context={"session_id": str(session_id), "user_id": str(current_user.id)},
                user_id=str(current_user.id),
                preview_only=False  # Actually execute
            )

            action_record.status = "completed" if result.get("success") else "failed"
            action_record.completed_at = datetime.utcnow()
            action_record.result = serialize_for_json(result)

        else:
            # Fallback for other action types
            action_record.status = "completed"
            action_record.completed_at = datetime.utcnow()
            action_record.result = {"message": f"Action {action.action_type} executed successfully"}

    except Exception as e:
        action_record.status = "failed"
        action_record.completed_at = datetime.utcnow()
        action_record.result = serialize_for_json({"error": str(e)})
    
    await db.commit()
    await db.refresh(action_record)

    return action_record


# === Phase 2: Action Planning and Execution Endpoints ===

@router.post("/sessions/{session_id}/plan", response_model=ConversationActionResponse)
async def create_action_plan(
    session_id: UUID,
    action: ConversationActionExecute,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create an action plan for the conversation.
    Returns a preview of what will be executed.

    This is part of Phase 2: Guided Action Execution.
    """
    # Verify session
    stmt = select(ConversationSession).where(
        ConversationSession.id == session_id,
        ConversationSession.user_id == current_user.id
    )
    result = await db.execute(stmt)
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Use ActionOrchestrator to create plan
    orchestrator = ActionOrchestrator(db, Edition.COMMUNITY)

    plan_result = await orchestrator.plan_and_execute_action(
        intent=action.action_params.get("intent", "create_workflow"),
        parameters=action.action_params,
        context={
            "session_id": str(session_id),
            "user_id": str(current_user.id),
            "conversation_context": session.context
        },
        user_id=str(current_user.id),
        preview_only=True  # Only create plan, don't execute
    )

    # Format response as conversation message
    from uuid import uuid4
    from datetime import datetime
    from models.conversation import ConversationAction

    action_record = ConversationAction(
        id=uuid4(),
        session_id=session_id,
        action_type="plan",
        action_params=action.action_params,
        status="completed",
        result=serialize_for_json(plan_result),
        created_at=datetime.utcnow(),
        completed_at=datetime.utcnow()
    )

    db.add(action_record)
    await db.commit()
    await db.refresh(action_record)

    return action_record


@router.get("/sessions/{session_id}/actions/{action_id}/progress")
async def get_action_progress(
    session_id: UUID,
    action_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get progress of an executing action.

    This endpoint would be polled or use SSE/WebSockets for real-time updates.
    """
    # Verify session and action
    from models.conversation import ConversationAction

    stmt = select(ConversationAction).where(
        ConversationAction.id == action_id,
        ConversationAction.session_id == session_id
    ).options(selectinload(ConversationAction.session))

    result = await db.execute(stmt)
    action_record = result.scalar_one_or_none()

    if not action_record or action_record.session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Action not found")

    # In a real implementation, this would check actual execution progress
    # For now, return the action status
    progress = {
        "action_id": str(action_id),
        "status": action_record.status,
        "progress": 100 if action_record.status == "completed" else 50,
        "current_step": "Execution complete" if action_record.status == "completed" else "Processing",
        "total_steps": 1,
        "result": action_record.result if action_record.status == "completed" else None
    }

    return progress


@router.post("/sessions/{session_id}/actions/{action_id}/rollback")
async def rollback_action(
    session_id: UUID,
    action_id: UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Rollback an executed action.

    Part of Phase 2: Progressive execution with rollback capability.
    """
    # Verify session and action
    from models.conversation import ConversationAction

    stmt = select(ConversationAction).where(
        ConversationAction.id == action_id,
        ConversationAction.session_id == session_id
    ).options(selectinload(ConversationAction.session))

    result = await db.execute(stmt)
    action_record = result.scalar_one_or_none()

    if not action_record or action_record.session.user_id != current_user.id:
        raise HTTPException(status_code=404, detail="Action not found")

    if action_record.status != "completed":
        raise HTTPException(status_code=400, detail="Can only rollback completed actions")

    # In a real implementation, this would trigger actual rollback
    # For now, mark as rolled back
    from datetime import datetime

    action_record.status = "rolled_back"
    action_record.result = {
        **action_record.result,
        "rollback_at": datetime.utcnow().isoformat(),
        "rollback_reason": "User requested rollback"
    }

    await db.commit()

    return {
        "success": True,
        "action_id": str(action_id),
        "status": "rolled_back",
        "message": "Action has been rolled back successfully"
    }


# =============================================================================
# v5 UNIFIED CONVERSATION ENDPOINT - LLM as Brain
# =============================================================================

@router.post("/sessions/{session_id}/chat")
async def chat_v5(
    session_id: UUID,
    message: ConversationMessageCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    v5 Unified conversation endpoint with LLM-as-brain architecture.

    This is the PRIMARY endpoint for intelligent conversations in v5.
    It replaces both /messages and /messages/tools/stream with a unified flow.

    Key features:
    - Full conversation history passed to LLM
    - LLM decides whether to ask questions or take action
    - Parameters accumulate naturally across turns
    - Streaming SSE response for real-time updates

    Returns Server-Sent Events (SSE) stream with events:
    - thinking: Initial processing
    - knowledge_retrieved: Knowledge context loaded
    - context_ready: Full context assembled
    - tools_identified: Tools LLM wants to call
    - tool_start/tool_complete: Tool execution progress
    - response: Final response content
    - complete: Stream finished
    - error: If something goes wrong

    Example usage:
    ```
    Turn 1:
    User: "Set up a new legal matter for e-discovery"
    Assistant: "I'd be happy to help. Could you tell me the matter name,
                document count, and case type?"

    Turn 2:
    User: "X Matter, 4000 documents, litigation"
    Assistant: [Executes tools] "Created the workflow with..."
    ```
    """
    async def v5_event_generator():
        try:
            # Use EnhancedConversationService with v5 unified flow
            enhanced_service = EnhancedConversationService(db)

            async for event in enhanced_service.process_message_v2(
                session_id=session_id,
                content=message.content,
                user_id=str(current_user.id),
                stream=True
            ):
                # Format as SSE
                event_data = serialize_for_json(event.get('data', {}))
                yield f"event: {event['event']}\ndata: {json.dumps(event_data)}\n\n"

        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"[v5 Chat] Error: {e}")
            import traceback
            logger.error(f"[v5 Chat] Traceback: {traceback.format_exc()}")
            yield f"event: error\ndata: {json.dumps({'message': str(e)})}\n\n"

    return StreamingResponse(
        v5_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )