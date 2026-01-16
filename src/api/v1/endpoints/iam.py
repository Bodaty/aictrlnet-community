"""Internal Agent Messaging (IAM) endpoints for Community Edition."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Path
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import uuid

from core.database import get_db
from core.security import get_current_active_user
from services.iam import IAMService
from schemas.iam import (
    IAMAgentCreate, IAMAgentUpdate, IAMAgentResponse,
    IAMMessageCreate, IAMMessageResponse, IAMMessageFilter,
    IAMSessionCreate, IAMSessionUpdate, IAMSessionResponse,
    IAMFlowData, IAMSystemMetrics, IAMAgentMetrics,
    IAMCommunicationPatterns, IAMErrorLog,
    IAMMessageHistory
)

router = APIRouter()


# Agent management endpoints
@router.post("/agents", response_model=IAMAgentResponse)
async def create_agent(
    agent_data: IAMAgentCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Create a new IAM agent."""
    service = IAMService(db)
    return await service.create_agent(agent_data)


@router.get("/agents", response_model=List[IAMAgentResponse])
async def list_agents(
    agent_type: Optional[str] = Query(None, description="Filter by agent type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """List IAM agents."""
    service = IAMService(db)
    return await service.list_agents(
        agent_type=agent_type,
        status=status,
        skip=skip,
        limit=limit
    )


@router.get("/agents/active", response_model=List[IAMAgentResponse])
async def get_active_agents(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get currently active agents."""
    service = IAMService(db)
    return await service.get_active_agents()


# Agent discovery GET endpoint (for HitLai compatibility)
@router.get("/agents/discover", response_model=List[IAMAgentResponse])
async def discover_agents_get(
    capabilities: Optional[str] = Query(None, description="Comma-separated list of capabilities"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Discover agents by capabilities (GET version for frontend compatibility)."""
    service = IAMService(db)
    # Parse comma-separated capabilities
    capability_list = []
    if capabilities:
        capability_list = [cap.strip() for cap in capabilities.split(",") if cap.strip()]
    return await service.discover_agents(capabilities=capability_list)


@router.get("/agents/{agent_id}", response_model=IAMAgentResponse)
async def get_agent(
    agent_id: uuid.UUID = Path(..., description="Agent ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get an IAM agent by ID."""
    service = IAMService(db)
    agent = await service.get_agent(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.put("/agents/{agent_id}", response_model=IAMAgentResponse)
async def update_agent(
    agent_id: uuid.UUID = Path(..., description="Agent ID"),
    agent_update: IAMAgentUpdate = ...,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Update an IAM agent."""
    service = IAMService(db)
    agent = await service.update_agent(agent_id, agent_update)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent


@router.post("/agents/{agent_id}/heartbeat")
async def agent_heartbeat(
    agent_id: uuid.UUID = Path(..., description="Agent ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Update agent heartbeat."""
    service = IAMService(db)
    success = await service.update_agent_heartbeat(agent_id)
    if not success:
        raise HTTPException(status_code=404, detail="Agent not found")
    return {"status": "ok", "timestamp": datetime.utcnow()}


# Message endpoints
@router.post("/messages", response_model=IAMMessageResponse)
async def send_message(
    message_data: IAMMessageCreate,
    sender_id: uuid.UUID = Query(..., description="Sender agent ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Send an IAM message."""
    service = IAMService(db)
    # Verify sender exists
    sender = await service.get_agent(sender_id)
    if not sender:
        raise HTTPException(status_code=404, detail="Sender agent not found")
    
    return await service.send_message(sender_id, message_data)


@router.get("/messages", response_model=IAMMessageHistory)
async def get_messages(
    source_id: Optional[uuid.UUID] = Query(None, description="Filter by sender ID"),
    destination_id: Optional[uuid.UUID] = Query(None, description="Filter by recipient ID"),
    message_type: Optional[str] = Query(None, description="Filter by message type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    correlation_id: Optional[str] = Query(None, description="Filter by correlation ID"),
    session_id: Optional[uuid.UUID] = Query(None, description="Filter by session ID"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM messages with filtering."""
    service = IAMService(db)
    filter_params = IAMMessageFilter(
        source_id=source_id,
        destination_id=destination_id,
        message_type=message_type,
        status=status,
        correlation_id=correlation_id,
        session_id=session_id,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset
    )
    return await service.get_messages(filter_params)


@router.get("/messages/{message_id}", response_model=Dict[str, Any])
async def get_message_details(
    message_id: uuid.UUID = Path(..., description="Message ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get detailed information about a message."""
    service = IAMService(db)
    details = await service.get_message_details(message_id)
    if not details:
        raise HTTPException(status_code=404, detail="Message not found")
    return details


# Session endpoints
@router.post("/sessions", response_model=IAMSessionResponse)
async def create_session(
    session_data: IAMSessionCreate,
    initiator_id: uuid.UUID = Query(..., description="Initiator agent ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Create a new IAM session."""
    service = IAMService(db)
    # Verify initiator exists
    initiator = await service.get_agent(initiator_id)
    if not initiator:
        raise HTTPException(status_code=404, detail="Initiator agent not found")
    
    return await service.create_session(initiator_id, session_data)


@router.get("/sessions/{session_id}", response_model=IAMSessionResponse)
async def get_session(
    session_id: uuid.UUID = Path(..., description="Session ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get a session by ID."""
    service = IAMService(db)
    session = await service.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


@router.put("/sessions/{session_id}", response_model=IAMSessionResponse)
async def update_session(
    session_id: uuid.UUID = Path(..., description="Session ID"),
    session_update: IAMSessionUpdate = ...,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Update a session."""
    service = IAMService(db)
    session = await service.update_session(session_id, session_update)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# Monitoring endpoints
@router.get("/monitoring/messages", response_model=IAMMessageHistory)
async def get_message_history(
    source_id: Optional[uuid.UUID] = Query(None, description="Filter by sender ID"),
    destination_id: Optional[uuid.UUID] = Query(None, description="Filter by recipient ID"),
    type: Optional[str] = Query(None, description="Filter by message type"),
    status: Optional[str] = Query(None, description="Filter by status"),
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM message history for monitoring."""
    service = IAMService(db)
    filter_params = IAMMessageFilter(
        source_id=source_id,
        destination_id=destination_id,
        message_type=type,
        status=status,
        start_time=start_time,
        end_time=end_time,
        limit=limit,
        offset=offset
    )
    return await service.get_messages(filter_params)


@router.get("/monitoring/flow", response_model=IAMFlowData)
async def get_flow_data(
    timeRange: str = Query("1h", description="Time range (e.g., '1h', '24h', '7d')"),
    agent_ids: Optional[List[uuid.UUID]] = Query(None, description="Filter by agent IDs"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM flow data for visualization."""
    service = IAMService(db)
    return await service.get_flow_data(
        time_range=timeRange,
        agent_ids=agent_ids
    )


@router.get("/monitoring/messages/{message_id}", response_model=Dict[str, Any])
async def get_monitoring_message_details(
    message_id: uuid.UUID = Path(..., description="Message ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get detailed message information for monitoring."""
    service = IAMService(db)
    details = await service.get_message_details(message_id)
    if not details:
        raise HTTPException(status_code=404, detail="Message not found")
    return details


@router.get("/monitoring/metrics", response_model=IAMSystemMetrics)
async def get_system_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM system metrics."""
    service = IAMService(db)
    return await service.get_system_metrics()


@router.get("/monitoring/agents/active", response_model=List[IAMAgentResponse])
async def get_monitoring_active_agents(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get active agents for monitoring."""
    service = IAMService(db)
    return await service.get_active_agents()


@router.get("/monitoring/agents/{agent_id}/metrics", response_model=IAMAgentMetrics)
async def get_agent_metrics(
    agent_id: uuid.UUID = Path(..., description="Agent ID"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get metrics for a specific agent."""
    service = IAMService(db)
    metrics = await service.get_agent_metrics(agent_id)
    if not metrics:
        raise HTTPException(status_code=404, detail="Agent not found")
    return metrics


@router.get("/monitoring/patterns", response_model=IAMCommunicationPatterns)
async def get_communication_patterns(
    timeRange: str = Query("24h", description="Time range for analysis"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM communication patterns."""
    service = IAMService(db)
    return await service.get_communication_patterns(time_range=timeRange)


@router.get("/monitoring/errors", response_model=List[IAMErrorLog])
async def get_error_logs(
    start_time: Optional[datetime] = Query(None, description="Start time"),
    end_time: Optional[datetime] = Query(None, description="End time"),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM error logs."""
    service = IAMService(db)
    return await service.get_error_logs(
        start_time=start_time,
        end_time=end_time,
        limit=limit
    )


# Visualization endpoints (for HitLai compatibility)
@router.get("/flows", response_model=IAMFlowData)
async def get_iam_flow_data(
    workflowId: Optional[str] = Query(None, description="Filter by workflow ID"),
    agentId: Optional[uuid.UUID] = Query(None, description="Filter by agent ID"),
    timeStart: Optional[int] = Query(None, description="Start timestamp (ms)"),
    timeEnd: Optional[int] = Query(None, description="End timestamp (ms)"),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM flow data (HitLai compatibility endpoint)."""
    service = IAMService(db)
    # Default to last hour if no time range specified
    time_range = "1h"
    agent_ids = [agentId] if agentId else None
    return await service.get_flow_data(time_range=time_range, agent_ids=agent_ids)


@router.get("/metrics", response_model=Dict[str, Any])
async def get_iam_performance_metrics(
    workflowId: Optional[str] = Query(None, description="Filter by workflow ID"),
    agentId: Optional[uuid.UUID] = Query(None, description="Filter by agent ID"),
    timeRange: str = Query("1h", description="Time range"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM performance metrics (HitLai compatibility endpoint)."""
    service = IAMService(db)
    
    if agentId:
        metrics = await service.get_agent_metrics(agentId)
        if not metrics:
            raise HTTPException(status_code=404, detail="Agent not found")
        return metrics.model_dump()
    else:
        system_metrics = await service.get_system_metrics()
        return system_metrics.model_dump()


# Health endpoint
@router.get("/health", response_model=Dict[str, Any])
async def get_iam_health(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM system health status."""
    service = IAMService(db)
    return await service.get_health_status()


# Events endpoint  
@router.get("/events", response_model=List[Dict[str, Any]])
async def get_iam_events(
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    agent_id: Optional[str] = Query(None, description="Filter by agent ID"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Get IAM events."""
    service = IAMService(db)
    return await service.get_events(
        event_type=event_type,
        agent_id=agent_id,
        skip=skip,
        limit=limit
    )


# Agent discovery endpoint
@router.post("/agents/discover", response_model=List[IAMAgentResponse])
async def discover_agents(
    capabilities: List[str],
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user)
):
    """Discover agents by capabilities."""
    service = IAMService(db)
    return await service.discover_agents(capabilities=capabilities)


