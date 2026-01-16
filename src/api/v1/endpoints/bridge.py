"""Bridge endpoints for system integration."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from services.bridge import BridgeService

router = APIRouter()


class BridgeConnectionRequest(BaseModel):
    """Bridge connection request."""
    name: str = Field(..., min_length=1, max_length=256)
    source_type: str = Field(..., min_length=1, max_length=50)
    target_type: str = Field(..., min_length=1, max_length=50)
    config: Dict[str, Any] = Field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None


class BridgeConnectionResponse(BaseModel):
    """Bridge connection response."""
    id: str
    name: str
    source_type: str
    target_type: str
    status: str
    config: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class BridgeStatusResponse(BaseModel):
    """Bridge status response."""
    connection_id: str
    status: str
    last_sync: Optional[str] = None
    sync_count: int = 0
    error_count: int = 0
    last_error: Optional[str] = None


class BridgeSyncRequest(BaseModel):
    """Bridge sync request."""
    connection_id: str
    force: bool = False
    options: Optional[Dict[str, Any]] = None


class BridgeSyncResponse(BaseModel):
    """Bridge sync response."""
    success: bool
    message: str
    sync_id: Optional[str] = None
    items_processed: int = 0
    items_created: int = 0
    items_updated: int = 0
    items_failed: int = 0
    duration_ms: Optional[int] = None


class BridgeDataResponse(BaseModel):
    """Bridge data response."""
    items: List[Dict[str, Any]]
    total: int
    page: int
    limit: int
    connection_id: str


@router.get("/connections", response_model=List[BridgeConnectionResponse])
async def list_bridge_connections(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """List all bridge connections."""
    service = BridgeService(db)
    connections = await service.list_connections(
        skip=skip,
        limit=limit,
        status=status
    )
    
    return [
        BridgeConnectionResponse(
            id=conn.id,
            name=conn.name,
            source_type=conn.source_type,
            target_type=conn.target_type,
            status=conn.status,
            config=conn.config or {},
            metadata=conn.bridge_metadata,
            created_at=conn.created_at.isoformat(),
            updated_at=conn.updated_at.isoformat()
        )
        for conn in connections
    ]


@router.post("/connections", response_model=BridgeConnectionResponse)
async def create_bridge_connection(
    request: BridgeConnectionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a new bridge connection."""
    service = BridgeService(db)
    
    connection = await service.create_connection(
        name=request.name,
        source_type=request.source_type,
        target_type=request.target_type,
        config=request.config,
        metadata=request.metadata
    )
    
    return BridgeConnectionResponse(
        id=connection.id,
        name=connection.name,
        source_type=connection.source_type,
        target_type=connection.target_type,
        status=connection.status,
        config=connection.config or {},
        metadata=connection.bridge_metadata,
        created_at=connection.created_at.isoformat(),
        updated_at=connection.updated_at.isoformat()
    )


@router.get("/connections/{connection_id}", response_model=BridgeConnectionResponse)
async def get_bridge_connection(
    connection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get a specific bridge connection."""
    service = BridgeService(db)
    connection = await service.get_connection(connection_id)
    
    if not connection:
        raise HTTPException(status_code=404, detail="Bridge connection not found")
    
    return BridgeConnectionResponse(
        id=connection.id,
        name=connection.name,
        source_type=connection.source_type,
        target_type=connection.target_type,
        status=connection.status,
        config=connection.config or {},
        metadata=connection.bridge_metadata,
        created_at=connection.created_at.isoformat(),
        updated_at=connection.updated_at.isoformat()
    )


@router.get("/connections/{connection_id}/status", response_model=BridgeStatusResponse)
async def get_bridge_status(
    connection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get bridge connection status."""
    service = BridgeService(db)
    status = await service.get_connection_status(connection_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Bridge connection not found")
    
    return BridgeStatusResponse(**status)


@router.post("/connections/{connection_id}/sync", response_model=BridgeSyncResponse)
async def sync_bridge_connection(
    connection_id: str,
    request: Optional[BridgeSyncRequest] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Synchronize a bridge connection."""
    service = BridgeService(db)
    
    sync_options = {}
    force = False
    
    if request:
        sync_options = request.options or {}
        force = request.force
        if request.connection_id != connection_id:
            raise HTTPException(
                status_code=400,
                detail="Connection ID in path and body must match"
            )
    
    try:
        result = await service.sync_connection(
            connection_id=connection_id,
            force=force,
            options=sync_options
        )
        
        return BridgeSyncResponse(**result)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")


@router.get("/connections/{connection_id}/data", response_model=BridgeDataResponse)
async def get_bridge_data(
    connection_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get data from a bridge connection."""
    service = BridgeService(db)
    
    try:
        data = await service.get_connection_data(
            connection_id=connection_id,
            skip=skip,
            limit=limit
        )
        
        return BridgeDataResponse(**data)
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/connections/{connection_id}")
async def delete_bridge_connection(
    connection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Delete a bridge connection."""
    service = BridgeService(db)
    success = await service.delete_connection(connection_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Bridge connection not found")
    
    return {"message": "Bridge connection deleted successfully"}


@router.get("/types", response_model=Dict[str, List[str]])
async def get_bridge_types(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get available bridge connection types."""
    service = BridgeService(db)
    types = await service.get_supported_types()
    
    return types


@router.get("/sessions")
async def list_bridge_sessions(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """List active bridge sessions."""
    service = BridgeService(db)
    sessions = await service.get_active_sessions()
    
    return {
        "sessions": sessions,
        "total": len(sessions),
        "active": sum(1 for s in sessions if s.get("status") == "active")
    }


@router.get("/metrics")
async def get_bridge_metrics(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get bridge system metrics."""
    service = BridgeService(db)
    metrics = await service.get_metrics()
    
    return {
        "total_connections": metrics.get("total_connections", 0),
        "active_connections": metrics.get("active_connections", 0),
        "total_syncs": metrics.get("total_syncs", 0),
        "successful_syncs": metrics.get("successful_syncs", 0),
        "failed_syncs": metrics.get("failed_syncs", 0),
        "last_sync_time": metrics.get("last_sync_time"),
        "average_sync_duration_ms": metrics.get("average_sync_duration_ms", 0),
        "data_transferred_bytes": metrics.get("data_transferred_bytes", 0)
    }


@router.get("/connections/{connection_id}/logs")
async def get_bridge_logs(
    connection_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=1000),
    level: Optional[str] = Query(None, description="Filter by log level"),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get logs for a bridge connection."""
    service = BridgeService(db)
    
    try:
        logs = await service.get_connection_logs(
            connection_id=connection_id,
            skip=skip,
            limit=limit,
            level=level
        )
        
        return {"logs": logs}
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))