"""Resource pool configuration endpoints for Community Edition."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from core.database import get_db
from core.security import get_current_user
from schemas.resource_pool import (
    ResourcePoolConfigCreate,
    ResourcePoolConfigUpdate,
    ResourcePoolConfigResponse
)
from services.basic_resource_pool import BasicResourcePoolService
from core.exceptions import ValidationError

router = APIRouter()


@router.get("/configs", response_model=List[ResourcePoolConfigResponse])
async def list_resource_pool_configs(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    resource_type: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """List resource pool configurations."""
    try:
        service = BasicResourcePoolService(db)
        configs = await service.list_configs(
            resource_type=resource_type,
            is_active=is_active,
            skip=skip,
            limit=limit
        )
        # Return empty list if no configs found
        if configs is None:
            return []
        return configs
    except Exception as e:
        # Log the error for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error listing resource pool configs: {str(e)}")
        # Return empty list instead of 404 for list endpoints
        return []


@router.get("/configs/{config_id}", response_model=ResourcePoolConfigResponse)
async def get_resource_pool_config(
    config_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get a specific resource pool configuration."""
    service = BasicResourcePoolService(db)
    config = await service.get_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Resource pool configuration not found")
    return config


@router.post("/configs", response_model=ResourcePoolConfigResponse)
async def create_resource_pool_config(
    config: ResourcePoolConfigCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new resource pool configuration."""
    service = BasicResourcePoolService(db)
    try:
        return await service.create_config(config)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.put("/configs/{config_id}", response_model=ResourcePoolConfigResponse)
async def update_resource_pool_config(
    config_id: str,
    config: ResourcePoolConfigUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Update a resource pool configuration."""
    service = BasicResourcePoolService(db)
    updated_config = await service.update_config(config_id, config)
    if not updated_config:
        raise HTTPException(status_code=404, detail="Resource pool configuration not found")
    return updated_config


@router.delete("/configs/{config_id}")
async def delete_resource_pool_config(
    config_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Delete a resource pool configuration."""
    service = BasicResourcePoolService(db)
    if not await service.delete_config(config_id):
        raise HTTPException(status_code=404, detail="Resource pool configuration not found")
    return {"detail": "Resource pool configuration deleted"}


@router.get("/configs/{config_id}/status")
async def get_pool_status(
    config_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get runtime status of a pool by config ID."""
    service = BasicResourcePoolService(db)
    config = await service.get_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="Resource pool configuration not found")
    
    status = service.get_pool_status(config.name)
    if not status:
        return {
            "name": config.name,
            "status": "not_initialized",
            "is_active": config.enabled
        }
    
    return status


@router.get("/runtime")
async def get_runtime_pools(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Get all runtime pools and their status."""
    service = BasicResourcePoolService(db)
    await service.initialize()  # Ensure pools are loaded
    return service.pool_manager.list_pools()


@router.get("/types")
async def get_resource_types(
    current_user: dict = Depends(get_current_user)
):
    """Get available resource types."""
    return {
        "types": [
            {"id": "compute", "name": "Compute Resources", "description": "CPU and memory resources"},
            {"id": "storage", "name": "Storage Resources", "description": "Disk and object storage"},
            {"id": "network", "name": "Network Resources", "description": "Bandwidth and connections"},
            {"id": "ai_model", "name": "AI Model Resources", "description": "AI/ML model instances"},
            {"id": "database", "name": "Database Resources", "description": "Database connections"},
            {"id": "adapter", "name": "Adapter Resources", "description": "AI/LLM adapter connections"},
            {"id": "cache", "name": "Cache Resources", "description": "Cache connections"}
        ]
    }


# Convenience endpoint for creating resource pools directly
@router.post("/", response_model=ResourcePoolConfigResponse)
async def create_resource_pool(
    config: ResourcePoolConfigCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """Create a new resource pool (alias for /configs)."""
    service = BasicResourcePoolService(db)
    try:
        return await service.create_config(config)
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))