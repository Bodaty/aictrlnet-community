"""Adapter Configuration API endpoints for managing user adapter settings."""

import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
from pydantic import UUID4

from core.database import get_db
from core.dependencies import get_edition, get_current_user_safe
from core.security import get_current_active_user
from models.adapter_config import UserAdapterConfig
from schemas.adapter_config import (
    AdapterConfigCreate,
    AdapterConfigUpdate,
    AdapterConfigResponse,
    AdapterConfigTestRequest,
    AdapterConfigTestResponse,
    AdapterConfigListResponse,
    AdapterConfigBulkTestRequest,
    AdapterConfigBulkTestResponse,
    AdapterConfigWithCapabilities,
    TestStatus
)
from services.adapter_config import AdapterConfigService
from adapters.registry import adapter_registry


logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/config", response_model=AdapterConfigListResponse)
async def list_adapter_configs(
    adapter_type: Optional[str] = Query(None, description="Filter by adapter type"),
    enabled_only: bool = Query(False, description="Only show enabled configurations"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """List user's adapter configurations."""
    service = AdapterConfigService(db)
    
    configs, total = await service.list_user_configs(
        user_id=current_user.get('id'),
        adapter_type=adapter_type,
        enabled_only=enabled_only,
        skip=skip,
        limit=limit
    )
    
    # Convert to response models, handling encrypted credentials
    response_configs = []
    for config in configs:
        config_dict = config.to_dict()
        # Don't include credentials in list response for security
        config_dict.pop('credentials', None)
        # Add settings from the model
        config_dict['settings'] = config.settings
        # Fix metadata field name
        config_dict['metadata'] = config_dict.pop('metadata', {})
        response_configs.append(AdapterConfigResponse(**config_dict))
    
    return AdapterConfigListResponse(
        configs=response_configs,
        total=total
    )


@router.post("/config", response_model=AdapterConfigResponse)
async def create_adapter_config(
    config_data: AdapterConfigCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Create a new adapter configuration."""
    service = AdapterConfigService(db)
    
    # Verify adapter type exists in registry
    if config_data.adapter_type not in adapter_registry.adapter_classes:
        raise HTTPException(
            status_code=400,
            detail=f"Adapter type '{config_data.adapter_type}' not found in registry"
        )
    
    # Create configuration
    config = await service.create_config(
        user_id=current_user.get('id'),
        config_data=config_data
    )
    
    # Convert to response model
    config_dict = config.to_dict()
    # Don't include encrypted credentials in response
    config_dict.pop('credentials', None)
    config_dict['settings'] = config.settings
    config_dict['credentials'] = None  # Indicate credentials are saved but not shown
    config_dict['metadata'] = config_dict.pop('metadata', {})
    
    return AdapterConfigResponse(**config_dict)


@router.get("/config/{config_id}", response_model=AdapterConfigWithCapabilities)
async def get_adapter_config(
    config_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Get a specific adapter configuration with capabilities."""
    service = AdapterConfigService(db)
    
    config = await service.get_config(
        config_id=config_id,
        user_id=current_user.get('id')
    )
    
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    # Enhance with registry information
    response = AdapterConfigWithCapabilities(**config.to_dict())
    
    if config.adapter_type in adapter_registry.adapter_classes:
        response.is_registered = True
        
        # Get capabilities from registry if instance exists
        adapter_instance = adapter_registry.adapters.get(config.adapter_type)
        if adapter_instance and hasattr(adapter_instance, 'get_capabilities'):
            try:
                caps = adapter_instance.get_capabilities()
                response.capabilities = [
                    {
                        "name": cap.name,
                        "description": cap.description,
                        "category": cap.category,
                    }
                    for cap in caps
                ]
            except:
                pass
        
        # Get category and description
        if adapter_instance:
            if hasattr(adapter_instance, 'config') and hasattr(adapter_instance.config, 'category'):
                response.category = str(adapter_instance.config.category.value if hasattr(adapter_instance.config.category, 'value') else adapter_instance.config.category)
            
        adapter_class = adapter_registry.adapter_classes[config.adapter_type]
        response.description = adapter_class.__doc__ or f"{config.adapter_type} adapter"
    
    return response


@router.put("/config/{config_id}", response_model=AdapterConfigResponse)
async def update_adapter_config(
    config_id: UUID4,
    update_data: AdapterConfigUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Update an adapter configuration."""
    service = AdapterConfigService(db)
    
    config = await service.update_config(
        config_id=config_id,
        user_id=current_user.get('id'),
        update_data=update_data
    )
    
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return config


@router.delete("/config/{config_id}")
async def delete_adapter_config(
    config_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Delete an adapter configuration."""
    service = AdapterConfigService(db)
    
    success = await service.delete_config(
        config_id=config_id,
        user_id=current_user.get('id')
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    return {"message": "Configuration deleted successfully"}


@router.post("/config/{config_id}/test", response_model=AdapterConfigTestResponse)
async def test_adapter_config(
    config_id: UUID4,
    test_request: AdapterConfigTestRequest = AdapterConfigTestRequest(),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Test an adapter configuration."""
    service = AdapterConfigService(db)
    
    # Get configuration
    config = await service.get_config(
        config_id=config_id,
        user_id=current_user.get('id')
    )
    
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    # Test the configuration
    result = await service.test_config(
        config=config,
        timeout=test_request.timeout
    )
    
    return result


@router.post("/config/test-bulk", response_model=AdapterConfigBulkTestResponse)
async def test_adapter_configs_bulk(
    test_request: AdapterConfigBulkTestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Test multiple adapter configurations."""
    service = AdapterConfigService(db)
    
    # Verify all configs belong to user
    configs = []
    for config_id in test_request.config_ids:
        config = await service.get_config(
            config_id=config_id,
            user_id=current_user.get('id')
        )
        if not config:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration {config_id} not found"
            )
        configs.append(config)
    
    # Test configurations
    result = await service.test_configs_bulk(
        configs=configs,
        parallel=test_request.parallel,
        timeout=test_request.timeout
    )
    
    return result


@router.post("/config/{config_id}/activate")
async def activate_adapter_config(
    config_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Activate an adapter configuration and create runtime instance."""
    service = AdapterConfigService(db)
    
    # Get configuration
    config = await service.get_config(
        config_id=config_id,
        user_id=current_user.get('id')
    )
    
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    # Activate the adapter
    success = await service.activate_adapter(config)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to activate adapter"
        )
    
    return {
        "message": "Adapter activated successfully",
        "adapter_type": config.adapter_type,
        "config_id": str(config_id)
    }


@router.post("/config/{config_id}/deactivate")
async def deactivate_adapter_config(
    config_id: UUID4,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Deactivate an adapter configuration and remove runtime instance."""
    service = AdapterConfigService(db)
    
    # Get configuration
    config = await service.get_config(
        config_id=config_id,
        user_id=current_user.get('id')
    )
    
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")
    
    # Deactivate the adapter
    success = await service.deactivate_adapter(config)
    
    if not success:
        raise HTTPException(
            status_code=500,
            detail="Failed to deactivate adapter"
        )
    
    return {
        "message": "Adapter deactivated successfully",
        "adapter_type": config.adapter_type,
        "config_id": str(config_id)
    }