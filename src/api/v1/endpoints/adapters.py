"""Adapter-related endpoints."""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from core.database import get_db
from core.dependencies import get_edition
from core.security import get_current_active_user
from schemas.adapter import (
    AdapterResponse,
    AdapterDiscoverResponse,
    AdapterCategoryResponse,
    AdapterAvailabilityRequest,
    AdapterAvailabilityResponse,
)
from services.adapter import AdapterService

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/discover", response_model=AdapterDiscoverResponse, deprecated=True)
async def discover_adapters(
    response: Response,
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in name and description"),
    include_unavailable: bool = Query(False, description="Include adapters not available in current edition"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
    edition: str = Depends(get_edition),
):
    """[DEPRECATED] Discover available adapters filtered by user's edition.
    
    **DEPRECATED**: This endpoint queries the database for adapter metadata.
    Please use `/api/v1/adapters/registry/list` instead, which provides:
    - Real-time adapter status from the runtime registry
    - Actual capabilities from initialized adapters
    - Discovery mode information
    - Better performance (no database query)
    
    This endpoint will be removed in a future version.
    """
    # Log usage of deprecated endpoint for monitoring
    user_email = getattr(current_user, 'email', 'unknown') if hasattr(current_user, 'email') else current_user.get('email', 'unknown') if isinstance(current_user, dict) else 'unknown'
    logger.warning(
        f"DEPRECATED endpoint /adapters/discover called by user {user_email} "
        f"with params: category={category}, search={search}, edition={edition}"
    )
    
    import warnings
    warnings.warn(
        "The /adapters/discover endpoint is deprecated. "
        "Use /api/v1/adapters/registry/list instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Add deprecation headers for API clients
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = "</api/v1/adapters/registry/list>; rel=\"alternate\""
    response.headers["X-Deprecated-Since"] = "2025-01-10"
    response.headers["X-Sunset-Date"] = "2025-04-01"
    response.headers["Warning"] = '299 - "This endpoint is deprecated. Use /api/v1/adapters/registry/list instead."'
    
    service = AdapterService(db)
    adapters, total = await service.discover_adapters(
        edition=edition,
        category=category,
        search=search,
        skip=skip,
        limit=limit,
        include_unavailable=include_unavailable
    )
    
    return AdapterDiscoverResponse(
        adapters=adapters,
        total=total,
        edition=edition,
        categories=await service.get_categories()
    )


@router.get("/categories", response_model=List[AdapterCategoryResponse])
async def get_adapter_categories(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get all adapter categories with counts."""
    service = AdapterService(db)
    return await service.get_categories_with_counts()


@router.get("/{adapter_id}", response_model=AdapterResponse)
async def get_adapter(
    adapter_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
    edition: str = Depends(get_edition),
):
    """Get detailed information about a specific adapter."""
    service = AdapterService(db)
    adapter = await service.get_adapter(adapter_id, edition)
    
    if not adapter:
        raise HTTPException(status_code=404, detail="Adapter not found")
    
    return adapter


@router.post("/check-availability", response_model=AdapterAvailabilityResponse)
async def check_adapter_availability(
    request: AdapterAvailabilityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
    edition: str = Depends(get_edition),
):
    """Check if multiple adapters are available for the user."""
    service = AdapterService(db)
    availability = await service.check_availability(
        adapter_ids=request.adapter_ids,
        edition=edition
    )
    
    return AdapterAvailabilityResponse(
        available=availability["available"],
        unavailable=availability["unavailable"],
        edition=edition
    )


@router.get("/", response_model=List[AdapterResponse])
async def list_adapters(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
    edition: str = Depends(get_edition),
):
    """List available adapters."""
    service = AdapterService(db)
    adapters = await service.list_adapters(
        edition=edition,
        skip=skip,
        limit=limit
    )
    return adapters