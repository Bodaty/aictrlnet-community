"""Community Marketplace API endpoints.

Provides browsing, searching, installing (limit 10), reviewing, and
retrieving marketplace items. Publishing is available in Business edition.

Mounted at prefix ``/marketplace`` by the community router.
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.dependencies import get_current_user_safe
from schemas.marketplace import (
    MarketplaceItemResponse,
    MarketplaceItemListResponse,
    MarketplaceSearchRequest,
    MarketplaceReviewCreate,
    MarketplaceReviewResponse,
    MarketplaceReviewListResponse,
    MarketplaceInstallRequest,
    MarketplaceInstallResponse,
    MarketplaceInstallationListResponse,
    MarketplaceUninstallResponse,
)
from services.marketplace_service import MarketplaceService
from api.v1.endpoints._auth_helpers import get_safe_user_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["marketplace"])


# ── Browse / List ────────────────────────────────────────────────────────────


@router.get(
    "/",
    response_model=MarketplaceItemListResponse,
)
async def browse_marketplace(
    category: Optional[str] = Query(None, description="Filter by category: workflow, template, adapter, agent"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Browse published marketplace items."""
    service = MarketplaceService(db)
    return await service.browse(category=category, limit=limit, offset=offset)


# ── Search ───────────────────────────────────────────────────────────────────


@router.post(
    "/search",
    response_model=MarketplaceItemListResponse,
)
async def search_marketplace(
    request: MarketplaceSearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Search marketplace items with filters and sorting."""
    service = MarketplaceService(db)
    return await service.search(request)


@router.get(
    "/search",
    response_model=MarketplaceItemListResponse,
)
async def search_marketplace_get(
    q: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Search marketplace items via GET (convenience alias for POST /search)."""
    service = MarketplaceService(db)
    request = MarketplaceSearchRequest(query=q or "", category=category, limit=limit, offset=offset)
    return await service.search(request)


# ── Category convenience routes (placed BEFORE /{item_id} catch-all) ─────────


@router.get(
    "/categories",
    response_model=Dict[str, Any],
)
async def list_categories(
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """List available marketplace categories."""
    return {
        "categories": [
            {"name": "workflow", "label": "Workflows", "description": "Automation workflows"},
            {"name": "template", "label": "Templates", "description": "Workflow templates"},
            {"name": "adapter", "label": "Adapters", "description": "Platform adapters"},
            {"name": "agent", "label": "Agents", "description": "AI agents"},
        ]
    }


@router.get(
    "/templates",
    response_model=MarketplaceItemListResponse,
)
async def browse_templates(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Browse marketplace templates."""
    service = MarketplaceService(db)
    return await service.browse(category="template", limit=limit, offset=offset)


@router.get(
    "/adapters",
    response_model=MarketplaceItemListResponse,
)
async def browse_adapters(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Browse marketplace adapters."""
    service = MarketplaceService(db)
    return await service.browse(category="adapter", limit=limit, offset=offset)


@router.get(
    "/agents",
    response_model=MarketplaceItemListResponse,
)
async def browse_agents(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Browse marketplace agents."""
    service = MarketplaceService(db)
    return await service.browse(category="agent", limit=limit, offset=offset)


# ── Get Item ─────────────────────────────────────────────────────────────────


@router.get(
    "/{item_id}",
    response_model=MarketplaceItemResponse,
)
async def get_marketplace_item(
    item_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Get a single marketplace item by ID."""
    service = MarketplaceService(db)
    item = await service.get_item(item_id)
    if item is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Marketplace item not found",
        )
    return item


# ── Install / Uninstall ─────────────────────────────────────────────────────


@router.post(
    "/{item_id}/install",
    response_model=MarketplaceInstallResponse,
    status_code=status.HTTP_201_CREATED,
)
async def install_item(
    item_id: str,
    request: MarketplaceInstallRequest = None,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Install a marketplace item. Community edition limits to 10 active installs."""
    if request is None:
        request = MarketplaceInstallRequest()
    user_id = get_safe_user_id(current_user)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found",
        )
    service = MarketplaceService(db)
    try:
        return await service.install(item_id, user_id, request)
    except ValueError as exc:
        error_msg = str(exc)
        # Return 404 for "not found" errors, 400 for other validation errors
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )


@router.post(
    "/installations/{installation_id}/uninstall",
    response_model=MarketplaceUninstallResponse,
)
async def uninstall_item(
    installation_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Uninstall a previously installed marketplace item."""
    user_id = get_safe_user_id(current_user)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found",
        )
    service = MarketplaceService(db)
    try:
        return await service.uninstall(installation_id, user_id)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )


# ── My Installations ────────────────────────────────────────────────────────


@router.get(
    "/installations/me",
    response_model=MarketplaceInstallationListResponse,
)
async def my_installations(
    status_filter: Optional[str] = Query("installed", description="Filter: installed, uninstalled"),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """List the current user's marketplace installations."""
    user_id = get_safe_user_id(current_user)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found",
        )
    service = MarketplaceService(db)
    return await service.list_installations(user_id, status_filter=status_filter)


# ── Reviews ──────────────────────────────────────────────────────────────────


@router.get(
    "/{item_id}/reviews",
    response_model=MarketplaceReviewListResponse,
)
async def get_reviews(
    item_id: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Get reviews for a marketplace item."""
    service = MarketplaceService(db)
    return await service.get_reviews(item_id, limit=limit, offset=offset)


@router.post(
    "/{item_id}/reviews",
    response_model=MarketplaceReviewResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_review(
    item_id: str,
    request: MarketplaceReviewCreate,
    db: AsyncSession = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
):
    """Submit a review for a marketplace item (one review per user per item)."""
    user_id = get_safe_user_id(current_user)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID not found",
        )
    service = MarketplaceService(db)
    try:
        return await service.create_review(item_id, user_id, request)
    except ValueError as exc:
        error_msg = str(exc)
        # Return 404 for "not found" errors, 400 for other validation errors
        if "not found" in error_msg.lower():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=error_msg,
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=error_msg,
        )
