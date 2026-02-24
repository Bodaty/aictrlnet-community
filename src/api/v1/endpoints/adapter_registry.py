"""Adapter Registry API endpoints for discovery.

Simplified version that queries the database for adapter metadata
instead of relying on complex runtime registry initialization.
"""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, text, func, select
from sqlalchemy.future import select as future_select

from core.dependencies import get_edition
from core.security import get_current_active_user
from core.database import get_db
from models.community_complete import Adapter
from adapters.registry import adapter_registry

router = APIRouter()


class AdapterInfo(BaseModel):
    """Information about an available adapter."""
    id: str
    type: str  # Adapter type/id
    name: str
    category: str
    description: Optional[str]
    capabilities: Optional[List[Dict[str, Any]]]
    min_edition: str
    available: bool
    edition: str  # Current user's edition
    

class AdapterRegistryResponse(BaseModel):
    """Response for adapter registry endpoints."""
    adapters: List[AdapterInfo]
    total: int
    edition: str
    categories: List[str]


@router.get("/registry/list", response_model=AdapterRegistryResponse)
async def list_available_adapters(
    category: Optional[str] = Query(None, description="Filter by category"),
    search: Optional[str] = Query(None, description="Search in adapter name and description"),
    edition_filter: Optional[str] = Query(None, description="Filter by minimum edition requirement"),
    current_user: dict = Depends(get_current_active_user),
    edition: str = Depends(get_edition),
    db: AsyncSession = Depends(get_db),
):
    """List all available adapters from the database.
    
    This endpoint queries the database for adapter metadata.
    No complex runtime registry initialization required.
    Adapters are filtered based on the user's edition (accretive model).
    """
    
    # Build query
    stmt = select(Adapter).where(Adapter.enabled == True)
    
    # Filter by edition (accretive model)
    # Community sees only community adapters
    # Business sees community + business adapters  
    # Enterprise sees all adapters
    edition_hierarchy = {
        "community": ["community"],
        "business": ["community", "business"],
        "enterprise": ["community", "business", "enterprise"]
    }
    allowed_editions = edition_hierarchy.get(edition.lower(), ["community"])
    stmt = stmt.where(Adapter.min_edition.in_(allowed_editions))
    
    # Apply filters
    if category:
        stmt = stmt.where(Adapter.category == category)
    
    if search:
        search_pattern = f"%{search}%"
        stmt = stmt.where(
            or_(
                Adapter.name.ilike(search_pattern),
                Adapter.description.ilike(search_pattern)
            )
        )
    
    if edition_filter:
        stmt = stmt.where(Adapter.min_edition == edition_filter)
    
    # Execute query
    result = await db.execute(stmt)
    adapters_db = result.scalars().all()
    
    # Build a set of known runtime adapter type names for lookups
    runtime_types = set(adapter_registry._adapter_classes.keys())

    # Convert to response format
    adapters = []
    for adapter in adapters_db:
        # Convert string capabilities to dictionary format for compatibility
        capabilities_list = []
        if adapter.capabilities:
            for cap in adapter.capabilities:
                # Convert string to dict format
                capabilities_list.append({
                    "name": cap,
                    "description": cap.replace("_", " ").title(),
                    "category": adapter.category
                })

        # Derive the registry type key from the DB name.
        # "Claude" → "claude", "OpenAI" → "openai", etc.
        # Fall back to UUID if the lowercased name isn't in the runtime registry.
        registry_type = adapter.name.lower().replace(" ", "-")
        if registry_type not in runtime_types:
            # Try exact lowercase
            registry_type = adapter.name.lower().replace(" ", "")
        if registry_type not in runtime_types:
            registry_type = str(adapter.id)

        adapter_info = AdapterInfo(
            id=adapter.id,
            type=registry_type,
            name=adapter.name,
            category=adapter.category,
            description=adapter.description,
            capabilities=capabilities_list,
            min_edition=adapter.min_edition,
            available=adapter.available,
            edition=edition
        )
        adapters.append(adapter_info)
    
    # Get unique categories
    categories = list(set(a.category for a in adapters))
    categories.sort()
    
    return AdapterRegistryResponse(
        adapters=adapters,
        total=len(adapters),
        edition=edition,
        categories=categories
    )



@router.get("/registry/stats")
async def get_registry_stats(
    current_user: dict = Depends(get_current_active_user),
    edition: str = Depends(get_edition),
    db: AsyncSession = Depends(get_db),
):
    """Get statistics about available adapters from the database."""
    
    # Get edition-filtered counts
    edition_hierarchy = {
        "community": ["community"],
        "business": ["community", "business"],
        "enterprise": ["community", "business", "enterprise"]
    }
    allowed_editions = edition_hierarchy.get(edition.lower(), ["community"])
    
    # Total adapters available to this edition
    stmt = select(func.count(Adapter.id)).where(
        and_(
            Adapter.enabled == True,
            Adapter.min_edition.in_(allowed_editions)
        )
    )
    result = await db.execute(stmt)
    total_available = result.scalar()
    
    # Count by category
    categories_count = {}
    stmt = select(
        Adapter.category,
        func.count(Adapter.id)
    ).where(
        and_(
            Adapter.enabled == True,
            Adapter.min_edition.in_(allowed_editions)
        )
    ).group_by(Adapter.category)
    result = await db.execute(stmt)
    category_results = result.all()
    
    for category, count in category_results:
        categories_count[category] = count
    
    # Count by edition requirement
    edition_counts = {}
    stmt = select(
        Adapter.min_edition,
        func.count(Adapter.id)
    ).where(
        Adapter.enabled == True
    ).group_by(Adapter.min_edition)
    result = await db.execute(stmt)
    edition_results = result.all()
    
    for min_edition, count in edition_results:
        edition_counts[min_edition] = count
    
    return {
        "edition": edition,
        "total_available": total_available,
        "categories": categories_count,
        "edition_requirements": edition_counts,
        "database_status": "healthy"
    }


@router.get("/registry/{adapter_id}", response_model=AdapterInfo)
async def get_adapter_info(
    adapter_id: str,
    current_user: dict = Depends(get_current_active_user),
    edition: str = Depends(get_edition),
    db: AsyncSession = Depends(get_db),
):
    """Get detailed information about a specific adapter from the database."""
    
    # Query database for adapter
    stmt = select(Adapter).where(Adapter.id == adapter_id)
    result = await db.execute(stmt)
    adapter = result.scalar_one_or_none()
    
    if not adapter:
        raise HTTPException(status_code=404, detail=f"Adapter '{adapter_id}' not found")
    
    # Check if user's edition has access
    edition_hierarchy = {
        "community": ["community"],
        "business": ["community", "business"],
        "enterprise": ["community", "business", "enterprise"]
    }
    allowed_editions = edition_hierarchy.get(edition.lower(), ["community"])
    
    if adapter.min_edition not in allowed_editions:
        raise HTTPException(
            status_code=403, 
            detail=f"Adapter '{adapter_id}' requires {adapter.min_edition} edition"
        )
    
    # Convert string capabilities to dictionary format for compatibility
    capabilities_list = []
    if adapter.capabilities:
        for cap in adapter.capabilities:
            capabilities_list.append({
                "name": cap,
                "description": cap.replace("_", " ").title(),
                "category": adapter.category
            })
    
    return AdapterInfo(
        id=adapter.id,
        type=adapter.id,
        name=adapter.name,
        category=adapter.category,
        description=adapter.description,
        capabilities=capabilities_list,
        min_edition=adapter.min_edition,
        available=adapter.available,
        edition=edition
    )


