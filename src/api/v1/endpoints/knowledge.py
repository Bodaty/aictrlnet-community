"""
Knowledge Service API Endpoints

Provides access to system knowledge, capabilities, and intelligent retrieval
for the AICtrlNet Intelligent Assistant.
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.dependencies import get_current_user_safe
from services.knowledge.system_manifest_service import get_manifest_service
from services.knowledge.knowledge_indexer import KnowledgeIndexer
from services.knowledge.knowledge_retrieval_service import KnowledgeRetrievalService
from schemas.knowledge import (
    CapabilitySummaryResponse,
    KnowledgeRetrievalRequest,
    KnowledgeRetrievalResponse,
    SystemManifestResponse,
    FeatureDetailResponse,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
    SuggestionRequest,
    SuggestionResponse,
    KnowledgeStatsResponse,
    KnowledgeItemResponse,
    SuggestionItem
)

router = APIRouter(prefix="/knowledge", tags=["knowledge"])


@router.get("/capabilities", response_model=CapabilitySummaryResponse)
async def get_system_capabilities(
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> CapabilitySummaryResponse:
    """
    Get system capabilities summary.

    Returns counts of templates, agents, adapters, and overall automation coverage.
    This endpoint provides the data for the System tab in the UI.
    """
    retrieval_service = KnowledgeRetrievalService(db)
    await retrieval_service.initialize()

    capabilities = await retrieval_service.get_capabilities_summary()

    # Add user-specific context
    capabilities["user_edition"] = current_user.get("edition", "community")
    capabilities["personalized"] = True

    return CapabilitySummaryResponse(**capabilities)


@router.post("/retrieve", response_model=KnowledgeRetrievalResponse)
async def retrieve_knowledge(
    request: KnowledgeRetrievalRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> KnowledgeRetrievalResponse:
    """
    Retrieve relevant knowledge for a query.

    Uses RAG-based retrieval to find relevant templates, agents, features,
    and patterns based on the user's query and context.
    """
    retrieval_service = KnowledgeRetrievalService(db)
    await retrieval_service.initialize()

    # Enhance context with user information
    enhanced_context = request.context or {}
    enhanced_context["user_id"] = current_user.get("id")
    enhanced_context["edition"] = current_user.get("edition", "community")

    # Retrieve relevant knowledge
    knowledge_items = await retrieval_service.find_relevant_knowledge(
        query=request.query,
        context=enhanced_context,
        limit=request.limit
    )

    # Convert KnowledgeItem objects to response format
    response_items = []
    import uuid
    for item in knowledge_items:
        # KnowledgeItem from service doesn't have all fields that KnowledgeItemResponse expects
        # Create a response with the fields we have
        # Generate a deterministic UUID from the item id if it's not already a UUID
        try:
            item_uuid = uuid.UUID(item.id)
        except (ValueError, AttributeError):
            # Create a deterministic UUID v5 from the item id string
            item_uuid = uuid.uuid5(uuid.NAMESPACE_DNS, str(item.id))

        response_items.append(KnowledgeItemResponse(
            id=item_uuid,
            item_type=item.type,
            category=item.data.get("category", "general"),
            name=item.name,
            description=item.data.get("description", ""),
            content=item.data,
            tags=item.data.get("tags", []),
            keywords=item.data.get("keywords", item.data.get("search_terms", [])),
            edition_required=item.data.get("requirements", {}).get("edition", "community"),
            usage_count=0,
            relevance_score=item.relevance,
            success_rate=None,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            last_accessed=None,
            is_active=True,
            is_deprecated=False
        ))

    return KnowledgeRetrievalResponse(
        query=request.query,
        items=response_items,
        total=len(response_items),
        context_used=bool(request.context)
    )


@router.get("/manifest")
async def get_system_manifest(
    section: Optional[str] = Query(None, description="Specific section to retrieve"),
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> Dict:
    """
    Get the system manifest with all features and capabilities.

    This provides comprehensive information about the AICtrlNet system.
    Optional section parameter to get specific parts (features, templates, agents, etc.).
    """
    manifest_service = await get_manifest_service(db)

    # Get full manifest or specific section
    if section:
        if section not in manifest_service.manifest:
            raise HTTPException(404, f"Section '{section}' not found in manifest")
        return {section: manifest_service.manifest[section]}

    # Filter based on user edition
    user_edition = current_user.get("edition", "community")
    filtered_manifest = _filter_manifest_by_edition(manifest_service.manifest, user_edition)

    return filtered_manifest


@router.get("/feature/{feature_name}")
async def get_feature_details(
    feature_name: str,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> Dict:
    """
    Get detailed information about a specific feature.

    Returns comprehensive details including endpoints, UI locations,
    related templates, and examples.
    """
    manifest_service = await get_manifest_service(db)

    details = await manifest_service.get_feature_details(feature_name)

    if not details:
        raise HTTPException(404, f"Feature '{feature_name}' not found")

    # Check if user has access to this feature
    required_edition = details.get("feature", {}).get("required_edition", "community")
    user_edition = current_user.get("edition", "community")

    if not _has_edition_access(user_edition, required_edition):
        details["access_restricted"] = True
        details["upgrade_required"] = required_edition

    return details


@router.post("/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    request: KnowledgeSearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> KnowledgeSearchResponse:
    """
    Search across all knowledge types.

    Searches templates, agents, adapters, and features for matching items.
    Can filter by specific types.
    """
    indexer = KnowledgeIndexer(db)
    await indexer.build_index()

    # Perform search
    results = await indexer.search(request.query, limit=request.limit)

    # Filter by types if specified
    if request.types:
        filtered_results = {}
        for type_name in request.types:
            if type_name in results:
                filtered_results[type_name] = results[type_name]
        results = filtered_results

    # Add metadata
    total_count = sum(len(items) for items in results.values())

    return KnowledgeSearchResponse(
        query=request.query,
        results=results,
        total=total_count,
        types_searched=request.types or ["all"]
    )


@router.post("/suggestions", response_model=SuggestionResponse)
async def get_suggestions(
    request: SuggestionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> SuggestionResponse:
    """
    Get intelligent suggestions based on current context.

    Returns suggested next actions based on the current action
    and user context.
    """
    retrieval_service = KnowledgeRetrievalService(db)
    await retrieval_service.initialize()

    # Enhance context
    enhanced_context = request.context or {}
    enhanced_context["user_id"] = current_user.get("id")
    enhanced_context["edition"] = current_user.get("edition", "community")

    # Get suggestions
    suggestions = await retrieval_service.suggest_next_actions(
        current_action=request.current_action or "",
        context=enhanced_context
    )

    # Convert to SuggestionItem objects
    suggestion_items = [
        SuggestionItem(
            action=s.get("action", ""),
            description=s.get("description", ""),
            confidence=s.get("confidence", 0.5),
            category=s.get("category", "general"),
            related_items=s.get("related_items", [])
        ) for s in suggestions
    ]

    return SuggestionResponse(
        current_action=request.current_action,
        suggestions=suggestion_items,
        total=len(suggestion_items)
    )


@router.get("/stats", response_model=KnowledgeStatsResponse)
async def get_knowledge_stats(
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> KnowledgeStatsResponse:
    """
    Get statistics about the knowledge base.

    Returns counts and metadata about indexed items.
    """
    manifest_service = await get_manifest_service(db)
    stats = manifest_service.manifest.get("statistics", {})

    # Add indexing status
    indexer = KnowledgeIndexer(db)
    await indexer.build_index()

    stats["indexed_items"] = {
        "templates": len(indexer.index.get("templates", {})),
        "agents": len(indexer.index.get("agents", {})),
        "adapters": len(indexer.index.get("adapters", {})),
        "patterns": len(indexer.index.get("patterns", {}))
    }

    stats["last_indexed"] = indexer.index.get("metadata", {}).get("indexed_at")

    return KnowledgeStatsResponse(**stats)


@router.post("/refresh")
async def refresh_knowledge_index(
    db: AsyncSession = Depends(get_db),
    current_user: Dict = Depends(get_current_user_safe)
) -> Dict:
    """
    Refresh the knowledge index and manifest.

    Rebuilds the system manifest and knowledge index to capture
    any new templates, agents, or features.

    Note: This operation may take a few seconds.
    """
    # Check if user is admin or has appropriate permissions
    # For now, allow all authenticated users

    try:
        # Refresh manifest
        manifest_service = await get_manifest_service(db)
        await manifest_service.generate_manifest()

        # Refresh index
        indexer = KnowledgeIndexer(db)
        await indexer.build_index()

        return {
            "message": "Knowledge index refreshed successfully",
            "success": True
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to refresh knowledge index: {str(e)}")


# Helper functions

def _filter_manifest_by_edition(manifest: Dict, edition: str) -> Dict:
    """Filter manifest based on user edition."""
    filtered = manifest.copy()

    # Filter features by edition
    if "features" in filtered:
        edition_features = {}
        for feature_name, feature_data in filtered["features"].items():
            required_edition = feature_data.get("required_edition", "community")
            if _has_edition_access(edition, required_edition):
                edition_features[feature_name] = feature_data
        filtered["features"] = edition_features

    return filtered


def _has_edition_access(user_edition: str, required_edition: str) -> bool:
    """Check if user edition has access to a feature."""
    edition_hierarchy = {
        "community": 0,
        "business": 1,
        "enterprise": 2
    }

    user_level = edition_hierarchy.get(user_edition, 0)
    required_level = edition_hierarchy.get(required_edition, 0)

    return user_level >= required_level