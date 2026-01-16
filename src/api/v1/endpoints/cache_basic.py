"""Cache basic endpoints for Community Edition."""

from typing import Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user

router = APIRouter()

# Simple in-memory cache stats for Community edition
_cache_stats = {
    "hits": 0,
    "misses": 0,
    "entries": 0,
    "size_bytes": 0,
    "start_time": datetime.utcnow()
}


@router.get("/health", response_model=Dict[str, Any])
async def get_cache_health_basic(
    current_user = Depends(get_current_active_user),
):
    """
    Get cache service health (Community Edition).
    
    Returns basic health status only.
    """
    uptime = (datetime.utcnow() - _cache_stats["start_time"]).total_seconds()
    
    return {
        "status": "healthy",
        "type": "in-memory",
        "edition": "community",
        "uptime_seconds": int(uptime),
        "features": {
            "view_stats": True,
            "clear_cache": False,
            "manage_keys": False,
            "configure": False
        },
        "upgrade_prompt": {
            "message": "Upgrade for full cache management capabilities",
            "url": "/upgrade/business",
            "benefits": [
                "Clear cache on demand",
                "Pattern-based cache clearing",
                "View and manage cache keys",
                "Performance metrics",
                "Cache configuration"
            ]
        }
    }


@router.get("/basic-stats", response_model=Dict[str, Any])
async def get_cache_basic_stats(
    current_user = Depends(get_current_active_user),
):
    """
    Get basic cache statistics (Community Edition).
    
    Read-only access to basic cache metrics.
    """
    total_requests = _cache_stats["hits"] + _cache_stats["misses"]
    hit_rate = 0.0
    if total_requests > 0:
        hit_rate = (_cache_stats["hits"] / total_requests) * 100
    
    uptime = (datetime.utcnow() - _cache_stats["start_time"]).total_seconds()
    
    return {
        "edition": "community",
        "type": "basic_stats",
        "metrics": {
            "hit_rate": round(hit_rate, 2),
            "total_requests": total_requests,
            "hits": _cache_stats["hits"],
            "misses": _cache_stats["misses"],
            "approximate_entries": _cache_stats["entries"],
            "approximate_size": f"{_cache_stats['size_bytes'] / 1024:.1f}KB"
        },
        "uptime": {
            "seconds": int(uptime),
            "readable": f"{int(uptime / 3600)}h {int((uptime % 3600) / 60)}m"
        },
        "limitations": {
            "detailed_stats": False,
            "cache_management": False,
            "key_inspection": False,
            "performance_tuning": False
        },
        "upgrade_benefits": {
            "business": {
                "message": "Get detailed cache analytics and management",
                "features": [
                    "Detailed performance metrics",
                    "Clear cache by pattern",
                    "Memory usage breakdown",
                    "Hit/miss analysis by endpoint"
                ]
            },
            "enterprise": {
                "message": "Advanced cache optimization and analytics",
                "features": [
                    "All Business features",
                    "Cache warming strategies",
                    "Multi-region distribution",
                    "AI-driven optimization",
                    "Predictive cache management"
                ]
            }
        }
    }


@router.get("/info", response_model=Dict[str, Any])
async def get_cache_info_basic(
    current_user = Depends(get_current_active_user),
):
    """
    Get cache information (Community Edition).
    
    Returns basic information about cache capabilities.
    """
    return {
        "edition": "community",
        "cache_type": "in-memory",
        "persistence": False,
        "distributed": False,
        "features_available": {
            "basic_stats": True,
            "health_check": True,
            "detailed_stats": False,
            "clear_cache": False,
            "manage_keys": False,
            "configure": False,
            "analytics": False
        },
        "current_usage": {
            "description": "Basic caching for improved performance",
            "backends": ["memory"],
            "ttl_default": 300,  # 5 minutes
            "eviction_policy": "LRU"
        },
        "management_options": {
            "available": False,
            "message": "Cache management requires Business edition or higher",
            "upgrade_url": "/upgrade/business"
        }
    }


# Simulate cache activity for demo purposes
@router.post("/simulate", response_model=Dict[str, Any])
async def simulate_cache_activity(
    request: Dict[str, Any],
    current_user = Depends(get_current_active_user),
):
    """
    Simulate cache activity for testing (Community Edition).
    
    This endpoint is for demonstration purposes only.
    """
    action = request.get("action", "hit")
    count = min(request.get("count", 1), 10)  # Limit to 10 for safety
    
    if action == "hit":
        _cache_stats["hits"] += count
    elif action == "miss":
        _cache_stats["misses"] += count
    elif action == "add_entry":
        _cache_stats["entries"] += count
        _cache_stats["size_bytes"] += count * 1024  # Assume 1KB per entry
    else:
        raise HTTPException(status_code=400, detail="Invalid action")
    
    return {
        "status": "success",
        "action": action,
        "count": count,
        "updated_stats": {
            "hits": _cache_stats["hits"],
            "misses": _cache_stats["misses"],
            "entries": _cache_stats["entries"]
        }
    }


# Endpoints that require Business edition or higher
@router.delete("/clear")
async def clear_cache_not_available(
    current_user = Depends(get_current_active_user),
):
    """Clear cache - not available in Community Edition."""
    raise HTTPException(
        status_code=403,
        detail={
            "error": "feature_not_available",
            "message": "Cache clearing is available in Business edition and above",
            "upgrade_url": "/upgrade/business",
            "benefits": [
                "Clear entire cache",
                "Clear by pattern matching",
                "Scheduled cache clearing",
                "Cache warming after clear"
            ]
        }
    )


@router.get("/keys")
async def list_cache_keys_not_available(
    current_user = Depends(get_current_active_user),
):
    """List cache keys - not available in Community Edition."""
    raise HTTPException(
        status_code=403,
        detail={
            "error": "feature_not_available", 
            "message": "Cache key inspection requires Enterprise edition",
            "upgrade_url": "/upgrade/enterprise",
            "benefits": [
                "View all cache keys",
                "Search keys by pattern",
                "Inspect key values",
                "Analyze key usage patterns",
                "Export cache contents"
            ]
        }
    )