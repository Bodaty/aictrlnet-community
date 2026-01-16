"""Memory basic endpoints for Community Edition."""

from typing import Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user

router = APIRouter()

# In-memory storage for Community edition (session-scoped, not persistent)
_memory_store = {}
_memory_stats = {
    "total_keys": 0,
    "total_size": 0,
    "max_keys": 1000,
    "max_size": 10 * 1024 * 1024  # 10MB
}


def _estimate_size(value: Any) -> int:
    """Estimate memory size of a value."""
    return len(str(value))


def _get_user_memory(user_id: str) -> Dict[str, Any]:
    """Get memory store for a specific user."""
    if user_id not in _memory_store:
        _memory_store[user_id] = {}
    return _memory_store[user_id]


@router.get("/health", response_model=Dict[str, Any])
async def get_memory_health_basic(
    current_user = Depends(get_current_active_user),
):
    """
    Get memory service health (Community Edition).
    
    Returns basic health status for in-memory storage.
    """
    user_memory = _get_user_memory(str(current_user.id))
    user_size = sum(_estimate_size(v) for v in user_memory.values())
    
    return {
        "status": "healthy",
        "type": "in-memory",
        "persistence": False,
        "edition": "community",
        "user_stats": {
            "keys": len(user_memory),
            "size_bytes": user_size,
            "size_readable": f"{user_size / 1024:.1f}KB"
        },
        "limitations": {
            "max_keys": 1000,
            "max_size": "10MB",
            "persistence": "none",
            "scope": "session-only",
            "sharing": "disabled"
        },
        "upgrade_prompt": {
            "message": "Upgrade for persistent distributed memory",
            "url": "/upgrade/business",
            "benefits": [
                "Redis-backed persistence",
                "Distributed across instances",
                "Unlimited storage",
                "Search and indexing",
                "Bulk operations",
                "Import/export capabilities"
            ]
        }
    }


@router.get("/stats", response_model=Dict[str, Any])
async def get_memory_stats_basic(
    current_user = Depends(get_current_active_user),
):
    """
    Get basic memory statistics (Community Edition).
    
    Returns stats for current user's session memory only.
    """
    user_memory = _get_user_memory(str(current_user.id))
    user_size = sum(_estimate_size(v) for v in user_memory.values())
    
    return {
        "type": "session",
        "user_id": str(current_user.id),
        "entries": len(user_memory),
        "size_bytes": user_size,
        "size_readable": f"{user_size / 1024:.1f}KB",
        "limits": {
            "current_keys": len(user_memory),
            "max_keys": _memory_stats["max_keys"],
            "keys_remaining": _memory_stats["max_keys"] - len(user_memory),
            "current_size": user_size,
            "max_size": _memory_stats["max_size"],
            "size_remaining": _memory_stats["max_size"] - user_size
        },
        "features_available": {
            "get": True,
            "set": True,
            "delete": True,
            "list": False,
            "search": False,
            "bulk": False,
            "export": False
        }
    }


@router.get("/{key}", response_model=Dict[str, Any])
async def get_memory_value_basic(
    key: str,
    current_user = Depends(get_current_active_user),
):
    """
    Get value from memory (Community Edition).
    
    Session-scoped only, not shared between users or sessions.
    """
    user_memory = _get_user_memory(str(current_user.id))
    
    if key not in user_memory:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "key_not_found",
                "message": f"Key '{key}' not found in session memory",
                "note": "Community edition memory is session-scoped and non-persistent"
            }
        )
    
    return {
        "key": key,
        "value": user_memory[key],
        "metadata": {
            "type": "session",
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": _estimate_size(user_memory[key])
        }
    }


@router.put("/{key}", response_model=Dict[str, Any])
async def set_memory_value_basic(
    key: str,
    request: Dict[str, Any],
    current_user = Depends(get_current_active_user),
):
    """
    Set value in memory (Community Edition).
    
    Session-scoped only with size and count limits.
    """
    user_memory = _get_user_memory(str(current_user.id))
    value = request.get("value")
    
    if value is None:
        raise HTTPException(status_code=400, detail="Value is required")
    
    # Check key limit
    if key not in user_memory and len(user_memory) >= _memory_stats["max_keys"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "key_limit_exceeded",
                "message": f"Maximum of {_memory_stats['max_keys']} keys allowed in Community edition",
                "upgrade_url": "/upgrade/business",
                "benefits": [
                    "Unlimited memory keys",
                    "Persistent storage",
                    "Distributed memory",
                    "Advanced search capabilities"
                ]
            }
        )
    
    # Check size limit
    new_size = _estimate_size(value)
    current_size = sum(_estimate_size(v) for k, v in user_memory.items() if k != key)
    
    if current_size + new_size > _memory_stats["max_size"]:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "size_limit_exceeded",
                "message": "Memory size limit of 10MB exceeded",
                "current_size": f"{current_size / 1024:.1f}KB",
                "requested_size": f"{new_size / 1024:.1f}KB",
                "upgrade_url": "/upgrade/business",
                "benefits": [
                    "Unlimited memory storage",
                    "Redis-backed persistence",
                    "Distributed memory access"
                ]
            }
        )
    
    # Set the value
    user_memory[key] = value
    
    return {
        "status": "success",
        "key": key,
        "metadata": {
            "type": "session",
            "created_at": datetime.utcnow().isoformat(),
            "size_bytes": new_size,
            "total_keys": len(user_memory),
            "total_size": current_size + new_size
        }
    }


@router.delete("/{key}", response_model=Dict[str, Any])
async def delete_memory_value_basic(
    key: str,
    current_user = Depends(get_current_active_user),
):
    """
    Delete value from memory (Community Edition).
    
    Removes from session memory only.
    """
    user_memory = _get_user_memory(str(current_user.id))
    
    if key not in user_memory:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "key_not_found",
                "message": f"Key '{key}' not found in session memory"
            }
        )
    
    del user_memory[key]
    
    return {
        "status": "success",
        "message": f"Key '{key}' deleted from session memory",
        "remaining_keys": len(user_memory)
    }


@router.get("/", response_model=Dict[str, Any])
async def list_memory_keys_basic(
    current_user = Depends(get_current_active_user),
):
    """
    List memory capabilities (Community Edition).
    
    Note: Key listing is not available in Community edition.
    """
    user_memory = _get_user_memory(str(current_user.id))
    
    return {
        "feature": "key_listing",
        "available": False,
        "edition_required": "business",
        "current_stats": {
            "total_keys": len(user_memory),
            "total_size": sum(_estimate_size(v) for v in user_memory.values())
        },
        "upgrade_benefits": {
            "message": "Upgrade to Business edition for full memory management",
            "url": "/upgrade/business",
            "features": [
                "List all memory keys",
                "Search memory entries",
                "Bulk operations",
                "Memory contexts",
                "Import/export data",
                "Persistent storage"
            ]
        }
    }