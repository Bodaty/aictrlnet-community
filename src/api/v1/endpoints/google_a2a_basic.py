"""Google A2A Protocol basic endpoints for Community Edition.

Limited to discovery and read-only operations with rate limits.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, date
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from core.database import get_db
from core.security import get_current_active_user
from models.user import User

router = APIRouter()

# Rate limiting for Community edition (in-memory)
_rate_limits = {}

def check_rate_limit(user_id: str, endpoint: str, daily_limit: int) -> bool:
    """Check if user has exceeded rate limit for endpoint."""
    today = date.today().isoformat()
    key = f"{user_id}:a2a:{endpoint}:{today}"
    
    if key not in _rate_limits:
        _rate_limits[key] = 0
    
    if _rate_limits[key] >= daily_limit:
        return False
    
    _rate_limits[key] += 1
    return True


@router.get("/.well-known/agent.json")
async def get_agent_card_basic():
    """Get basic agent card for Community Edition (public endpoint)."""
    return {
        "protocolVersion": "0.2.5",
        "implementation": "aictrlnet-fastapi-community",
        "name": "AICtrlNet Community Edition",
        "description": "Open source AI Control Network with basic A2A support",
        "version": "1.0.0",
        "capabilities": {
            "supportsSync": True,
            "supportsStreaming": False,
            "supportsAsync": False,
            "supportedMethods": ["agents/list", "agents/get"],  # Read-only
            "features": [
                "basic_workflow_execution",
                "agent_discovery"
            ]
        },
        "endpoints": {
            "discovery": "/.well-known/agent.json"
        },
        "authentication": {
            "bearer": True,
            "oauth2": False,
            "apiKey": False
        },
        "limitations": {
            "daily_queries": 100,
            "write_operations": False,
            "sessions": False,
            "federation": False
        },
        "upgrade": {
            "message": "Upgrade to Business edition for full A2A protocol support",
            "url": "/upgrade/business",
            "benefits": [
                "Unlimited A2A queries",
                "Write operations (create/update/delete)",
                "Stateful sessions",
                "JSON-RPC endpoint",
                "OAuth2 authentication"
            ]
        }
    }


@router.get("/agents", response_model=Dict[str, Any])
async def list_agents_basic(
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(10, ge=1, le=50),  # Limited to 50
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List A2A agents (Community Edition - limited and rate-limited)."""
    # Check rate limit
    if not check_rate_limit(str(current_user.id), "list", 100):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "You've reached the daily limit of 100 A2A queries",
                "upgrade_url": "/upgrade/business",
                "benefits": [
                    "Unlimited A2A queries",
                    "Full CRUD operations",
                    "Stateful sessions",
                    "Advanced filtering"
                ]
            }
        )
    
    # For Community edition, return mock data or limited real data
    # In production, this would query real A2A agents with limitations
    return {
        "agents": [
            {
                "id": "demo-agent-1",
                "name": "Demo Workflow Agent",
                "url": "https://demo.example.com",
                "status": "active",
                "capabilities": ["workflow_execution", "task_processing"],
                "description": "Example A2A agent for demonstration"
            }
        ],
        "total": 1,
        "limit": limit,
        "offset": offset,
        "edition_limits": {
            "max_results": 50,
            "daily_queries": 100,
            "queries_remaining": 100 - _rate_limits.get(
                f"{current_user.id}:a2a:list:{date.today().isoformat()}", 0
            )
        },
        "upgrade_prompt": {
            "message": "Access full agent registry with Business edition",
            "benefits": [
                "Unlimited agent listings",
                "Register your own agents",
                "Advanced capability filtering",
                "Real-time health monitoring"
            ]
        }
    }


@router.get("/agents/{agent_id}", response_model=Dict[str, Any])
async def get_agent_basic(
    agent_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get agent details (Community Edition - rate-limited)."""
    # Check rate limit
    if not check_rate_limit(str(current_user.id), "get", 100):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "You've reached the daily limit of 100 A2A queries",
                "upgrade_url": "/upgrade/business"
            }
        )
    
    # For Community edition, return limited information
    if agent_id == "demo-agent-1":
        return {
            "id": "demo-agent-1",
            "name": "Demo Workflow Agent",
            "url": "https://demo.example.com",
            "status": "active",
            "capabilities": ["workflow_execution", "task_processing"],
            "description": "Example A2A agent for demonstration",
            "created_at": datetime.utcnow().isoformat(),
            "edition_note": "Community edition provides read-only access to agent information"
        }
    else:
        raise HTTPException(status_code=404, detail="Agent not found")


@router.get("/capabilities", response_model=Dict[str, Any])
async def list_capabilities_basic(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """List available A2A capabilities (Community Edition)."""
    return {
        "edition": "community",
        "capabilities": {
            "discovery": {
                "available": True,
                "description": "Discover A2A agents and their capabilities",
                "limitations": ["Read-only", "100 queries/day"]
            },
            "query": {
                "available": True,
                "description": "Query agent information",
                "limitations": ["Read-only", "Basic filtering only"]
            },
            "create": {
                "available": False,
                "description": "Register new agents",
                "required_edition": "business"
            },
            "sessions": {
                "available": False,
                "description": "Stateful agent sessions",
                "required_edition": "business"
            },
            "tasks": {
                "available": False,
                "description": "Submit tasks to agents",
                "required_edition": "business"
            },
            "federation": {
                "available": False,
                "description": "Multi-agent orchestration",
                "required_edition": "enterprise"
            }
        },
        "upgrade_benefits": {
            "business": [
                "Full CRUD operations on agents",
                "Stateful sessions",
                "Task submission and tracking",
                "JSON-RPC endpoint",
                "OAuth2 authentication"
            ],
            "enterprise": [
                "All Business features",
                "Agent federation",
                "AI-powered agent matching",
                "Custom authentication",
                "Unlimited agents"
            ]
        }
    }


@router.get("/health", response_model=Dict[str, Any])
async def a2a_health_basic(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Check A2A service health (Community Edition)."""
    queries_today = _rate_limits.get(
        f"{current_user.id}:a2a:list:{date.today().isoformat()}", 0
    ) + _rate_limits.get(
        f"{current_user.id}:a2a:get:{date.today().isoformat()}", 0
    )
    
    return {
        "status": "healthy",
        "service": "google_a2a_basic",
        "edition": "community",
        "features": {
            "discovery": True,
            "query": True,
            "create": False,
            "update": False,
            "delete": False,
            "sessions": False,
            "tasks": False
        },
        "rate_limits": {
            "daily_limit": 100,
            "queries_today": queries_today,
            "queries_remaining": max(0, 100 - queries_today)
        },
        "upgrade_url": "/upgrade/business"
    }


# Endpoints that require Business edition or higher

@router.post("/agents")
async def create_agent_not_available(
    current_user: User = Depends(get_current_active_user)
):
    """Create agent - not available in Community Edition."""
    raise HTTPException(
        status_code=403,
        detail={
            "error": "feature_not_available",
            "message": "Agent registration requires Business edition",
            "upgrade_url": "/upgrade/business",
            "benefits": [
                "Register unlimited A2A agents",
                "Full agent management",
                "Health monitoring",
                "Capability mapping"
            ]
        }
    )


@router.post("/sessions")
async def create_session_not_available(
    current_user: User = Depends(get_current_active_user)
):
    """Create session - not available in Community Edition."""
    raise HTTPException(
        status_code=403,
        detail={
            "error": "feature_not_available",
            "message": "Stateful sessions require Business edition",
            "upgrade_url": "/upgrade/business",
            "benefits": [
                "Stateful agent conversations",
                "Context preservation",
                "Session management",
                "Multi-turn interactions"
            ]
        }
    )


@router.post("/tasks")
async def create_task_not_available(
    current_user: User = Depends(get_current_active_user)
):
    """Create task - not available in Community Edition."""
    raise HTTPException(
        status_code=403,
        detail={
            "error": "feature_not_available",
            "message": "Task submission requires Business edition",
            "upgrade_url": "/upgrade/business",
            "benefits": [
                "Submit tasks to A2A agents",
                "Track task progress",
                "Handle task results",
                "Error handling and retries"
            ]
        }
    )


@router.post("/rpc")
async def jsonrpc_not_available(
    current_user: User = Depends(get_current_active_user)
):
    """JSON-RPC endpoint - not available in Community Edition."""
    raise HTTPException(
        status_code=403,
        detail={
            "error": "feature_not_available",
            "message": "JSON-RPC endpoint requires Business edition",
            "upgrade_url": "/upgrade/business",
            "benefits": [
                "Full JSON-RPC 2.0 support",
                "All A2A protocol methods",
                "Batch requests",
                "Async notifications"
            ]
        }
    )