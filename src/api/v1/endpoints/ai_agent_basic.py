"""AI Agent basic endpoints for Community Edition."""

from typing import Dict, Any
from datetime import datetime, date, timedelta
from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from core.cache import get_cache
from core.upgrade_hints import attach_upgrade_hints

router = APIRouter()

# Rate limiting tracking (in-memory for Community)
_rate_limits = {}

def check_rate_limit(user_id: str, endpoint: str, daily_limit: int) -> bool:
    """Check if user has exceeded rate limit for endpoint."""
    today = date.today().isoformat()
    key = f"{user_id}:{endpoint}:{today}"
    
    if key not in _rate_limits:
        _rate_limits[key] = 0
    
    if _rate_limits[key] >= daily_limit:
        return False
    
    _rate_limits[key] += 1
    return True


@router.get("/status", response_model=Dict[str, Any])
async def get_ai_agent_status_basic(
    response: Response,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """
    Get basic AI agent system status (Community Edition).

    Returns limited information about NLP and Workflow agents only.
    """
    attach_upgrade_hints(response, "agents")
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "edition": "community",
        "agents": {
            "nlp": {
                "name": "NLP Agent",
                "status": "active",
                "health": "healthy",
                "last_active": datetime.utcnow().isoformat(),
                "rate_limit": {
                    "daily_limit": 100,
                    "remaining": max(0, 100 - _rate_limits.get(
                        f"{current_user.id}:nlp:{date.today().isoformat()}", 0
                    ))
                }
            },
            "workflow": {
                "name": "Workflow Generation Agent",
                "status": "active",
                "health": "healthy",
                "last_active": datetime.utcnow().isoformat(),
                "rate_limit": {
                    "daily_limit": 10,
                    "remaining": max(0, 10 - _rate_limits.get(
                        f"{current_user.id}:workflow:{date.today().isoformat()}", 0
                    ))
                }
            }
        },
        "system": {
            "models_available": ["llama3.2:1b"],
            "default_model": "llama3.2:1b",
            "ollama_status": "connected",
            "cache_enabled": False,  # Community uses in-memory only
            "streaming_enabled": False,  # Disabled in Community
            "rate_limits": {
                "nlp_daily": 100,
                "workflow_daily": 10
            }
        },
        "upgrade_prompt": {
            "message": "Unlock advanced AI agents and unlimited requests",
            "url": "/upgrade/business",
            "benefits": [
                "Analytics and Approval AI agents",
                "Unlimited requests",
                "Advanced models (llama3.1:8b, mixtral)",
                "Streaming support",
                "Semantic caching"
            ]
        }
    }


@router.get("/capabilities", response_model=Dict[str, Any])
async def get_ai_agent_capabilities_basic(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """
    List available AI agent capabilities (Community Edition).
    
    Returns basic NLP and workflow capabilities only.
    """
    return {
        "capabilities": [
            {
                "id": "nlp_analysis",
                "name": "Basic NLP Analysis",
                "description": "Analyze text for intent and entities",
                "daily_limit": 100,
                "model": "llama3.2:1b"
            },
            {
                "id": "workflow_suggestion",
                "name": "Workflow Suggestions",
                "description": "Get workflow suggestions from natural language",
                "daily_limit": 10,
                "model": "llama3.2:1b"
            }
        ],
        "limitations": {
            "rate_limited": True,
            "basic_models_only": True,
            "no_streaming": True,
            "no_custom_config": True
        },
        "upgrade_benefits": {
            "business": [
                "Unlimited requests",
                "Advanced models",
                "Analytics agent",
                "Approval agent",
                "Streaming responses"
            ],
            "enterprise": [
                "All Business features",
                "Vision AI agent",
                "Code AI agent",
                "A2A coordination",
                "Custom model training"
            ]
        }
    }


@router.post("/nlp/analyze", response_model=Dict[str, Any])
async def analyze_text_basic(
    request: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """
    Basic NLP analysis (Community Edition).
    
    Rate limited to 100 requests per day.
    """
    # Check rate limit
    if not check_rate_limit(current_user.id, "nlp", 100):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "You've reached the daily limit of 100 NLP requests",
                "upgrade_url": "/upgrade/business",
                "benefits": [
                    "Unlimited AI agent requests",
                    "Advanced models and caching",
                    "Persistent memory storage",
                    "Full cache management"
                ]
            }
        )
    
    text = request.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Simulate basic NLP analysis
    # In production, this would call the ML microservice
    return {
        "status": "success",
        "analysis": {
            "text": text,
            "intent": "general_query" if "?" in text else "statement",
            "entities": [],  # Basic version doesn't extract entities
            "sentiment": "neutral",  # Basic version has simple sentiment
            "model_used": "llama3.2:1b",
            "processing_time": 0.1
        },
        "rate_limit": {
            "used_today": _rate_limits.get(
                f"{current_user.id}:nlp:{date.today().isoformat()}", 0
            ),
            "daily_limit": 100,
            "resets_at": (datetime.now() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
        }
    }


@router.post("/workflow/suggest", response_model=Dict[str, Any])
async def suggest_workflow_basic(
    request: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """
    Basic workflow suggestions (Community Edition).
    
    Rate limited to 10 requests per day.
    """
    # Check rate limit
    if not check_rate_limit(current_user.id, "workflow", 10):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": "You've reached the daily limit of 10 workflow suggestions",
                "upgrade_url": "/upgrade/business",
                "benefits": [
                    "Unlimited workflow generation",
                    "Advanced workflow patterns",
                    "Multi-step workflows",
                    "Custom workflow templates"
                ]
            }
        )
    
    description = request.get("description", "")
    if not description:
        raise HTTPException(status_code=400, detail="Description is required")
    
    # Simulate basic workflow suggestion
    # In production, this would call the ML microservice
    return {
        "status": "success",
        "suggestion": {
            "description": description,
            "workflow": {
                "name": f"Suggested workflow for: {description[:50]}...",
                "steps": [
                    {
                        "type": "task",
                        "name": "Step 1",
                        "description": "Initial task"
                    },
                    {
                        "type": "task", 
                        "name": "Step 2",
                        "description": "Follow-up task"
                    }
                ],
                "complexity": "basic"
            },
            "model_used": "llama3.2:1b",
            "processing_time": 0.2
        },
        "rate_limit": {
            "used_today": _rate_limits.get(
                f"{current_user.id}:workflow:{date.today().isoformat()}", 0
            ),
            "daily_limit": 10,
            "resets_at": (datetime.now() + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
        },
        "upgrade_note": "Upgrade to Business for complex multi-step workflows and unlimited generation"
    }