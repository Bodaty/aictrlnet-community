"""Agent execution endpoints for Community Edition.

Provides basic agent execution with limitations:
- 3 agents maximum
- Single API provider
- Fast tier models only
- 100 daily executions
"""

from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from services.agent_config_service import AgentConfigService
from services.agent_execution_basic import BasicAgentExecutor

router = APIRouter()

# Initialize services
config_service = AgentConfigService()
executor = BasicAgentExecutor()


# Request/Response schemas
class AgentExecutionRequest(BaseModel):
    """Request for agent execution."""
    agent_name: str = Field(..., description="Name of the agent to execute")
    task: Dict[str, Any] = Field(..., description="Task definition")
    

class APIConfigRequest(BaseModel):
    """Request to configure API."""
    provider: str = Field(..., description="API provider (ollama or openai)")
    config: Dict[str, Any] = Field(..., description="Provider configuration")


class EnableAgentRequest(BaseModel):
    """Request to enable an agent."""
    agent_name: str = Field(..., description="Name of the agent to enable")


@router.post("/execute", response_model=Dict[str, Any])
async def execute_agent(
    request: AgentExecutionRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Execute a basic agent task (Community Edition).
    
    Limitations:
    - Only fast tier models
    - No memory/context
    - 100 daily executions
    - 3 agents maximum
    """
    result = await executor.execute_agent(
        db=db,
        user_id=current_user.id,
        agent_name=request.agent_name,
        task=request.task
    )
    
    if not result["success"]:
        # Return 402 Payment Required for upgrade prompts
        if "upgrade_prompt" in result:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=result
            )
        # Return 400 for other errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result
        )
    
    return result


@router.get("/status", response_model=Dict[str, Any])
async def get_agent_status(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get status of available agents for current user."""
    return await executor.get_agent_status(db, current_user.id)


@router.get("/config", response_model=Dict[str, Any])
async def get_agent_config(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get current agent configuration."""
    config = await config_service.get_user_agent_config(db, current_user.id)
    
    # Don't expose sensitive data
    if config.get("api_config", {}).get("api_key"):
        config["api_config"]["api_key"] = "***configured***"
    
    return {
        "config": config,
        "limits": {
            "max_agents": config_service.MAX_AGENTS,
            "allowed_providers": config_service.ALLOWED_PROVIDERS,
            "allowed_agents": config_service.ALLOWED_AGENTS,
            "model_tier": "fast",
            "daily_executions": 100
        },
        "edition": "community"
    }


@router.post("/config/api", response_model=Dict[str, Any])
async def configure_api(
    request: APIConfigRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Configure API for agent execution."""
    try:
        result = await config_service.update_api_key(
            db=db,
            user_id=current_user.id,
            provider=request.provider,
            config=request.config
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/config/enable-agent", response_model=Dict[str, Any])
async def enable_agent(
    request: EnableAgentRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Enable an agent for the user (up to limit)."""
    result = await config_service.enable_agent(
        db=db,
        user_id=current_user.id,
        agent_name=request.agent_name
    )
    
    if not result["success"]:
        # Return 402 if it's an upgrade issue
        if "upgrade_url" in result:
            raise HTTPException(
                status_code=status.HTTP_402_PAYMENT_REQUIRED,
                detail=result
            )
        # Return 400 for other errors
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=result
        )
    
    return result


@router.get("/available-agents", response_model=Dict[str, Any])
async def get_available_agents(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get list of available agents in Community Edition."""
    config = await config_service.get_user_agent_config(db, current_user.id)
    enabled = config.get("enabled_agents", ["basic_nlp"])
    
    agents = []
    for agent_name in config_service.ALLOWED_AGENTS:
        agents.append({
            "name": agent_name,
            "enabled": agent_name in enabled,
            "description": _get_agent_description(agent_name),
            "capabilities": _get_agent_capabilities(agent_name)
        })
    
    return {
        "agents": agents,
        "enabled_count": len(enabled),
        "max_agents": config_service.MAX_AGENTS,
        "edition": "community",
        "business_agents_preview": [
            "Sales Optimization Agent",
            "Customer Success Agent", 
            "Marketing Campaign Agent",
            "Data Analytics Agent",
            "HR Automation Agent",
            "Financial Planning Agent",
            "... and 27 more agents"
        ]
    }


def _get_agent_description(agent_name: str) -> str:
    """Get description for agent."""
    descriptions = {
        "basic_nlp": "Natural language processing and text analysis",
        "basic_workflow": "Simple workflow generation from descriptions",
        "basic_assistant": "General AI assistance for various tasks"
    }
    return descriptions.get(agent_name, "AI agent")


def _get_agent_capabilities(agent_name: str) -> list:
    """Get capabilities for agent."""
    capabilities = {
        "basic_nlp": [
            "Text analysis",
            "Sentiment detection",
            "Entity extraction",
            "Summarization"
        ],
        "basic_workflow": [
            "Simple workflow design",
            "Task breakdown",
            "Process mapping"
        ],
        "basic_assistant": [
            "Question answering",
            "Task assistance",
            "Information lookup"
        ]
    }
    return capabilities.get(agent_name, [])