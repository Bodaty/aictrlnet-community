"""Pydantic schemas for Basic Agent (Community Edition)."""

from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, Field, validator
from uuid import UUID


class BasicAgentBase(BaseModel):
    """Base schema for basic agents."""
    name: str = Field(..., min_length=1, max_length=255, description="Agent name")
    description: Optional[str] = Field(None, description="Agent description")
    agent_type: str = Field("assistant", description="Agent type: assistant, support, or task")
    model: str = Field("llama3.2:1b", description="Model to use (fast tier only)")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Temperature for responses")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")

    @validator('agent_type')
    def validate_agent_type(cls, v):
        allowed = ["assistant", "support", "task"]
        if v not in allowed:
            raise ValueError(f"Agent type must be one of: {', '.join(allowed)}")
        return v

    @validator('model')
    def validate_model(cls, v):
        allowed = ["llama3.2:1b", "llama3.2:3b"]
        if v not in allowed:
            raise ValueError(f"Model must be one of: {', '.join(allowed)} (fast tier only)")
        return v


class BasicAgentCreate(BasicAgentBase):
    """Schema for creating a basic agent."""
    pass


class BasicAgentUpdate(BaseModel):
    """Schema for updating a basic agent."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    agent_type: Optional[str] = None
    model: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0.0, le=1.0)
    system_prompt: Optional[str] = None
    capabilities: Optional[List[str]] = None
    is_active: Optional[bool] = None

    @validator('agent_type')
    def validate_agent_type(cls, v):
        if v is not None:
            allowed = ["assistant", "support", "task"]
            if v not in allowed:
                raise ValueError(f"Agent type must be one of: {', '.join(allowed)}")
        return v

    @validator('model')
    def validate_model(cls, v):
        if v is not None:
            allowed = ["llama3.2:1b", "llama3.2:3b"]
            if v not in allowed:
                raise ValueError(f"Model must be one of: {', '.join(allowed)} (fast tier only)")
        return v


class BasicAgentResponse(BasicAgentBase):
    """Response schema for basic agent."""
    id: UUID
    user_id: str
    is_active: bool = True
    execution_count: int = 0
    last_executed: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BasicAgentExecuteRequest(BaseModel):
    """Request schema for executing a basic agent."""
    prompt: str = Field(..., min_length=1, description="Prompt to execute")
    context: Optional[dict] = Field(None, description="Additional context for execution")


class BasicAgentExecuteResponse(BaseModel):
    """Response schema for agent execution."""
    agent_id: UUID
    execution_id: str
    response: str
    model_used: str
    execution_time: float
    tokens_used: Optional[int] = None
    timestamp: datetime


class BasicAgentLimitResponse(BaseModel):
    """Response for agent limit check."""
    current_count: int
    max_allowed: int
    can_create: bool
    upgrade_message: Optional[str] = None