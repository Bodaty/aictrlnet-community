"""Pydantic schemas for Personal Agent Hub (Community Edition)."""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, validator


# ---------------------------------------------------------------------------
# Personality & Preferences sub-schemas
# ---------------------------------------------------------------------------

class PersonalityConfig(BaseModel):
    """Agent personality settings."""
    tone: str = Field("friendly", description="Communication tone: friendly, formal, casual, technical")
    style: str = Field("concise", description="Response style: concise, detailed, conversational")
    expertise_areas: List[str] = Field(default_factory=list, description="Areas of expertise")

    @validator("tone")
    def validate_tone(cls, v):
        allowed = ["friendly", "formal", "casual", "technical"]
        if v not in allowed:
            raise ValueError(f"Tone must be one of: {', '.join(allowed)}")
        return v

    @validator("style")
    def validate_style(cls, v):
        allowed = ["concise", "detailed", "conversational"]
        if v not in allowed:
            raise ValueError(f"Style must be one of: {', '.join(allowed)}")
        return v


class NotificationSettings(BaseModel):
    """Notification preferences."""
    enabled: bool = True
    frequency: str = Field("daily", description="Notification frequency: realtime, daily, weekly")


class AutoActionSettings(BaseModel):
    """Auto-action preferences."""
    enabled: bool = False
    require_confirmation: bool = True


class PreferencesConfig(BaseModel):
    """User preference settings for the personal agent."""
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)
    auto_actions: AutoActionSettings = Field(default_factory=AutoActionSettings)


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------

class PersonalAgentConfigBase(BaseModel):
    """Base schema for personal agent configuration."""
    agent_name: str = Field("My Assistant", min_length=1, max_length=255, description="Display name for your agent")
    personality: PersonalityConfig = Field(default_factory=PersonalityConfig, description="Agent personality")
    preferences: PreferencesConfig = Field(default_factory=PreferencesConfig, description="User preferences")


class PersonalAgentConfigUpdate(BaseModel):
    """Schema for updating personal agent configuration."""
    agent_name: Optional[str] = Field(None, min_length=1, max_length=255)
    personality: Optional[PersonalityConfig] = None
    preferences: Optional[PreferencesConfig] = None
    status: Optional[str] = None

    @validator("status")
    def validate_status(cls, v):
        if v is not None:
            allowed = ["active", "paused", "disabled"]
            if v not in allowed:
                raise ValueError(f"Status must be one of: {', '.join(allowed)}")
        return v


class PersonalAgentConfigResponse(BaseModel):
    """Response schema for personal agent configuration."""
    id: str
    user_id: str
    agent_name: str
    personality: Dict[str, Any] = {}
    preferences: Dict[str, Any] = {}
    active_workflows: List[str] = []
    max_workflows: int = 5
    status: str = "active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True


# ---------------------------------------------------------------------------
# Ask schemas
# ---------------------------------------------------------------------------

class PersonalAgentAskRequest(BaseModel):
    """Request schema for asking the personal agent a question."""
    message: str = Field(..., min_length=1, max_length=5000, description="Question or instruction")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the question")


class PersonalAgentAskResponse(BaseModel):
    """Response schema for a personal agent answer."""
    message: str
    response: str
    model_used: str = "llama3.2:1b"
    memory_id: Optional[str] = None
    processing_time: float = 0.0
    timestamp: datetime


# ---------------------------------------------------------------------------
# Activity feed schemas
# ---------------------------------------------------------------------------

class ActivityFeedItem(BaseModel):
    """Single item in the activity feed."""
    id: str
    memory_type: str
    content: Dict[str, Any] = {}
    importance_score: float = 0.5
    created_at: Optional[datetime] = None


class ActivityFeedResponse(BaseModel):
    """Response schema for the activity feed."""
    items: List[ActivityFeedItem] = []
    total: int = 0
    limit: int = 20


# ---------------------------------------------------------------------------
# Workflow management schemas
# ---------------------------------------------------------------------------

class WorkflowAddResponse(BaseModel):
    """Response after adding a personal workflow."""
    workflow_id: str
    active_workflows: List[str]
    current_count: int
    max_allowed: int
    message: str


class WorkflowRemoveResponse(BaseModel):
    """Response after removing a personal workflow."""
    workflow_id: str
    active_workflows: List[str]
    current_count: int
    message: str


class WorkflowLimitResponse(BaseModel):
    """Response when workflow limit is reached."""
    current_count: int
    max_allowed: int
    can_add: bool
    upgrade_message: Optional[str] = None
