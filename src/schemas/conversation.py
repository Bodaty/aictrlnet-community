"""Pydantic schemas for Conversation models.

These schemas define the request/response models for the multi-turn
conversation system API endpoints.
"""

from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from uuid import UUID


# Enums as string literals for better compatibility
# v4: Added "executing_tools" state for tool-calling conversation flow
ConversationStateType = Literal[
    "greeting",           # Initial state
    "gathering_intent",   # Understanding user intent
    "clarifying_details", # Getting missing parameters
    "confirming_action",  # User confirmation before execution
    "executing_tools",    # v4: LLM is calling tools (intermediate state)
    "executing_action",   # Running the action plan
    "workflow_created",   # Workflow was created from conversation
    "completed",          # Successfully finished
    "abandoned",          # User gave up
    "ended"               # Session ended
]
MessageRoleType = Literal["user", "assistant", "system"]
ActionStatusType = Literal["pending", "in_progress", "completed", "failed", "cancelled"]


# Request Schemas
class ConversationSessionCreate(BaseModel):
    """Request schema for creating a new conversation session."""
    initial_message: Optional[str] = None
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    session_config: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "multi_turn_enabled": True,
        "edition": "community"
    })


class ConversationMessageCreate(BaseModel):
    """Request schema for sending a message in a conversation."""
    content: str = Field(..., min_length=1, max_length=10000)
    role: Optional[MessageRoleType] = "user"
    message_config: Optional[Dict[str, Any]] = Field(default_factory=dict)
    file_id: Optional[str] = Field(None, description="ID of an uploaded file to attach to this message")


class ConversationActionExecute(BaseModel):
    """Request schema for executing a conversation action."""
    action_type: str
    action_params: Dict[str, Any] = Field(default_factory=dict)
    confirm: bool = True


class ConversationIntentCreate(BaseModel):
    """Request schema for creating a conversation intent."""
    name: str = Field(..., min_length=1, max_length=100)
    category: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = None
    required_params: List[str] = Field(default_factory=list)
    optional_params: List[str] = Field(default_factory=list)
    example_phrases: List[str] = Field(default_factory=list)
    clarification_questions: List[Dict[str, Any]] = Field(default_factory=list)
    service_endpoint: Optional[str] = None
    action_template: Optional[Dict[str, Any]] = None


# Response Schemas
class ConversationMessageResponse(BaseModel):
    """Response schema for a conversation message."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: UUID
    session_id: UUID
    role: MessageRoleType
    content: str
    timestamp: datetime
    message_config: Dict[str, Any] = Field(default_factory=dict)
    suggested_actions: List[Dict[str, Any]] = Field(default_factory=list)
    detected_intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    entities: Dict[str, Any] = Field(default_factory=dict)
    channel_type: Optional[str] = Field("web", description="Channel this message arrived from")
    llm_model_used: Optional[str] = None
    token_count: Optional[int] = None
    processing_time_ms: Optional[int] = None


class ConversationActionResponse(BaseModel):
    """Response schema for a conversation action."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: UUID
    session_id: UUID
    action_type: str
    action_params: Dict[str, Any]
    status: ActionStatusType
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    workflow_id: Optional[str] = None
    agent_id: Optional[UUID] = None
    pod_id: Optional[UUID] = None


class ConversationSessionResponse(BaseModel):
    """Response schema for a conversation session."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )

    id: UUID
    user_id: str
    name: Optional[str] = None
    state: ConversationStateType
    started_at: datetime
    last_activity: datetime
    ended_at: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    extracted_params: Dict[str, Any] = Field(default_factory=dict)
    primary_intent: Optional[str] = None
    intent_confidence: Optional[float] = None
    session_config: Dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    requires_human: bool = False
    messages: List[ConversationMessageResponse] = Field(default_factory=list)
    actions: List[ConversationActionResponse] = Field(default_factory=list)


class ConversationIntentResponse(BaseModel):
    """Response schema for a conversation intent."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: UUID
    name: str
    category: str
    description: Optional[str] = None
    required_params: List[str]
    optional_params: List[str]
    example_phrases: List[str]
    clarification_questions: List[Dict[str, Any]]
    service_endpoint: Optional[str] = None
    action_template: Optional[Dict[str, Any]] = None
    usage_count: int = 0
    success_rate: float = 0.0
    created_at: datetime
    updated_at: datetime
    is_active: bool = True


class ConversationPatternResponse(BaseModel):
    """Response schema for a conversation pattern."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: UUID
    pattern_hash: str
    pattern_type: str
    conversation_flow: Dict[str, Any]
    success_criteria: Dict[str, Any]
    occurrence_count: int
    success_count: int
    average_turns: float
    average_duration_seconds: float
    first_seen: datetime
    last_seen: datetime
    confidence_score: float
    is_promoted: bool
    promoted_at: Optional[datetime] = None


# Conversation API Response Models
class ConversationResponse(BaseModel):
    """Main response for conversation API calls."""
    session_id: UUID
    message: ConversationMessageResponse
    state: ConversationStateType
    context: Dict[str, Any] = Field(default_factory=dict)
    quick_actions: List[Dict[str, Any]] = Field(default_factory=list)
    requires_clarification: bool = False
    clarification_options: List[Dict[str, Any]] = Field(default_factory=list)
    # Company automation result for rich UX display
    automation_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Company automation result with plan details for UX rendering"
    )


class ConversationListResponse(BaseModel):
    """Response for listing conversations."""
    conversations: List[ConversationSessionResponse]
    total: int
    page: int = 1
    per_page: int = 20


class IntentDetectionResponse(BaseModel):
    """Response for intent detection."""
    intent: str
    confidence: float
    entities: Dict[str, Any] = Field(default_factory=dict)
    requires_params: List[str] = Field(default_factory=list)
    has_params: List[str] = Field(default_factory=list)
    missing_params: List[str] = Field(default_factory=list)