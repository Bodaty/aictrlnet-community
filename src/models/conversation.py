"""Conversation models for Multi-Turn NLP System.

These models support the conversational interface that guides users through
AICtrlNet's capabilities using multi-turn dialogue with context persistence.
Part of the Community Edition as this is a core feature.
"""

from sqlalchemy import Column, String, Text, JSON, Boolean, DateTime, ForeignKey, Integer, Float
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base


class ConversationSession(Base):
    """Represents a multi-turn conversation session with a user."""
    __tablename__ = "conversation_sessions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(String(36), ForeignKey('users.id'), nullable=False, index=True)

    # Human-readable conversation name (auto-generated from first message or manually set)
    name = Column(String(200), nullable=True)

    # Session state - using String instead of Enum to avoid migration issues
    state = Column(String(50), default="greeting", nullable=False)
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    ended_at = Column(DateTime, nullable=True)
    
    # Context and extracted information
    context = Column(JSON, default={}, nullable=False)
    extracted_params = Column(JSON, default={}, nullable=False)
    primary_intent = Column(String(100), nullable=True, index=True)
    intent_confidence = Column(Float, nullable=True)
    
    # Session metadata - avoid using 'metadata' as column name
    session_config = Column(JSON, default={
        "user_agent": None,
        "ip_address": None,
        "client_version": None,
        "edition": "community",
        "multi_turn_enabled": True
    })
    
    # Flags
    is_active = Column(Boolean, default=True, nullable=False)
    requires_human = Column(Boolean, default=False, nullable=False)
    
    # Relationships
    messages = relationship("ConversationMessage", back_populates="session", cascade="all, delete-orphan")
    actions = relationship("ConversationAction", back_populates="session", cascade="all, delete-orphan")
    user = relationship("User", backref="conversation_sessions")


class ConversationMessage(Base):
    """Individual messages within a conversation session."""
    __tablename__ = "conversation_messages"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('conversation_sessions.id', ondelete='CASCADE'), 
                       nullable=False, index=True)
    
    # Message content
    role = Column(String(20), nullable=False)  # user, assistant, system
    content = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Message metadata - avoid using 'metadata' as column name
    message_config = Column(JSON, default={}, nullable=False)
    suggested_actions = Column(JSON, default=[], nullable=False)  # Quick action buttons
    
    # Intent detection for this message
    detected_intent = Column(String(100), nullable=True)
    intent_confidence = Column(Float, nullable=True)
    entities = Column(JSON, default={}, nullable=False)
    
    # LLM tracking
    llm_model_used = Column(String(100), nullable=True)
    token_count = Column(Integer, nullable=True)
    processing_time_ms = Column(Integer, nullable=True)
    
    # Relationships
    session = relationship("ConversationSession", back_populates="messages")


class ConversationAction(Base):
    """Actions taken or pending within a conversation."""
    __tablename__ = "conversation_actions"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey('conversation_sessions.id', ondelete='CASCADE'), 
                       nullable=False, index=True)
    
    # Action details
    action_type = Column(String(100), nullable=False)  # create_workflow, form_pod, etc.
    action_params = Column(JSON, default={}, nullable=False)
    status = Column(String(20), default="pending", nullable=False)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    
    # Results
    result = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # References to created resources
    workflow_id = Column(String(36), nullable=True)
    agent_id = Column(UUID(as_uuid=True), nullable=True)
    pod_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Relationships
    session = relationship("ConversationSession", back_populates="actions")


class ConversationIntent(Base):
    """Predefined intents that the conversation system can recognize."""
    __tablename__ = "conversation_intents"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Intent definition
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(String(50), nullable=False)  # workflow, agent, pod, discovery, troubleshooting
    description = Column(Text, nullable=True)
    
    # Parameters required for this intent
    required_params = Column(JSON, default=[], nullable=False)
    optional_params = Column(JSON, default=[], nullable=False)
    
    # Example phrases that trigger this intent
    example_phrases = Column(JSON, default=[], nullable=False)
    
    # Clarification questions to ask
    clarification_questions = Column(JSON, default=[], nullable=False)
    
    # Routing information
    service_endpoint = Column(String(200), nullable=True)
    action_template = Column(JSON, nullable=True)
    
    # Statistics
    usage_count = Column(Integer, default=0, nullable=False)
    success_rate = Column(Float, default=0.0, nullable=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)


class ConversationPattern(Base):
    """Learned patterns from successful conversations for continuous improvement."""
    __tablename__ = "conversation_patterns"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Pattern identification
    pattern_hash = Column(String(64), unique=True, nullable=False, index=True)
    pattern_type = Column(String(50), nullable=False)  # intent_flow, clarification_sequence, etc.
    
    # Pattern content
    conversation_flow = Column(JSON, nullable=False)  # Sequence of states/intents
    success_criteria = Column(JSON, nullable=False)
    
    # Performance metrics
    occurrence_count = Column(Integer, default=1, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    average_turns = Column(Float, default=0.0, nullable=False)
    average_duration_seconds = Column(Float, default=0.0, nullable=False)
    
    # Learning metadata
    first_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, default=datetime.utcnow, nullable=False)
    confidence_score = Column(Float, default=0.0, nullable=False)
    
    # Promotion to template
    is_promoted = Column(Boolean, default=False, nullable=False)
    promoted_at = Column(DateTime, nullable=True)
