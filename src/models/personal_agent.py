"""Personal Agent Hub models for Community Edition.

Provides per-user personal AI agent configuration and memory storage.
Business edition extends with advanced memory and unlimited workflows.
Enterprise edition adds org-wide agent policies and federation.
"""

import uuid
from datetime import datetime
from sqlalchemy import (
    Column, String, Text, Boolean, DateTime, JSON, ForeignKey,
    Integer, Float, Index
)
from sqlalchemy.orm import relationship

from .base import Base


class PersonalAgentConfig(Base):
    """Personal agent configuration for a user.

    Each user gets one personal agent that can be customized with
    personality, preferences, and up to 5 personal workflows (Community).
    """
    __tablename__ = "personal_agent_configs"
    __table_args__ = (
        Index("ix_personal_agent_configs_user_id", "user_id"),
        Index("ix_personal_agent_configs_status", "status"),
        {"extend_existing": True},
    )

    # Core fields
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    # Agent identity
    agent_name = Column(String(255), nullable=False, default="My Assistant")

    # Personality configuration
    personality = Column(JSON, default=lambda: {
        "tone": "friendly",
        "style": "concise",
        "expertise_areas": [],
    })

    # User preferences
    preferences = Column(JSON, default=lambda: {
        "notifications": {"enabled": True, "frequency": "daily"},
        "auto_actions": {"enabled": False, "require_confirmation": True},
    })

    # Personal workflows (list of workflow IDs)
    active_workflows = Column(JSON, default=list)

    # Community limit
    max_workflows = Column(Integer, nullable=False, default=5)

    # Status: active, paused, disabled
    status = Column(String(20), nullable=False, default="active")

    # Onboarding interview state tracking
    onboarding_state = Column(JSON, default=lambda: {
        "status": "not_started",
        "current_chapter": 1,
        "completed_chapters": [],
    })

    # User context gathered during onboarding (role, intent, comfort level)
    user_context = Column(JSON, default=dict)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    memories = relationship(
        "PersonalAgentMemory",
        back_populates="config",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "agent_name": self.agent_name,
            "personality": self.personality or {},
            "preferences": self.preferences or {},
            "active_workflows": self.active_workflows or [],
            "max_workflows": self.max_workflows,
            "status": self.status,
            "onboarding_state": self.onboarding_state or {},
            "user_context": self.user_context or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

    def __repr__(self):
        return (
            f"<PersonalAgentConfig(id={self.id}, user_id='{self.user_id}', "
            f"agent_name='{self.agent_name}', status='{self.status}')>"
        )


class PersonalAgentMemory(Base):
    """Memory entries for a personal agent.

    Stores interaction history, learned preferences, contextual data,
    and derived learnings so the agent can improve over time.
    """
    __tablename__ = "personal_agent_memories"
    __table_args__ = (
        Index("ix_personal_agent_memories_config_id", "config_id"),
        Index("ix_personal_agent_memories_memory_type", "memory_type"),
        Index("ix_personal_agent_memories_importance", "importance_score"),
        Index("ix_personal_agent_memories_created_at", "created_at"),
        {"extend_existing": True},
    )

    # Core fields
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    config_id = Column(
        String(36),
        ForeignKey("personal_agent_configs.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Memory classification: interaction, preference, context, learning
    memory_type = Column(String(50), nullable=False, default="interaction")

    # Structured content (question, answer, context, tags, etc.)
    content = Column(JSON, nullable=False, default=dict)

    # Importance for memory retrieval (0.0 = trivial, 1.0 = critical)
    importance_score = Column(Float, nullable=False, default=0.5)

    # Optional expiry for ephemeral context
    expires_at = Column(DateTime, nullable=True)

    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    config = relationship("PersonalAgentConfig", back_populates="memories")

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "config_id": self.config_id,
            "memory_type": self.memory_type,
            "content": self.content or {},
            "importance_score": self.importance_score,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    def __repr__(self):
        return (
            f"<PersonalAgentMemory(id={self.id}, type='{self.memory_type}', "
            f"importance={self.importance_score})>"
        )


# Community Edition limits
COMMUNITY_MAX_WORKFLOWS = 5
ALLOWED_MEMORY_TYPES = ["interaction", "preference", "context", "learning"]
ALLOWED_STATUSES = ["active", "paused", "disabled"]
