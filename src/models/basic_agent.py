"""Basic Agent model for Community Edition.

Provides simple agent storage with limits suitable for Community tier.
Business edition uses EnhancedAgent for ML-powered capabilities.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Text, Boolean, DateTime, JSON, ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import Base


class BasicAgent(Base):
    """Basic agent configuration for Community Edition.

    Limited to 5 agents per user with simple LLM execution.
    No AI frameworks or ML capabilities - those require Business Edition.
    """
    __tablename__ = "basic_agents"

    # Core fields
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    agent_type = Column(String(50), nullable=False, default="assistant")  # assistant, support, task

    # User ownership
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    user = relationship("User", back_populates="basic_agents")

    # Configuration
    model = Column(String(100), default="llama3.2:1b")  # Fast tier models only
    temperature = Column(Integer, default=7)  # 0-10 scale (0.0-1.0 * 10)
    system_prompt = Column(Text)

    # Capabilities (simple list, no ML discovery)
    capabilities = Column(JSON, default=list)  # ["chat", "summarize", "translate"]

    # Status
    is_active = Column(Boolean, default=True)
    execution_count = Column(Integer, default=0)
    last_executed = Column(DateTime, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "name": self.name,
            "description": self.description,
            "agent_type": self.agent_type,
            "model": self.model,
            "temperature": self.temperature / 10.0,  # Convert back to 0-1 scale
            "system_prompt": self.system_prompt,
            "capabilities": self.capabilities or [],
            "is_active": self.is_active,
            "execution_count": self.execution_count,
            "last_executed": self.last_executed.isoformat() if self.last_executed else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }

    def __repr__(self):
        return f"<BasicAgent(id={self.id}, name='{self.name}', type='{self.agent_type}')>"


# Agent type limits for Community Edition
COMMUNITY_AGENT_LIMIT = 5
ALLOWED_AGENT_TYPES = ["assistant", "support", "task"]
ALLOWED_MODELS = ["llama3.2:1b", "llama3.2:3b"]  # Fast tier only