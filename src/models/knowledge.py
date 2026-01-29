"""Knowledge models for Intelligent Assistant System.

These models support the knowledge service that provides system awareness
and intelligent retrieval capabilities for the AICtrlNet Intelligent Assistant.
Part of the Community Edition as this is a foundation feature.
"""

from sqlalchemy import Column, String, Text, JSON, Boolean, DateTime, ForeignKey, Integer, Float, Index
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base


class KnowledgeItem(Base):
    """Represents a piece of knowledge about the AICtrlNet system."""
    __tablename__ = "knowledge_items"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Item categorization
    item_type = Column(String(50), nullable=False, index=True)  # template, agent, adapter, feature, endpoint
    category = Column(String(100), nullable=False, index=True)  # workflow, integration, ai, etc.
    name = Column(String(200), nullable=False, index=True)
    
    # Content
    description = Column(Text, nullable=False)
    content = Column(JSON, nullable=False)  # Full structured content
    
    # Metadata for retrieval
    tags = Column(JSON, default=[], nullable=False)
    keywords = Column(JSON, default=[], nullable=False)
    semantic_embedding = Column(JSON, nullable=True)  # For future vector search
    
    # Usage and performance
    usage_count = Column(Integer, default=0, nullable=False)
    relevance_score = Column(Float, default=1.0, nullable=False)
    success_rate = Column(Float, nullable=True)
    
    # Source information
    source_file = Column(String(500), nullable=True)
    source_version = Column(String(50), nullable=True)
    edition_required = Column(String(20), default="community", nullable=False)
    
    # Relationships and references
    related_items = Column(JSON, default=[], nullable=False)  # List of item IDs
    dependencies = Column(JSON, default=[], nullable=False)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_accessed = Column(DateTime, nullable=True)
    
    # Flags
    is_active = Column(Boolean, default=True, nullable=False)
    is_deprecated = Column(Boolean, default=False, nullable=False)
    
    # Indexes for efficient retrieval
    __table_args__ = (
        Index('idx_knowledge_type_category', 'item_type', 'category'),
        Index('idx_knowledge_name_type', 'name', 'item_type'),
        Index('idx_knowledge_usage', 'usage_count', 'relevance_score'),
    )


class KnowledgeIndex(Base):
    """Maintains an index of all system knowledge for fast retrieval."""
    __tablename__ = "knowledge_index"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Index metadata
    index_version = Column(String(20), nullable=False)
    index_type = Column(String(50), nullable=False)  # full, partial, category
    
    # Index content (optimized structure for fast search)
    index_data = Column(JSON, nullable=False)
    
    # Statistics
    total_items = Column(Integer, default=0, nullable=False)
    item_counts = Column(JSON, default={}, nullable=False)  # Count by type
    
    # Build information
    built_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    build_duration_ms = Column(Integer, nullable=True)
    last_refresh = Column(DateTime, nullable=True)
    
    # Configuration
    index_config = Column(JSON, default={}, nullable=False)
    
    # Status
    is_current = Column(Boolean, default=True, nullable=False)
    needs_rebuild = Column(Boolean, default=False, nullable=False)


class KnowledgeQuery(Base):
    """Tracks queries made to the knowledge system for learning and optimization."""
    __tablename__ = "knowledge_queries"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Query details
    query_text = Column(Text, nullable=False)
    query_type = Column(String(50), nullable=False)  # search, retrieve, suggest
    context = Column(JSON, default={}, nullable=False)
    
    # User information
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    session_id = Column(UUID(as_uuid=True), ForeignKey('conversation_sessions.id'), nullable=True)
    
    # Results
    results_returned = Column(JSON, default=[], nullable=False)  # Item IDs returned
    result_count = Column(Integer, default=0, nullable=False)
    top_result_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Performance
    query_time_ms = Column(Integer, nullable=True)
    retrieval_method = Column(String(50), nullable=True)  # keyword, semantic, hybrid
    
    # Feedback
    was_helpful = Column(Boolean, nullable=True)
    user_selected_item = Column(UUID(as_uuid=True), nullable=True)
    feedback_notes = Column(Text, nullable=True)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", backref="knowledge_queries")
    session = relationship("ConversationSession", backref="knowledge_queries")


class SystemManifest(Base):
    """Stores the generated system manifest for quick access."""
    __tablename__ = "system_manifests"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Manifest version and type
    manifest_version = Column(String(20), nullable=False)
    manifest_type = Column(String(50), default="full", nullable=False)
    
    # Manifest content
    manifest_data = Column(JSON, nullable=False)
    
    # Statistics from manifest
    statistics = Column(JSON, default={}, nullable=False)
    feature_count = Column(Integer, default=0, nullable=False)
    endpoint_count = Column(Integer, default=0, nullable=False)
    template_count = Column(Integer, default=0, nullable=False)
    agent_count = Column(Integer, default=0, nullable=False)
    adapter_count = Column(Integer, default=0, nullable=False)
    
    # Generation metadata
    generated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    generation_time_ms = Column(Integer, nullable=True)
    
    # Status
    is_current = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    
    # Index for finding current manifest
    __table_args__ = (
        Index('idx_manifest_current', 'is_current', 'manifest_type'),
    )


class LearnedPattern(Base):
    """Patterns learned from user interactions for improving the assistant.

    Multi-Tier Learning System:
    - user: Pattern specific to individual user (highest priority for that user)
    - organization: Pattern shared across organization (Enterprise only)
    - global: Pattern validated for all users (proven best practices)
    """
    __tablename__ = "learned_patterns"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # Multi-tier scoping (NEW)
    scope = Column(String(20), default="user", nullable=False)  # user, organization, global
    user_id = Column(String(36), ForeignKey('users.id'), nullable=True, index=True)
    organization_id = Column(UUID(as_uuid=True), nullable=True, index=True)  # For Enterprise edition

    # Escalation tracking (NEW)
    promoted_from_scope = Column(String(20), nullable=True)  # Track escalation source
    contributing_users_count = Column(Integer, default=1, nullable=False)  # How many users contributed
    promotion_candidate = Column(Boolean, default=False, nullable=False)  # Ready for promotion

    # Privacy controls (NEW)
    is_shareable = Column(Boolean, default=True, nullable=False)  # User consent to share
    contains_sensitive_data = Column(Boolean, default=False, nullable=False)  # Auto-detected or flagged
    anonymized = Column(Boolean, default=False, nullable=False)  # Has PII been removed

    # Pattern identification
    pattern_type = Column(String(50), nullable=False)  # query_sequence, action_flow, preference
    pattern_signature = Column(String(200), nullable=False, index=True)

    # Pattern content
    pattern_data = Column(JSON, nullable=False)
    context_requirements = Column(JSON, default={}, nullable=False)

    # Learning metrics
    occurrence_count = Column(Integer, default=1, nullable=False)
    success_count = Column(Integer, default=0, nullable=False)
    confidence_score = Column(Float, default=0.0, nullable=False)

    # Application
    is_active = Column(Boolean, default=False, nullable=False)  # Whether to use this pattern
    activation_threshold = Column(Float, default=0.7, nullable=False)
    last_applied = Column(DateTime, nullable=True)
    application_count = Column(Integer, default=0, nullable=False)

    # Timestamps
    first_observed = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_observed = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Review and validation
    is_validated = Column(Boolean, default=False, nullable=False)
    validated_by = Column(String(36), nullable=True)
    validated_at = Column(DateTime, nullable=True)

    # Index for pattern matching
    __table_args__ = (
        Index('idx_pattern_active', 'is_active', 'confidence_score'),
        Index('idx_pattern_signature', 'pattern_signature', 'pattern_type'),
        Index('idx_pattern_scope', 'scope', 'user_id', 'organization_id'),  # NEW: Multi-tier index
        Index('idx_pattern_promotion', 'promotion_candidate', 'scope'),  # NEW: For promotion queries
    )
