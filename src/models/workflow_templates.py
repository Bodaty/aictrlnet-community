"""Workflow template models for Community Edition."""

from sqlalchemy import Column, String, Text, JSON, Boolean, DateTime, ForeignKey, Index, Integer, DECIMAL, Enum
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

# CRITICAL: Use the correct Base class with relative import like all other models!
from .base import Base


class WorkflowTemplate(Base):
    """Workflow template definition.
    
    Owned by Community Edition - stores workflow template metadata and 
    references to template definition files.
    """
    __tablename__ = "workflow_templates"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # Basic information
    name = Column(String(255), nullable=False)
    description = Column(Text)
    category = Column(String(50))
    tags = Column(ARRAY(Text), default=[])  # Array of tags
    edition = Column(String(20), default='community')
    
    # Ownership and permissions
    owner_id = Column(String(36), ForeignKey('users.id', ondelete='SET NULL'))
    is_public = Column(Boolean, default=False)
    is_system = Column(Boolean, default=False)
    
    # Template relationships
    parent_template_id = Column(UUID(as_uuid=True), ForeignKey('workflow_templates.id', ondelete='SET NULL'))
    version = Column(Integer, default=1)
    
    # File storage
    definition_path = Column(String(500), nullable=False)
    thumbnail_path = Column(String(500))
    
    # Template metadata
    complexity = Column(Enum('simple', 'moderate', 'complex', 'advanced', name='workflow_complexity'), default='moderate', nullable=False)
    estimated_duration = Column(String(50))
    required_adapters = Column(ARRAY(Text), default=[])
    required_capabilities = Column(ARRAY(Text), default=[])
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    rating = Column(DECIMAL(3, 2))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    published_at = Column(DateTime)
    
    # Status
    is_active = Column(Boolean, default=True)

    # Soft delete
    is_deleted = Column(Boolean, default=False)
    deleted_at = Column(DateTime)
    
    # Relationships
    owner = relationship("User", foreign_keys=[owner_id])
    parent_template = relationship("WorkflowTemplate", remote_side=[id])
    permissions = relationship("WorkflowTemplatePermission", back_populates="template", cascade="all, delete-orphan")
    usage_history = relationship("WorkflowTemplateUsage", back_populates="template")
    reviews = relationship("WorkflowTemplateReview", back_populates="template", cascade="all, delete-orphan")
    
    # Indexes
    __table_args__ = (
        Index('idx_workflow_templates_category', 'category'),
        Index('idx_workflow_templates_owner', 'owner_id'),
        Index('idx_workflow_templates_public', 'is_public'),
        Index('idx_workflow_templates_system', 'is_system'),
        # Unique constraint for system templates
        Index('unique_system_template_name', 'name', unique=True, postgresql_where=Column('is_system') == True),
    )


class WorkflowTemplatePermission(Base):
    """Permissions for workflow templates.
    
    Owned by Community Edition - manages access control for templates.
    """
    __tablename__ = "workflow_template_permissions"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # References
    template_id = Column(UUID(as_uuid=True), ForeignKey('workflow_templates.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'))
    
    # Permission details
    permission = Column(String(20), nullable=False)  # view, use, edit, delete
    granted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    granted_by = Column(String(36), ForeignKey('users.id', ondelete='SET NULL'))
    
    # Relationships
    template = relationship("WorkflowTemplate", back_populates="permissions")
    user = relationship("User", foreign_keys=[user_id])
    grantor = relationship("User", foreign_keys=[granted_by])
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_template_permissions_template', 'template_id'),
        Index('idx_template_permissions_user', 'user_id'),
        Index('unique_user_template_permission', 'template_id', 'user_id', 'permission', 
              unique=True, postgresql_where=Column('user_id').isnot(None)),
    )


class WorkflowTemplateUsage(Base):
    """Usage history for workflow templates.
    
    Owned by Community Edition - tracks template instantiation.
    """
    __tablename__ = "workflow_template_usage"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # References
    template_id = Column(UUID(as_uuid=True), ForeignKey('workflow_templates.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    workflow_id = Column(String(36), nullable=True)  # Reference to workflow_instances.id if created
    
    # Usage details
    instantiated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    customization_level = Column(String(20))  # none, minor, major
    
    # Relationships
    template = relationship("WorkflowTemplate", back_populates="usage_history")
    user = relationship("User")
    # workflow relationship removed - no foreign key constraint
    
    # Indexes
    __table_args__ = (
        Index('idx_template_usage_template', 'template_id'),
        Index('idx_template_usage_user', 'user_id'),
        Index('idx_template_usage_date', 'instantiated_at'),
    )


class WorkflowTemplateReview(Base):
    """Reviews and ratings for workflow templates.
    
    Owned by Community Edition - allows users to rate and review templates.
    """
    __tablename__ = "workflow_template_reviews"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # References
    template_id = Column(UUID(as_uuid=True), ForeignKey('workflow_templates.id', ondelete='CASCADE'), nullable=False)
    user_id = Column(String(36), ForeignKey('users.id', ondelete='CASCADE'), nullable=False)
    
    # Review details
    rating = Column(Integer, nullable=False)  # CheckConstraint added in migration
    review = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Relationships
    template = relationship("WorkflowTemplate", back_populates="reviews")
    user = relationship("User")
    
    # Indexes and constraints
    __table_args__ = (
        Index('idx_template_reviews_template', 'template_id'),
        Index('idx_template_reviews_user', 'user_id'),
        Index('unique_user_template_review', 'template_id', 'user_id', unique=True),
    )