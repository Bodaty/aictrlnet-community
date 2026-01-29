"""User Adapter Configuration model for storing user-specific adapter settings."""

from sqlalchemy import Column, String, JSON, Boolean, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
import uuid

from .base import Base


class UserAdapterConfig(Base):
    """Store user-specific adapter configurations.
    
    This model stores user configurations for adapters, including:
    - Encrypted credentials
    - Custom settings
    - Enabled/disabled state
    - Test status and history
    
    The adapter_type field references the type in the adapter registry,
    not the database adapters table.
    """
    __tablename__ = "adapter_configs"
    
    # Primary key
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    # User relationship - users.id is String, not UUID
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="adapter_configs")
    
    # Adapter reference (to registry type, not DB)
    adapter_type = Column(String(100), nullable=False, index=True)
    
    # User's custom name for this configuration
    name = Column(String(255), nullable=True)
    display_name = Column(String(255), nullable=True)
    
    # Configuration data
    credentials = Column(JSON, nullable=True)  # Encrypted in service layer
    settings = Column(JSON, nullable=True)     # Custom settings
    
    # Status fields
    enabled = Column(Boolean, default=True, nullable=False)
    test_status = Column(String(50), nullable=True)  # 'success', 'failed', 'untested'
    test_message = Column(String(500), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    last_tested_at = Column(DateTime(timezone=True), nullable=True)
    
    # Metadata
    version = Column(String(50), nullable=True)  # Adapter version when configured
    metadata_field = Column('metadata', JSON, default={})  # Renamed to avoid SQLAlchemy conflict
    
    # Ensure unique configurations per user/type/name combination
    # Use extend_existing=True to avoid conflicts when table already exists from migrations
    __table_args__ = (
        UniqueConstraint('user_id', 'adapter_type', 'name', name='_user_adapter_name_uc'),
        {'extend_existing': True}
    )
    
    def __repr__(self):
        return f"<UserAdapterConfig(id={self.id}, user_id={self.user_id}, type={self.adapter_type}, name={self.name})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": str(self.id),
            "user_id": str(self.user_id),
            "adapter_type": self.adapter_type,
            "name": self.name,
            "display_name": self.display_name,
            "enabled": self.enabled,
            "test_status": self.test_status,
            "test_message": self.test_message,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_tested_at": self.last_tested_at.isoformat() if self.last_tested_at else None,
            "version": self.version,
            "metadata": self.metadata_field
        }
    
    def to_dict_with_credentials(self):
        """Convert to dictionary including credentials (use with caution)."""
        data = self.to_dict()
        data["credentials"] = self.credentials  # Should be decrypted by service layer
        data["settings"] = self.settings
        return data