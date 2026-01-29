"""API Key models for secure API authentication."""

from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, JSON, Integer
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .base import Base


class APIKey(Base):
    """API Key model for secure authentication."""
    
    __tablename__ = "api_keys"
    
    # Public key ID (shown to user)
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # User relationship
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    # Relationship defined in User model to avoid circular imports
    
    # Key identification
    name = Column(String(255), nullable=False)  # User-friendly name
    description = Column(String(500))  # Optional description
    
    # Key components (never store plain text!)
    key_prefix = Column(String(20), nullable=False, index=True)  # First chars for identification
    key_suffix = Column(String(10), nullable=False)  # Last chars for display
    key_hash = Column(String(255), nullable=False)  # Hashed full key
    key_salt = Column(String(255), nullable=False)  # Salt for hashing
    
    # Security settings
    scopes = Column(JSON, default=list)  # List of permissions
    allowed_ips = Column(JSON, default=list)  # IP whitelist (empty = all allowed)
    
    # Usage tracking
    last_used_at = Column(DateTime)
    last_used_ip = Column(String(45))  # IPv4 or IPv6
    usage_count = Column(Integer, default=0)
    
    # Lifecycle
    expires_at = Column(DateTime)  # Optional expiration
    is_active = Column(Boolean, default=True)
    revoked_at = Column(DateTime)
    revoked_reason = Column(String(500))
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<APIKey {self.name} ({self.key_prefix}...{self.key_suffix})>"
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "key_identifier": f"{self.key_prefix}...{self.key_suffix}",
            "scopes": self.scopes or [],
            "allowed_ips": self.allowed_ips or [],
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "last_used_ip": self.last_used_ip,
            "usage_count": self.usage_count,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class APIKeyLog(Base):
    """Log API key usage for security auditing."""
    
    __tablename__ = "api_key_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    api_key_id = Column(String, ForeignKey("api_keys.id"), nullable=False)
    
    # Request details
    endpoint = Column(String(500))
    method = Column(String(10))
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Response details
    status_code = Column(Integer)
    response_time_ms = Column(Integer)
    
    # Additional context
    error_message = Column(String(500))
    request_metadata = Column('metadata', JSON)
    
    # Timestamp
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship to parent API key
    # api_key = relationship("APIKey")  # Commented to avoid circular dependency