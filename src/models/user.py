"""User model for authentication and authorization."""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Boolean, DateTime, JSON
from sqlalchemy.orm import relationship
from .base import Base


class User(Base):
    """User account model."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, nullable=False, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    tenant_id = Column(String, index=True, nullable=True)  # NULL for Community/Business, required for Enterprise
    edition = Column(String, default="community")
    preferences = Column(JSON, default={})
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime)
    
    # Password reset fields
    password_reset_token = Column(String, nullable=True)
    password_reset_expires = Column(DateTime, nullable=True)
    
    # Email verification fields
    email_verified = Column(Boolean, default=False)
    email_verification_token = Column(String, nullable=True)
    
    # MFA fields
    mfa_enabled = Column(Boolean, default=False)
    mfa_secret = Column(String(255), nullable=True)  # Encrypted TOTP secret
    mfa_backup_codes = Column(String, nullable=True)  # Encrypted JSON array
    mfa_enrolled_at = Column(DateTime, nullable=True)
    mfa_last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    subscriptions = relationship("Subscription", back_populates="user", lazy="select")
    platform_credentials = relationship("PlatformCredential", back_populates="user", lazy="select")
    platform_webhooks = relationship("PlatformWebhook", back_populates="user", lazy="select")
    adapter_configs = relationship("UserAdapterConfig", back_populates="user", lazy="select", cascade="all, delete-orphan")
    basic_agents = relationship("BasicAgent", back_populates="user", lazy="select", cascade="all, delete-orphan")

    # Note: relationships to roles are defined in Business Edition
    # This keeps the User model in Community Edition simple