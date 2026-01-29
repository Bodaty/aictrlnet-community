"""Platform Integration models for Community Edition"""
from sqlalchemy import Column, String, Integer, JSON, Text, DateTime, Boolean, Enum as SQLEnum, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import enum

from .base import Base


class PlatformType(enum.Enum):
    """Supported automation platforms"""
    N8N = "n8n"
    ZAPIER = "zapier"
    MAKE = "make"
    POWER_AUTOMATE = "power_automate"
    IFTTT = "ifttt"
    CUSTOM = "custom"


class AuthMethod(enum.Enum):
    """Platform authentication methods"""
    API_KEY = "api_key"
    OAUTH2 = "oauth2"
    BASIC = "basic"
    TOKEN = "token"
    CUSTOM = "custom"


class PlatformCredential(Base):
    """Platform credentials storage (encrypted)"""
    __tablename__ = "platform_credentials"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    platform = Column(String(50), nullable=False)
    auth_method = Column(String(50), nullable=False)
    
    # Encrypted credential data
    encrypted_data = Column(Text, nullable=False)  # JSON encrypted
    
    # Access control
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    is_shared = Column(Boolean, default=False)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime)
    
    # Usage tracking
    execution_count = Column(Integer, default=0)
    last_error = Column(Text)
    
    # Configuration
    config_metadata = Column('metadata', JSON, default={})  # Renamed to avoid SQLAlchemy reserved word
    
    # Relationships
    user = relationship("User", back_populates="platform_credentials")
    platform_executions = relationship("PlatformExecution", back_populates="credential")


class PlatformExecution(Base):
    """Track platform workflow executions"""
    __tablename__ = "platform_executions"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Execution context
    workflow_id = Column(String(255), nullable=False, index=True)
    node_id = Column(String(255), nullable=False)
    execution_id = Column(String(255), nullable=False, index=True)
    
    # Platform details
    platform = Column(String(50), nullable=False)
    external_workflow_id = Column(String(255), nullable=False)
    external_execution_id = Column(String(255))
    
    # Credentials used
    credential_id = Column(Integer, ForeignKey("platform_credentials.id"))
    
    # Execution data
    input_data = Column(JSON, default={})
    output_data = Column(JSON, default={})
    error_data = Column(JSON)
    
    # Timing
    started_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    completed_at = Column(DateTime)
    duration_ms = Column(Integer)
    
    # Status tracking
    status = Column(String(50), nullable=False, default="pending")  # pending, running, completed, failed
    
    # Cost tracking (if available from platform)
    estimated_cost = Column(Integer, default=0)  # In cents
    
    # Metadata
    execution_metadata = Column('metadata', JSON, default={})  # Renamed to avoid SQLAlchemy reserved word
    
    # Relationships
    credential = relationship("PlatformCredential", back_populates="platform_executions")


class PlatformAdapter(Base):
    """Registry of available platform adapters"""
    __tablename__ = "platform_adapters"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), unique=True, nullable=False)
    
    # Adapter configuration
    adapter_class = Column(String(255), nullable=False)  # Python class path
    version = Column(String(50), nullable=False)
    
    # Capabilities
    capabilities = Column(JSON, default={})  # What this adapter can do
    supported_auth_methods = Column(JSON, default=[])  # List of AuthMethod values
    
    # Status
    is_active = Column(Boolean, default=True)
    is_beta = Column(Boolean, default=False)
    
    # Configuration schema
    config_schema = Column(JSON, default={})  # JSON Schema for configuration
    
    # Documentation
    documentation_url = Column(String(500))
    icon_url = Column(String(500))
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class PlatformHealth(Base):
    """Track platform health and availability"""
    __tablename__ = "platform_health"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), nullable=False, index=True)
    
    # Health status
    is_healthy = Column(Boolean, default=True)
    last_check_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    response_time_ms = Column(Integer)
    
    # Error tracking
    consecutive_failures = Column(Integer, default=0)
    last_error = Column(Text)
    
    # Availability metrics (last 24 hours)
    uptime_percentage = Column(Integer, default=100)  # 0-100
    total_checks = Column(Integer, default=0)
    failed_checks = Column(Integer, default=0)
    
    # Performance metrics
    avg_response_time_ms = Column(Integer)
    p95_response_time_ms = Column(Integer)
    
    # Metadata
    health_metadata = Column('metadata', JSON, default={})  # Renamed to avoid SQLAlchemy reserved word


class PlatformWebhook(Base):
    """Platform webhook configuration"""
    __tablename__ = "platform_webhooks"
    
    id = Column(Integer, primary_key=True, index=True)
    platform = Column(String(50), nullable=False, index=True)
    webhook_url = Column(String(500), nullable=False)
    secret = Column(String(255), nullable=False)
    events = Column(JSON, default=list)  # List of event types to subscribe to
    user_id = Column(String(36), ForeignKey("users.id"), nullable=True)
    
    is_active = Column(Boolean, default=True)
    verified = Column(Boolean, default=False)
    
    # Stats
    last_triggered_at = Column(DateTime, nullable=True)
    total_deliveries = Column(Integer, default=0)
    successful_deliveries = Column(Integer, default=0)
    failed_deliveries = Column(Integer, default=0)
    consecutive_failures = Column(Integer, default=0)
    
    # Metadata
    webhook_metadata = Column('metadata', JSON, default={})  # Renamed to avoid SQLAlchemy reserved word
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="platform_webhooks")
    deliveries = relationship("PlatformWebhookDelivery", back_populates="webhook", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<PlatformWebhook {self.id}: {self.platform} -> {self.webhook_url}>"


class PlatformWebhookDelivery(Base):
    """Webhook delivery attempts and status"""
    __tablename__ = "platform_webhook_deliveries"
    
    id = Column(Integer, primary_key=True, index=True)
    webhook_id = Column(Integer, ForeignKey("platform_webhooks.id"), nullable=False, index=True)
    
    event_type = Column(String(50), nullable=False)
    payload = Column(JSON, nullable=False)
    
    status = Column(String(20), nullable=False, default="pending")  # pending, delivered, failed, retrying, cancelled
    attempts = Column(Integer, default=0)
    
    # Response info
    response_status = Column(Integer, nullable=True)
    response_headers = Column(JSON, nullable=True)
    response_body = Column(Text, nullable=True)
    
    # Timing
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    delivered_at = Column(DateTime, nullable=True)
    next_retry_at = Column(DateTime, nullable=True, index=True)
    
    # Relationships
    webhook = relationship("PlatformWebhook", back_populates="deliveries")
    
    def __repr__(self):
        return f"<PlatformWebhookDelivery {self.id}: {self.event_type} - {self.status}>"