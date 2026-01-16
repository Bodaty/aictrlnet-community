"""Adapter models and schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field
import uuid


class AdapterCategory(str, Enum):
    """Categories of adapters."""
    AI = "ai"
    AI_AGENT = "ai_agent"
    COMMUNICATION = "communication"
    HUMAN = "human"
    PAYMENT = "payment"
    DATA = "data"
    INTEGRATION = "integration"
    UTILITY = "utility"


class Edition(str, Enum):
    """Edition types."""
    COMMUNITY = "community"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class AdapterStatus(str, Enum):
    """Adapter status."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


class AdapterCapability(BaseModel):
    """A capability provided by an adapter."""
    name: str
    description: str
    category: str = "general"
    
    # Parameters for this capability
    parameters: Dict[str, Any] = {}
    required_parameters: List[str] = []
    
    # Response format
    response_format: Dict[str, Any] = {}
    
    # Performance characteristics
    async_supported: bool = True
    estimated_duration_seconds: Optional[float] = None
    rate_limit: Optional[int] = None  # requests per minute
    
    # Cost information
    cost_per_request: Optional[float] = None
    cost_currency: str = "USD"


class AdapterConfig(BaseModel):
    """Configuration for an adapter.
    
    NOTE: This is the Pydantic model for adapter registry metadata.
    For user-specific adapter configurations stored in the database,
    see models.adapter_config.UserAdapterConfig (SQLAlchemy model).
    
    This model defines the configuration structure for adapter instances
    in the registry, NOT user-specific settings.
    """
    # Basic info
    name: str
    version: str = "1.0.0"
    description: Optional[str] = None
    category: AdapterCategory
    required_edition: Edition = Edition.COMMUNITY
    
    # Authentication
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    credentials: Dict[str, Any] = {}
    
    # Connection settings
    base_url: Optional[str] = None
    timeout_seconds: int = 30
    max_retries: int = 3
    retry_delay_seconds: int = 1
    
    # Rate limiting
    rate_limit_per_minute: Optional[int] = None
    concurrent_requests: int = 10
    
    # Feature flags
    features: Dict[str, bool] = {}
    
    # Edition requirements
    required_edition: str = "community"
    
    # Custom settings
    custom_config: Dict[str, Any] = {}


class AdapterMetrics(BaseModel):
    """Runtime metrics for an adapter."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    average_response_time_ms: float = 0.0
    last_response_time_ms: Optional[float] = None
    
    total_cost: float = 0.0
    
    last_used: Optional[datetime] = None
    last_error: Optional[str] = None
    last_error_time: Optional[datetime] = None
    
    # Rate limiting
    requests_this_minute: int = 0
    minute_start: datetime = Field(default_factory=datetime.utcnow)


class AdapterRequest(BaseModel):
    """Request to an adapter."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    capability: str
    parameters: Dict[str, Any] = {}
    
    # Request metadata
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    # Options
    timeout_override: Optional[int] = None
    priority: str = "normal"  # low, normal, high
    
    # Context
    context: Dict[str, Any] = {}


class AdapterResponse(BaseModel):
    """Response from an adapter."""
    request_id: str
    capability: str
    status: str = "success"  # success, error, partial
    
    # Response data
    data: Any = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    
    # Metadata
    duration_ms: float
    cost: Optional[float] = None
    tokens_used: Optional[int] = None
    
    # Additional info
    metadata: Dict[str, Any] = {}
    warnings: List[str] = []


class AdapterInfo(BaseModel):
    """Information about a registered adapter."""
    id: str
    name: str
    category: AdapterCategory
    version: str
    description: Optional[str] = None
    
    # Status
    status: AdapterStatus
    status_message: Optional[str] = None
    
    # Capabilities
    capabilities: List[AdapterCapability] = []
    
    # Configuration
    required_config: List[str] = []
    optional_config: List[str] = []
    
    # Edition and availability
    required_edition: str = "community"
    available: bool = True
    availability_message: Optional[str] = None
    
    # Metrics
    metrics: AdapterMetrics = Field(default_factory=AdapterMetrics)
    
    # Registration
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    last_health_check: Optional[datetime] = None