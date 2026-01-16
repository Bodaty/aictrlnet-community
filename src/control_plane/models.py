"""Control plane models for component registration and management."""

from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import uuid


class ComponentType(str, Enum):
    """Types of components that can be registered."""
    ADAPTER = "adapter"
    NODE = "node"
    SERVICE = "service"
    WORKFLOW = "workflow"
    MCP_SERVER = "mcp_server"


class ComponentStatus(str, Enum):
    """Component registration status."""
    PENDING = "pending"
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    FAILED = "failed"


class ComponentCapability(BaseModel):
    """A capability provided by a component."""
    name: str
    description: str
    version: str = "1.0.0"
    parameters: Dict[str, Any] = {}
    required_permissions: List[str] = []


class Component(BaseModel):
    """Registered component in the control plane."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    type: ComponentType
    version: str
    description: Optional[str] = None
    status: ComponentStatus = ComponentStatus.PENDING
    
    # Registration details
    registered_at: datetime = Field(default_factory=datetime.utcnow)
    registered_by: str  # User or system that registered it
    last_heartbeat: Optional[datetime] = None
    
    # JWT token for this component
    token: Optional[str] = None
    token_expires_at: Optional[datetime] = None
    
    # Capabilities and metadata
    capabilities: List[ComponentCapability] = []
    metadata: Dict[str, Any] = {}
    
    # Edition and feature gating
    edition: str = "community"  # community, business, enterprise
    required_edition: str = "community"
    
    # Health and performance
    health_score: float = 100.0  # 0-100
    reputation_score: float = 100.0  # 0-100
    error_count: int = 0
    success_count: int = 0
    
    # Configuration
    config: Dict[str, Any] = {}
    endpoint_url: Optional[str] = None
    webhook_url: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "name": "openai-adapter",
                "type": "adapter",
                "version": "1.0.0",
                "description": "OpenAI API adapter for GPT models",
                "registered_by": "system",
                "capabilities": [
                    {
                        "name": "chat_completion",
                        "description": "Generate chat completions",
                        "parameters": {
                            "model": "string",
                            "messages": "array"
                        }
                    }
                ],
                "edition": "community"
            }
        }


class ComponentRegistrationRequest(BaseModel):
    """Request to register a new component."""
    name: str
    type: ComponentType
    version: str
    description: Optional[str] = None
    capabilities: List[ComponentCapability] = []
    metadata: Dict[str, Any] = {}
    edition: str = "community"
    endpoint_url: Optional[str] = None
    webhook_url: Optional[str] = None
    config: Dict[str, Any] = {}


class ComponentRegistrationResponse(BaseModel):
    """Response after component registration."""
    component: Component
    token: str
    expires_at: datetime
    
    
class ComponentHeartbeat(BaseModel):
    """Heartbeat from a component."""
    component_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    status: ComponentStatus = ComponentStatus.ACTIVE
    health_score: float = 100.0
    metrics: Dict[str, Any] = {}
    

class ComponentEvent(BaseModel):
    """Event related to a component."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    component_id: str
    event_type: str  # registered, updated, failed, etc.
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = {}
    severity: str = "info"  # debug, info, warning, error, critical