"""API Key schemas for request/response validation."""

from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict


class APIKeyCreate(BaseModel):
    """Schema for creating a new API key."""
    name: str = Field(..., min_length=1, max_length=255, description="Name for the API key")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    scopes: List[str] = Field(default_factory=list, description="List of permission scopes")
    allowed_ips: List[str] = Field(default_factory=list, description="IP whitelist (empty = all allowed)")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days (max 365)")
    
    @validator('scopes')
    def validate_scopes(cls, v):
        """Validate scope format."""
        valid_scopes = [
            "read:all", "write:all",
            "read:tasks", "write:tasks",
            "read:workflows", "write:workflows", 
            "read:agents", "write:agents",
            "read:adapters", "write:adapters",
            "admin:all"
        ]
        for scope in v:
            if scope not in valid_scopes:
                raise ValueError(f"Invalid scope: {scope}")
        return v
    
    @validator('allowed_ips')
    def validate_ips(cls, v):
        """Basic IP validation."""
        import ipaddress
        for ip in v:
            try:
                # Try parsing as IP address or network
                if '/' in ip:
                    ipaddress.ip_network(ip)
                else:
                    ipaddress.ip_address(ip)
            except ValueError:
                raise ValueError(f"Invalid IP address or network: {ip}")
        return v


class APIKeyResponse(BaseModel):
    """Schema for API key response (without sensitive data)."""
    id: str
    name: str
    description: Optional[str]
    key_identifier: str = Field(..., description="Partial key for identification (prefix...suffix)")
    scopes: List[str]
    allowed_ips: List[str]
    last_used_at: Optional[datetime]
    last_used_ip: Optional[str]
    usage_count: int
    expires_at: Optional[datetime]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class APIKeyCreateResponse(APIKeyResponse):
    """Response when creating a new API key (includes the full key once)."""
    api_key: str = Field(..., description="Full API key - SAVE THIS! It won't be shown again")


class APIKeyUpdate(BaseModel):
    """Schema for updating an API key."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = Field(None, max_length=500)
    scopes: Optional[List[str]] = None
    allowed_ips: Optional[List[str]] = None
    is_active: Optional[bool] = None
    
    @validator('scopes')
    def validate_scopes(cls, v):
        """Validate scope format if provided."""
        if v is not None:
            valid_scopes = [
                "read:all", "write:all",
                "read:tasks", "write:tasks",
                "read:workflows", "write:workflows",
                "read:agents", "write:agents",
                "read:adapters", "write:adapters",
                "admin:all"
            ]
            for scope in v:
                if scope not in valid_scopes:
                    raise ValueError(f"Invalid scope: {scope}")
        return v


class APIKeyListResponse(BaseModel):
    """Response for listing API keys."""
    keys: List[APIKeyResponse]
    total: int


class APIKeyRevoke(BaseModel):
    """Schema for revoking an API key."""
    reason: Optional[str] = Field(None, max_length=500, description="Reason for revocation")


class APIKeyVerifyRequest(BaseModel):
    """Request to verify an API key."""
    api_key: str = Field(..., description="API key to verify")


class APIKeyVerifyResponse(BaseModel):
    """Response for API key verification."""
    is_valid: bool
    user_id: Optional[str] = None
    scopes: List[str] = Field(default_factory=list)
    expires_at: Optional[datetime] = None
    message: Optional[str] = None