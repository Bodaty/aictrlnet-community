"""API Key schemas for request/response validation."""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field, validator, ConfigDict


def _validate_scopes(v: List[str]) -> List[str]:
    """Reject unknown or legacy scopes at creation time.

    Legacy scopes (``read:all`` / ``write:all``) were dropped at request-time
    in Wave 2 Phase B. Keys submitted with them would silently fail every
    scope check, so fail fast at create/update instead. Existing stored keys
    with legacy scopes render in the UI with a "deprecated" badge; they can
    be corrected via PUT.
    """
    # Imported inside the function so test collection doesn't pay the cost.
    from mcp_server.scopes import ALL_SCOPES
    unknown = [s for s in v if s not in ALL_SCOPES]
    if unknown:
        raise ValueError(
            f"Unknown or legacy scope(s): {unknown}. "
            "See GET /api/v1/api-keys/available-scopes for the valid list."
        )
    return v


def _validate_ips(v: List[str]) -> List[str]:
    """Basic IP / CIDR validation."""
    import ipaddress
    for ip in v:
        try:
            if '/' in ip:
                ipaddress.ip_network(ip)
            else:
                ipaddress.ip_address(ip)
        except ValueError:
            raise ValueError(f"Invalid IP address or network: {ip}")
    return v


def _validate_rate_limit_per_tool(v: Optional[Dict[str, Dict[str, int]]]) -> Optional[Dict[str, Dict[str, int]]]:
    """Accept ``{"tool_name": {"per_minute": N, "per_day": M}}`` shape."""
    if v is None:
        return v
    for tool, caps in v.items():
        if not isinstance(caps, dict):
            raise ValueError(f"rate_limit_per_tool[{tool}] must be an object")
        for k in caps:
            if k not in ("per_minute", "per_day"):
                raise ValueError(
                    f"rate_limit_per_tool[{tool}] only supports keys 'per_minute' and 'per_day'"
                )
        for k, n in caps.items():
            if not isinstance(n, int) or n < 1:
                raise ValueError(f"rate_limit_per_tool[{tool}][{k}] must be a positive integer")
    return v


class APIKeyCreate(BaseModel):
    """Schema for creating a new API key."""
    name: str = Field(..., min_length=1, max_length=255, description="Name for the API key")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    scopes: List[str] = Field(default_factory=list, description="List of permission scopes")
    allowed_ips: List[str] = Field(default_factory=list, description="IP whitelist (empty = all allowed)")
    expires_in_days: Optional[int] = Field(None, ge=1, le=365, description="Expiration in days (max 365)")
    rate_limit_per_tool: Optional[Dict[str, Dict[str, int]]] = Field(
        None,
        description='Per-tool rate caps: {"tool_name": {"per_minute": N, "per_day": M}}',
    )

    _v_scopes = validator('scopes', allow_reuse=True)(_validate_scopes)
    _v_ips = validator('allowed_ips', allow_reuse=True)(_validate_ips)
    _v_rlpt = validator('rate_limit_per_tool', allow_reuse=True)(_validate_rate_limit_per_tool)


class APIKeyResponse(BaseModel):
    """Schema for API key response (without sensitive data)."""
    id: str
    name: str
    description: Optional[str]
    key_identifier: str = Field(..., description="Partial key for identification (prefix...suffix)")
    scopes: List[str]
    allowed_ips: List[str]
    rate_limit_per_tool: Dict[str, Dict[str, int]] = Field(default_factory=dict)
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
    rate_limit_per_tool: Optional[Dict[str, Dict[str, int]]] = None
    is_active: Optional[bool] = None

    @validator('scopes')
    def _validate_scopes_optional(cls, v):
        if v is None:
            return v
        return _validate_scopes(v)

    @validator('allowed_ips')
    def _validate_ips_optional(cls, v):
        if v is None:
            return v
        return _validate_ips(v)

    @validator('rate_limit_per_tool')
    def _validate_rlpt_optional(cls, v):
        return _validate_rate_limit_per_tool(v)


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