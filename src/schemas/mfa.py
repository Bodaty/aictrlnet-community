"""MFA schemas for request/response validation."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


# Request schemas
class MFAInitRequest(BaseModel):
    """Request to initialize MFA enrollment."""
    device_name: Optional[str] = Field(None, max_length=100, description="Name for the MFA device")


class MFACompleteRequest(BaseModel):
    """Request to complete MFA enrollment."""
    verification_code: str = Field(..., pattern=r"^\d{6}$", description="6-digit TOTP code")
    device_name: Optional[str] = Field(None, max_length=100, description="Name for the MFA device")


class MFAVerifyRequest(BaseModel):
    """Request to verify MFA code during login."""
    session_token: str = Field(..., description="Temporary session token from login")
    code: str = Field(..., min_length=6, max_length=8, description="TOTP code or backup code")


class MFADisableRequest(BaseModel):
    """Request to disable MFA."""
    password: str = Field(..., description="User's password for verification")


class MFAResetRequest(BaseModel):
    """Request to reset MFA (admin only)."""
    reason: str = Field(..., description="Reason for reset")


# Response schemas
class MFAStatusResponse(BaseModel):
    """MFA status response."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    mfa_enabled: bool
    enrolled_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    devices: Optional[List[Dict[str, Any]]] = None  # Business/Enterprise only


class MFAInitResponse(BaseModel):
    """MFA initialization response."""
    qr_code: str = Field(..., description="Base64 encoded QR code image")
    manual_entry_key: str = Field(..., description="Secret key for manual entry")
    expires_in: int = Field(..., description="Seconds until enrollment expires")


class MFACompleteResponse(BaseModel):
    """MFA enrollment completion response."""
    backup_codes: List[str] = Field(..., description="One-time backup codes")
    message: str = Field(default="MFA successfully enabled")


class MFAVerifyResponse(BaseModel):
    """MFA verification response."""
    access_token: str
    token_type: str = "bearer"
    backup_code_used: bool = False
    remaining_backup_codes: Optional[int] = None


class MFADisableResponse(BaseModel):
    """MFA disable response."""
    message: str = Field(default="MFA successfully disabled")


class MFADeviceResponse(BaseModel):
    """MFA device information (Business/Enterprise)."""
    model_config = ConfigDict(from_attributes=True,
        protected_namespaces=()
    )
    
    id: str
    device_name: str
    device_type: str = "totp"
    is_primary: bool
    last_used_at: Optional[datetime] = None
    created_at: datetime


class MFARecoveryCodesResponse(BaseModel):
    """MFA recovery codes response."""
    recovery_codes: List[str]
    generated_at: datetime


# Login-related schemas
class LoginRequest(BaseModel):
    """Login request."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response with MFA support."""
    access_token: Optional[str] = None
    token_type: Optional[str] = None
    mfa_required: bool = False
    session_token: Optional[str] = None
    expires_in: Optional[int] = None