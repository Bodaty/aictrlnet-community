"""User schemas for request/response validation."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from datetime import datetime


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    username: Optional[str] = None
    full_name: Optional[str] = None
    edition: Optional[str] = Field(default="community", description="User edition")
    is_active: bool = True


class UserCreate(UserBase):
    """Schema for creating a user."""
    password: str = Field(..., min_length=8, description="User password")
    is_superuser: bool = False


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    email: Optional[EmailStr] = None
    username: Optional[str] = None
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    edition: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


class UserResponse(UserBase):
    """Schema for user response."""
    id: str
    roles: List[str] = Field(default_factory=list)
    is_superuser: bool
    created_at: Optional[datetime]
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True


class NotificationSettings(BaseModel):
    """Notification preferences."""
    email: bool = True
    inApp: bool = True
    workflow: bool = True


class AppSettingsUpdate(BaseModel):
    """Schema for updating user application settings."""
    # Legacy default model (for backward compatibility)
    aiModel: Optional[str] = Field(
        default="llama3.1-local",
        description="Default AI model (legacy - for backward compatibility)"
    )

    # New tiered model preferences
    preferredFastModel: Optional[str] = Field(
        default="llama3.2:1b",
        description="Fast tier model (~1-2s per call) for quick tasks like intent classification"
    )
    preferredBalancedModel: Optional[str] = Field(
        default="llama3.2:3b",
        description="Balanced tier model (~3-5s per call) for semantic matching and analysis"
    )
    preferredQualityModel: Optional[str] = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        description="Quality tier model (~20-25s per call) for workflow generation and complex tasks"
    )

    # UI/UX settings
    theme: Optional[str] = Field(default="light", description="UI theme (light/dark/auto)")
    language: Optional[str] = Field(default="en", description="Preferred language")
    notifications: Optional[NotificationSettings] = Field(
        default_factory=NotificationSettings,
        description="Notification preferences"
    )

    # Workflow preferences
    defaultWorkflowEngine: Optional[str] = Field(
        default="standard",
        description="Default workflow engine to use"
    )
    autoSaveInterval: Optional[int] = Field(
        default=30,
        description="Auto-save interval in seconds"
    )

    # Dashboard settings
    dashboardLayout: Optional[str] = Field(
        default="grid",
        description="Dashboard layout preference"
    )

    class Config:
        extra = "allow"  # Allow additional fields for extensibility


class AppSettingsResponse(AppSettingsUpdate):
    """Schema for app settings response (same as update for now)."""
    pass