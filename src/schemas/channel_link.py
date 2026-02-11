"""Schemas for channel account linking."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ChannelLinkRequest(BaseModel):
    """Request a linking code for a specific channel."""
    channel_type: str = Field(..., description="Channel to link: telegram, whatsapp, twilio, slack, discord")


class ChannelLinkCodeResponse(BaseModel):
    """Response with the linking code the user must send via the channel."""
    code: str = Field(..., description="6-digit code to send via the channel")
    channel_type: str
    expires_in_seconds: int = 600
    instructions: str


class ChannelLinkResponse(BaseModel):
    """Details of a linked channel identity."""
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    user_id: str
    channel_type: str
    channel_user_id: str
    display_name: Optional[str] = None
    is_active: bool
    linked_at: datetime


class ChannelUnlinkRequest(BaseModel):
    """Unlink a channel identity."""
    channel_type: str
    channel_user_id: str
