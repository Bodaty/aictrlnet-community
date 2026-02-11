"""ChannelLink model — maps external messaging identities to authenticated users.

Before a Telegram/WhatsApp/SMS user can interact with AICtrlNet through a
channel webhook, they must *link* their channel identity to their platform
account.  The flow:

1. Authenticated user calls POST /api/v1/channels/link  ->  receives a 6-digit code.
2. User sends that code via the channel (e.g. "/link 482901" to the Telegram bot).
3. The webhook verifies the code, creates a ChannelLink row, and the user is
   authenticated for all future messages on that channel.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import Base


class ChannelLink(Base):
    """Maps a (channel_type, channel_user_id) pair to an authenticated user."""
    __tablename__ = "channel_links"
    __table_args__ = (
        UniqueConstraint("channel_type", "channel_user_id", name="uq_channel_identity"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    # The AICtrlNet user this channel identity belongs to
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)

    # External identity
    channel_type = Column(String(50), nullable=False, index=True)     # telegram, whatsapp, twilio, slack, discord
    channel_user_id = Column(String(255), nullable=False, index=True) # platform-specific sender ID

    # Optional display info
    display_name = Column(String(200), nullable=True)

    # Linking state
    is_active = Column(Boolean, default=True, nullable=False)
    linked_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    unlinked_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", backref="channel_links")


class ChannelLinkCode(Base):
    """Short-lived verification code for the account-linking handshake."""
    __tablename__ = "channel_link_codes"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)

    user_id = Column(String(36), ForeignKey("users.id"), nullable=False)
    code = Column(String(6), nullable=False, index=True)  # 6-digit code
    channel_type = Column(String(50), nullable=False)      # which channel this code is for
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
