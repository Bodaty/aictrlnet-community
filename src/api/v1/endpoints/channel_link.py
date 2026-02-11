"""Channel account linking endpoints.

Authenticated users generate a 6-digit code, then send it via their
messaging channel to prove ownership.  Once verified, the channel identity
is permanently linked to their AICtrlNet account.
"""

import logging
import random
import string
from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_user
from models.channel_link import ChannelLink, ChannelLinkCode
from schemas.channel_link import (
    ChannelLinkRequest,
    ChannelLinkCodeResponse,
    ChannelLinkResponse,
    ChannelUnlinkRequest,
)

logger = logging.getLogger(__name__)
router = APIRouter()

VALID_CHANNELS = {"telegram", "whatsapp", "twilio", "slack", "discord"}
CODE_TTL_SECONDS = 600  # 10 minutes


def _generate_code() -> str:
    """Generate a 6-digit numeric linking code."""
    return "".join(random.choices(string.digits, k=6))


def _link_instructions(channel_type: str, code: str) -> str:
    """Human-readable instructions for the user."""
    channel_hints = {
        "telegram": f"Send /link {code} to the AICtrlNet bot on Telegram.",
        "whatsapp": f"Send the message 'link {code}' to the AICtrlNet WhatsApp number.",
        "twilio": f"Reply with 'link {code}' to the AICtrlNet SMS number.",
        "slack": f"In the AICtrlNet Slack app, type /link {code}",
        "discord": f"In the AICtrlNet Discord bot DM, type /link {code}",
    }
    return channel_hints.get(channel_type, f"Send 'link {code}' via {channel_type}.")


@router.post("/link", response_model=ChannelLinkCodeResponse)
async def request_link_code(
    body: ChannelLinkRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Generate a 6-digit code to link a channel identity to this account.

    The user must then send this code via the target channel within 10 minutes.
    """
    user_id = current_user.id if hasattr(current_user, "id") else current_user.get("id")

    if body.channel_type not in VALID_CHANNELS:
        raise HTTPException(status_code=400, detail=f"Invalid channel. Must be one of: {', '.join(sorted(VALID_CHANNELS))}")

    # Expire any previous unused codes for this user + channel
    prev = await db.execute(
        select(ChannelLinkCode).filter(
            and_(
                ChannelLinkCode.user_id == str(user_id),
                ChannelLinkCode.channel_type == body.channel_type,
                ChannelLinkCode.used == False,
            )
        )
    )
    for old_code in prev.scalars().all():
        old_code.used = True

    code = _generate_code()
    link_code = ChannelLinkCode(
        user_id=str(user_id),
        code=code,
        channel_type=body.channel_type,
        expires_at=datetime.utcnow() + timedelta(seconds=CODE_TTL_SECONDS),
    )
    db.add(link_code)
    await db.commit()

    return ChannelLinkCodeResponse(
        code=code,
        channel_type=body.channel_type,
        expires_in_seconds=CODE_TTL_SECONDS,
        instructions=_link_instructions(body.channel_type, code),
    )


@router.get("/links", response_model=List[ChannelLinkResponse])
async def list_linked_channels(
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """List all active channel links for the current user."""
    user_id = current_user.id if hasattr(current_user, "id") else current_user.get("id")

    result = await db.execute(
        select(ChannelLink).filter(
            and_(ChannelLink.user_id == str(user_id), ChannelLink.is_active == True)
        )
    )
    return result.scalars().all()


@router.delete("/links")
async def unlink_channel(
    body: ChannelUnlinkRequest,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Unlink a channel identity from the current user's account."""
    user_id = current_user.id if hasattr(current_user, "id") else current_user.get("id")

    result = await db.execute(
        select(ChannelLink).filter(
            and_(
                ChannelLink.user_id == str(user_id),
                ChannelLink.channel_type == body.channel_type,
                ChannelLink.channel_user_id == body.channel_user_id,
                ChannelLink.is_active == True,
            )
        )
    )
    link = result.scalar_one_or_none()
    if not link:
        raise HTTPException(status_code=404, detail="Channel link not found")

    link.is_active = False
    link.unlinked_at = datetime.utcnow()
    await db.commit()

    return {"status": "unlinked", "channel_type": body.channel_type, "channel_user_id": body.channel_user_id}
