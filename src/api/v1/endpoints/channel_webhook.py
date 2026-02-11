"""Channel webhook endpoint — single entry point for all messaging platforms.

POST /api/v1/channels/{channel_type}/webhook

Security model:
- Every channel user MUST be linked to an authenticated AICtrlNet account via
  the ChannelLink table before they can interact.
- If an unlinked user sends a message, they receive a "please link your account"
  reply and no conversation session is created.
- The `/link <code>` (or `link <code>`) command completes the account-linking
  handshake inline during the webhook call.
"""

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.database import get_db
from services.channel_normalizer import ChannelNormalizer, InboundMessage
from services.conversation_manager import ConversationManagerService
from models.channel_link import ChannelLink, ChannelLinkCode
from adapters.registry import adapter_registry
from adapters.models import AdapterRequest

logger = logging.getLogger(__name__)
settings = get_settings()
normalizer = ChannelNormalizer()

router = APIRouter()

# Normalizer dispatch table
_NORMALIZERS = {
    "slack": normalizer.normalize_slack,
    "telegram": normalizer.normalize_telegram,
    "whatsapp": normalizer.normalize_whatsapp,
    "twilio": normalizer.normalize_twilio,
    "discord": normalizer.normalize_discord,
    "email": normalizer.normalize_email,
}

# Regex to detect a linking command: "/link 123456" or "link 123456"
_LINK_PATTERN = re.compile(r"^/?link\s+(\d{6})$", re.IGNORECASE)


@router.post("/channels/{channel_type}/webhook")
async def channel_webhook(
    channel_type: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Receive webhook from any messaging platform.

    1. Validates/normalizes the payload.
    2. Checks for a linking command — if so, completes the account-linking flow.
    3. Looks up ChannelLink to authenticate the sender.
    4. If authenticated, routes to the conversation manager.
    5. If NOT authenticated, replies with "please link your account" instructions.
    """
    if channel_type not in _NORMALIZERS:
        raise HTTPException(status_code=400, detail=f"Unsupported channel: {channel_type}")

    # --- Platform-specific verification challenges ---

    # Slack URL verification
    if channel_type == "slack":
        body = await request.json()
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge", "")}

    # Discord signature verification + PING/PONG
    if channel_type == "discord":
        raw_body = await request.body()
        sig = request.headers.get("X-Signature-Ed25519", "")
        ts = request.headers.get("X-Signature-Timestamp", "")
        discord_pub_key = getattr(settings, "DISCORD_PUBLIC_KEY", "")
        if discord_pub_key and not normalizer.validate_discord_signature(raw_body, sig, ts, discord_pub_key):
            raise HTTPException(status_code=401, detail="Invalid Discord signature")
        try:
            body = json.loads(raw_body)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")
        if body.get("type") == 1:  # PING
            return {"type": 1}  # PONG

    # Email webhook secret validation
    if channel_type == "email":
        token = request.headers.get("X-Webhook-Secret", "")
        email_secret = getattr(settings, "EMAIL_WEBHOOK_SECRET", "")
        if email_secret and not normalizer.validate_email_webhook(token, email_secret):
            raise HTTPException(status_code=401, detail="Invalid email webhook signature")

    # --- Parse payload ---
    try:
        if channel_type in ("twilio", "email"):
            form = await request.form()
            payload: Dict[str, Any] = dict(form)
        elif channel_type == "discord":
            payload = body  # Already parsed above during signature verification
        else:
            payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    # --- Normalize ---
    normalize_fn = _NORMALIZERS[channel_type]
    message: Optional[InboundMessage] = normalize_fn(payload)

    if message is None:
        # Non-message event (typing indicator, delivery receipt, etc.)
        return Response(status_code=200)

    # --- Check for linking command ---
    link_match = _LINK_PATTERN.match((message.message_text or "").strip())
    if link_match:
        code_str = link_match.group(1)
        return await _handle_link_command(
            db, channel_type, message.sender_id, code_str, message,
            display_name=message.platform_metadata.get("display_name"),
        )

    # --- Authenticate sender via ChannelLink ---
    link = await _lookup_channel_link(db, channel_type, message.sender_id)
    if link is None:
        # Unlinked user — tell them how to link
        await _send_reply_via_channel(
            channel_type, message,
            "You haven't linked your AICtrlNet account to this channel yet.\n\n"
            "To link:\n"
            "1. Log in to AICtrlNet (web UI)\n"
            f'2. Go to Settings > Channels and request a linking code for "{channel_type}"\n'
            "3. Send the code here as: link <6-digit-code>\n\n"
            "Example: link 482901",
        )
        return _format_channel_response(channel_type, {"text": "Account not linked"}, message)

    # --- Route to conversation manager with the REAL authenticated user_id ---
    try:
        manager = ConversationManagerService(db)
        session = await manager.find_or_create_channel_session(
            channel_type=message.channel_type,
            sender_id=message.sender_id,
            user_id=link.user_id,  # authenticated user
            platform_metadata=message.platform_metadata,
        )

        response = await manager.process_message(
            session_id=session.id,
            content=message.message_text,
            user_id=session.user_id,
            channel_type=message.channel_type,
            external_message_id=message.external_message_id,
        )

        reply_text = _extract_reply_text(response)
        await _send_reply_via_channel(channel_type, message, reply_text)

        return _format_channel_response(channel_type, response, message)

    except Exception as e:
        logger.error(f"Channel webhook error ({channel_type}): {e}", exc_info=True)
        return Response(status_code=200)


@router.get("/channels/whatsapp/webhook")
async def whatsapp_verify(request: Request):
    """WhatsApp (Meta) webhook verification endpoint."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    verify_token = settings.WHATSAPP_VERIFY_TOKEN if hasattr(settings, "WHATSAPP_VERIFY_TOKEN") else ""

    if mode == "subscribe" and token == verify_token:
        return Response(content=challenge, media_type="text/plain")

    raise HTTPException(status_code=403, detail="Verification failed")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _lookup_channel_link(
    db: AsyncSession, channel_type: str, sender_id: str
) -> Optional[ChannelLink]:
    """Find an active ChannelLink for the given channel identity."""
    result = await db.execute(
        select(ChannelLink).filter(
            and_(
                ChannelLink.channel_type == channel_type,
                ChannelLink.channel_user_id == sender_id,
                ChannelLink.is_active == True,
            )
        )
    )
    return result.scalar_one_or_none()


async def _handle_link_command(
    db: AsyncSession,
    channel_type: str,
    sender_id: str,
    code_str: str,
    inbound: InboundMessage,
    display_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify a 6-digit linking code and create the ChannelLink.

    Returns a channel-formatted response with the result.
    """
    # Look up the code
    result = await db.execute(
        select(ChannelLinkCode).filter(
            and_(
                ChannelLinkCode.code == code_str,
                ChannelLinkCode.channel_type == channel_type,
                ChannelLinkCode.used == False,
                ChannelLinkCode.expires_at > datetime.utcnow(),
            )
        )
    )
    link_code = result.scalar_one_or_none()

    if link_code is None:
        reply = "Invalid or expired linking code. Please generate a new one from the AICtrlNet web UI."
        await _send_reply_via_channel(channel_type, inbound, reply)
        return _format_channel_response(channel_type, {"text": reply}, inbound)

    # Check if this channel identity is already linked to someone
    existing = await _lookup_channel_link(db, channel_type, sender_id)
    if existing:
        link_code.used = True
        await db.commit()
        reply = f"This {channel_type} account is already linked to an AICtrlNet user. Unlink first if you want to re-link."
        await _send_reply_via_channel(channel_type, inbound, reply)
        return _format_channel_response(channel_type, {"text": reply}, inbound)

    # Create the link
    link_code.used = True
    new_link = ChannelLink(
        user_id=link_code.user_id,
        channel_type=channel_type,
        channel_user_id=sender_id,
        display_name=display_name,
    )
    db.add(new_link)
    await db.commit()

    logger.info(f"Channel linked: {channel_type}:{sender_id} -> user {link_code.user_id}")

    reply = f"Account linked successfully! You can now use AICtrlNet through {channel_type}."
    await _send_reply_via_channel(channel_type, inbound, reply)
    return _format_channel_response(channel_type, {"text": reply}, inbound)


def _extract_reply_text(response: Any) -> str:
    """Extract plain text from conversation response."""
    if hasattr(response, "message"):
        return response.message.content if hasattr(response.message, "content") else str(response.message)
    if isinstance(response, dict):
        msg = response.get("message", {})
        return msg.get("content", "") if isinstance(msg, dict) else str(msg)
    return str(response)


async def _send_reply_via_channel(
    channel_type: str, inbound: InboundMessage, text: str
) -> None:
    """Send the assistant reply back through the originating channel adapter.

    Best-effort: if the adapter isn't registered or the send fails we log and
    continue — the webhook response body still carries the reply for platforms
    that support synchronous replies.
    """
    if not text:
        return

    adapter_map = {
        "telegram": ("telegram", "send_message", lambda: {"chat_id": inbound.platform_metadata.get("chat_id", inbound.sender_id), "text": text}),
        "whatsapp": ("whatsapp", "send_message", lambda: {"to": inbound.sender_id, "text": text}),
        "twilio": ("twilio", "send_sms", lambda: {"to": inbound.sender_id, "body": text}),
        "slack": ("slack", "send_message", lambda: {"channel": inbound.platform_metadata.get("channel", inbound.sender_id), "text": text}),
        "discord": ("discord", "send_message", lambda: {"channel_id": inbound.platform_metadata.get("channel_id", inbound.sender_id), "content": text}),
        "email": ("email", "send_email", lambda: {"to": inbound.sender_id, "subject": f"Re: {inbound.platform_metadata.get('subject', 'AICtrlNet')}", "body": text}),
    }

    entry = adapter_map.get(channel_type)
    if not entry:
        return

    adapter_name, capability, params_fn = entry
    try:
        adapter_class = adapter_registry.get_adapter_class(adapter_name)
        if not adapter_class:
            logger.debug(f"Adapter '{adapter_name}' not registered — skipping outbound reply")
            return

        adapter = adapter_class({})
        req = AdapterRequest(capability=capability, parameters=params_fn())
        await adapter.execute(req)
    except Exception as e:
        logger.warning(f"Failed to send reply via {channel_type}: {e}")


def _format_channel_response(
    channel_type: str, response: Any, inbound: InboundMessage
) -> Dict[str, Any]:
    """Format conversation response for the originating channel."""
    text = response.get("text", "") if isinstance(response, dict) else _extract_reply_text(response)

    if channel_type == "twilio":
        return Response(
            content=f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{text}</Message></Response>',
            media_type="application/xml",
        )

    return {"text": text, "channel": channel_type}
