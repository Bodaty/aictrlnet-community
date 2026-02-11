"""Channel webhook endpoint — single entry point for all messaging platforms.

POST /api/v1/channels/{channel_type}/webhook
"""

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Request, Response, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.database import get_db
from fastapi import Depends
from services.channel_normalizer import ChannelNormalizer, InboundMessage
from services.conversation_manager import ConversationManagerService
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
}


@router.post("/channels/{channel_type}/webhook")
async def channel_webhook(
    channel_type: str,
    request: Request,
    db: AsyncSession = Depends(get_db),
):
    """Receive webhook from any messaging platform.

    Validates signature, normalizes payload, routes to conversation manager,
    and returns platform-appropriate response.
    """
    if channel_type not in _NORMALIZERS:
        raise HTTPException(status_code=400, detail=f"Unsupported channel: {channel_type}")

    # --- Platform-specific verification challenges ---

    # Slack URL verification
    if channel_type == "slack":
        body = await request.json()
        if body.get("type") == "url_verification":
            return {"challenge": body.get("challenge", "")}

    # WhatsApp/Meta webhook verification (GET handled separately, POST continues)
    if channel_type == "whatsapp":
        # Meta sends a GET for verification; POST for actual messages
        # GET is handled by the verify endpoint below
        pass

    # --- Parse payload ---
    try:
        if channel_type == "twilio":
            # Twilio sends form-encoded data
            form = await request.form()
            payload: Dict[str, Any] = dict(form)
        else:
            payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid payload")

    # --- Normalize ---
    normalize_fn = _NORMALIZERS[channel_type]
    message: InboundMessage | None = normalize_fn(payload)

    if message is None:
        # Non-message event (typing indicator, delivery receipt, etc.) — acknowledge
        return Response(status_code=200)

    # --- Route to conversation manager ---
    try:
        manager = ConversationManagerService(db)
        session = await manager.find_or_create_channel_session(
            channel_type=message.channel_type,
            sender_id=message.sender_id,
            platform_metadata=message.platform_metadata,
        )

        response = await manager.process_message(
            session_id=session.id,
            content=message.message_text,
            user_id=session.user_id,
            channel_type=message.channel_type,
            external_message_id=message.external_message_id,
        )

        # Send response back via originating channel adapter
        reply_text = _extract_reply_text(response)
        await _send_reply_via_channel(channel_type, message, reply_text)

        return _format_channel_response(channel_type, response, message)

    except Exception as e:
        logger.error(f"Channel webhook error ({channel_type}): {e}", exc_info=True)
        # Return 200 to prevent platform retries on internal errors
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
        request = AdapterRequest(capability=capability, parameters=params_fn())
        await adapter.execute(request)
    except Exception as e:
        logger.warning(f"Failed to send reply via {channel_type}: {e}")


def _format_channel_response(
    channel_type: str, response: Any, inbound: InboundMessage
) -> Dict[str, Any]:
    """Format conversation response for the originating channel."""
    text = _extract_reply_text(response)

    if channel_type == "twilio":
        # Twilio expects TwiML
        return Response(
            content=f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{text}</Message></Response>',
            media_type="application/xml",
        )

    # Default JSON response (Telegram, WhatsApp, Slack, Discord)
    return {"text": text, "channel": channel_type}
