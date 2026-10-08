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
from xml.sax.saxutils import escape as xml_escape

from fastapi import APIRouter, Request, Response, HTTPException, Depends
from sqlalchemy import and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.database import bind_session_tenant, get_db, get_session_maker
from core.tenant_context import set_current_tenant_id
from services import channel_outbound
from services.channel_normalizer import ChannelNormalizer, InboundMessage
from services.conversation_service_registry import get_conversation_service_class
from services.conversation_turn_runner import TurnFailed, run_turn
from models.channel_link import ChannelLink, ChannelLinkCode
from models.user import User

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

_TURN_FAILED_REPLY = "Sorry, I couldn't complete that. Please try again in a moment."

# Regex to detect a linking command: "/link 123456" or "link 123456"
_LINK_PATTERN = re.compile(r"^/?link\s+(\d{6})$", re.IGNORECASE)

# Per-channel signing-secret setting names. In a deploy environment the relevant
# secret MUST be set, otherwise inbound webhooks are unauthenticated and anyone
# could forge messages — so we fail closed rather than fall through to the
# "if secret:" verification branches below (which silently skip when unset).
_CHANNEL_SECRET_ATTR = {
    "slack": "SLACK_SIGNING_SECRET",
    "discord": "DISCORD_PUBLIC_KEY",
    "email": "EMAIL_WEBHOOK_SECRET",
    "whatsapp": "WHATSAPP_APP_SECRET",
    "telegram": "TELEGRAM_SECRET_TOKEN",
    "twilio": "TWILIO_AUTH_TOKEN",
}
_DEPLOY_ENVIRONMENTS = {"production", "prod", "staging", "stage"}


def _is_deploy_env() -> bool:
    return (getattr(settings, "ENVIRONMENT", "") or "").strip().lower() in _DEPLOY_ENVIRONMENTS


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
    4. If authenticated, runs a v5 turn as the linked user, in their tenant.
    5. If NOT authenticated, replies with "please link your account" instructions.
    """
    if channel_type not in _NORMALIZERS:
        raise HTTPException(status_code=400, detail=f"Unsupported channel: {channel_type}")

    # Fail closed in deploy environments: no signing secret => cannot verify
    # authenticity => refuse the webhook. Development keeps the permissive path
    # so local testing without secrets still works.
    secret_attr = _CHANNEL_SECRET_ATTR.get(channel_type)
    if _is_deploy_env() and secret_attr and not getattr(settings, secret_attr, ""):
        logger.error(
            "Rejecting %s webhook: %s is not configured in a deploy environment",
            channel_type, secret_attr,
        )
        raise HTTPException(status_code=503, detail="Channel webhook signing secret not configured")

    # --- Platform-specific verification challenges ---

    # Slack signature verification + URL verification
    if channel_type == "slack":
        raw_body = await request.body()
        slack_secret = getattr(settings, "SLACK_SIGNING_SECRET", "")
        if slack_secret:
            sig = request.headers.get("X-Slack-Signature", "")
            ts = request.headers.get("X-Slack-Request-Timestamp", "")
            # Replay protection: reject stale timestamps (Slack guidance: 5 min).
            import time as _time
            try:
                if abs(_time.time() - int(ts)) > 300:
                    raise HTTPException(status_code=401, detail="Slack request timestamp too old")
            except (ValueError, TypeError):
                raise HTTPException(status_code=401, detail="Invalid Slack timestamp")
            if not normalizer.validate_slack_signature(raw_body, ts, sig, slack_secret):
                raise HTTPException(status_code=401, detail="Invalid Slack signature")
        try:
            body = json.loads(raw_body) if raw_body else {}
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid payload")
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

    # WhatsApp (Meta) X-Hub-Signature-256 validation
    if channel_type == "whatsapp":
        wa_secret = getattr(settings, "WHATSAPP_APP_SECRET", "")
        if wa_secret:
            raw_body = await request.body()
            sig = request.headers.get("X-Hub-Signature-256", "")
            if not normalizer.validate_whatsapp_signature(raw_body, sig, wa_secret):
                raise HTTPException(status_code=401, detail="Invalid WhatsApp signature")

    # Telegram secret-token header validation
    if channel_type == "telegram":
        tg_secret = getattr(settings, "TELEGRAM_SECRET_TOKEN", "")
        if tg_secret:
            token = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if not normalizer.validate_telegram_secret(token, tg_secret):
                raise HTTPException(status_code=401, detail="Invalid Telegram secret token")

    # Twilio X-Twilio-Signature validation (HMAC over the full URL + POST params).
    # Reconstruct the externally-visible URL via X-Forwarded-Proto so verification
    # works behind the Cloud Run / nginx TLS terminator.
    if channel_type == "twilio":
        tw_token = getattr(settings, "TWILIO_AUTH_TOKEN", "")
        if tw_token:
            form = await request.form()
            params = {k: str(v) for k, v in form.items()}
            sig = request.headers.get("X-Twilio-Signature", "")
            proto = request.headers.get("X-Forwarded-Proto", request.url.scheme)
            url = str(request.url.replace(scheme=proto))
            if not normalizer.validate_twilio_signature(url, params, sig, tw_token):
                raise HTTPException(status_code=401, detail="Invalid Twilio signature")

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

    # --- Run a v5 turn as the linked user, in the linked user's tenant ---
    user_id = str(link.user_id)
    tenant_id = await _user_tenant(user_id)
    if tenant_id:
        # The webhook is unauthenticated, so the request started in the default
        # tenant; everything the turn does (tools, jobs) must run in the user's.
        await db.rollback()
        set_current_tenant_id(tenant_id)
        bind_session_tenant(db, tenant_id)
    try:
        service = get_conversation_service_class()(db)
        session = await service.find_or_create_channel_session(
            channel_type=message.channel_type,
            sender_id=message.sender_id,
            user_id=user_id,
            platform_metadata=message.platform_metadata,
        )
        result = await run_turn(
            service, session.id, message.message_text, user_id,
            channel_type=message.channel_type,
            external_message_id=message.external_message_id,
        )
        reply_text = result.get("content") or ""
    except TurnFailed as e:
        logger.warning(f"Channel turn failed ({channel_type}): {e}")
        reply_text = _TURN_FAILED_REPLY
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Channel webhook error ({channel_type}): {e}", exc_info=True)
        reply_text = _TURN_FAILED_REPLY

    await _send_reply_via_channel(channel_type, message, reply_text)
    return _format_channel_response(channel_type, {"text": reply_text}, message)


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
    # Throttle 6-digit code guessing per channel identity: 1e6 space, so cap
    # attempts hard (per-process window if Redis is down). Without this an attacker could
    # brute-force a valid, unexpired code from a single channel account.
    from core.rate_limit import enforce_rate_limit
    await enforce_rate_limit(
        "channel_link", f"{channel_type}:{sender_id}", limit=8, window_seconds=900,
        detail="Too many linking attempts. Please wait and generate a fresh code.",
    )

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


async def _user_tenant(user_id: str) -> Optional[str]:
    """The linked user's tenant. `users` is tenant-isolated and the request has
    no tenant yet, so this one read runs with the admin RLS bypass."""
    async with get_session_maker()() as admin:
        async with admin.begin():
            await admin.execute(text("SET LOCAL app.is_admin = 'true'"))
            return (await admin.execute(
                select(User.tenant_id).where(User.id == user_id)
            )).scalar_one_or_none()


async def _send_reply_via_channel(
    channel_type: str, inbound: InboundMessage, text: str
) -> None:
    """Send the assistant reply back through the originating channel adapter.

    Best-effort: the webhook response body still carries the reply for platforms
    that support synchronous replies. Twilio gets its reply in the TwiML body
    only — sending it through the adapter as well would deliver it twice.
    """
    if channel_type == "twilio":
        return
    delivery = await channel_outbound.send_text(
        channel_type, inbound.sender_id, text, inbound.platform_metadata
    )
    if not delivery.delivered:
        logger.debug(f"Outbound {channel_type} reply not sent: {delivery.reason}")


def _format_channel_response(
    channel_type: str, response: Any, inbound: InboundMessage
) -> Dict[str, Any]:
    """Format conversation response for the originating channel."""
    text = response.get("text", "")

    if channel_type == "twilio":
        return Response(
            content=f'<?xml version="1.0" encoding="UTF-8"?><Response><Message>{xml_escape(text)}</Message></Response>',
            media_type="application/xml",
        )

    return {"text": text, "channel": channel_type}
