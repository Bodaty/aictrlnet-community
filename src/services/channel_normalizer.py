"""Channel normalizer — transforms platform-specific payloads into generic InboundMessage.

Stateless. One method per platform. All business logic downstream is channel-unaware.
"""

import hashlib
import hmac
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InboundMessage:
    """Generic inbound message from any channel."""
    channel_type: str
    sender_id: str
    message_text: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    external_message_id: str = ""
    platform_metadata: Dict[str, Any] = field(default_factory=dict)


class ChannelNormalizer:
    """Normalizes platform-specific webhook payloads into generic InboundMessage."""

    @staticmethod
    def normalize_slack(payload: Dict[str, Any]) -> Optional[InboundMessage]:
        """Normalize Slack Events API payload."""
        event = payload.get("event", {})
        event_type = event.get("type")

        if event_type != "message" or event.get("subtype"):
            return None  # Skip non-message events, bot messages, edits, etc.

        return InboundMessage(
            channel_type="slack",
            sender_id=event.get("user", ""),
            message_text=event.get("text", ""),
            attachments=[
                {"type": f.get("filetype", "file"), "url": f.get("url_private", ""), "name": f.get("name", "")}
                for f in event.get("files", [])
            ],
            external_message_id=event.get("ts", ""),
            platform_metadata={
                "team_id": payload.get("team_id", ""),
                "channel_id": event.get("channel", ""),
                "thread_ts": event.get("thread_ts"),
            },
        )

    @staticmethod
    def normalize_telegram(payload: Dict[str, Any]) -> Optional[InboundMessage]:
        """Normalize Telegram Bot API Update payload."""
        message = payload.get("message") or payload.get("edited_message")
        if not message:
            return None

        attachments = []
        if message.get("photo"):
            # Telegram sends multiple sizes; take the largest
            photo = message["photo"][-1]
            attachments.append({"type": "photo", "file_id": photo.get("file_id", "")})
        if message.get("document"):
            doc = message["document"]
            attachments.append({
                "type": "document",
                "file_id": doc.get("file_id", ""),
                "name": doc.get("file_name", ""),
                "mime_type": doc.get("mime_type", ""),
            })

        sender = message.get("from", {})
        chat = message.get("chat", {})

        return InboundMessage(
            channel_type="telegram",
            sender_id=str(sender.get("id", "")),
            message_text=message.get("text", "") or message.get("caption", ""),
            attachments=attachments,
            external_message_id=str(message.get("message_id", "")),
            platform_metadata={
                "chat_id": chat.get("id"),
                "chat_type": chat.get("type"),
                "username": sender.get("username"),
                "first_name": sender.get("first_name"),
            },
        )

    @staticmethod
    def normalize_whatsapp(payload: Dict[str, Any]) -> Optional[InboundMessage]:
        """Normalize Meta Cloud API (WhatsApp) webhook payload."""
        # WhatsApp webhook structure: entry[].changes[].value.messages[]
        entries = payload.get("entry", [])
        if not entries:
            return None

        changes = entries[0].get("changes", [])
        if not changes:
            return None

        value = changes[0].get("value", {})
        messages = value.get("messages", [])
        if not messages:
            return None

        msg = messages[0]
        msg_type = msg.get("type", "text")
        text = ""
        attachments = []

        if msg_type == "text":
            text = msg.get("text", {}).get("body", "")
        elif msg_type in ("image", "video", "audio", "document"):
            media = msg.get(msg_type, {})
            text = media.get("caption", "")
            attachments.append({
                "type": msg_type,
                "media_id": media.get("id", ""),
                "mime_type": media.get("mime_type", ""),
            })

        contacts = value.get("contacts", [{}])
        sender_name = contacts[0].get("profile", {}).get("name", "") if contacts else ""

        return InboundMessage(
            channel_type="whatsapp",
            sender_id=msg.get("from", ""),
            message_text=text,
            attachments=attachments,
            external_message_id=msg.get("id", ""),
            platform_metadata={
                "phone_number_id": value.get("metadata", {}).get("phone_number_id", ""),
                "display_phone": value.get("metadata", {}).get("display_phone_number", ""),
                "sender_name": sender_name,
                "timestamp": msg.get("timestamp", ""),
            },
        )

    @staticmethod
    def normalize_twilio(payload: Dict[str, Any]) -> Optional[InboundMessage]:
        """Normalize Twilio SMS/MMS webhook payload (form data converted to dict)."""
        body = payload.get("Body", "")
        sender = payload.get("From", "")

        if not sender:
            return None

        # Collect media attachments (Twilio uses MediaUrl0, MediaUrl1, etc.)
        attachments = []
        num_media = int(payload.get("NumMedia", "0"))
        for i in range(num_media):
            url = payload.get(f"MediaUrl{i}", "")
            content_type = payload.get(f"MediaContentType{i}", "")
            if url:
                attachments.append({"type": "media", "url": url, "content_type": content_type})

        return InboundMessage(
            channel_type="twilio",
            sender_id=sender,
            message_text=body,
            attachments=attachments,
            external_message_id=payload.get("MessageSid", ""),
            platform_metadata={
                "account_sid": payload.get("AccountSid", ""),
                "to": payload.get("To", ""),
                "from_city": payload.get("FromCity", ""),
                "from_country": payload.get("FromCountry", ""),
            },
        )

    @staticmethod
    def normalize_discord(payload: Dict[str, Any]) -> Optional[InboundMessage]:
        """Normalize Discord gateway message payload."""
        # Discord sends different event types; we handle MESSAGE_CREATE
        if payload.get("t") != "MESSAGE_CREATE":
            data = payload  # Direct message object
        else:
            data = payload.get("d", {})

        # Skip bot messages
        author = data.get("author", {})
        if author.get("bot"):
            return None

        attachments = [
            {"type": att.get("content_type", "file"), "url": att.get("url", ""), "name": att.get("filename", "")}
            for att in data.get("attachments", [])
        ]

        return InboundMessage(
            channel_type="discord",
            sender_id=author.get("id", ""),
            message_text=data.get("content", ""),
            attachments=attachments,
            external_message_id=data.get("id", ""),
            platform_metadata={
                "channel_id": data.get("channel_id", ""),
                "guild_id": data.get("guild_id"),
                "username": author.get("username", ""),
            },
        )

    # === Signature Validation ===

    @staticmethod
    def validate_slack_signature(body: bytes, timestamp: str, signature: str, signing_secret: str) -> bool:
        """Validate Slack request signature."""
        base = f"v0:{timestamp}:{body.decode('utf-8')}"
        computed = "v0=" + hmac.new(signing_secret.encode(), base.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(computed, signature)

    @staticmethod
    def validate_whatsapp_signature(body: bytes, signature: str, app_secret: str) -> bool:
        """Validate WhatsApp (Meta) X-Hub-Signature-256."""
        computed = hmac.new(app_secret.encode(), body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(f"sha256={computed}", signature)

    @staticmethod
    def validate_twilio_signature(url: str, params: Dict[str, str], signature: str, auth_token: str) -> bool:
        """Validate Twilio X-Twilio-Signature."""
        # Twilio signature: HMAC-SHA1 of URL + sorted POST params
        data = url
        for key in sorted(params.keys()):
            data += key + params[key]
        import base64
        computed = base64.b64encode(
            hmac.new(auth_token.encode(), data.encode(), hashlib.sha1).digest()
        ).decode()
        return hmac.compare_digest(computed, signature)

    @staticmethod
    def validate_telegram_secret(token: str, secret_token: str) -> bool:
        """Validate Telegram X-Telegram-Bot-Api-Secret-Token header."""
        return hmac.compare_digest(token, secret_token)

    @staticmethod
    def validate_discord_signature(body: bytes, signature: str, timestamp: str, public_key: str) -> bool:
        """Validate Discord webhook signature using Ed25519.

        Discord signs webhooks with Ed25519. The message to verify is
        the concatenation of the timestamp and request body.
        """
        try:
            from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
            key = Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key))
            key.verify(bytes.fromhex(signature), timestamp.encode() + body)
            return True
        except Exception:
            return False

    @staticmethod
    def validate_email_webhook(token: str, expected_token: str) -> bool:
        """Validate email inbound webhook using shared secret token."""
        if not expected_token:
            return True  # No secret configured — skip validation (dev mode)
        return hmac.compare_digest(token, expected_token)

    # === Email Normalizer ===

    @staticmethod
    def normalize_email(payload: Dict[str, Any]) -> Optional[InboundMessage]:
        """Normalize inbound email payload (SendGrid Inbound Parse format)."""
        import json as _json
        import re as _re

        from_addr = payload.get("from", "") or payload.get("sender_email", "")
        text = payload.get("text", "") or payload.get("plain", "")
        subject = payload.get("subject", "")

        # Extract just the email address from "Name <email>" format
        email_match = _re.search(r'<(.+?)>', from_addr)
        sender_email = email_match.group(1) if email_match else from_addr.strip()

        if not sender_email:
            return None

        attachments = []
        att_info = payload.get("attachment-info", {})
        if isinstance(att_info, str):
            try:
                att_info = _json.loads(att_info)
            except Exception:
                att_info = {}
        for key, info in att_info.items():
            attachments.append({
                "type": info.get("type", "file"),
                "name": info.get("filename", key),
            })

        return InboundMessage(
            channel_type="email",
            sender_id=sender_email,
            message_text=f"[Subject: {subject}]\n{text}" if subject else text,
            attachments=attachments,
            external_message_id=payload.get("headers", "")[:64],
            platform_metadata={
                "subject": subject,
                "to": payload.get("to", ""),
                "cc": payload.get("cc", ""),
                "from_display": from_addr,
                "spam_score": payload.get("spam_score", ""),
                "SPF": payload.get("SPF", ""),
                "dkim": payload.get("dkim", ""),
            },
        )
