"""Deliver a text message to a channel identity through its channel adapter.

Used by the channel webhook (the assistant's reply) and by the MCP
`send_channel_message` tool (a message to the caller's own linked account).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from adapters.models import AdapterRequest
from adapters.registry import adapter_registry

logger = logging.getLogger(__name__)

CHANNEL_TYPES = ("telegram", "whatsapp", "twilio", "slack", "discord", "email")


@dataclass
class Delivery:
    delivered: bool
    reason: Optional[str] = None


def _request(channel_type: str, recipient_id: str, text: str, meta: Dict[str, Any]):
    if channel_type == "telegram":
        return "telegram", "send_message", {"chat_id": meta.get("chat_id", recipient_id), "text": text}
    if channel_type == "whatsapp":
        return "whatsapp", "send_message", {"to": recipient_id, "text": text}
    if channel_type == "twilio":
        return "twilio", "send_sms", {"to": recipient_id, "body": text}
    if channel_type == "slack":
        return "slack", "send_message", {"channel": meta.get("channel", recipient_id), "text": text}
    if channel_type == "discord":
        return "discord", "send_message", {"channel_id": meta.get("channel_id", recipient_id), "content": text}
    if channel_type == "email":
        subject = meta.get("subject")
        return "email", "send_email", {
            "to": recipient_id,
            "subject": f"Re: {subject}" if subject else "AICtrlNet",
            "body": text,
        }
    return None


async def send_text(
    channel_type: str,
    recipient_id: str,
    text: str,
    platform_metadata: Optional[Dict[str, Any]] = None,
) -> Delivery:
    """Send `text` to `recipient_id` on `channel_type`. Never raises."""
    if not text:
        return Delivery(False, "empty message")
    entry = _request(channel_type, recipient_id, text, platform_metadata or {})
    if entry is None:
        return Delivery(False, f"unsupported channel '{channel_type}'")
    adapter_name, capability, params = entry
    adapter_class = adapter_registry.get_adapter_class(adapter_name)
    if not adapter_class:
        return Delivery(False, f"the {adapter_name} adapter is not registered")
    try:
        response = await adapter_class({}).execute(
            AdapterRequest(capability=capability, parameters=params)
        )
    except Exception as e:
        logger.warning(f"Failed to send via {channel_type}: {e}")
        return Delivery(False, f"{channel_type} send failed: {type(e).__name__}")
    if getattr(response, "status", "success") == "error":
        return Delivery(False, response.error or f"{channel_type} send failed")
    return Delivery(True)
