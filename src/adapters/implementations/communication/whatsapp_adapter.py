"""WhatsApp adapter implementation using Meta Cloud API."""

import hashlib
import hmac
import logging
from typing import Any, Dict, List, Optional
import httpx
from datetime import datetime
import json

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class WhatsAppAdapter(BaseAdapter):
    """Adapter for WhatsApp Business API integration via Meta Cloud API.

    Supports sending text messages, template messages, media messages,
    read receipts, and processing inbound webhooks from Meta.

    Configuration:
        - api_key or credentials.access_token: Meta permanent/temporary access token
        - credentials.phone_number_id: WhatsApp Business phone number ID
        - credentials.app_secret: Meta app secret for webhook signature validation
        - credentials.verify_token: Webhook verification token for subscription setup
        - custom_config.api_version: Graph API version (default v18.0)
        - custom_config.discovery_only: If True, skip credential validation
    """

    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.COMMUNICATION
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None

        # API version and base URL
        self.api_version = (
            config.custom_config.get("api_version", "v18.0")
            if config.custom_config else "v18.0"
        )
        self.base_url = config.base_url or f"https://graph.facebook.com/{self.api_version}"

        # Discovery mode support
        self.discovery_only = (
            config.custom_config.get("discovery_only", False)
            if config.custom_config else False
        )

        # Authentication - access token from api_key or credentials
        self.access_token = config.api_key or (
            config.credentials.get("access_token") if config.credentials else None
        )

        # WhatsApp Business phone number ID
        self.phone_number_id = (
            config.credentials.get("phone_number_id") if config.credentials else None
        )

        # App secret for webhook signature validation
        self.app_secret = (
            config.credentials.get("app_secret") if config.credentials else None
        )

        # Verify token for webhook subscription
        self.verify_token = (
            config.credentials.get("verify_token") if config.credentials else None
        )

        # Business account info populated on initialize
        self.business_account_info: Optional[Dict[str, Any]] = None

        # Validate credentials outside discovery mode
        if not self.discovery_only:
            if not self.access_token:
                raise ValueError(
                    "WhatsApp access token is required. "
                    "Provide via api_key or credentials.access_token"
                )
            if not self.phone_number_id:
                raise ValueError(
                    "WhatsApp phone_number_id is required in credentials"
                )

    async def initialize(self) -> None:
        """Initialize the WhatsApp adapter and verify access token."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("WhatsApp adapter initialized in discovery mode")
            return

        # Create HTTP client with auth header
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            timeout=self.config.timeout_seconds,
        )

        # Verify access token by fetching phone number details
        try:
            response = await self.client.get(
                f"/{self.phone_number_id}",
                params={"fields": "verified_name,code_verification_status,display_phone_number,quality_rating,id"}
            )
            response.raise_for_status()
            result = response.json()

            self.business_account_info = {
                "phone_number_id": result.get("id"),
                "verified_name": result.get("verified_name"),
                "display_phone_number": result.get("display_phone_number"),
                "quality_rating": result.get("quality_rating"),
                "code_verification_status": result.get("code_verification_status"),
            }

            logger.info(
                f"WhatsApp adapter initialized. "
                f"Business name: {self.business_account_info['verified_name']}, "
                f"Phone: {self.business_account_info['display_phone_number']}"
            )
        except httpx.HTTPStatusError as e:
            error_body = e.response.text
            logger.error(
                f"Failed to verify WhatsApp access token: "
                f"HTTP {e.response.status_code} - {error_body}"
            )
            raise ValueError(
                f"WhatsApp API authentication failed: {e.response.status_code}. "
                f"Check your access_token and phone_number_id."
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize WhatsApp adapter: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the adapter and close the HTTP client."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("WhatsApp adapter shutdown")

    def get_capabilities(self) -> List[AdapterCapability]:
        """Return WhatsApp adapter capabilities."""
        return [
            AdapterCapability(
                name="send_message",
                description="Send a text message to a WhatsApp user",
                category="messaging",
                parameters={
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number in E.164 format (e.g., +1234567890)"
                    },
                    "text": {
                        "type": "string",
                        "description": "Message text body"
                    },
                    "preview_url": {
                        "type": "boolean",
                        "description": "Enable URL preview in message",
                        "default": False
                    },
                    "reply_to": {
                        "type": "string",
                        "description": "Message ID to reply to (context)"
                    },
                },
                required_parameters=["to", "text"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="send_template",
                description=(
                    "Send a pre-approved template message. "
                    "Required to initiate conversations outside the 24-hour window."
                ),
                category="messaging",
                parameters={
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number in E.164 format"
                    },
                    "template_name": {
                        "type": "string",
                        "description": "Name of the approved message template"
                    },
                    "language_code": {
                        "type": "string",
                        "description": "Template language code (e.g., en_US)",
                        "default": "en_US"
                    },
                    "components": {
                        "type": "array",
                        "description": (
                            "Template components with parameter values "
                            "(header, body, button parameters)"
                        )
                    },
                },
                required_parameters=["to", "template_name"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="send_media",
                description="Send an image, document, video, or audio message",
                category="messaging",
                parameters={
                    "to": {
                        "type": "string",
                        "description": "Recipient phone number in E.164 format"
                    },
                    "media_type": {
                        "type": "string",
                        "description": "Type of media: image, document, video, audio, sticker"
                    },
                    "media_url": {
                        "type": "string",
                        "description": "Public URL of the media file"
                    },
                    "media_id": {
                        "type": "string",
                        "description": "WhatsApp media ID (alternative to media_url)"
                    },
                    "caption": {
                        "type": "string",
                        "description": "Caption for image, video, or document"
                    },
                    "filename": {
                        "type": "string",
                        "description": "Filename for document type"
                    },
                    "reply_to": {
                        "type": "string",
                        "description": "Message ID to reply to (context)"
                    },
                },
                required_parameters=["to", "media_type"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.0,
            ),
            AdapterCapability(
                name="mark_read",
                description="Mark a message as read (sends blue check marks to the sender)",
                category="messaging",
                parameters={
                    "message_id": {
                        "type": "string",
                        "description": "ID of the message to mark as read"
                    },
                },
                required_parameters=["message_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0,
            ),
        ]

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to the WhatsApp Cloud API."""
        # Validate request
        self.validate_request(request)

        # Route to appropriate handler
        handlers = {
            "send_message": self._handle_send_message,
            "send_template": self._handle_send_template,
            "send_media": self._handle_send_media,
            "mark_read": self._handle_mark_read,
        }

        handler = handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")

        return await handler(request)

    # -------------------------------------------------------------------------
    # Capability handlers
    # -------------------------------------------------------------------------

    async def _handle_send_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a text message."""
        start_time = datetime.utcnow()

        try:
            to = request.parameters["to"]
            text = request.parameters["text"]
            preview_url = request.parameters.get("preview_url", False)

            payload: Dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "text",
                "text": {
                    "preview_url": preview_url,
                    "body": text,
                },
            }

            # Add context for reply
            if "reply_to" in request.parameters:
                payload["context"] = {
                    "message_id": request.parameters["reply_to"]
                }

            response = await self.client.post(
                f"/{self.phone_number_id}/messages",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Extract message ID from response
            message_id = None
            messages = result.get("messages", [])
            if messages:
                message_id = messages[0].get("id")

            # Publish event
            await event_bus.publish(
                "adapter.whatsapp.message_sent",
                {
                    "to": to,
                    "message_id": message_id,
                    "type": "text",
                },
                source_id=self.id,
                source_type="adapter",
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "messaging_product": result.get("messaging_product"),
                    "contacts": result.get("contacts", []),
                    "messages": result.get("messages", []),
                    "message_id": message_id,
                },
                duration_ms=duration_ms,
                metadata={"wa_api_version": self.api_version},
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _handle_send_template(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a template message.

        Template messages are required to initiate conversations outside the
        24-hour messaging window. Templates must be pre-approved by Meta.
        """
        start_time = datetime.utcnow()

        try:
            to = request.parameters["to"]
            template_name = request.parameters["template_name"]
            language_code = request.parameters.get("language_code", "en_US")
            components = request.parameters.get("components", [])

            template_object: Dict[str, Any] = {
                "name": template_name,
                "language": {
                    "code": language_code,
                },
            }

            if components:
                template_object["components"] = components

            payload: Dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": "template",
                "template": template_object,
            }

            response = await self.client.post(
                f"/{self.phone_number_id}/messages",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            message_id = None
            messages = result.get("messages", [])
            if messages:
                message_id = messages[0].get("id")

            # Publish event
            await event_bus.publish(
                "adapter.whatsapp.template_sent",
                {
                    "to": to,
                    "message_id": message_id,
                    "template_name": template_name,
                    "language_code": language_code,
                },
                source_id=self.id,
                source_type="adapter",
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "messaging_product": result.get("messaging_product"),
                    "contacts": result.get("contacts", []),
                    "messages": result.get("messages", []),
                    "message_id": message_id,
                    "template_name": template_name,
                },
                duration_ms=duration_ms,
                metadata={
                    "wa_api_version": self.api_version,
                    "template_language": language_code,
                },
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _handle_send_media(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a media message (image, document, video, audio, sticker)."""
        start_time = datetime.utcnow()

        try:
            to = request.parameters["to"]
            media_type = request.parameters["media_type"]

            # Validate media type
            supported_types = {"image", "document", "video", "audio", "sticker"}
            if media_type not in supported_types:
                raise ValueError(
                    f"Unsupported media_type '{media_type}'. "
                    f"Must be one of: {', '.join(sorted(supported_types))}"
                )

            # Build media object - either by URL or by media ID
            media_object: Dict[str, Any] = {}

            if "media_id" in request.parameters:
                media_object["id"] = request.parameters["media_id"]
            elif "media_url" in request.parameters:
                media_object["link"] = request.parameters["media_url"]
            else:
                raise ValueError(
                    "Either media_url or media_id must be provided"
                )

            # Add caption where supported (image, video, document)
            if "caption" in request.parameters and media_type in {"image", "video", "document"}:
                media_object["caption"] = request.parameters["caption"]

            # Add filename for documents
            if "filename" in request.parameters and media_type == "document":
                media_object["filename"] = request.parameters["filename"]

            payload: Dict[str, Any] = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to,
                "type": media_type,
                media_type: media_object,
            }

            # Add context for reply
            if "reply_to" in request.parameters:
                payload["context"] = {
                    "message_id": request.parameters["reply_to"]
                }

            response = await self.client.post(
                f"/{self.phone_number_id}/messages",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            message_id = None
            messages = result.get("messages", [])
            if messages:
                message_id = messages[0].get("id")

            # Publish event
            await event_bus.publish(
                "adapter.whatsapp.media_sent",
                {
                    "to": to,
                    "message_id": message_id,
                    "media_type": media_type,
                },
                source_id=self.id,
                source_type="adapter",
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "messaging_product": result.get("messaging_product"),
                    "contacts": result.get("contacts", []),
                    "messages": result.get("messages", []),
                    "message_id": message_id,
                    "media_type": media_type,
                },
                duration_ms=duration_ms,
                metadata={"wa_api_version": self.api_version},
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    async def _handle_mark_read(self, request: AdapterRequest) -> AdapterResponse:
        """Handle marking a message as read."""
        start_time = datetime.utcnow()

        try:
            message_id = request.parameters["message_id"]

            payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id,
            }

            response = await self.client.post(
                f"/{self.phone_number_id}/messages",
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "success": result.get("success", True),
                    "message_id": message_id,
                },
                duration_ms=duration_ms,
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000,
            )

    # -------------------------------------------------------------------------
    # Webhook handling
    # -------------------------------------------------------------------------

    async def process_webhook(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Process an inbound Meta webhook payload for WhatsApp messages.

        Meta sends webhook notifications to your callback URL when events
        occur (message received, status update, etc.). This method parses
        the payload and publishes appropriate events on the event bus.

        Args:
            payload: The JSON body from the Meta webhook POST request.

        Returns:
            A dict summarizing the processed events.

        Webhook payload structure (Meta Cloud API):
            {
                "object": "whatsapp_business_account",
                "entry": [
                    {
                        "id": "<WHATSAPP_BUSINESS_ACCOUNT_ID>",
                        "changes": [
                            {
                                "field": "messages",
                                "value": {
                                    "messaging_product": "whatsapp",
                                    "metadata": {
                                        "display_phone_number": "...",
                                        "phone_number_id": "..."
                                    },
                                    "contacts": [...],
                                    "messages": [...],
                                    "statuses": [...]
                                }
                            }
                        ]
                    }
                ]
            }
        """
        processed_events: List[Dict[str, Any]] = []

        if payload.get("object") != "whatsapp_business_account":
            logger.warning(
                f"Ignoring webhook with unexpected object type: "
                f"{payload.get('object')}"
            )
            return {"processed": 0, "events": []}

        entries = payload.get("entry", [])

        for entry in entries:
            business_account_id = entry.get("id")
            changes = entry.get("changes", [])

            for change in changes:
                if change.get("field") != "messages":
                    continue

                value = change.get("value", {})
                wa_metadata = value.get("metadata", {})

                # Process inbound messages
                messages = value.get("messages", [])
                contacts = value.get("contacts", [])

                # Build contact lookup
                contact_map = {}
                for contact in contacts:
                    wa_id = contact.get("wa_id")
                    if wa_id:
                        contact_map[wa_id] = {
                            "wa_id": wa_id,
                            "name": contact.get("profile", {}).get("name"),
                        }

                for message in messages:
                    event_data = self._parse_inbound_message(
                        message, contact_map, wa_metadata, business_account_id
                    )

                    if event_data:
                        await event_bus.publish(
                            "adapter.whatsapp.message_received",
                            event_data,
                            source_id=self.id,
                            source_type="adapter",
                        )
                        processed_events.append(event_data)

                # Process status updates (sent, delivered, read, failed)
                statuses = value.get("statuses", [])

                for status in statuses:
                    status_data = {
                        "message_id": status.get("id"),
                        "status": status.get("status"),
                        "timestamp": status.get("timestamp"),
                        "recipient_id": status.get("recipient_id"),
                        "business_account_id": business_account_id,
                        "phone_number_id": wa_metadata.get("phone_number_id"),
                    }

                    # Include error details if present
                    errors = status.get("errors", [])
                    if errors:
                        status_data["errors"] = [
                            {
                                "code": err.get("code"),
                                "title": err.get("title"),
                                "message": err.get("message"),
                                "error_data": err.get("error_data"),
                            }
                            for err in errors
                        ]

                    # Include conversation info if present
                    conversation = status.get("conversation", {})
                    if conversation:
                        status_data["conversation"] = {
                            "id": conversation.get("id"),
                            "origin_type": conversation.get("origin", {}).get("type"),
                            "expiration_timestamp": conversation.get("expiration_timestamp"),
                        }

                    # Include pricing info if present
                    pricing = status.get("pricing", {})
                    if pricing:
                        status_data["pricing"] = {
                            "billable": pricing.get("billable"),
                            "pricing_model": pricing.get("pricing_model"),
                            "category": pricing.get("category"),
                        }

                    await event_bus.publish(
                        "adapter.whatsapp.status_update",
                        status_data,
                        source_id=self.id,
                        source_type="adapter",
                    )
                    processed_events.append(status_data)

        return {
            "processed": len(processed_events),
            "events": processed_events,
        }

    def _parse_inbound_message(
        self,
        message: Dict[str, Any],
        contact_map: Dict[str, Dict[str, Any]],
        wa_metadata: Dict[str, Any],
        business_account_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Parse a single inbound message from the webhook payload.

        Args:
            message: The message object from the webhook.
            contact_map: Mapping of wa_id to contact info.
            wa_metadata: Metadata from the webhook value object.
            business_account_id: The WhatsApp Business Account ID.

        Returns:
            Parsed event data dict, or None if parsing fails.
        """
        try:
            msg_from = message.get("from")
            msg_id = message.get("id")
            msg_timestamp = message.get("timestamp")
            msg_type = message.get("type")

            event_data: Dict[str, Any] = {
                "message_id": msg_id,
                "from": msg_from,
                "timestamp": msg_timestamp,
                "type": msg_type,
                "business_account_id": business_account_id,
                "phone_number_id": wa_metadata.get("phone_number_id"),
                "display_phone_number": wa_metadata.get("display_phone_number"),
            }

            # Add contact info if available
            contact = contact_map.get(msg_from)
            if contact:
                event_data["contact_name"] = contact.get("name")

            # Add context (if this is a reply)
            context = message.get("context")
            if context:
                event_data["context"] = {
                    "message_id": context.get("id"),
                    "from": context.get("from"),
                }

            # Parse message content based on type
            if msg_type == "text":
                text_obj = message.get("text", {})
                event_data["text"] = text_obj.get("body")

            elif msg_type == "image":
                image_obj = message.get("image", {})
                event_data["media"] = {
                    "media_id": image_obj.get("id"),
                    "mime_type": image_obj.get("mime_type"),
                    "sha256": image_obj.get("sha256"),
                    "caption": image_obj.get("caption"),
                }

            elif msg_type == "video":
                video_obj = message.get("video", {})
                event_data["media"] = {
                    "media_id": video_obj.get("id"),
                    "mime_type": video_obj.get("mime_type"),
                    "sha256": video_obj.get("sha256"),
                    "caption": video_obj.get("caption"),
                }

            elif msg_type == "audio":
                audio_obj = message.get("audio", {})
                event_data["media"] = {
                    "media_id": audio_obj.get("id"),
                    "mime_type": audio_obj.get("mime_type"),
                    "sha256": audio_obj.get("sha256"),
                    "voice": audio_obj.get("voice", False),
                }

            elif msg_type == "document":
                doc_obj = message.get("document", {})
                event_data["media"] = {
                    "media_id": doc_obj.get("id"),
                    "mime_type": doc_obj.get("mime_type"),
                    "sha256": doc_obj.get("sha256"),
                    "filename": doc_obj.get("filename"),
                    "caption": doc_obj.get("caption"),
                }

            elif msg_type == "sticker":
                sticker_obj = message.get("sticker", {})
                event_data["media"] = {
                    "media_id": sticker_obj.get("id"),
                    "mime_type": sticker_obj.get("mime_type"),
                    "sha256": sticker_obj.get("sha256"),
                    "animated": sticker_obj.get("animated", False),
                }

            elif msg_type == "location":
                location_obj = message.get("location", {})
                event_data["location"] = {
                    "latitude": location_obj.get("latitude"),
                    "longitude": location_obj.get("longitude"),
                    "name": location_obj.get("name"),
                    "address": location_obj.get("address"),
                }

            elif msg_type == "contacts":
                event_data["contacts"] = message.get("contacts", [])

            elif msg_type == "interactive":
                interactive_obj = message.get("interactive", {})
                interactive_type = interactive_obj.get("type")

                if interactive_type == "button_reply":
                    button_reply = interactive_obj.get("button_reply", {})
                    event_data["interactive"] = {
                        "type": "button_reply",
                        "button_id": button_reply.get("id"),
                        "button_title": button_reply.get("title"),
                    }
                elif interactive_type == "list_reply":
                    list_reply = interactive_obj.get("list_reply", {})
                    event_data["interactive"] = {
                        "type": "list_reply",
                        "list_id": list_reply.get("id"),
                        "list_title": list_reply.get("title"),
                        "list_description": list_reply.get("description"),
                    }
                else:
                    event_data["interactive"] = {
                        "type": interactive_type,
                        "raw": interactive_obj,
                    }

            elif msg_type == "reaction":
                reaction_obj = message.get("reaction", {})
                event_data["reaction"] = {
                    "message_id": reaction_obj.get("message_id"),
                    "emoji": reaction_obj.get("emoji"),
                }

            elif msg_type == "button":
                button_obj = message.get("button", {})
                event_data["button"] = {
                    "text": button_obj.get("text"),
                    "payload": button_obj.get("payload"),
                }

            elif msg_type == "order":
                order_obj = message.get("order", {})
                event_data["order"] = {
                    "catalog_id": order_obj.get("catalog_id"),
                    "product_items": order_obj.get("product_items", []),
                    "text": order_obj.get("text"),
                }

            elif msg_type == "system":
                system_obj = message.get("system", {})
                event_data["system"] = {
                    "body": system_obj.get("body"),
                    "identity": system_obj.get("identity"),
                    "new_wa_id": system_obj.get("new_wa_id"),
                    "type": system_obj.get("type"),
                }

            else:
                # Unknown message type - preserve raw data
                event_data["raw_message"] = message
                logger.warning(f"Unknown WhatsApp message type: {msg_type}")

            return event_data

        except Exception as e:
            logger.error(f"Failed to parse inbound WhatsApp message: {str(e)}")
            return None

    # -------------------------------------------------------------------------
    # Webhook signature validation
    # -------------------------------------------------------------------------

    def validate_webhook_signature(
        self, payload_body: bytes, signature_header: str
    ) -> bool:
        """Validate X-Hub-Signature-256 from Meta webhook requests.

        Meta signs webhook payloads with HMAC-SHA256 using the app secret.
        This method verifies that signature to ensure the request is authentic.

        Args:
            payload_body: The raw request body bytes.
            signature_header: The value of the X-Hub-Signature-256 header
                              (e.g., "sha256=abc123...").

        Returns:
            True if the signature is valid, False otherwise.
        """
        if not self.app_secret:
            logger.warning(
                "Cannot validate webhook signature: app_secret not configured. "
                "Set credentials.app_secret for production use."
            )
            return False

        if not signature_header:
            logger.warning("Missing X-Hub-Signature-256 header")
            return False

        # The header format is "sha256=<hex_digest>"
        if not signature_header.startswith("sha256="):
            logger.warning(
                f"Invalid signature header format: {signature_header[:20]}..."
            )
            return False

        expected_signature = signature_header[7:]  # Strip "sha256=" prefix

        # Compute HMAC-SHA256
        computed_signature = hmac.new(
            key=self.app_secret.encode("utf-8"),
            msg=payload_body,
            digestmod=hashlib.sha256,
        ).hexdigest()

        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(computed_signature, expected_signature)

    def verify_webhook_subscription(
        self, mode: str, token: str, challenge: str
    ) -> Optional[str]:
        """Verify a webhook subscription request from Meta.

        When you configure a webhook URL in Meta, it sends a GET request
        with hub.mode, hub.verify_token, and hub.challenge parameters.

        Args:
            mode: The hub.mode parameter (should be "subscribe").
            token: The hub.verify_token parameter.
            challenge: The hub.challenge parameter.

        Returns:
            The challenge string if verification succeeds, None otherwise.
        """
        if not self.verify_token:
            logger.warning(
                "Cannot verify webhook subscription: verify_token not configured. "
                "Set credentials.verify_token."
            )
            return None

        if mode == "subscribe" and token == self.verify_token:
            logger.info("Webhook subscription verified successfully")
            return challenge

        logger.warning(
            f"Webhook verification failed. mode={mode}, "
            f"token_match={token == self.verify_token}"
        )
        return None

    # -------------------------------------------------------------------------
    # Health check
    # -------------------------------------------------------------------------

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform WhatsApp-specific health check.

        Queries the phone number endpoint to verify the access token
        is still valid and the phone number is operational.
        """
        try:
            if self.discovery_only:
                return {"status": "healthy", "mode": "discovery"}

            if not self.client:
                return {"status": "unhealthy", "error": "HTTP client not initialized"}

            response = await self.client.get(
                f"/{self.phone_number_id}",
                params={"fields": "verified_name,quality_rating,display_phone_number"}
            )
            response.raise_for_status()
            result = response.json()

            return {
                "status": "healthy",
                "verified_name": result.get("verified_name"),
                "display_phone_number": result.get("display_phone_number"),
                "quality_rating": result.get("quality_rating"),
                "api_version": self.api_version,
            }

        except httpx.HTTPStatusError as e:
            return {
                "status": "unhealthy",
                "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
            }
