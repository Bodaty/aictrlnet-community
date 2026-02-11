"""Telegram Bot API adapter implementation."""

import asyncio
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


class TelegramAdapter(BaseAdapter):
    """Adapter for Telegram Bot API integration."""

    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.COMMUNICATION
        super().__init__(config)

        self.client: Optional[httpx.AsyncClient] = None

        # Check for discovery mode
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False

        # Bot token from api_key or credentials
        self.bot_token = config.api_key or (config.credentials.get("bot_token") if config.credentials else None)

        # Skip validation in discovery mode
        if not self.discovery_only and not self.bot_token:
            raise ValueError("Telegram bot token is required")

        # Build base URL for Telegram Bot API
        if self.bot_token:
            self.base_url = config.base_url or f"https://api.telegram.org/bot{self.bot_token}"
        else:
            self.base_url = None

        # Bot info populated during initialize()
        self.bot_info: Optional[Dict[str, Any]] = None

    async def initialize(self) -> None:
        """Initialize the Telegram adapter."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("Telegram adapter initialized in discovery mode")
            return

        # Create HTTP client (no auth header needed -- token is in the URL)
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.config.timeout_seconds
        )

        # Validate bot token with getMe
        try:
            response = await self.client.get("/getMe")
            response.raise_for_status()
            result = response.json()

            if not result.get("ok"):
                raise ValueError(f"Telegram getMe failed: {result.get('description', 'Unknown error')}")

            bot_data = result.get("result", {})
            self.bot_info = {
                "id": bot_data.get("id"),
                "is_bot": bot_data.get("is_bot"),
                "first_name": bot_data.get("first_name"),
                "username": bot_data.get("username"),
                "can_join_groups": bot_data.get("can_join_groups"),
                "can_read_all_group_messages": bot_data.get("can_read_all_group_messages"),
                "supports_inline_queries": bot_data.get("supports_inline_queries")
            }

            logger.info(f"Telegram adapter initialized for bot: @{self.bot_info['username']}")
        except Exception as e:
            logger.error(f"Failed to initialize Telegram adapter: {str(e)}")
            raise

    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Telegram adapter shutdown")

    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Telegram adapter capabilities."""
        capabilities = [
            AdapterCapability(
                name="send_message",
                description="Send a text message to a Telegram chat",
                category="messaging",
                parameters={
                    "chat_id": {"type": "string", "description": "Unique identifier for the target chat or @channelusername"},
                    "text": {"type": "string", "description": "Text of the message to be sent (1-4096 characters)"},
                    "parse_mode": {"type": "string", "description": "Mode for parsing entities: MarkdownV2, HTML, or Markdown"},
                    "reply_to_message_id": {"type": "integer", "description": "If the message is a reply, ID of the original message"},
                    "disable_web_page_preview": {"type": "boolean", "description": "Disables link previews for links in this message"},
                    "disable_notification": {"type": "boolean", "description": "Sends the message silently"}
                },
                required_parameters=["chat_id", "text"],
                async_supported=True,
                estimated_duration_seconds=0.5
            ),
            AdapterCapability(
                name="send_photo",
                description="Send a photo to a Telegram chat",
                category="messaging",
                parameters={
                    "chat_id": {"type": "string", "description": "Unique identifier for the target chat or @channelusername"},
                    "photo": {"type": "string", "description": "Photo URL or file_id to send"},
                    "caption": {"type": "string", "description": "Photo caption (0-1024 characters)"},
                    "parse_mode": {"type": "string", "description": "Mode for parsing entities in the caption"},
                    "reply_to_message_id": {"type": "integer", "description": "If the message is a reply, ID of the original message"},
                    "disable_notification": {"type": "boolean", "description": "Sends the message silently"}
                },
                required_parameters=["chat_id", "photo"],
                async_supported=True,
                estimated_duration_seconds=1.0
            ),
            AdapterCapability(
                name="send_document",
                description="Send a document to a Telegram chat",
                category="messaging",
                parameters={
                    "chat_id": {"type": "string", "description": "Unique identifier for the target chat or @channelusername"},
                    "document": {"type": "string", "description": "Document URL or file_id to send"},
                    "caption": {"type": "string", "description": "Document caption (0-1024 characters)"},
                    "parse_mode": {"type": "string", "description": "Mode for parsing entities in the caption"},
                    "reply_to_message_id": {"type": "integer", "description": "If the message is a reply, ID of the original message"},
                    "disable_notification": {"type": "boolean", "description": "Sends the message silently"}
                },
                required_parameters=["chat_id", "document"],
                async_supported=True,
                estimated_duration_seconds=1.5
            ),
            AdapterCapability(
                name="get_chat_info",
                description="Get information about a Telegram chat",
                category="chat_management",
                parameters={
                    "chat_id": {"type": "string", "description": "Unique identifier for the target chat or @channelusername"}
                },
                required_parameters=["chat_id"],
                async_supported=True,
                estimated_duration_seconds=0.3
            )
        ]

        return capabilities

    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to Telegram Bot API."""
        # Validate request
        self.validate_request(request)

        # Route to appropriate handler
        handlers = {
            "send_message": self._handle_send_message,
            "send_photo": self._handle_send_photo,
            "send_document": self._handle_send_document,
            "get_chat_info": self._handle_get_chat_info
        }

        handler = handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")

        return await handler(request)

    async def _handle_send_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a text message."""
        start_time = datetime.utcnow()

        try:
            data = {
                "chat_id": request.parameters["chat_id"],
                "text": request.parameters["text"]
            }

            # Add optional parameters
            for param in ["parse_mode", "reply_to_message_id", "disable_web_page_preview", "disable_notification"]:
                if param in request.parameters:
                    data[param] = request.parameters[param]

            response = await self.client.post("/sendMessage", json=data)
            response.raise_for_status()

            result = response.json()

            if not result.get("ok"):
                raise ValueError(f"Telegram API error: {result.get('description', 'Unknown error')}")

            message_data = result.get("result", {})
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Publish message sent event
            await event_bus.publish(
                "adapter.telegram.message_sent",
                {
                    "chat_id": data["chat_id"],
                    "message_id": message_data.get("message_id"),
                    "reply_to_message_id": data.get("reply_to_message_id")
                },
                source_id=self.id,
                source_type="adapter"
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "message_id": message_data.get("message_id"),
                    "chat": message_data.get("chat"),
                    "date": message_data.get("date"),
                    "text": message_data.get("text")
                },
                duration_ms=duration_ms,
                metadata={"ok": True}
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )

    async def _handle_send_photo(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a photo."""
        start_time = datetime.utcnow()

        try:
            data = {
                "chat_id": request.parameters["chat_id"],
                "photo": request.parameters["photo"]
            }

            # Add optional parameters
            for param in ["caption", "parse_mode", "reply_to_message_id", "disable_notification"]:
                if param in request.parameters:
                    data[param] = request.parameters[param]

            response = await self.client.post("/sendPhoto", json=data)
            response.raise_for_status()

            result = response.json()

            if not result.get("ok"):
                raise ValueError(f"Telegram API error: {result.get('description', 'Unknown error')}")

            message_data = result.get("result", {})
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Publish photo sent event
            await event_bus.publish(
                "adapter.telegram.photo_sent",
                {
                    "chat_id": data["chat_id"],
                    "message_id": message_data.get("message_id"),
                    "photo_sizes": len(message_data.get("photo", []))
                },
                source_id=self.id,
                source_type="adapter"
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "message_id": message_data.get("message_id"),
                    "chat": message_data.get("chat"),
                    "date": message_data.get("date"),
                    "photo": message_data.get("photo"),
                    "caption": message_data.get("caption")
                },
                duration_ms=duration_ms,
                metadata={"ok": True}
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )

    async def _handle_send_document(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a document."""
        start_time = datetime.utcnow()

        try:
            data = {
                "chat_id": request.parameters["chat_id"],
                "document": request.parameters["document"]
            }

            # Add optional parameters
            for param in ["caption", "parse_mode", "reply_to_message_id", "disable_notification"]:
                if param in request.parameters:
                    data[param] = request.parameters[param]

            response = await self.client.post("/sendDocument", json=data)
            response.raise_for_status()

            result = response.json()

            if not result.get("ok"):
                raise ValueError(f"Telegram API error: {result.get('description', 'Unknown error')}")

            message_data = result.get("result", {})
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            # Publish document sent event
            await event_bus.publish(
                "adapter.telegram.document_sent",
                {
                    "chat_id": data["chat_id"],
                    "message_id": message_data.get("message_id"),
                    "file_name": message_data.get("document", {}).get("file_name")
                },
                source_id=self.id,
                source_type="adapter"
            )

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "message_id": message_data.get("message_id"),
                    "chat": message_data.get("chat"),
                    "date": message_data.get("date"),
                    "document": message_data.get("document"),
                    "caption": message_data.get("caption")
                },
                duration_ms=duration_ms,
                metadata={"ok": True}
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )

    async def _handle_get_chat_info(self, request: AdapterRequest) -> AdapterResponse:
        """Handle getting chat information."""
        start_time = datetime.utcnow()

        try:
            data = {"chat_id": request.parameters["chat_id"]}

            response = await self.client.post("/getChat", json=data)
            response.raise_for_status()

            result = response.json()

            if not result.get("ok"):
                raise ValueError(f"Telegram API error: {result.get('description', 'Unknown error')}")

            chat_data = result.get("result", {})
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "id": chat_data.get("id"),
                    "type": chat_data.get("type"),
                    "title": chat_data.get("title"),
                    "username": chat_data.get("username"),
                    "first_name": chat_data.get("first_name"),
                    "last_name": chat_data.get("last_name"),
                    "description": chat_data.get("description"),
                    "invite_link": chat_data.get("invite_link"),
                    "permissions": chat_data.get("permissions"),
                    "photo": chat_data.get("photo")
                },
                duration_ms=duration_ms,
                metadata={"ok": True}
            )

        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )

    async def process_update(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an inbound Telegram webhook update.

        Telegram sends updates via webhook for incoming messages, edited messages,
        channel posts, callback queries, inline queries, etc.

        Args:
            payload: The raw update JSON from Telegram's webhook.

        Returns:
            Parsed update data with normalized fields, or None if unrecognized.
        """
        update_id = payload.get("update_id")

        # Determine update type and extract relevant data
        if "message" in payload:
            message = payload["message"]
            parsed = {
                "update_id": update_id,
                "type": "message",
                "message_id": message.get("message_id"),
                "chat_id": message.get("chat", {}).get("id"),
                "chat_type": message.get("chat", {}).get("type"),
                "from_user": {
                    "id": message.get("from", {}).get("id"),
                    "username": message.get("from", {}).get("username"),
                    "first_name": message.get("from", {}).get("first_name"),
                    "is_bot": message.get("from", {}).get("is_bot", False)
                },
                "text": message.get("text"),
                "date": message.get("date"),
                "reply_to_message_id": message.get("reply_to_message", {}).get("message_id") if message.get("reply_to_message") else None,
                "entities": message.get("entities", []),
                "photo": message.get("photo"),
                "document": message.get("document"),
                "caption": message.get("caption")
            }

            # Publish inbound message event
            await event_bus.publish(
                "adapter.telegram.message_received",
                {
                    "update_id": update_id,
                    "chat_id": parsed["chat_id"],
                    "message_id": parsed["message_id"],
                    "from_user_id": parsed["from_user"]["id"],
                    "text": parsed["text"]
                },
                source_id=self.id,
                source_type="adapter"
            )

            return parsed

        elif "edited_message" in payload:
            message = payload["edited_message"]
            parsed = {
                "update_id": update_id,
                "type": "edited_message",
                "message_id": message.get("message_id"),
                "chat_id": message.get("chat", {}).get("id"),
                "chat_type": message.get("chat", {}).get("type"),
                "from_user": {
                    "id": message.get("from", {}).get("id"),
                    "username": message.get("from", {}).get("username"),
                    "first_name": message.get("from", {}).get("first_name"),
                    "is_bot": message.get("from", {}).get("is_bot", False)
                },
                "text": message.get("text"),
                "date": message.get("date"),
                "edit_date": message.get("edit_date")
            }

            await event_bus.publish(
                "adapter.telegram.message_edited",
                {
                    "update_id": update_id,
                    "chat_id": parsed["chat_id"],
                    "message_id": parsed["message_id"]
                },
                source_id=self.id,
                source_type="adapter"
            )

            return parsed

        elif "callback_query" in payload:
            callback = payload["callback_query"]
            parsed = {
                "update_id": update_id,
                "type": "callback_query",
                "callback_query_id": callback.get("id"),
                "from_user": {
                    "id": callback.get("from", {}).get("id"),
                    "username": callback.get("from", {}).get("username"),
                    "first_name": callback.get("from", {}).get("first_name"),
                    "is_bot": callback.get("from", {}).get("is_bot", False)
                },
                "chat_id": callback.get("message", {}).get("chat", {}).get("id") if callback.get("message") else None,
                "message_id": callback.get("message", {}).get("message_id") if callback.get("message") else None,
                "data": callback.get("data"),
                "inline_message_id": callback.get("inline_message_id")
            }

            await event_bus.publish(
                "adapter.telegram.callback_query",
                {
                    "update_id": update_id,
                    "callback_query_id": parsed["callback_query_id"],
                    "chat_id": parsed["chat_id"],
                    "data": parsed["data"]
                },
                source_id=self.id,
                source_type="adapter"
            )

            return parsed

        elif "channel_post" in payload:
            post = payload["channel_post"]
            parsed = {
                "update_id": update_id,
                "type": "channel_post",
                "message_id": post.get("message_id"),
                "chat_id": post.get("chat", {}).get("id"),
                "chat_title": post.get("chat", {}).get("title"),
                "text": post.get("text"),
                "date": post.get("date"),
                "caption": post.get("caption")
            }

            await event_bus.publish(
                "adapter.telegram.channel_post",
                {
                    "update_id": update_id,
                    "chat_id": parsed["chat_id"],
                    "message_id": parsed["message_id"]
                },
                source_id=self.id,
                source_type="adapter"
            )

            return parsed

        else:
            logger.warning(f"Unrecognized Telegram update type, update_id={update_id}")
            return None

    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Telegram-specific health check."""
        try:
            if self.client:
                response = await self.client.get("/getMe")
                result = response.json()

                if result.get("ok"):
                    bot_data = result.get("result", {})
                    return {
                        "status": "healthy",
                        "bot_id": bot_data.get("id"),
                        "bot_username": bot_data.get("username"),
                        "bot_name": bot_data.get("first_name")
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": result.get("description", "Unknown error")
                    }
            else:
                # Discovery mode or not initialized
                return {"status": "healthy", "mode": "discovery"}

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
