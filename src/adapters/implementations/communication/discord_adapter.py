"""Discord adapter implementation."""

import asyncio
import logging
from typing import Any, Dict, List, Optional, Union
import httpx
import json
from datetime import datetime, timedelta
import base64

from adapters.base_adapter import BaseAdapter
from adapters.models import (
    AdapterCapability, AdapterRequest, AdapterResponse,
    AdapterConfig, AdapterCategory
)
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class DiscordAdapter(BaseAdapter):
    """Adapter for Discord integration."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.COMMUNICATION
        super().__init__(config)
        
        # Discovery mode support
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False
        
        self.client: Optional[httpx.AsyncClient] = None
        
        # Discord API configuration
        self.base_url = "https://discord.com/api/v10"
        self.bot_token = config.credentials.get("bot_token") if config.credentials else None
        self.client_id = config.credentials.get("client_id") if config.credentials else None
        self.client_secret = config.credentials.get("client_secret") if config.credentials else None
        
        # Webhook URL for simple webhook messages
        self.webhook_url = config.credentials.get("webhook_url") if config.credentials else None
        
        # Gateway connection (for real-time events)
        self.gateway_url = None
        self.gateway_ws = None
        
        # Skip validation in discovery mode
        if not self.discovery_only and not self.bot_token and not self.webhook_url:
            raise ValueError("Either bot_token or webhook_url is required")
    
    async def initialize(self) -> None:
        """Initialize the Discord adapter."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("Discord adapter initialized in discovery mode")
            return
        
        # Create HTTP client
        headers = {}
        if self.bot_token:
            headers["Authorization"] = f"Bot {self.bot_token}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=30.0,
            limits=httpx.Limits(
                max_keepalive_connections=20,
                max_connections=100,
                keepalive_expiry=30.0
            )
        )
        
        # Test connection if bot token provided
        if self.bot_token:
            try:
                response = await self.client.get("/users/@me")
                response.raise_for_status()
                bot_info = response.json()
                logger.info(f"Discord adapter initialized. Bot: {bot_info['username']}#{bot_info['discriminator']}")
            except Exception as e:
                logger.warning(f"Failed to verify bot token: {str(e)}")
        
        logger.info("Discord adapter initialized successfully")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.gateway_ws:
            await self.gateway_ws.close()
            self.gateway_ws = None
        
        if self.client:
            await self.client.aclose()
            self.client = None
        
        logger.info("Discord adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Discord adapter capabilities."""
        return [
            # Messaging
            AdapterCapability(
                name="send_message",
                description="Send a message to a Discord channel",
                category="messaging",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "content": {"type": "string", "description": "Message content"},
                    "embeds": {"type": "array", "description": "Rich embed objects"},
                    "components": {"type": "array", "description": "Interactive components"},
                    "attachments": {"type": "array", "description": "File attachments"},
                    "reply_to": {"type": "string", "description": "Message ID to reply to"},
                    "tts": {"type": "boolean", "description": "Text-to-speech", "default": False}
                },
                required_parameters=["channel_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="send_embed",
                description="Send a rich embed message",
                category="messaging",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "title": {"type": "string", "description": "Embed title"},
                    "description": {"type": "string", "description": "Embed description"},
                    "color": {"type": "integer", "description": "Embed color (decimal)"},
                    "fields": {"type": "array", "description": "Embed fields"},
                    "thumbnail": {"type": "string", "description": "Thumbnail URL"},
                    "image": {"type": "string", "description": "Image URL"},
                    "footer": {"type": "object", "description": "Footer object"},
                    "author": {"type": "object", "description": "Author object"}
                },
                required_parameters=["channel_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="send_webhook_message",
                description="Send a message via webhook URL",
                category="messaging",
                parameters={
                    "webhook_url": {"type": "string", "description": "Webhook URL (overrides default)"},
                    "content": {"type": "string", "description": "Message content"},
                    "username": {"type": "string", "description": "Override webhook username"},
                    "avatar_url": {"type": "string", "description": "Override webhook avatar"},
                    "embeds": {"type": "array", "description": "Rich embed objects"},
                    "thread_id": {"type": "string", "description": "Thread ID to send to"}
                },
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="edit_message",
                description="Edit an existing message",
                category="messaging",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "message_id": {"type": "string", "description": "Message ID"},
                    "content": {"type": "string", "description": "New content"},
                    "embeds": {"type": "array", "description": "New embeds"},
                    "components": {"type": "array", "description": "New components"}
                },
                required_parameters=["channel_id", "message_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="delete_message",
                description="Delete a message",
                category="messaging",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "message_id": {"type": "string", "description": "Message ID"}
                },
                required_parameters=["channel_id", "message_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            
            # Channel Management
            AdapterCapability(
                name="create_channel",
                description="Create a new channel",
                category="channel_management",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "name": {"type": "string", "description": "Channel name"},
                    "type": {"type": "integer", "description": "Channel type (0=text, 2=voice)", "default": 0},
                    "topic": {"type": "string", "description": "Channel topic"},
                    "category_id": {"type": "string", "description": "Parent category ID"},
                    "position": {"type": "integer", "description": "Sort position"},
                    "nsfw": {"type": "boolean", "description": "NSFW channel", "default": False}
                },
                required_parameters=["guild_id", "name"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="list_channels",
                description="List channels in a guild",
                category="channel_management",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "type_filter": {"type": "integer", "description": "Filter by channel type"}
                },
                required_parameters=["guild_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="create_thread",
                description="Create a thread in a channel",
                category="channel_management",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "name": {"type": "string", "description": "Thread name"},
                    "message_id": {"type": "string", "description": "Message to create thread from"},
                    "auto_archive_duration": {"type": "integer", "description": "Auto-archive minutes", "default": 1440},
                    "type": {"type": "integer", "description": "Thread type (public/private)"}
                },
                required_parameters=["channel_id", "name"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            
            # User Management
            AdapterCapability(
                name="add_role",
                description="Add a role to a user",
                category="user_management",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "user_id": {"type": "string", "description": "User ID"},
                    "role_id": {"type": "string", "description": "Role ID"}
                },
                required_parameters=["guild_id", "user_id", "role_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="remove_role",
                description="Remove a role from a user",
                category="user_management",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "user_id": {"type": "string", "description": "User ID"},
                    "role_id": {"type": "string", "description": "Role ID"}
                },
                required_parameters=["guild_id", "user_id", "role_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="ban_user",
                description="Ban a user from a guild",
                category="user_management",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "user_id": {"type": "string", "description": "User ID"},
                    "reason": {"type": "string", "description": "Ban reason"},
                    "delete_message_days": {"type": "integer", "description": "Days of messages to delete", "default": 0}
                },
                required_parameters=["guild_id", "user_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="kick_user",
                description="Kick a user from a guild",
                category="user_management",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "user_id": {"type": "string", "description": "User ID"},
                    "reason": {"type": "string", "description": "Kick reason"}
                },
                required_parameters=["guild_id", "user_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            
            # Voice
            AdapterCapability(
                name="join_voice_channel",
                description="Join a voice channel",
                category="voice",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "channel_id": {"type": "string", "description": "Voice channel ID"},
                    "self_mute": {"type": "boolean", "description": "Self mute", "default": False},
                    "self_deaf": {"type": "boolean", "description": "Self deafen", "default": False}
                },
                required_parameters=["guild_id", "channel_id"],
                async_supported=True,
                estimated_duration_seconds=2.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="leave_voice_channel",
                description="Leave current voice channel",
                category="voice",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"}
                },
                required_parameters=["guild_id"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            
            # Reactions
            AdapterCapability(
                name="add_reaction",
                description="Add a reaction to a message",
                category="reactions",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "message_id": {"type": "string", "description": "Message ID"},
                    "emoji": {"type": "string", "description": "Emoji (unicode or custom ID)"}
                },
                required_parameters=["channel_id", "message_id", "emoji"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="remove_reaction",
                description="Remove a reaction from a message",
                category="reactions",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "message_id": {"type": "string", "description": "Message ID"},
                    "emoji": {"type": "string", "description": "Emoji (unicode or custom ID)"},
                    "user_id": {"type": "string", "description": "User ID (@me for self)"}
                },
                required_parameters=["channel_id", "message_id", "emoji"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            
            # Guild Management
            AdapterCapability(
                name="create_role",
                description="Create a new role",
                category="guild_management",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "name": {"type": "string", "description": "Role name"},
                    "color": {"type": "integer", "description": "Role color (decimal)"},
                    "permissions": {"type": "string", "description": "Permission bitfield"},
                    "hoist": {"type": "boolean", "description": "Display separately", "default": False},
                    "mentionable": {"type": "boolean", "description": "Allow mentions", "default": False}
                },
                required_parameters=["guild_id", "name"],
                async_supported=True,
                estimated_duration_seconds=1.0,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="create_invite",
                description="Create an invite to a channel",
                category="guild_management",
                parameters={
                    "channel_id": {"type": "string", "description": "Channel ID"},
                    "max_age": {"type": "integer", "description": "Expiration in seconds", "default": 86400},
                    "max_uses": {"type": "integer", "description": "Max number of uses", "default": 0},
                    "temporary": {"type": "boolean", "description": "Temporary membership", "default": False},
                    "unique": {"type": "boolean", "description": "Create unique invite", "default": False}
                },
                required_parameters=["channel_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            
            # Utility
            AdapterCapability(
                name="get_guild_info",
                description="Get information about a guild",
                category="utility",
                parameters={
                    "guild_id": {"type": "string", "description": "Guild ID"},
                    "with_counts": {"type": "boolean", "description": "Include member counts", "default": True}
                },
                required_parameters=["guild_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="get_user_info",
                description="Get information about a user",
                category="utility",
                parameters={
                    "user_id": {"type": "string", "description": "User ID"}
                },
                required_parameters=["user_id"],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            ),
            AdapterCapability(
                name="health_check",
                description="Check Discord API connectivity",
                category="monitoring",
                parameters={},
                required_parameters=[],
                async_supported=True,
                estimated_duration_seconds=0.5,
                cost_per_request=0.0
            )
        ]
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a Discord operation."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        capability_handlers = {
            "send_message": self._handle_send_message,
            "send_embed": self._handle_send_embed,
            "send_webhook_message": self._handle_send_webhook_message,
            "edit_message": self._handle_edit_message,
            "delete_message": self._handle_delete_message,
            "create_channel": self._handle_create_channel,
            "list_channels": self._handle_list_channels,
            "create_thread": self._handle_create_thread,
            "add_role": self._handle_add_role,
            "remove_role": self._handle_remove_role,
            "ban_user": self._handle_ban_user,
            "kick_user": self._handle_kick_user,
            "join_voice_channel": self._handle_join_voice_channel,
            "leave_voice_channel": self._handle_leave_voice_channel,
            "add_reaction": self._handle_add_reaction,
            "remove_reaction": self._handle_remove_reaction,
            "create_role": self._handle_create_role,
            "create_invite": self._handle_create_invite,
            "get_guild_info": self._handle_get_guild_info,
            "get_user_info": self._handle_get_user_info,
            "health_check": self._handle_health_check
        }
        
        handler = capability_handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")
        
        return await handler(request)
    
    async def _handle_send_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a message."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            
            # Build message payload
            payload = {}
            
            if "content" in request.parameters:
                payload["content"] = request.parameters["content"]
            
            if "embeds" in request.parameters:
                payload["embeds"] = request.parameters["embeds"]
            
            if "components" in request.parameters:
                payload["components"] = request.parameters["components"]
            
            if "attachments" in request.parameters:
                # Handle file attachments
                files = []
                for i, attachment in enumerate(request.parameters["attachments"]):
                    file_data = base64.b64decode(attachment["content"])
                    files.append(("files", (attachment["filename"], file_data)))
                
                # Send with multipart form data
                response = await self.client.post(
                    f"/channels/{channel_id}/messages",
                    data={"payload_json": json.dumps(payload)},
                    files=files
                )
            else:
                # Send regular JSON
                if "reply_to" in request.parameters:
                    payload["message_reference"] = {"message_id": request.parameters["reply_to"]}
                
                if request.parameters.get("tts", False):
                    payload["tts"] = True
                
                response = await self.client.post(
                    f"/channels/{channel_id}/messages",
                    json=payload
                )
            
            response.raise_for_status()
            message_data = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish event
            await event_bus.publish(
                "adapter.discord.message_sent",
                {
                    "channel_id": channel_id,
                    "message_id": message_data["id"],
                    "has_embeds": len(message_data.get("embeds", [])) > 0,
                    "has_attachments": len(message_data.get("attachments", [])) > 0
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "message_id": message_data["id"],
                    "channel_id": message_data["channel_id"],
                    "author": message_data["author"],
                    "timestamp": message_data["timestamp"],
                    "content": message_data.get("content"),
                    "embeds": message_data.get("embeds", []),
                    "attachments": message_data.get("attachments", [])
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_send_embed(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending an embed message."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            
            # Build embed object
            embed = {}
            
            if "title" in request.parameters:
                embed["title"] = request.parameters["title"]
            
            if "description" in request.parameters:
                embed["description"] = request.parameters["description"]
            
            if "color" in request.parameters:
                embed["color"] = request.parameters["color"]
            
            if "fields" in request.parameters:
                embed["fields"] = request.parameters["fields"]
            
            if "thumbnail" in request.parameters:
                embed["thumbnail"] = {"url": request.parameters["thumbnail"]}
            
            if "image" in request.parameters:
                embed["image"] = {"url": request.parameters["image"]}
            
            if "footer" in request.parameters:
                embed["footer"] = request.parameters["footer"]
            
            if "author" in request.parameters:
                embed["author"] = request.parameters["author"]
            
            embed["timestamp"] = datetime.utcnow().isoformat()
            
            # Send message with embed
            response = await self.client.post(
                f"/channels/{channel_id}/messages",
                json={"embeds": [embed]}
            )
            
            response.raise_for_status()
            message_data = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "message_id": message_data["id"],
                    "channel_id": message_data["channel_id"],
                    "timestamp": message_data["timestamp"],
                    "embed": message_data["embeds"][0] if message_data.get("embeds") else None
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_send_webhook_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a webhook message."""
        start_time = datetime.utcnow()
        
        try:
            webhook_url = request.parameters.get("webhook_url", self.webhook_url)
            if not webhook_url:
                raise ValueError("No webhook URL provided")
            
            # Build webhook payload
            payload = {}
            
            if "content" in request.parameters:
                payload["content"] = request.parameters["content"]
            
            if "username" in request.parameters:
                payload["username"] = request.parameters["username"]
            
            if "avatar_url" in request.parameters:
                payload["avatar_url"] = request.parameters["avatar_url"]
            
            if "embeds" in request.parameters:
                payload["embeds"] = request.parameters["embeds"]
            
            # Add thread_id as query param if provided
            params = {}
            if "thread_id" in request.parameters:
                params["thread_id"] = request.parameters["thread_id"]
            
            # Send webhook
            response = await self.client.post(
                webhook_url,
                json=payload,
                params=params
            )
            
            response.raise_for_status()
            
            # Webhook responses may be empty
            response_data = {}
            if response.content:
                try:
                    response_data = response.json()
                except:
                    pass
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "webhook_executed": True,
                    "status_code": response.status_code,
                    "response": response_data
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_edit_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle editing a message."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            message_id = request.parameters["message_id"]
            
            # Build edit payload
            payload = {}
            
            if "content" in request.parameters:
                payload["content"] = request.parameters["content"]
            
            if "embeds" in request.parameters:
                payload["embeds"] = request.parameters["embeds"]
            
            if "components" in request.parameters:
                payload["components"] = request.parameters["components"]
            
            # Edit message
            response = await self.client.patch(
                f"/channels/{channel_id}/messages/{message_id}",
                json=payload
            )
            
            response.raise_for_status()
            message_data = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "message_id": message_data["id"],
                    "channel_id": message_data["channel_id"],
                    "edited_timestamp": message_data.get("edited_timestamp"),
                    "content": message_data.get("content")
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_delete_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle deleting a message."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            message_id = request.parameters["message_id"]
            
            # Delete message
            response = await self.client.delete(
                f"/channels/{channel_id}/messages/{message_id}"
            )
            
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "deleted": True,
                    "channel_id": channel_id,
                    "message_id": message_id
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_channel(self, request: AdapterRequest) -> AdapterResponse:
        """Handle creating a channel."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            
            # Build channel data
            channel_data = {
                "name": request.parameters["name"],
                "type": request.parameters.get("type", 0)
            }
            
            if "topic" in request.parameters:
                channel_data["topic"] = request.parameters["topic"]
            
            if "category_id" in request.parameters:
                channel_data["parent_id"] = request.parameters["category_id"]
            
            if "position" in request.parameters:
                channel_data["position"] = request.parameters["position"]
            
            if "nsfw" in request.parameters:
                channel_data["nsfw"] = request.parameters["nsfw"]
            
            # Create channel
            response = await self.client.post(
                f"/guilds/{guild_id}/channels",
                json=channel_data
            )
            
            response.raise_for_status()
            channel_info = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "channel_id": channel_info["id"],
                    "name": channel_info["name"],
                    "type": channel_info["type"],
                    "guild_id": channel_info["guild_id"],
                    "position": channel_info["position"],
                    "parent_id": channel_info.get("parent_id")
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_list_channels(self, request: AdapterRequest) -> AdapterResponse:
        """Handle listing channels."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            type_filter = request.parameters.get("type_filter")
            
            # Get channels
            response = await self.client.get(f"/guilds/{guild_id}/channels")
            response.raise_for_status()
            channels = response.json()
            
            # Filter by type if requested
            if type_filter is not None:
                channels = [ch for ch in channels if ch["type"] == type_filter]
            
            # Process channels
            processed_channels = []
            for channel in channels:
                processed_channels.append({
                    "id": channel["id"],
                    "name": channel["name"],
                    "type": channel["type"],
                    "position": channel["position"],
                    "parent_id": channel.get("parent_id"),
                    "topic": channel.get("topic"),
                    "nsfw": channel.get("nsfw", False)
                })
            
            # Sort by position
            processed_channels.sort(key=lambda x: x["position"])
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "channels": processed_channels,
                    "channel_count": len(processed_channels),
                    "guild_id": guild_id
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_thread(self, request: AdapterRequest) -> AdapterResponse:
        """Handle creating a thread."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            
            # Build thread data
            thread_data = {
                "name": request.parameters["name"],
                "auto_archive_duration": request.parameters.get("auto_archive_duration", 1440)
            }
            
            if "type" in request.parameters:
                thread_data["type"] = request.parameters["type"]
            
            # Create thread from message or start new
            if "message_id" in request.parameters:
                # Create from existing message
                response = await self.client.post(
                    f"/channels/{channel_id}/messages/{request.parameters['message_id']}/threads",
                    json=thread_data
                )
            else:
                # Start new thread
                response = await self.client.post(
                    f"/channels/{channel_id}/threads",
                    json=thread_data
                )
            
            response.raise_for_status()
            thread_info = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "thread_id": thread_info["id"],
                    "name": thread_info["name"],
                    "parent_id": thread_info["parent_id"],
                    "owner_id": thread_info["owner_id"],
                    "message_count": thread_info.get("message_count", 0),
                    "member_count": thread_info.get("member_count", 0)
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_add_role(self, request: AdapterRequest) -> AdapterResponse:
        """Handle adding a role to a user."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            user_id = request.parameters["user_id"]
            role_id = request.parameters["role_id"]
            
            # Add role
            response = await self.client.put(
                f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}"
            )
            
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "role_added": True,
                    "guild_id": guild_id,
                    "user_id": user_id,
                    "role_id": role_id
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_remove_role(self, request: AdapterRequest) -> AdapterResponse:
        """Handle removing a role from a user."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            user_id = request.parameters["user_id"]
            role_id = request.parameters["role_id"]
            
            # Remove role
            response = await self.client.delete(
                f"/guilds/{guild_id}/members/{user_id}/roles/{role_id}"
            )
            
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "role_removed": True,
                    "guild_id": guild_id,
                    "user_id": user_id,
                    "role_id": role_id
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_ban_user(self, request: AdapterRequest) -> AdapterResponse:
        """Handle banning a user."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            user_id = request.parameters["user_id"]
            
            # Build ban data
            ban_data = {}
            
            if "reason" in request.parameters:
                ban_data["reason"] = request.parameters["reason"]
            
            if "delete_message_days" in request.parameters:
                ban_data["delete_message_days"] = request.parameters["delete_message_days"]
            
            # Ban user
            response = await self.client.put(
                f"/guilds/{guild_id}/bans/{user_id}",
                json=ban_data
            )
            
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "user_banned": True,
                    "guild_id": guild_id,
                    "user_id": user_id,
                    "reason": request.parameters.get("reason")
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_kick_user(self, request: AdapterRequest) -> AdapterResponse:
        """Handle kicking a user."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            user_id = request.parameters["user_id"]
            
            # Kick user
            headers = {}
            if "reason" in request.parameters:
                headers["X-Audit-Log-Reason"] = request.parameters["reason"]
            
            response = await self.client.delete(
                f"/guilds/{guild_id}/members/{user_id}",
                headers=headers
            )
            
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "user_kicked": True,
                    "guild_id": guild_id,
                    "user_id": user_id,
                    "reason": request.parameters.get("reason")
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_join_voice_channel(self, request: AdapterRequest) -> AdapterResponse:
        """Handle joining a voice channel."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            channel_id = request.parameters["channel_id"]
            self_mute = request.parameters.get("self_mute", False)
            self_deaf = request.parameters.get("self_deaf", False)
            
            # Voice state update via gateway
            # This is a simplified response - actual implementation would use gateway
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "voice_state_update": True,
                    "guild_id": guild_id,
                    "channel_id": channel_id,
                    "self_mute": self_mute,
                    "self_deaf": self_deaf,
                    "note": "Voice connection requires gateway connection"
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_leave_voice_channel(self, request: AdapterRequest) -> AdapterResponse:
        """Handle leaving a voice channel."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            
            # Voice state update via gateway
            # This is a simplified response - actual implementation would use gateway
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "voice_state_update": True,
                    "guild_id": guild_id,
                    "channel_id": None,
                    "note": "Voice disconnection requires gateway connection"
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_add_reaction(self, request: AdapterRequest) -> AdapterResponse:
        """Handle adding a reaction."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            message_id = request.parameters["message_id"]
            emoji = request.parameters["emoji"]
            
            # URL encode emoji
            from urllib.parse import quote
            encoded_emoji = quote(emoji)
            
            # Add reaction
            response = await self.client.put(
                f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded_emoji}/@me"
            )
            
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "reaction_added": True,
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "emoji": emoji
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_remove_reaction(self, request: AdapterRequest) -> AdapterResponse:
        """Handle removing a reaction."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            message_id = request.parameters["message_id"]
            emoji = request.parameters["emoji"]
            user_id = request.parameters.get("user_id", "@me")
            
            # URL encode emoji
            from urllib.parse import quote
            encoded_emoji = quote(emoji)
            
            # Remove reaction
            response = await self.client.delete(
                f"/channels/{channel_id}/messages/{message_id}/reactions/{encoded_emoji}/{user_id}"
            )
            
            response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "reaction_removed": True,
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "emoji": emoji,
                    "user_id": user_id
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_role(self, request: AdapterRequest) -> AdapterResponse:
        """Handle creating a role."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            
            # Build role data
            role_data = {
                "name": request.parameters["name"]
            }
            
            if "color" in request.parameters:
                role_data["color"] = request.parameters["color"]
            
            if "permissions" in request.parameters:
                role_data["permissions"] = request.parameters["permissions"]
            
            if "hoist" in request.parameters:
                role_data["hoist"] = request.parameters["hoist"]
            
            if "mentionable" in request.parameters:
                role_data["mentionable"] = request.parameters["mentionable"]
            
            # Create role
            response = await self.client.post(
                f"/guilds/{guild_id}/roles",
                json=role_data
            )
            
            response.raise_for_status()
            role_info = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "role_id": role_info["id"],
                    "name": role_info["name"],
                    "color": role_info["color"],
                    "position": role_info["position"],
                    "permissions": role_info["permissions"],
                    "hoist": role_info["hoist"],
                    "mentionable": role_info["mentionable"]
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_create_invite(self, request: AdapterRequest) -> AdapterResponse:
        """Handle creating an invite."""
        start_time = datetime.utcnow()
        
        try:
            channel_id = request.parameters["channel_id"]
            
            # Build invite data
            invite_data = {
                "max_age": request.parameters.get("max_age", 86400),
                "max_uses": request.parameters.get("max_uses", 0),
                "temporary": request.parameters.get("temporary", False),
                "unique": request.parameters.get("unique", False)
            }
            
            # Create invite
            response = await self.client.post(
                f"/channels/{channel_id}/invites",
                json=invite_data
            )
            
            response.raise_for_status()
            invite_info = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "code": invite_info["code"],
                    "url": f"https://discord.gg/{invite_info['code']}",
                    "channel": invite_info.get("channel"),
                    "guild": invite_info.get("guild"),
                    "expires_at": invite_info.get("expires_at"),
                    "max_age": invite_info.get("max_age"),
                    "max_uses": invite_info.get("max_uses"),
                    "uses": invite_info.get("uses", 0)
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_get_guild_info(self, request: AdapterRequest) -> AdapterResponse:
        """Handle getting guild information."""
        start_time = datetime.utcnow()
        
        try:
            guild_id = request.parameters["guild_id"]
            with_counts = request.parameters.get("with_counts", True)
            
            # Get guild info
            params = {"with_counts": "true"} if with_counts else {}
            response = await self.client.get(f"/guilds/{guild_id}", params=params)
            
            response.raise_for_status()
            guild_info = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "id": guild_info["id"],
                    "name": guild_info["name"],
                    "icon": guild_info.get("icon"),
                    "owner_id": guild_info["owner_id"],
                    "member_count": guild_info.get("approximate_member_count"),
                    "presence_count": guild_info.get("approximate_presence_count"),
                    "description": guild_info.get("description"),
                    "features": guild_info.get("features", []),
                    "premium_tier": guild_info.get("premium_tier"),
                    "created_at": guild_info["id"]  # Can extract timestamp from snowflake
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_get_user_info(self, request: AdapterRequest) -> AdapterResponse:
        """Handle getting user information."""
        start_time = datetime.utcnow()
        
        try:
            user_id = request.parameters["user_id"]
            
            # Get user info
            response = await self.client.get(f"/users/{user_id}")
            response.raise_for_status()
            user_info = response.json()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "id": user_info["id"],
                    "username": user_info["username"],
                    "discriminator": user_info["discriminator"],
                    "avatar": user_info.get("avatar"),
                    "bot": user_info.get("bot", False),
                    "system": user_info.get("system", False),
                    "banner": user_info.get("banner"),
                    "accent_color": user_info.get("accent_color"),
                    "public_flags": user_info.get("public_flags", 0)
                },
                duration_ms=duration_ms,
                cost=0.0
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_health_check(self, request: AdapterRequest) -> AdapterResponse:
        """Handle health check."""
        return await self._perform_health_check_response(request)
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Discord health check."""
        try:
            if self.bot_token:
                # Check bot connection
                response = await self.client.get("/users/@me")
                response.raise_for_status()
                bot_info = response.json()
                
                return {
                    "status": "healthy",
                    "auth_type": "bot",
                    "bot_name": f"{bot_info['username']}#{bot_info['discriminator']}",
                    "bot_id": bot_info["id"]
                }
            elif self.webhook_url:
                return {
                    "status": "healthy",
                    "auth_type": "webhook",
                    "webhook_configured": True
                }
            else:
                return {
                    "status": "unhealthy",
                    "error": "No authentication configured"
                }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def _perform_health_check_response(self, request: AdapterRequest) -> AdapterResponse:
        """Perform health check and return as response."""
        start_time = datetime.utcnow()
        
        health_status = await self._perform_health_check()
        duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        return AdapterResponse(
            request_id=request.id,
            capability=request.capability,
            status="success" if health_status["status"] == "healthy" else "error",
            data=health_status,
            duration_ms=duration_ms,
            cost=0.0
        )