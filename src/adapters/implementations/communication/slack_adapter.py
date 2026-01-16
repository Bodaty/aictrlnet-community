"""Slack adapter implementation."""

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


class SlackAdapter(BaseAdapter):
    """Adapter for Slack API integration."""
    
    def __init__(self, config: AdapterConfig):
        # Ensure category is set correctly
        config.category = AdapterCategory.COMMUNICATION
        super().__init__(config)
        
        self.client: Optional[httpx.AsyncClient] = None
        self.base_url = config.base_url or "https://slack.com/api"
        
        # Check for discovery mode
        self.discovery_only = config.custom_config.get("discovery_only", False) if config.custom_config else False
        
        # Support both bot token and webhook URL
        self.bot_token = config.api_key or (config.credentials.get("bot_token") if config.credentials else None)
        self.webhook_url = config.credentials.get("webhook_url") if config.credentials else None
        
        # Skip validation in discovery mode
        if not self.discovery_only and not self.bot_token and not self.webhook_url:
            raise ValueError("Either Slack bot token or webhook URL is required")
    
    async def initialize(self) -> None:
        """Initialize the Slack adapter."""
        # Skip initialization in discovery mode
        if self.discovery_only:
            logger.info("Slack adapter initialized in discovery mode")
            return
            
        # Create HTTP client
        headers = {}
        if self.bot_token:
            headers["Authorization"] = f"Bearer {self.bot_token}"
        
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=self.config.timeout_seconds
        )
        
        # Test authentication if using bot token
        if self.bot_token:
            try:
                response = await self.client.post("/auth.test")
                response.raise_for_status()
                result = response.json()
                
                if not result.get("ok"):
                    raise ValueError(f"Slack auth failed: {result.get('error')}")
                
                self.bot_info = {
                    "user_id": result.get("user_id"),
                    "team": result.get("team"),
                    "user": result.get("user")
                }
                
                logger.info(f"Slack adapter initialized for team: {self.bot_info['team']}")
            except Exception as e:
                logger.error(f"Failed to initialize Slack adapter: {str(e)}")
                raise
        else:
            logger.info("Slack adapter initialized with webhook URL")
    
    async def shutdown(self) -> None:
        """Shutdown the adapter."""
        if self.client:
            await self.client.aclose()
            self.client = None
        logger.info("Slack adapter shutdown")
    
    def get_capabilities(self) -> List[AdapterCapability]:
        """Return Slack adapter capabilities."""
        capabilities = []
        
        if self.bot_token:
            capabilities.extend([
                AdapterCapability(
                    name="send_message",
                    description="Send a message to a Slack channel or user",
                    category="messaging",
                    parameters={
                        "channel": {"type": "string", "description": "Channel ID or name (e.g., #general, @user)"},
                        "text": {"type": "string", "description": "Message text"},
                        "blocks": {"type": "array", "description": "Block Kit blocks for rich formatting"},
                        "thread_ts": {"type": "string", "description": "Thread timestamp to reply to"},
                        "attachments": {"type": "array", "description": "Legacy attachments"}
                    },
                    required_parameters=["channel", "text"],
                    async_supported=True,
                    estimated_duration_seconds=0.5
                ),
                AdapterCapability(
                    name="upload_file",
                    description="Upload a file to Slack",
                    category="file_sharing",
                    parameters={
                        "channels": {"type": "string", "description": "Comma-separated channel IDs"},
                        "content": {"type": "string", "description": "File content"},
                        "filename": {"type": "string", "description": "Filename"},
                        "title": {"type": "string", "description": "File title"},
                        "initial_comment": {"type": "string", "description": "Initial comment"}
                    },
                    required_parameters=["channels", "content"],
                    async_supported=True,
                    estimated_duration_seconds=2.0
                ),
                AdapterCapability(
                    name="get_channel_info",
                    description="Get information about a Slack channel",
                    category="channel_management",
                    parameters={
                        "channel": {"type": "string", "description": "Channel ID"}
                    },
                    required_parameters=["channel"],
                    async_supported=True,
                    estimated_duration_seconds=0.3
                ),
                AdapterCapability(
                    name="list_channels",
                    description="List Slack channels",
                    category="channel_management",
                    parameters={
                        "types": {"type": "string", "description": "Channel types (public_channel,private_channel)", "default": "public_channel"},
                        "limit": {"type": "integer", "description": "Number of channels to return", "default": 100}
                    },
                    required_parameters=[],
                    async_supported=True,
                    estimated_duration_seconds=0.5
                ),
                AdapterCapability(
                    name="add_reaction",
                    description="Add a reaction to a message",
                    category="messaging",
                    parameters={
                        "channel": {"type": "string", "description": "Channel ID"},
                        "timestamp": {"type": "string", "description": "Message timestamp"},
                        "name": {"type": "string", "description": "Reaction name (e.g., thumbsup)"}
                    },
                    required_parameters=["channel", "timestamp", "name"],
                    async_supported=True,
                    estimated_duration_seconds=0.3
                )
            ])
        
        if self.webhook_url:
            capabilities.append(
                AdapterCapability(
                    name="send_webhook",
                    description="Send a message via webhook URL",
                    category="messaging",
                    parameters={
                        "text": {"type": "string", "description": "Message text"},
                        "blocks": {"type": "array", "description": "Block Kit blocks"},
                        "attachments": {"type": "array", "description": "Legacy attachments"}
                    },
                    required_parameters=["text"],
                    async_supported=True,
                    estimated_duration_seconds=0.5
                )
            )
        
        return capabilities
    
    async def execute(self, request: AdapterRequest) -> AdapterResponse:
        """Execute a request to Slack."""
        # Validate request
        self.validate_request(request)
        
        # Route to appropriate handler
        handlers = {
            "send_message": self._handle_send_message,
            "send_webhook": self._handle_send_webhook,
            "upload_file": self._handle_upload_file,
            "get_channel_info": self._handle_get_channel_info,
            "list_channels": self._handle_list_channels,
            "add_reaction": self._handle_add_reaction
        }
        
        handler = handlers.get(request.capability)
        if not handler:
            raise ValueError(f"Unknown capability: {request.capability}")
        
        return await handler(request)
    
    async def _handle_send_message(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending a message."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "channel": request.parameters["channel"],
                "text": request.parameters["text"]
            }
            
            # Add optional parameters
            for param in ["blocks", "thread_ts", "attachments"]:
                if param in request.parameters:
                    data[param] = request.parameters[param]
            
            response = await self.client.post("/chat.postMessage", json=data)
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get("ok"):
                raise ValueError(f"Slack API error: {result.get('error')}")
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Publish message sent event
            await event_bus.publish(
                "adapter.slack.message_sent",
                {
                    "channel": data["channel"],
                    "ts": result.get("ts"),
                    "thread_ts": data.get("thread_ts")
                },
                source_id=self.id,
                source_type="adapter"
            )
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "channel": result.get("channel"),
                    "ts": result.get("ts"),
                    "message": result.get("message")
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
    
    async def _handle_send_webhook(self, request: AdapterRequest) -> AdapterResponse:
        """Handle sending via webhook."""
        if not self.webhook_url:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error="Webhook URL not configured"
            )
        
        start_time = datetime.utcnow()
        
        try:
            data = {"text": request.parameters["text"]}
            
            # Add optional parameters
            for param in ["blocks", "attachments"]:
                if param in request.parameters:
                    data[param] = request.parameters[param]
            
            # Use a new client for webhook (different base URL)
            async with httpx.AsyncClient() as webhook_client:
                response = await webhook_client.post(
                    self.webhook_url,
                    json=data,
                    timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={"sent": True},
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_upload_file(self, request: AdapterRequest) -> AdapterResponse:
        """Handle file upload."""
        start_time = datetime.utcnow()
        
        try:
            data = {
                "channels": request.parameters["channels"],
                "content": request.parameters["content"],
                "filename": request.parameters.get("filename", "file.txt"),
                "title": request.parameters.get("title"),
                "initial_comment": request.parameters.get("initial_comment")
            }
            
            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}
            
            response = await self.client.post("/files.upload", data=data)
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get("ok"):
                raise ValueError(f"Slack API error: {result.get('error')}")
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "file": result.get("file"),
                    "file_id": result.get("file", {}).get("id")
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _handle_get_channel_info(self, request: AdapterRequest) -> AdapterResponse:
        """Handle getting channel information."""
        start_time = datetime.utcnow()
        
        try:
            params = {"channel": request.parameters["channel"]}
            
            response = await self.client.get("/conversations.info", params=params)
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get("ok"):
                raise ValueError(f"Slack API error: {result.get('error')}")
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={"channel": result.get("channel")},
                duration_ms=duration_ms
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
            params = {
                "types": request.parameters.get("types", "public_channel"),
                "limit": request.parameters.get("limit", 100)
            }
            
            response = await self.client.get("/conversations.list", params=params)
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get("ok"):
                raise ValueError(f"Slack API error: {result.get('error')}")
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={
                    "channels": result.get("channels", []),
                    "response_metadata": result.get("response_metadata")
                },
                duration_ms=duration_ms
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
            data = {
                "channel": request.parameters["channel"],
                "timestamp": request.parameters["timestamp"],
                "name": request.parameters["name"]
            }
            
            response = await self.client.post("/reactions.add", json=data)
            response.raise_for_status()
            
            result = response.json()
            
            if not result.get("ok"):
                raise ValueError(f"Slack API error: {result.get('error')}")
            
            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="success",
                data={"ok": True},
                duration_ms=duration_ms
            )
            
        except Exception as e:
            return AdapterResponse(
                request_id=request.id,
                capability=request.capability,
                status="error",
                error=str(e),
                duration_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
            )
    
    async def _perform_health_check(self) -> Dict[str, Any]:
        """Perform Slack-specific health check."""
        try:
            if self.bot_token:
                response = await self.client.post("/auth.test")
                result = response.json()
                
                if result.get("ok"):
                    return {
                        "status": "healthy",
                        "team": result.get("team"),
                        "user": result.get("user")
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": result.get("error")
                    }
            else:
                # For webhook-only, just return healthy
                return {"status": "healthy", "mode": "webhook"}
                
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }