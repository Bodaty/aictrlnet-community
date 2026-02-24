"""Notification node implementation for sending notifications."""

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from ..base_node import BaseNode
from ..models import NodeConfig
from ..template_utils import resolve_templates
from events.event_bus import event_bus
from adapters.registry import adapter_registry
from adapters.models import AdapterConfig, AdapterCategory, AdapterRequest


logger = logging.getLogger(__name__)


class NotificationNode(BaseNode):
    """Node for sending notifications through various channels.

    Supports:
    - Email notifications
    - SMS notifications (via Twilio adapter)
    - Slack messages
    - Discord messages
    - WhatsApp messages
    - Telegram messages
    - Webhook notifications
    - Push notifications
    - In-app notifications
    """
    
    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the notification node. Returns output dict for BaseNode.run() to wrap."""
        # Build template context and resolve templates in parameters
        tmpl_ctx = {"input_data": input_data, **input_data}
        resolved_params = resolve_templates(dict(self.config.parameters), tmpl_ctx)
        self.config.parameters.update(resolved_params)

        # Get notification configuration
        channel = self.config.parameters.get("channel", "email")
        recipients = self._get_recipients(input_data)
        message = self._build_message(input_data, context)

        # For channels that can work without explicit recipients, allow empty list
        if not recipients and channel in ("in_app", "webhook"):
            logger.warning(
                f"Notification node {self.config.id}: no recipients for "
                f"channel={channel} — logging message and skipping send."
            )
            return {
                "channel": channel,
                "sent_to": [],
                "skipped": True,
                "message_subject": message.get("subject"),
                "message_body": message.get("body"),
            }

        # Send notification based on channel
        if channel == "email":
            output_data = await self._send_email(recipients, message)
        elif channel == "sms":
            output_data = await self._send_sms(recipients, message)
        elif channel == "slack":
            output_data = await self._send_slack(recipients, message)
        elif channel == "discord":
            output_data = await self._send_discord(recipients, message)
        elif channel == "whatsapp":
            output_data = await self._send_whatsapp(recipients, message)
        elif channel == "telegram":
            output_data = await self._send_telegram(recipients, message)
        elif channel == "webhook":
            output_data = await self._send_webhook(recipients, message)
        elif channel == "push":
            output_data = await self._send_push(recipients, message)
        elif channel == "in_app":
            output_data = await self._send_in_app(recipients, message)
        elif channel == "multi":
            output_data = await self._send_multi_channel(recipients, message)
        else:
            raise ValueError(f"Unsupported notification channel: {channel}")

        # Publish completion event
        await event_bus.publish(
            "node.executed",
            {
                "node_id": self.config.id,
                "node_type": "notification",
                "channel": channel,
                "recipients_count": len(recipients)
            }
        )

        return output_data
    
    @staticmethod
    def _create_adapter(adapter_class, adapter_id: str):
        """Create adapter instance with proper AdapterConfig."""
        config = AdapterConfig(
            name=adapter_id,
            category=AdapterCategory.COMMUNICATION,
            version="1.0.0",
            description=f"Notification adapter: {adapter_id}",
        )
        return adapter_class(config)

    def _get_recipients(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get notification recipients."""
        # Get from configuration or input
        recipients = self.config.parameters.get("recipients", [])
        if "recipients" in input_data:
            recipients = input_data["recipients"]
        
        # Ensure recipients is a list
        if not isinstance(recipients, list):
            recipients = [recipients]
        
        # Normalize recipient format
        normalized = []
        for recipient in recipients:
            if isinstance(recipient, str):
                # Simple string recipient (email, phone, etc.)
                normalized.append({"address": recipient})
            elif isinstance(recipient, dict):
                # Already formatted
                normalized.append(recipient)
        
        if not normalized:
            logger.warning(f"Notification node {self.config.id}: no recipients resolved")

        return normalized
    
    def _build_message(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Build notification message."""
        # Get message components
        subject = self.config.parameters.get("subject") or input_data.get("subject", "Notification")
        body = self.config.parameters.get("body") or input_data.get("body", "")
        template = self.config.parameters.get("template")
        
        # Apply template if specified
        if template:
            template_data = {
                **self.config.parameters,
                **input_data,
                **context,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Replace template variables
            if isinstance(template, str):
                body = template.format(**template_data)
            elif isinstance(template, dict):
                if "subject" in template:
                    subject = template["subject"].format(**template_data)
                if "body" in template:
                    body = template["body"].format(**template_data)
        
        # Build message structure
        message = {
            "subject": subject,
            "body": body,
            "type": self.config.parameters.get("message_type", "text"),
            "priority": self.config.parameters.get("priority", "normal"),
            "metadata": {
                "workflow_id": context.get("workflow_id"),
                "node_id": self.config.id,
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        # Add optional fields
        if self.config.parameters.get("attachments"):
            message["attachments"] = self.config.parameters["attachments"]
        if self.config.parameters.get("actions"):
            message["actions"] = self.config.parameters["actions"]
        if self.config.parameters.get("data"):
            message["data"] = self.config.parameters["data"]
        
        return message
    
    async def _send_email(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send email notification."""
        # Get email adapter
        adapter_id = self.config.parameters.get("email_adapter", "email")
        # Get adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Email adapter '{adapter_id}' not found")
        
        # Create adapter instance
        adapter = self._create_adapter(adapter_class, adapter_id)
        
        # Prepare email addresses
        to_addresses = [r.get("address") or r.get("email") for r in recipients]
        to_addresses = [addr for addr in to_addresses if addr]  # Filter None values
        
        # Build email request
        request = AdapterRequest(
            capability="send_email",
            parameters={
                "to": to_addresses,
                "subject": message["subject"],
                "body": message["body"],
                "html": message["type"] == "html",
                "attachments": message.get("attachments", []),
                "from": self.config.parameters.get("from_email"),
                "reply_to": self.config.parameters.get("reply_to")
            }
        )
        
        # Send email
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Email sending failed: {response.error}")
        
        return {
            "channel": "email",
            "sent_to": to_addresses,
            "message_id": response.data.get("message_id"),
            "adapters_used": [adapter_id]
        }
    
    async def _send_sms(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send SMS notification."""
        # For SMS, we'll use a webhook adapter to call SMS service
        adapter_id = self.config.parameters.get("sms_adapter", "webhook")
        # Get adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"SMS adapter '{adapter_id}' not found")
        
        # Create adapter instance
        adapter = self._create_adapter(adapter_class, adapter_id)
        
        # Prepare phone numbers
        phone_numbers = [r.get("address") or r.get("phone") for r in recipients]
        phone_numbers = [num for num in phone_numbers if num]  # Filter None values
        
        # Build SMS request (using webhook to SMS gateway)
        sms_endpoint = self.config.parameters.get("sms_endpoint")
        if not sms_endpoint:
            raise ValueError("sms_endpoint is required for SMS notifications")
        
        results = []
        for phone in phone_numbers:
            request = AdapterRequest(
                capability="webhook",
                parameters={
                    "url": sms_endpoint,
                    "method": "POST",
                    "body": {
                        "to": phone,
                        "message": message["body"][:160],  # SMS limit
                        "from": self.config.parameters.get("from_phone")
                    }
                }
            )
            
            response = await adapter.execute(request)
            results.append({
                "phone": phone,
                "success": response.status != "error",
                "error": response.error if response.status == "error" else None
            })
        
        successful = sum(1 for r in results if r["success"])
        
        return {
            "channel": "sms",
            "sent_to": phone_numbers,
            "results": results,
            "successful": successful,
            "failed": len(results) - successful,
            "adapters_used": [adapter_id]
        }
    
    async def _send_slack(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send Slack notification."""
        # Get Slack adapter
        adapter_id = self.config.parameters.get("slack_adapter", "slack")
        # Get adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Slack adapter '{adapter_id}' not found")
        
        # Create adapter instance
        adapter = self._create_adapter(adapter_class, adapter_id)
        
        # Prepare channels/users
        targets = []
        for r in recipients:
            target = r.get("address") or r.get("channel") or r.get("user")
            if target:
                targets.append(target)
        
        # Send to each target
        results = []
        for target in targets:
            # Determine if target is channel or user
            is_channel = target.startswith("#") or target.startswith("C")
            
            request = AdapterRequest(
                capability="send_message",
                parameters={
                    "channel": target if is_channel else None,
                    "user": target if not is_channel else None,
                    "text": message["body"],
                    "attachments": self._convert_to_slack_attachments(message),
                    "thread_ts": self.config.parameters.get("thread_ts")
                }
            )
            
            response = await adapter.execute(request)
            results.append({
                "target": target,
                "success": response.status != "error",
                "error": response.error if response.status == "error" else None,
                "ts": response.data.get("ts") if response.status != "error" else None
            })
        
        successful = sum(1 for r in results if r["success"])
        
        return {
            "channel": "slack",
            "sent_to": targets,
            "results": results,
            "successful": successful,
            "failed": len(results) - successful,
            "adapters_used": [adapter_id]
        }
    
    async def _send_discord(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send Discord notification."""
        # Get Discord adapter
        adapter_id = self.config.parameters.get("discord_adapter", "discord")
        # Get adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Discord adapter '{adapter_id}' not found")
        
        # Create adapter instance
        adapter = self._create_adapter(adapter_class, adapter_id)
        
        # Prepare channels
        channels = []
        for r in recipients:
            channel = r.get("address") or r.get("channel")
            if channel:
                channels.append(channel)
        
        # Send to each channel
        results = []
        for channel in channels:
            request = AdapterRequest(
                capability="send_message",
                parameters={
                    "channel_id": channel,
                    "content": message["body"],
                    "embed": self._convert_to_discord_embed(message) if message["type"] != "text" else None
                }
            )
            
            response = await adapter.execute(request)
            results.append({
                "channel": channel,
                "success": response.status != "error",
                "error": response.error if response.status == "error" else None,
                "message_id": response.data.get("id") if response.status != "error" else None
            })
        
        successful = sum(1 for r in results if r["success"])
        
        return {
            "channel": "discord",
            "sent_to": channels,
            "results": results,
            "successful": successful,
            "failed": len(results) - successful,
            "adapters_used": [adapter_id]
        }
    
    async def _send_whatsapp(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send WhatsApp notification via WhatsApp adapter."""
        adapter_id = self.config.parameters.get("whatsapp_adapter", "whatsapp")
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"WhatsApp adapter '{adapter_id}' not found")

        adapter = self._create_adapter(adapter_class, adapter_id)

        phone_numbers = [r.get("address") or r.get("phone") for r in recipients]
        phone_numbers = [p for p in phone_numbers if p]

        results = []
        for phone in phone_numbers:
            # Use template message if specified (required for 24-hour window)
            template_name = self.config.parameters.get("whatsapp_template")
            if template_name:
                request = AdapterRequest(
                    capability="send_template",
                    parameters={
                        "to": phone,
                        "template_name": template_name,
                        "language": self.config.parameters.get("whatsapp_language", "en"),
                        "components": self.config.parameters.get("whatsapp_components", []),
                    }
                )
            else:
                request = AdapterRequest(
                    capability="send_message",
                    parameters={
                        "to": phone,
                        "text": message["body"],
                    }
                )

            response = await adapter.execute(request)
            results.append({
                "phone": phone,
                "success": response.status != "error",
                "error": response.error if response.status == "error" else None,
                "message_id": response.data.get("message_id") if response.status != "error" else None,
            })

        successful = sum(1 for r in results if r["success"])

        return {
            "channel": "whatsapp",
            "sent_to": phone_numbers,
            "results": results,
            "successful": successful,
            "failed": len(results) - successful,
            "adapters_used": [adapter_id],
        }

    async def _send_telegram(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send Telegram notification via Telegram adapter."""
        adapter_id = self.config.parameters.get("telegram_adapter", "telegram")
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Telegram adapter '{adapter_id}' not found")

        adapter = self._create_adapter(adapter_class, adapter_id)

        chat_ids = [r.get("address") or r.get("chat_id") for r in recipients]
        chat_ids = [c for c in chat_ids if c]

        results = []
        for chat_id in chat_ids:
            parse_mode = self.config.parameters.get("telegram_parse_mode", "HTML")
            request = AdapterRequest(
                capability="send_message",
                parameters={
                    "chat_id": chat_id,
                    "text": message["body"],
                    "parse_mode": parse_mode,
                }
            )

            response = await adapter.execute(request)
            results.append({
                "chat_id": chat_id,
                "success": response.status != "error",
                "error": response.error if response.status == "error" else None,
                "message_id": response.data.get("message_id") if response.status != "error" else None,
            })

        successful = sum(1 for r in results if r["success"])

        return {
            "channel": "telegram",
            "sent_to": chat_ids,
            "results": results,
            "successful": successful,
            "failed": len(results) - successful,
            "adapters_used": [adapter_id],
        }

    async def _send_webhook(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send webhook notification."""
        # Get webhook adapter
        adapter_id = self.config.parameters.get("webhook_adapter", "webhook")
        # Get adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Webhook adapter '{adapter_id}' not found")
        
        # Create adapter instance
        adapter = self._create_adapter(adapter_class, adapter_id)
        
        # Prepare webhook URLs
        urls = []
        for r in recipients:
            url = r.get("address") or r.get("url") or r.get("webhook")
            if url:
                urls.append(url)
        
        # Send to each webhook
        results = []
        for url in urls:
            # Build webhook payload
            payload = {
                "notification": {
                    "subject": message["subject"],
                    "body": message["body"],
                    "type": message["type"],
                    "priority": message["priority"],
                    "metadata": message["metadata"]
                }
            }
            
            # Add custom fields
            if self.config.parameters.get("webhook_fields"):
                payload.update(self.config.parameters["webhook_fields"])
            
            request = AdapterRequest(
                capability="webhook",
                parameters={
                    "url": url,
                    "method": "POST",
                    "body": payload,
                    "headers": self.config.parameters.get("webhook_headers", {})
                }
            )
            
            response = await adapter.execute(request)
            results.append({
                "url": url,
                "success": response.status != "error",
                "error": response.error if response.status == "error" else None,
                "status_code": response.data.get("status_code") if response.status != "error" else None
            })
        
        successful = sum(1 for r in results if r["success"])
        
        return {
            "channel": "webhook",
            "sent_to": urls,
            "results": results,
            "successful": successful,
            "failed": len(results) - successful,
            "adapters_used": [adapter_id]
        }
    
    async def _send_push(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send push notification."""
        # Push notifications would typically use a service like Firebase, OneSignal, etc.
        # For now, we'll use webhook adapter to call push service
        
        push_endpoint = self.config.parameters.get("push_endpoint")
        if not push_endpoint:
            raise ValueError("push_endpoint is required for push notifications")
        
        # Get webhook adapter
        adapter_id = self.config.parameters.get("push_adapter", "webhook")
        # Get adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Push adapter '{adapter_id}' not found")
        
        # Create adapter instance
        adapter = self._create_adapter(adapter_class, adapter_id)
        
        # Prepare device tokens
        tokens = []
        for r in recipients:
            token = r.get("address") or r.get("token") or r.get("device_token")
            if token:
                tokens.append(token)
        
        # Send push notification request
        request = AdapterRequest(
            capability="webhook",
            parameters={
                "url": push_endpoint,
                "method": "POST",
                "body": {
                    "tokens": tokens,
                    "notification": {
                        "title": message["subject"],
                        "body": message["body"],
                        "data": message.get("data", {}),
                        "priority": message["priority"]
                    }
                },
                "headers": self.config.parameters.get("push_headers", {})
            }
        )
        
        response = await adapter.execute(request)
        
        if response.status == "error":
            raise Exception(f"Push notification failed: {response.error}")
        
        return {
            "channel": "push",
            "sent_to": tokens,
            "batch_id": response.data.get("batch_id"),
            "adapters_used": [adapter_id]
        }
    
    async def _send_in_app(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send in-app notification."""
        # In-app notifications are typically stored in database and shown in UI
        # We'll publish an event for the application to handle
        
        user_ids = []
        for r in recipients:
            user_id = r.get("address") or r.get("user_id") or r.get("id")
            if user_id:
                user_ids.append(user_id)
        
        # Publish in-app notification events
        for user_id in user_ids:
            await event_bus.publish(
                "notification.in_app",
                {
                    "user_id": user_id,
                    "notification": {
                        "id": f"notif-{datetime.utcnow().timestamp()}",
                        "subject": message["subject"],
                        "body": message["body"],
                        "type": message["type"],
                        "priority": message["priority"],
                        "actions": message.get("actions", []),
                        "data": message.get("data", {}),
                        "created_at": datetime.utcnow().isoformat(),
                        "read": False
                    }
                }
            )
        
        return {
            "channel": "in_app",
            "sent_to": user_ids,
            "events_published": len(user_ids)
        }
    
    async def _send_multi_channel(self, recipients: List[Dict[str, Any]], message: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification through multiple channels."""
        channels = self.config.parameters.get("channels", ["email", "slack"])
        results = {}
        all_adapters = []
        
        for channel in channels:
            try:
                # Route to appropriate channel handler
                if channel == "email":
                    result = await self._send_email(recipients, message)
                elif channel == "sms":
                    result = await self._send_sms(recipients, message)
                elif channel == "slack":
                    result = await self._send_slack(recipients, message)
                elif channel == "discord":
                    result = await self._send_discord(recipients, message)
                elif channel == "whatsapp":
                    result = await self._send_whatsapp(recipients, message)
                elif channel == "telegram":
                    result = await self._send_telegram(recipients, message)
                elif channel == "webhook":
                    result = await self._send_webhook(recipients, message)
                elif channel == "push":
                    result = await self._send_push(recipients, message)
                elif channel == "in_app":
                    result = await self._send_in_app(recipients, message)
                else:
                    result = {"error": f"Unsupported channel: {channel}"}
                
                results[channel] = result
                all_adapters.extend(result.get("adapters_used", []))
                
            except Exception as e:
                results[channel] = {"error": str(e)}
        
        # Calculate success metrics
        successful_channels = sum(1 for r in results.values() if "error" not in r)
        total_channels = len(channels)
        
        return {
            "channel": "multi",
            "channels": channels,
            "results": results,
            "successful_channels": successful_channels,
            "failed_channels": total_channels - successful_channels,
            "adapters_used": list(set(all_adapters))  # Unique adapters
        }
    
    def _convert_to_slack_attachments(self, message: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert message to Slack attachments format."""
        attachments = []
        
        if message.get("attachments"):
            # Convert generic attachments to Slack format
            for att in message["attachments"]:
                slack_att = {
                    "fallback": att.get("title", "Attachment"),
                    "title": att.get("title"),
                    "text": att.get("text"),
                    "color": att.get("color", "good")
                }
                if att.get("fields"):
                    slack_att["fields"] = att["fields"]
                attachments.append(slack_att)
        
        if message.get("actions"):
            # Add action buttons
            actions_att = {
                "fallback": "Actions",
                "callback_id": f"notification_{self.config.id}",
                "actions": [
                    {
                        "name": action.get("name", f"action_{i}"),
                        "text": action.get("label", action.get("name")),
                        "type": "button",
                        "value": action.get("value", action.get("name"))
                    }
                    for i, action in enumerate(message["actions"])
                ]
            }
            attachments.append(actions_att)
        
        return attachments
    
    def _convert_to_discord_embed(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Convert message to Discord embed format."""
        embed = {
            "title": message["subject"],
            "description": message["body"],
            "timestamp": datetime.utcnow().isoformat(),
            "color": self._get_discord_color(message["priority"])
        }
        
        if message.get("attachments"):
            # Add fields from attachments
            fields = []
            for att in message["attachments"]:
                if att.get("fields"):
                    for field in att["fields"]:
                        fields.append({
                            "name": field.get("title", "Field"),
                            "value": field.get("value", ""),
                            "inline": field.get("inline", True)
                        })
            if fields:
                embed["fields"] = fields
        
        return embed
    
    def _get_discord_color(self, priority: str) -> int:
        """Get Discord color based on priority."""
        colors = {
            "low": 0x2ECC71,     # Green
            "normal": 0x3498DB,   # Blue
            "high": 0xF39C12,     # Orange
            "urgent": 0xE74C3C    # Red
        }
        return colors.get(priority, 0x3498DB)  # Default to blue
    
    def validate_config(self) -> bool:
        """Validate node configuration."""
        channel = self.config.parameters.get("channel")
        if not channel:
            raise ValueError("channel parameter is required")
        
        valid_channels = [
            "email", "sms", "slack", "discord", "whatsapp", "telegram",
            "webhook", "push", "in_app", "multi"
        ]
        
        if channel not in valid_channels:
            raise ValueError(f"Invalid channel: {channel}. Must be one of {valid_channels}")
        
        # Validate channel-specific requirements
        if channel == "sms" and not self.config.parameters.get("sms_endpoint"):
            raise ValueError("sms_endpoint is required for SMS notifications")
        elif channel == "push" and not self.config.parameters.get("push_endpoint"):
            raise ValueError("push_endpoint is required for push notifications")
        elif channel == "multi" and not self.config.parameters.get("channels"):
            raise ValueError("channels list is required for multi-channel notifications")
        
        return True