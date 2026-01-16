"""Platform Webhook Service for handling incoming webhooks from external platforms"""
import os
import json
import hmac
import hashlib
import logging
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
import asyncio
from enum import Enum
from collections import defaultdict
import aiohttp
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from core.database import get_db
from models.platform_integration import (
    PlatformWebhook, 
    PlatformWebhookDelivery,
    PlatformExecution,
    PlatformType
)
from schemas.platform_integration import (
    WebhookEventType,
    WebhookDeliveryStatus
)
from services.platform_adapters import PlatformAdapterService

logger = logging.getLogger(__name__)


class WebhookVerificationError(Exception):
    """Raised when webhook verification fails"""
    pass


class WebhookHandler:
    """Base class for platform-specific webhook handlers"""
    
    async def verify_signature(self, headers: Dict[str, str], body: bytes, secret: str) -> bool:
        """Verify webhook signature"""
        raise NotImplementedError
    
    async def parse_event(self, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        """Parse webhook event into standard format"""
        raise NotImplementedError
    
    async def get_event_type(self, event: Dict[str, Any]) -> WebhookEventType:
        """Determine event type from webhook data"""
        raise NotImplementedError


class N8NWebhookHandler(WebhookHandler):
    """Handler for n8n webhooks"""
    
    async def verify_signature(self, headers: Dict[str, str], body: bytes, secret: str) -> bool:
        """Verify n8n webhook signature"""
        # n8n uses HMAC-SHA256
        signature = headers.get("x-n8n-signature", "")
        if not signature:
            return False
        
        expected = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)
    
    async def parse_event(self, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        """Parse n8n webhook event"""
        return {
            "workflow_id": body.get("workflowId"),
            "execution_id": body.get("executionId"),
            "status": body.get("status", "unknown"),
            "started_at": body.get("startedAt"),
            "finished_at": body.get("finishedAt"),
            "data": body.get("data", {}),
            "error": body.get("error"),
            "metadata": {
                "mode": body.get("mode"),
                "retry_of": body.get("retryOf")
            }
        }
    
    async def get_event_type(self, event: Dict[str, Any]) -> WebhookEventType:
        """Determine n8n event type"""
        status = event.get("status", "").lower()
        
        if status == "running":
            return WebhookEventType.EXECUTION_STARTED
        elif status == "success":
            return WebhookEventType.EXECUTION_COMPLETED
        elif status in ["error", "failed"]:
            return WebhookEventType.EXECUTION_FAILED
        elif status == "waiting":
            return WebhookEventType.EXECUTION_WAITING
        else:
            return WebhookEventType.OTHER


class ZapierWebhookHandler(WebhookHandler):
    """Handler for Zapier webhooks"""
    
    async def verify_signature(self, headers: Dict[str, str], body: bytes, secret: str) -> bool:
        """Verify Zapier webhook signature"""
        # Zapier uses custom header
        signature = headers.get("x-zapier-signature", "")
        if not signature:
            return False
        
        # Zapier format: timestamp.signature
        try:
            timestamp, sig = signature.split(".", 1)
            # Verify timestamp is recent (within 5 minutes)
            webhook_time = datetime.fromtimestamp(int(timestamp))
            if abs((datetime.utcnow() - webhook_time).total_seconds()) > 300:
                return False
            
            # Verify signature
            payload = f"{timestamp}.{body.decode()}"
            expected = hmac.new(
                secret.encode(),
                payload.encode(),
                hashlib.sha256
            ).hexdigest()
            
            return hmac.compare_digest(sig, expected)
        except (ValueError, AttributeError):
            return False
    
    async def parse_event(self, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Zapier webhook event"""
        return {
            "workflow_id": body.get("zap", {}).get("id"),
            "execution_id": body.get("id"),
            "status": body.get("status", "unknown"),
            "started_at": body.get("request_timestamp"),
            "finished_at": body.get("timestamp"),
            "data": body.get("data", {}),
            "error": body.get("error"),
            "metadata": {
                "zap_name": body.get("zap", {}).get("name"),
                "attempt": body.get("attempt", 1)
            }
        }
    
    async def get_event_type(self, event: Dict[str, Any]) -> WebhookEventType:
        """Determine Zapier event type"""
        status = event.get("status", "").lower()
        
        if status == "success":
            return WebhookEventType.EXECUTION_COMPLETED
        elif status in ["error", "halted"]:
            return WebhookEventType.EXECUTION_FAILED
        elif status == "throttled":
            return WebhookEventType.RATE_LIMITED
        else:
            return WebhookEventType.OTHER


class MakeWebhookHandler(WebhookHandler):
    """Handler for Make (Integromat) webhooks"""
    
    async def verify_signature(self, headers: Dict[str, str], body: bytes, secret: str) -> bool:
        """Verify Make webhook signature"""
        signature = headers.get("x-make-signature", "")
        if not signature:
            return False
        
        expected = hmac.new(
            secret.encode(),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected)
    
    async def parse_event(self, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Make webhook event"""
        return {
            "workflow_id": body.get("scenario_id"),
            "execution_id": body.get("execution_id"),
            "status": body.get("status", "unknown"),
            "started_at": body.get("timestamp"),
            "finished_at": body.get("finished_at"),
            "data": body.get("data", {}),
            "error": body.get("error"),
            "metadata": {
                "team_id": body.get("team_id"),
                "organization_id": body.get("organization_id"),
                "operations": body.get("operations_count")
            }
        }
    
    async def get_event_type(self, event: Dict[str, Any]) -> WebhookEventType:
        """Determine Make event type"""
        status = event.get("status", "").lower()
        
        if status == "running":
            return WebhookEventType.EXECUTION_STARTED
        elif status == "success":
            return WebhookEventType.EXECUTION_COMPLETED
        elif status == "error":
            return WebhookEventType.EXECUTION_FAILED
        elif status == "warning":
            return WebhookEventType.EXECUTION_WARNING
        else:
            return WebhookEventType.OTHER


class PowerAutomateWebhookHandler(WebhookHandler):
    """Handler for Power Automate webhooks"""
    
    async def verify_signature(self, headers: Dict[str, str], body: bytes, secret: str) -> bool:
        """Verify Power Automate webhook signature"""
        # Power Automate uses Bearer token in Authorization header
        auth_header = headers.get("authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        
        token = auth_header[7:]  # Remove "Bearer " prefix
        return token == secret
    
    async def parse_event(self, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        """Parse Power Automate webhook event"""
        return {
            "workflow_id": body.get("workflowId"),
            "execution_id": body.get("runId"),
            "status": body.get("status", "unknown"),
            "started_at": body.get("startTime"),
            "finished_at": body.get("endTime"),
            "data": body.get("outputs", {}),
            "error": body.get("error"),
            "metadata": {
                "environment_id": body.get("environmentId"),
                "trigger": body.get("trigger", {}).get("name")
            }
        }
    
    async def get_event_type(self, event: Dict[str, Any]) -> WebhookEventType:
        """Determine Power Automate event type"""
        status = event.get("status", "").lower()
        
        if status == "running":
            return WebhookEventType.EXECUTION_STARTED
        elif status == "succeeded":
            return WebhookEventType.EXECUTION_COMPLETED
        elif status == "failed":
            return WebhookEventType.EXECUTION_FAILED
        elif status == "cancelled":
            return WebhookEventType.EXECUTION_CANCELLED
        else:
            return WebhookEventType.OTHER


class IFTTTWebhookHandler(WebhookHandler):
    """Handler for IFTTT webhooks"""
    
    async def verify_signature(self, headers: Dict[str, str], body: bytes, secret: str) -> bool:
        """Verify IFTTT webhook signature"""
        # IFTTT uses simple token validation
        webhook_key = headers.get("x-ifttt-key", "")
        return webhook_key == secret
    
    async def parse_event(self, headers: Dict[str, str], body: Dict[str, Any]) -> Dict[str, Any]:
        """Parse IFTTT webhook event"""
        return {
            "workflow_id": body.get("applet_id"),
            "execution_id": body.get("run_id"),
            "status": "completed" if body.get("completed") else "failed",
            "started_at": body.get("started_at"),
            "finished_at": body.get("occurred_at"),
            "data": body.get("ingredients", {}),
            "error": body.get("error"),
            "metadata": {
                "user_id": body.get("user_id"),
                "source": body.get("source", {}).get("name")
            }
        }
    
    async def get_event_type(self, event: Dict[str, Any]) -> WebhookEventType:
        """Determine IFTTT event type"""
        if event.get("status") == "completed":
            return WebhookEventType.EXECUTION_COMPLETED
        else:
            return WebhookEventType.EXECUTION_FAILED


class PlatformWebhookService:
    """Service for managing platform webhooks"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.handlers: Dict[PlatformType, WebhookHandler] = {
            PlatformType.N8N: N8NWebhookHandler(),
            PlatformType.ZAPIER: ZapierWebhookHandler(),
            PlatformType.MAKE: MakeWebhookHandler(),
            PlatformType.POWER_AUTOMATE: PowerAutomateWebhookHandler(),
            PlatformType.IFTTT: IFTTTWebhookHandler()
        }
        self.delivery_queue = asyncio.Queue()
        self.retry_delays = [30, 60, 300, 900, 3600]  # Exponential backoff
    
    async def register_webhook(
        self,
        platform: PlatformType,
        webhook_url: str,
        events: List[WebhookEventType],
        secret: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PlatformWebhook:
        """Register a new webhook endpoint"""
        # Generate secret if not provided
        if not secret:
            secret = hashlib.sha256(os.urandom(32)).hexdigest()
        
        webhook = PlatformWebhook(
            platform=platform.value,
            webhook_url=webhook_url,
            secret=secret,
            events=[e.value for e in events],
            user_id=user_id,
            is_active=True,
            webhook_metadata=metadata or {}
        )
        
        self.db.add(webhook)
        await self.db.commit()
        await self.db.refresh(webhook)
        
        logger.info(f"Registered webhook for {platform.value}: {webhook_url}")
        return webhook
    
    async def process_webhook(
        self,
        platform: PlatformType,
        headers: Dict[str, str],
        body: bytes
    ) -> Dict[str, Any]:
        """Process incoming webhook from platform"""
        handler = self.handlers.get(platform)
        if not handler:
            raise ValueError(f"No handler for platform {platform.value}")
        
        # Parse body
        try:
            body_data = json.loads(body)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in webhook body")
        
        # Parse event
        event = await handler.parse_event(headers, body_data)
        event_type = await handler.get_event_type(event)
        
        # Store webhook event
        webhook_event = PlatformWebhook(
            platform=platform.value,
            webhook_url="incoming",  # This is an incoming webhook
            event_type=event_type.value,
            event_data=body_data,
            headers=dict(headers),
            is_active=True
        )
        
        self.db.add(webhook_event)
        await self.db.commit()
        
        # Update execution status if linked
        if event.get("execution_id"):
            await self._update_execution_status(platform, event)
        
        # Trigger outgoing webhooks
        await self._trigger_webhooks(platform, event_type, event)
        
        return {
            "event_id": webhook_event.id,
            "platform": platform.value,
            "event_type": event_type.value,
            "processed": True
        }
    
    async def _update_execution_status(
        self,
        platform: PlatformType,
        event: Dict[str, Any]
    ):
        """Update execution status based on webhook event"""
        execution = await self.db.execute(
            select(PlatformExecution).where(
                and_(
                    PlatformExecution.platform == platform.value,
                    PlatformExecution.external_execution_id == event["execution_id"]
                )
            )
        )
        execution = execution.scalar_one_or_none()
        
        if execution:
            # Update status
            status_map = {
                "running": "running",
                "success": "completed",
                "succeeded": "completed",
                "error": "failed",
                "failed": "failed",
                "cancelled": "cancelled",
                "halted": "failed",
                "warning": "completed"
            }
            
            new_status = status_map.get(event.get("status", "").lower(), "unknown")
            if new_status != "unknown":
                execution.status = new_status
            
            # Update timestamps
            if event.get("started_at"):
                execution.started_at = datetime.fromisoformat(event["started_at"])
            if event.get("finished_at"):
                execution.completed_at = datetime.fromisoformat(event["finished_at"])
            
            # Update data
            if event.get("data"):
                execution.output_data = event["data"]
            if event.get("error"):
                execution.error_data = {"error": event["error"]}
            
            # Calculate duration
            if execution.started_at and execution.completed_at:
                duration = (execution.completed_at - execution.started_at).total_seconds() * 1000
                execution.duration_ms = int(duration)
            
            await self.db.commit()
            logger.info(f"Updated execution {execution.id} status to {new_status}")
    
    async def _trigger_webhooks(
        self,
        platform: PlatformType,
        event_type: WebhookEventType,
        event_data: Dict[str, Any]
    ):
        """Trigger registered webhooks for this event"""
        # Find active webhooks for this platform and event
        webhooks = await self.db.execute(
            select(PlatformWebhook).where(
                and_(
                    PlatformWebhook.platform == platform.value,
                    PlatformWebhook.is_active == True,
                    PlatformWebhook.webhook_url != "incoming"
                )
            )
        )
        webhooks = webhooks.scalars().all()
        
        for webhook in webhooks:
            # Check if webhook is subscribed to this event type
            if event_type.value not in webhook.events:
                continue
            
            # Queue delivery
            delivery = PlatformWebhookDelivery(
                webhook_id=webhook.id,
                event_type=event_type.value,
                payload=event_data,
                status=WebhookDeliveryStatus.PENDING.value,
                next_retry_at=datetime.utcnow()
            )
            
            self.db.add(delivery)
            await self.db.commit()
            
            # Add to delivery queue
            await self.delivery_queue.put(delivery.id)
    
    async def deliver_webhook(self, delivery_id: int):
        """Deliver a webhook to its endpoint"""
        delivery = await self.db.get(PlatformWebhookDelivery, delivery_id)
        if not delivery:
            return
        
        webhook = await self.db.get(PlatformWebhook, delivery.webhook_id)
        if not webhook or not webhook.is_active:
            delivery.status = WebhookDeliveryStatus.CANCELLED.value
            await self.db.commit()
            return
        
        # Prepare request
        headers = {
            "Content-Type": "application/json",
            "X-AICtrlNet-Event": delivery.event_type,
            "X-AICtrlNet-Delivery": str(delivery.id),
            "X-AICtrlNet-Timestamp": datetime.utcnow().isoformat()
        }
        
        # Add signature
        payload_bytes = json.dumps(delivery.payload).encode()
        signature = hmac.new(
            webhook.secret.encode(),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        headers["X-AICtrlNet-Signature"] = signature
        
        # Attempt delivery
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    webhook.webhook_url,
                    json=delivery.payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    delivery.response_status = response.status
                    delivery.response_headers = dict(response.headers)
                    delivery.response_body = await response.text()
                    
                    if 200 <= response.status < 300:
                        delivery.status = WebhookDeliveryStatus.DELIVERED.value
                        delivery.delivered_at = datetime.utcnow()
                        webhook.last_triggered_at = datetime.utcnow()
                        webhook.total_deliveries += 1
                        webhook.successful_deliveries += 1
                    else:
                        await self._handle_delivery_failure(delivery, webhook)
                        
        except asyncio.TimeoutError:
            delivery.response_body = "Timeout"
            await self._handle_delivery_failure(delivery, webhook)
        except Exception as e:
            delivery.response_body = str(e)
            await self._handle_delivery_failure(delivery, webhook)
        
        await self.db.commit()
    
    async def _handle_delivery_failure(
        self,
        delivery: PlatformWebhookDelivery,
        webhook: PlatformWebhook
    ):
        """Handle webhook delivery failure"""
        delivery.attempts += 1
        webhook.total_deliveries += 1
        webhook.failed_deliveries += 1
        
        if delivery.attempts >= len(self.retry_delays):
            # Max retries reached
            delivery.status = WebhookDeliveryStatus.FAILED.value
            webhook.consecutive_failures += 1
            
            # Disable webhook after too many consecutive failures
            if webhook.consecutive_failures >= 10:
                webhook.is_active = False
                logger.warning(f"Disabled webhook {webhook.id} after 10 consecutive failures")
        else:
            # Schedule retry
            delay_seconds = self.retry_delays[delivery.attempts - 1]
            delivery.status = WebhookDeliveryStatus.RETRYING.value
            delivery.next_retry_at = datetime.utcnow() + timedelta(seconds=delay_seconds)
            
            # Re-queue for retry
            await asyncio.sleep(delay_seconds)
            await self.delivery_queue.put(delivery.id)
    
    async def process_delivery_queue(self):
        """Process webhook delivery queue"""
        while True:
            try:
                delivery_id = await self.delivery_queue.get()
                await self.deliver_webhook(delivery_id)
            except Exception as e:
                logger.error(f"Error processing webhook delivery: {e}")
            
            await asyncio.sleep(1)  # Rate limiting
    
    async def cleanup_old_deliveries(self, days: int = 30):
        """Clean up old webhook deliveries"""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        result = await self.db.execute(
            select(PlatformWebhookDelivery).where(
                PlatformWebhookDelivery.created_at < cutoff_date
            )
        )
        
        old_deliveries = result.scalars().all()
        for delivery in old_deliveries:
            await self.db.delete(delivery)
        
        await self.db.commit()
        logger.info(f"Cleaned up {len(old_deliveries)} old webhook deliveries")
    
    async def get_webhook_stats(self, webhook_id: int) -> Dict[str, Any]:
        """Get statistics for a webhook"""
        webhook = await self.db.get(PlatformWebhook, webhook_id)
        if not webhook:
            return {}
        
        # Get delivery stats
        deliveries = await self.db.execute(
            select(PlatformWebhookDelivery).where(
                PlatformWebhookDelivery.webhook_id == webhook_id
            )
        )
        deliveries = deliveries.scalars().all()
        
        stats = {
            "total_deliveries": webhook.total_deliveries,
            "successful_deliveries": webhook.successful_deliveries,
            "failed_deliveries": webhook.failed_deliveries,
            "success_rate": (
                webhook.successful_deliveries / webhook.total_deliveries * 100
                if webhook.total_deliveries > 0 else 0
            ),
            "consecutive_failures": webhook.consecutive_failures,
            "last_triggered": webhook.last_triggered_at,
            "recent_deliveries": []
        }
        
        # Add recent delivery details
        for delivery in sorted(deliveries, key=lambda d: d.created_at, reverse=True)[:10]:
            stats["recent_deliveries"].append({
                "id": delivery.id,
                "event_type": delivery.event_type,
                "status": delivery.status,
                "attempts": delivery.attempts,
                "response_status": delivery.response_status,
                "created_at": delivery.created_at,
                "delivered_at": delivery.delivered_at
            })
        
        return stats