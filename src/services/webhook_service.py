"""Service for managing webhooks with secure delivery and retry logic."""

import hmac
import hashlib
import json
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc
import httpx
import logging

from models import Webhook, WebhookDelivery, User
from schemas import (
    WebhookCreate, WebhookResponse, WebhookCreateResponse,
    WebhookUpdate, WebhookListResponse, WebhookTestRequest,
    WebhookTestResponse, WebhookDeliveryListResponse,
    WebhookDeliveryResponse
)

logger = logging.getLogger(__name__)


class WebhookService:
    """Service for managing webhooks."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.http_client = httpx.AsyncClient(timeout=30.0)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()
    
    async def create_webhook(
        self,
        user_id: str,
        data: WebhookCreate
    ) -> WebhookCreateResponse:
        """Create a new webhook for a user."""
        # Create the webhook record
        webhook = Webhook(
            user_id=user_id,
            name=data.name,
            description=data.description,
            url=str(data.url),
            events=data.events,
            secret=data.secret,
            custom_headers=data.custom_headers,
            max_retries=data.max_retries,
            retry_delay_seconds=data.retry_delay_seconds,
            timeout_seconds=data.timeout_seconds
        )
        
        self.db.add(webhook)
        await self.db.commit()
        await self.db.refresh(webhook)
        
        # Return response with secret (only time it's shown)
        response_data = webhook.to_dict()
        response_data['secret'] = data.secret
        
        return WebhookCreateResponse(**response_data)
    
    async def list_user_webhooks(
        self,
        user_id: str,
        is_active: Optional[bool] = None,
        event_type: Optional[str] = None
    ) -> WebhookListResponse:
        """List all webhooks for a user."""
        query = select(Webhook).where(Webhook.user_id == user_id)
        
        if is_active is not None:
            query = query.where(Webhook.is_active == is_active)
        
        # Filter by event type (supports wildcards)
        if event_type:
            # Use PostgreSQL's array operators
            query = query.where(
                Webhook.events.contains([event_type])
            )
        
        query = query.order_by(Webhook.created_at.desc())
        
        result = await self.db.execute(query)
        webhooks = result.scalars().all()
        
        return WebhookListResponse(
            webhooks=[WebhookResponse(**webhook.to_dict()) for webhook in webhooks],
            total=len(webhooks)
        )
    
    async def get_webhook(
        self,
        user_id: str,
        webhook_id: str
    ) -> Optional[WebhookResponse]:
        """Get a specific webhook."""
        result = await self.db.execute(
            select(Webhook).where(
                and_(
                    Webhook.id == webhook_id,
                    Webhook.user_id == user_id
                )
            )
        )
        webhook = result.scalar_one_or_none()
        
        if webhook:
            return WebhookResponse(**webhook.to_dict())
        return None
    
    async def update_webhook(
        self,
        user_id: str,
        webhook_id: str,
        data: WebhookUpdate
    ) -> Optional[WebhookResponse]:
        """Update a webhook."""
        result = await self.db.execute(
            select(Webhook).where(
                and_(
                    Webhook.id == webhook_id,
                    Webhook.user_id == user_id
                )
            )
        )
        webhook = result.scalar_one_or_none()
        
        if not webhook:
            return None
        
        # Update fields
        update_data = data.dict(exclude_unset=True)
        for field, value in update_data.items():
            if field == 'url' and value:
                # Resetting URL clears failure statistics
                webhook.consecutive_failures = 0
                webhook.last_failure_at = None
                value = str(value)  # Convert HttpUrl to string
            setattr(webhook, field, value)
        
        webhook.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(webhook)
        
        return WebhookResponse(**webhook.to_dict())
    
    async def delete_webhook(
        self,
        user_id: str,
        webhook_id: str
    ) -> bool:
        """Delete a webhook."""
        result = await self.db.execute(
            select(Webhook).where(
                and_(
                    Webhook.id == webhook_id,
                    Webhook.user_id == user_id
                )
            )
        )
        webhook = result.scalar_one_or_none()
        
        if not webhook:
            return False
        
        await self.db.delete(webhook)
        await self.db.commit()
        return True
    
    async def enable_webhook(
        self,
        user_id: str,
        webhook_id: str
    ) -> Optional[WebhookResponse]:
        """Enable a webhook."""
        result = await self.db.execute(
            select(Webhook).where(
                and_(
                    Webhook.id == webhook_id,
                    Webhook.user_id == user_id
                )
            )
        )
        webhook = result.scalar_one_or_none()
        
        if not webhook:
            return None
        
        webhook.is_active = True
        webhook.consecutive_failures = 0  # Reset failure count
        webhook.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(webhook)
        
        return WebhookResponse(**webhook.to_dict())
    
    async def disable_webhook(
        self,
        user_id: str,
        webhook_id: str,
        reason: Optional[str] = None
    ) -> Optional[WebhookResponse]:
        """Disable a webhook."""
        result = await self.db.execute(
            select(Webhook).where(
                and_(
                    Webhook.id == webhook_id,
                    Webhook.user_id == user_id
                )
            )
        )
        webhook = result.scalar_one_or_none()
        
        if not webhook:
            return None
        
        webhook.is_active = False
        webhook.updated_at = datetime.utcnow()
        
        # Log the reason if provided
        if reason:
            logger.info(f"Webhook {webhook_id} disabled: {reason}")
        
        await self.db.commit()
        await self.db.refresh(webhook)
        
        return WebhookResponse(**webhook.to_dict())
    
    async def test_webhook(
        self,
        webhook_id: str,
        data: WebhookTestRequest
    ) -> WebhookTestResponse:
        """Test a webhook by sending a test payload."""
        result = await self.db.execute(
            select(Webhook).where(Webhook.id == webhook_id)
        )
        webhook = result.scalar_one_or_none()
        
        if not webhook:
            return WebhookTestResponse(
                success=False,
                error_message="Webhook not found"
            )
        
        # Prepare test payload
        payload = {
            "event": data.event_type,
            "data": data.payload,
            "timestamp": datetime.utcnow().isoformat(),
            "test": True
        }
        
        # Send test request
        start_time = datetime.utcnow()
        try:
            headers = self._prepare_headers(webhook, payload)
            
            response = await self.http_client.post(
                webhook.url,
                json=payload,
                headers=headers,
                timeout=webhook.timeout_seconds
            )
            
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            return WebhookTestResponse(
                success=response.status_code < 300,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                response_body=response.text[:500]  # Limit response body
            )
            
        except httpx.TimeoutException:
            return WebhookTestResponse(
                success=False,
                error_message="Request timed out"
            )
        except Exception as e:
            return WebhookTestResponse(
                success=False,
                error_message=str(e)
            )
    
    async def list_webhook_deliveries(
        self,
        webhook_id: str,
        limit: int = 100,
        is_success: Optional[bool] = None
    ) -> WebhookDeliveryListResponse:
        """List recent delivery attempts for a webhook."""
        query = select(WebhookDelivery).where(
            WebhookDelivery.webhook_id == webhook_id
        )
        
        if is_success is not None:
            query = query.where(WebhookDelivery.is_success == is_success)
        
        query = query.order_by(desc(WebhookDelivery.created_at)).limit(limit)
        
        result = await self.db.execute(query)
        deliveries = result.scalars().all()
        
        return WebhookDeliveryListResponse(
            deliveries=[
                WebhookDeliveryResponse(**delivery.to_dict())
                for delivery in deliveries
            ],
            total=len(deliveries)
        )
    
    async def deliver_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        event_id: Optional[str] = None
    ):
        """Deliver an event to all matching webhooks."""
        # Find all active webhooks that match this event
        webhooks = await self._find_matching_webhooks(event_type)
        
        # Deliver to each webhook asynchronously
        tasks = []
        for webhook in webhooks:
            task = asyncio.create_task(
                self._deliver_to_webhook(webhook, event_type, event_data, event_id)
            )
            tasks.append(task)
        
        # Wait for all deliveries to complete
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _find_matching_webhooks(self, event_type: str) -> List[Webhook]:
        """Find all webhooks that match the given event type."""
        # Get all active webhooks
        result = await self.db.execute(
            select(Webhook).where(Webhook.is_active == True)
        )
        all_webhooks = result.scalars().all()
        
        matching_webhooks = []
        for webhook in all_webhooks:
            # Check if webhook matches this event
            if self._event_matches_patterns(event_type, webhook.events):
                matching_webhooks.append(webhook)
        
        return matching_webhooks
    
    def _event_matches_patterns(self, event_type: str, patterns: List[str]) -> bool:
        """Check if an event type matches any of the webhook patterns."""
        for pattern in patterns:
            if pattern == "*":
                return True
            elif pattern.endswith("*"):
                prefix = pattern[:-1]
                if event_type.startswith(prefix):
                    return True
            elif pattern == event_type:
                return True
        return False
    
    async def _deliver_to_webhook(
        self,
        webhook: Webhook,
        event_type: str,
        event_data: Dict[str, Any],
        event_id: Optional[str] = None,
        attempt_number: int = 1
    ):
        """Deliver an event to a specific webhook with retry logic."""
        # Prepare payload
        payload = {
            "event": event_type,
            "data": event_data,
            "timestamp": datetime.utcnow().isoformat(),
            "webhook_id": webhook.id,
            "event_id": event_id
        }
        
        # Create delivery log
        delivery = WebhookDelivery(
            webhook_id=webhook.id,
            event_type=event_type,
            event_id=event_id,
            attempt_number=attempt_number,
            payload=payload
        )
        
        start_time = datetime.utcnow()
        
        try:
            headers = self._prepare_headers(webhook, payload)
            
            response = await self.http_client.post(
                webhook.url,
                json=payload,
                headers=headers,
                timeout=webhook.timeout_seconds
            )
            
            response_time_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            
            # Update delivery log
            delivery.status_code = response.status_code
            delivery.response_time_ms = response_time_ms
            delivery.is_success = response.status_code < 300
            delivery.delivered_at = datetime.utcnow()
            
            if not delivery.is_success:
                delivery.error_message = f"HTTP {response.status_code}: {response.text[:200]}"
            
            # Update webhook statistics
            webhook.last_triggered_at = datetime.utcnow()
            webhook.total_deliveries += 1
            
            if delivery.is_success:
                webhook.last_success_at = datetime.utcnow()
                webhook.consecutive_failures = 0
            else:
                webhook.last_failure_at = datetime.utcnow()
                webhook.consecutive_failures += 1
                webhook.total_failures += 1
            
        except httpx.TimeoutException:
            delivery.is_success = False
            delivery.error_message = "Request timed out"
            webhook.consecutive_failures += 1
            webhook.total_failures += 1
            webhook.last_failure_at = datetime.utcnow()
            
        except Exception as e:
            delivery.is_success = False
            delivery.error_message = str(e)
            webhook.consecutive_failures += 1
            webhook.total_failures += 1
            webhook.last_failure_at = datetime.utcnow()
            logger.error(f"Webhook delivery error: {e}", exc_info=True)
        
        # Save delivery log and update webhook
        self.db.add(delivery)
        await self.db.commit()
        
        # Check if we need to retry
        if not delivery.is_success and attempt_number < webhook.max_retries:
            # Schedule retry
            delay = webhook.retry_delay_seconds * attempt_number
            await asyncio.sleep(delay)
            await self._deliver_to_webhook(
                webhook, event_type, event_data, event_id,
                attempt_number + 1
            )
        
        # Disable webhook if too many consecutive failures
        if webhook.consecutive_failures >= 10:
            webhook.is_active = False
            logger.warning(f"Webhook {webhook.id} disabled after 10 consecutive failures")
            await self.db.commit()
    
    def _prepare_headers(self, webhook: Webhook, payload: Dict[str, Any]) -> Dict[str, str]:
        """Prepare headers for webhook request."""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "AICtrlNet/1.0",
            "X-Webhook-Event": payload.get("event", ""),
            "X-Webhook-Delivery": str(webhook.total_deliveries + 1)
        }
        
        # Add custom headers
        if webhook.custom_headers:
            headers.update(webhook.custom_headers)
        
        # Add HMAC signature if secret is configured
        if webhook.secret:
            signature = self._calculate_signature(webhook.secret, payload)
            headers["X-Webhook-Signature"] = signature
        
        return headers
    
    def _calculate_signature(self, secret: str, payload: Dict[str, Any]) -> str:
        """Calculate HMAC-SHA256 signature for payload."""
        payload_bytes = json.dumps(payload, sort_keys=True).encode('utf-8')
        signature = hmac.new(
            secret.encode('utf-8'),
            payload_bytes,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"