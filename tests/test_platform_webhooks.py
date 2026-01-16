"""Tests for platform webhook functionality"""
import pytest
import json
import hmac
import hashlib
from datetime import datetime, timedelta
from unittest.mock import patch, AsyncMock, MagicMock
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import status

from src.services.platform_webhook_service import PlatformWebhookService
from src.models.platform_integration import PlatformWebhook, PlatformWebhookDelivery
from src.schemas.platform_integration import (
    PlatformType, WebhookEventType, WebhookDeliveryStatus,
    PlatformWebhookCreate, WebhookEventData
)


@pytest.fixture
async def webhook_service(db_session: AsyncSession):
    """Create webhook service instance"""
    service = PlatformWebhookService(db_session)
    yield service
    # Clean up
    if hasattr(service, 'delivery_task') and service.delivery_task:
        service.delivery_task.cancel()


@pytest.fixture
async def sample_webhook(db_session: AsyncSession) -> PlatformWebhook:
    """Create a sample webhook"""
    webhook = PlatformWebhook(
        platform="n8n",
        webhook_url="https://example.com/webhook",
        secret="test-secret-123",
        events=["execution.completed", "execution.failed"],
        user_id="test-user-123",
        is_active=True,
        verified=True
    )
    db_session.add(webhook)
    await db_session.commit()
    await db_session.refresh(webhook)
    return webhook


class TestWebhookRegistration:
    """Test webhook registration and management"""
    
    async def test_register_webhook(self, webhook_service: PlatformWebhookService):
        """Test registering a new webhook"""
        webhook_data = PlatformWebhookCreate(
            platform=PlatformType.N8N,
            webhook_url="https://example.com/webhook",
            events=[WebhookEventType.EXECUTION_COMPLETED],
            secret="my-secret"
        )
        
        webhook = await webhook_service.register_webhook(
            webhook_data=webhook_data,
            user_id="test-user-123"
        )
        
        assert webhook.id > 0
        assert webhook.platform == "n8n"
        assert webhook.webhook_url == "https://example.com/webhook"
        assert webhook.secret == "my-secret"
        assert webhook.events == ["execution.completed"]
        assert webhook.user_id == "test-user-123"
        assert webhook.is_active is True
        assert webhook.verified is False
    
    async def test_list_webhooks(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test listing webhooks"""
        webhooks = await webhook_service.list_webhooks(user_id="test-user-123")
        
        assert len(webhooks) == 1
        assert webhooks[0].id == sample_webhook.id
    
    async def test_get_webhook(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test getting a specific webhook"""
        webhook = await webhook_service.get_webhook(
            webhook_id=sample_webhook.id,
            user_id="test-user-123"
        )
        
        assert webhook is not None
        assert webhook.id == sample_webhook.id
    
    async def test_update_webhook(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test updating webhook"""
        updated = await webhook_service.update_webhook(
            webhook_id=sample_webhook.id,
            user_id="test-user-123",
            webhook_url="https://new-url.com/webhook",
            events=[WebhookEventType.EXECUTION_STARTED, WebhookEventType.EXECUTION_COMPLETED]
        )
        
        assert updated.webhook_url == "https://new-url.com/webhook"
        assert updated.events == ["execution.started", "execution.completed"]
        assert updated.verified is False  # Should reset verification
    
    async def test_delete_webhook(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test deleting webhook"""
        result = await webhook_service.delete_webhook(
            webhook_id=sample_webhook.id,
            user_id="test-user-123"
        )
        
        assert result is True
        
        # Verify it's deleted
        webhook = await webhook_service.get_webhook(
            webhook_id=sample_webhook.id,
            user_id="test-user-123"
        )
        assert webhook is None


class TestWebhookVerification:
    """Test webhook verification"""
    
    async def test_verify_webhook_success(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test successful webhook verification"""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "verified"}
            mock_post.return_value = mock_response
            
            result = await webhook_service.verify_webhook(
                webhook_id=sample_webhook.id,
                user_id="test-user-123"
            )
            
            assert result is True
            
            # Check webhook is marked as verified
            webhook = await webhook_service.get_webhook(sample_webhook.id, "test-user-123")
            assert webhook.verified is True
    
    async def test_verify_webhook_failure(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test failed webhook verification"""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = Exception("Connection failed")
            
            result = await webhook_service.verify_webhook(
                webhook_id=sample_webhook.id,
                user_id="test-user-123"
            )
            
            assert result is False


class TestWebhookProcessing:
    """Test webhook event processing"""
    
    async def test_process_n8n_webhook(self, webhook_service: PlatformWebhookService):
        """Test processing n8n webhook"""
        headers = {
            "x-n8n-signature": "test-signature",
            "content-type": "application/json"
        }
        body = {
            "workflowId": "123",
            "executionId": "456",
            "status": "success",
            "data": {"result": "completed"}
        }
        
        with patch.object(webhook_service.handlers[PlatformType.N8N], 'verify_signature', return_value=True):
            event = await webhook_service.process_incoming_webhook(
                platform=PlatformType.N8N,
                headers=headers,
                body=body
            )
            
            assert event is not None
            assert event.event_type == WebhookEventType.EXECUTION_COMPLETED
            assert event.platform == PlatformType.N8N
            assert event.event_data["executionId"] == "456"
    
    async def test_process_zapier_webhook(self, webhook_service: PlatformWebhookService):
        """Test processing Zapier webhook"""
        headers = {
            "x-zapier-signature": "test-signature",
            "content-type": "application/json"
        }
        body = {
            "zapId": "zap-123",
            "taskId": "task-456",
            "status": "error",
            "error": {"message": "API limit reached"}
        }
        
        with patch.object(webhook_service.handlers[PlatformType.ZAPIER], 'verify_signature', return_value=True):
            event = await webhook_service.process_incoming_webhook(
                platform=PlatformType.ZAPIER,
                headers=headers,
                body=body
            )
            
            assert event is not None
            assert event.event_type == WebhookEventType.EXECUTION_FAILED
            assert event.platform == PlatformType.ZAPIER
            assert "error" in event.event_data
    
    async def test_process_invalid_signature(self, webhook_service: PlatformWebhookService):
        """Test webhook with invalid signature"""
        headers = {"x-n8n-signature": "invalid"}
        body = {"test": "data"}
        
        with patch.object(webhook_service.handlers[PlatformType.N8N], 'verify_signature', return_value=False):
            with pytest.raises(ValueError, match="Invalid webhook signature"):
                await webhook_service.process_incoming_webhook(
                    platform=PlatformType.N8N,
                    headers=headers,
                    body=body
                )


class TestWebhookDelivery:
    """Test webhook delivery mechanism"""
    
    async def test_trigger_webhook_event(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test triggering webhook event"""
        event_data = WebhookEventData(
            webhook_key="test-key",
            platform=PlatformType.N8N,
            event_type=WebhookEventType.EXECUTION_COMPLETED,
            event_data={"executionId": "123", "status": "success"}
        )
        
        # Mock the delivery queue
        webhook_service.delivery_queue = AsyncMock()
        
        await webhook_service.trigger_webhook_event(event_data)
        
        # Verify webhooks were queued for delivery
        assert webhook_service.delivery_queue.put.called
    
    async def test_deliver_webhook_success(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test successful webhook delivery"""
        delivery = PlatformWebhookDelivery(
            webhook_id=sample_webhook.id,
            event_type="execution.completed",
            payload={"test": "data"},
            status="pending"
        )
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"x-response": "ok"}
            mock_response.text = "Success"
            mock_post.return_value = mock_response
            
            await webhook_service._deliver_webhook(sample_webhook, delivery)
            
            assert delivery.status == "delivered"
            assert delivery.response_status == 200
            assert delivery.attempts == 1
            assert delivery.delivered_at is not None
    
    async def test_deliver_webhook_retry(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test webhook delivery with retry"""
        delivery = PlatformWebhookDelivery(
            webhook_id=sample_webhook.id,
            event_type="execution.failed",
            payload={"test": "data"},
            status="pending"
        )
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = Exception("Connection timeout")
            
            await webhook_service._deliver_webhook(sample_webhook, delivery)
            
            assert delivery.status == "retrying"
            assert delivery.attempts == 1
            assert delivery.next_retry_at is not None
            assert delivery.next_retry_at > datetime.utcnow()
    
    async def test_webhook_max_retries(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook):
        """Test webhook max retries"""
        delivery = PlatformWebhookDelivery(
            webhook_id=sample_webhook.id,
            event_type="execution.failed",
            payload={"test": "data"},
            status="retrying",
            attempts=5  # Already at max
        )
        
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_post.side_effect = Exception("Still failing")
            
            await webhook_service._deliver_webhook(sample_webhook, delivery)
            
            assert delivery.status == "failed"
            assert delivery.attempts == 6
            assert delivery.next_retry_at is None


class TestWebhookStatistics:
    """Test webhook statistics tracking"""
    
    async def test_webhook_stats_update(self, webhook_service: PlatformWebhookService, sample_webhook: PlatformWebhook, db_session: AsyncSession):
        """Test webhook statistics are updated"""
        # Create some deliveries
        deliveries = [
            PlatformWebhookDelivery(
                webhook_id=sample_webhook.id,
                event_type="execution.completed",
                payload={},
                status="delivered",
                attempts=1,
                delivered_at=datetime.utcnow()
            ),
            PlatformWebhookDelivery(
                webhook_id=sample_webhook.id,
                event_type="execution.failed",
                payload={},
                status="failed",
                attempts=6
            )
        ]
        
        for delivery in deliveries:
            db_session.add(delivery)
        await db_session.commit()
        
        # Update stats
        await webhook_service._update_webhook_stats(sample_webhook.id)
        
        # Check updated webhook
        await db_session.refresh(sample_webhook)
        assert sample_webhook.total_deliveries == 2
        assert sample_webhook.successful_deliveries == 1
        assert sample_webhook.failed_deliveries == 1


class TestPlatformSpecificHandlers:
    """Test platform-specific webhook handlers"""
    
    def test_n8n_signature_generation(self, webhook_service: PlatformWebhookService):
        """Test n8n webhook signature generation"""
        handler = webhook_service.handlers[PlatformType.N8N]
        
        body = {"test": "data"}
        secret = "my-secret"
        
        signature = handler._generate_signature(json.dumps(body), secret)
        
        # Verify signature format
        assert signature.startswith("sha256=")
        
        # Verify signature is correct
        expected = "sha256=" + hmac.new(
            secret.encode(),
            json.dumps(body).encode(),
            hashlib.sha256
        ).hexdigest()
        assert signature == expected
    
    def test_zapier_event_parsing(self, webhook_service: PlatformWebhookService):
        """Test Zapier event parsing"""
        handler = webhook_service.handlers[PlatformType.ZAPIER]
        
        # Test success event
        body = {"status": "success", "taskId": "123"}
        event = handler.parse_event(body)
        assert event.event_type == WebhookEventType.EXECUTION_COMPLETED
        
        # Test error event
        body = {"status": "error", "error": "API limit"}
        event = handler.parse_event(body)
        assert event.event_type == WebhookEventType.EXECUTION_FAILED
        
        # Test halted event
        body = {"status": "halted", "reason": "User stopped"}
        event = handler.parse_event(body)
        assert event.event_type == WebhookEventType.EXECUTION_CANCELLED


@pytest.mark.integration
class TestWebhookIntegration:
    """Integration tests for webhook functionality"""
    
    async def test_full_webhook_flow(self, webhook_service: PlatformWebhookService, db_session: AsyncSession):
        """Test complete webhook flow from registration to delivery"""
        # 1. Register webhook
        webhook_data = PlatformWebhookCreate(
            platform=PlatformType.N8N,
            webhook_url="https://example.com/webhook",
            events=[WebhookEventType.EXECUTION_COMPLETED],
            secret="integration-test-secret"
        )
        
        webhook = await webhook_service.register_webhook(webhook_data, "test-user")
        assert webhook.id > 0
        
        # 2. Verify webhook
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            verified = await webhook_service.verify_webhook(webhook.id, "test-user")
            assert verified is True
        
        # 3. Process incoming webhook
        headers = {"x-n8n-signature": "test-sig"}
        body = {
            "workflowId": "wf-123",
            "executionId": "ex-456",
            "status": "success"
        }
        
        with patch.object(webhook_service.handlers[PlatformType.N8N], 'verify_signature', return_value=True):
            event = await webhook_service.process_incoming_webhook(
                PlatformType.N8N, headers, body
            )
            assert event is not None
        
        # 4. Trigger webhook delivery
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_post.return_value = mock_response
            
            await webhook_service.trigger_webhook_event(event)
            
            # Give time for async delivery
            import asyncio
            await asyncio.sleep(0.1)
        
        # 5. Check webhook stats
        await db_session.refresh(webhook)
        assert webhook.last_triggered_at is not None