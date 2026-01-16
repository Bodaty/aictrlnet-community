"""API endpoints for managing webhooks."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from schemas import (
    WebhookCreate, WebhookResponse, WebhookCreateResponse,
    WebhookUpdate, WebhookListResponse, WebhookTestRequest,
    WebhookTestResponse, WebhookDeliveryListResponse,
    SuccessResponse
)
from services.webhook_service import WebhookService

router = APIRouter()


@router.get("/webhooks", response_model=WebhookListResponse)
async def list_webhooks(
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    event_type: Optional[str] = Query(None, description="Filter by event type"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    List all webhooks for the current user.
    
    Available event types:
    - task.* - All task events
    - workflow.* - All workflow events
    - agent.* - All agent events
    - system.* - All system events
    - * - All events
    """
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    return await service.list_user_webhooks(
        user_id=user_id,
        is_active=is_active,
        event_type=event_type
    )


@router.post("/webhooks", response_model=WebhookCreateResponse)
async def create_webhook(
    data: WebhookCreate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Create a new webhook.
    
    The webhook secret (if provided) will be used to sign webhook payloads
    using HMAC-SHA256. The signature will be included in the X-Webhook-Signature header.
    
    Event patterns support wildcards:
    - task.created - Specific event
    - task.* - All task events
    - * - All events
    """
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    return await service.create_webhook(
        user_id=user_id,
        data=data
    )


@router.get("/webhooks/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Get details of a specific webhook."""
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    webhook = await service.get_webhook(
        user_id=user_id,
        webhook_id=webhook_id
    )
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return webhook


@router.put("/webhooks/{webhook_id}", response_model=WebhookResponse)
async def update_webhook(
    webhook_id: str,
    data: WebhookUpdate,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Update a webhook configuration.
    
    Note: Changing the URL will reset failure statistics.
    """
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    webhook = await service.update_webhook(
        user_id=user_id,
        webhook_id=webhook_id,
        data=data
    )
    
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return webhook


@router.post("/webhooks/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook(
    webhook_id: str,
    data: WebhookTestRequest,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Test a webhook by sending a sample payload.
    
    This will send a test event to the webhook URL and return the response.
    The test delivery is not recorded in the delivery history.
    """
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    
    # Verify ownership
    webhook = await service.get_webhook(user_id=user_id, webhook_id=webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    # Send test payload
    result = await service.test_webhook(
        webhook_id=webhook_id,
        event_type=data.event_type or "test.webhook",
        payload=data.payload or {
            "event": "test.webhook",
            "timestamp": "2025-01-23T10:00:00Z",
            "data": {
                "message": "This is a test webhook delivery",
                "test": True
            }
        }
    )
    
    return WebhookTestResponse(
        success=result["success"],
        status_code=result.get("status_code"),
        response_time_ms=result.get("response_time_ms"),
        response_body=result.get("response_body"),
        error=result.get("error")
    )


@router.post("/webhooks/{webhook_id}/enable", response_model=SuccessResponse)
async def enable_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Enable a disabled webhook."""
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    success = await service.enable_webhook(
        user_id=user_id,
        webhook_id=webhook_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return SuccessResponse(
        success=True,
        message="Webhook enabled successfully"
    )


@router.post("/webhooks/{webhook_id}/disable", response_model=SuccessResponse)
async def disable_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """Disable a webhook without deleting it."""
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    success = await service.disable_webhook(
        user_id=user_id,
        webhook_id=webhook_id
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return SuccessResponse(
        success=True,
        message="Webhook disabled successfully"
    )


@router.get("/webhooks/{webhook_id}/deliveries", response_model=WebhookDeliveryListResponse)
async def list_webhook_deliveries(
    webhook_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum deliveries to return"),
    is_success: Optional[bool] = Query(None, description="Filter by success status"),
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    List recent delivery attempts for a webhook.
    
    Returns the most recent deliveries first.
    """
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    
    # Verify ownership
    webhook = await service.get_webhook(user_id=user_id, webhook_id=webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    return await service.list_webhook_deliveries(
        webhook_id=webhook_id,
        limit=limit,
        is_success=is_success
    )


@router.delete("/webhooks/{webhook_id}", response_model=SuccessResponse)
async def delete_webhook(
    webhook_id: str,
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Delete a webhook permanently.
    
    This will also delete all delivery history for the webhook.
    """
    service = WebhookService(db)
    user_id = current_user.id if hasattr(current_user, 'id') else current_user.get("id", current_user.get("sub"))
    
    # Verify ownership
    webhook = await service.get_webhook(user_id=user_id, webhook_id=webhook_id)
    if not webhook:
        raise HTTPException(status_code=404, detail="Webhook not found")
    
    await service.delete_webhook(webhook_id)
    
    return SuccessResponse(
        success=True,
        message="Webhook deleted successfully"
    )