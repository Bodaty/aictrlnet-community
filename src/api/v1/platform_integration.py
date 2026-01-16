"""Platform Integration API endpoints"""
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import json
import logging
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from core.database import get_db
from core.dependencies import get_current_user_safe
from schemas.platform_integration import (
    PlatformType,
    PlatformCredentialCreate,
    PlatformCredentialUpdate,
    PlatformCredentialResponse,
    PlatformAdapterResponse,
    PlatformWorkflowInfo,
    PlatformExecutionCreate,
    PlatformExecutionResponse,
    PlatformHealthResponse,
    PlatformCapabilities,
    PlatformNodeConfig,
    WorkflowBrowserRequest,
    WorkflowBrowserResponse,
    TestExecutionRequest,
    TestExecutionResponse,
    PlatformNodeUIConfig,
    WebhookEventType,
    WebhookDeliveryStatus,
    PlatformWebhookCreate,
    PlatformWebhookUpdate,
    PlatformWebhookResponse,
    WebhookDeliveryResponse,
    WebhookEventRequest,
    WebhookVerificationRequest
)
from services.platform_credential_service import PlatformCredentialService
from services.platform_adapters import PlatformAdapterService, ExecutionRequest
from services.platform_usage_tracker import PlatformUsageTracker
from services.cloud_detection import cloud_detector
from services.platform_cloud_config import platform_cloud_config
from services.platform_usage_limiter import PlatformUsageLimiter
from services.platform_monitoring_service import PlatformMonitoringService
from services.platform_cost_optimizer import PlatformCostOptimizer
from services.platform_execution_cache import PlatformExecutionCache

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Platform Integration"])


@router.post("/credentials", response_model=PlatformCredentialResponse)
async def create_credential(
    credential_data: PlatformCredentialCreate,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Create a new platform credential"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    credential = await service.create_credential(
        user_id=current_user.get("id"),
        credential_data=credential_data
    )
    
    return credential


@router.get("/credentials", response_model=List[PlatformCredentialResponse])
async def list_credentials(
    platform: Optional[PlatformType] = None,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """List user's platform credentials"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    credentials = await service.list_credentials(
        user_id=current_user.get("id"),
        platform=platform.value if platform else None
    )
    
    return credentials


@router.get("/credentials/{credential_id}", response_model=PlatformCredentialResponse)
async def get_credential(
    credential_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific platform credential"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    credential = await service.get_credential(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    return credential


@router.patch("/credentials/{credential_id}", response_model=PlatformCredentialResponse)
async def update_credential(
    credential_id: int,
    update_data: PlatformCredentialUpdate,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Update a platform credential"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    credential = await service.update_credential(
        credential_id=credential_id,
        user_id=current_user.get("id"),
        update_data=update_data
    )
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    return credential


@router.delete("/credentials/{credential_id}")
async def delete_credential(
    credential_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Delete a platform credential"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    deleted = await service.delete_credential(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    return {"message": "Credential deleted successfully"}


@router.post("/credentials/{credential_id}/validate")
async def validate_credential(
    credential_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Validate a platform credential"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get credential
    credential = await service.get_credential(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found"
        )
    
    # Get credential data
    credential_data = await service.get_credential_data(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential_data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve credential data"
        )
    
    # Get adapter
    platform_type = PlatformType(credential.platform)
    adapter = adapter_service.get_adapter_instance(platform_type)
    
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {platform_type.value}"
        )
    
    # Validate
    is_valid, error_message = await adapter.validate_credentials(credential_data)
    
    return {
        "valid": is_valid,
        "error": error_message,
        "platform": platform_type.value
    }


@router.get("/adapters", response_model=List[PlatformAdapterResponse])
async def list_adapters(
    active_only: bool = Query(True, description="Only return active adapters"),
    db: AsyncSession = Depends(get_db)
):
    """List available platform adapters"""
    service = PlatformAdapterService(db)
    
    # Sync adapters from registry to database
    await service.sync_adapters_to_db()
    
    # Get adapters
    adapters = await service.list_adapters(active_only=active_only)
    
    return adapters


@router.get("/adapters/{platform}", response_model=PlatformAdapterResponse)
async def get_adapter(
    platform: PlatformType,
    db: AsyncSession = Depends(get_db)
):
    """Get information about a specific platform adapter"""
    service = PlatformAdapterService(db)
    
    adapter_info = await service.get_adapter_info(platform)
    
    if not adapter_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter not found for platform {platform.value}"
        )
    
    return adapter_info


@router.get("/adapters/{platform}/capabilities", response_model=PlatformCapabilities)
async def get_adapter_capabilities(
    platform: PlatformType,
    db: AsyncSession = Depends(get_db)
):
    """Get capabilities of a platform adapter"""
    service = PlatformAdapterService(db)
    
    adapter = service.get_adapter_instance(platform)
    
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Adapter not found for platform {platform.value}"
        )
    
    return adapter.get_capabilities()


@router.get("/workflows", response_model=List[PlatformWorkflowInfo])
async def list_platform_workflows(
    platform: PlatformType,
    credential_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """List workflows available on a platform"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get credential data
    credential_data = await service.get_credential_data(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found or unauthorized"
        )
    
    # Get adapter
    adapter = adapter_service.get_adapter_instance(platform)
    
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {platform.value}"
        )
    
    # List workflows
    try:
        workflows = await adapter.list_workflows(
            credentials=credential_data,
            limit=limit,
            offset=offset
        )
        return workflows
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list workflows: {str(e)}"
        )


@router.get("/workflows/{platform}/{workflow_id}", response_model=PlatformWorkflowInfo)
async def get_platform_workflow(
    platform: PlatformType,
    workflow_id: str,
    credential_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get information about a specific platform workflow"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get credential data
    credential_data = await service.get_credential_data(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found or unauthorized"
        )
    
    # Get adapter
    adapter = adapter_service.get_adapter_instance(platform)
    
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {platform.value}"
        )
    
    # Get workflow info
    try:
        workflow_info = await adapter.get_workflow_info(
            credentials=credential_data,
            workflow_id=workflow_id
        )
        
        if not workflow_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow not found on platform"
            )
        
        return workflow_info
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow info: {str(e)}"
        )


@router.post("/execute", response_model=PlatformExecutionResponse)
async def execute_platform_workflow(
    execution_request: PlatformExecutionCreate,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Execute a workflow on an external platform"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformExecution
    from datetime import datetime
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get credential data
    credential_data = await service.get_credential_data(
        credential_id=execution_request.credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found or unauthorized"
        )
    
    # Get adapter
    adapter = adapter_service.get_adapter_instance(execution_request.platform)
    
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {execution_request.platform.value}"
        )
    
    # Create execution record
    db_execution = PlatformExecution(
        workflow_id=execution_request.workflow_id,
        node_id=execution_request.node_id,
        execution_id=f"api-{datetime.utcnow().timestamp()}",
        platform=execution_request.platform.value,
        external_workflow_id=execution_request.external_workflow_id,
        credential_id=execution_request.credential_id,
        input_data=execution_request.input_data,
        status="running",
        started_at=datetime.utcnow(),
        execution_metadata=execution_request.execution_metadata
    )
    db.add(db_execution)
    db.commit()
    db.refresh(db_execution)
    
    # Execute workflow
    try:
        request = ExecutionRequest(
            workflow_id=execution_request.external_workflow_id,
            input_data=execution_request.input_data,
            timeout=300,  # 5 minutes default
            metadata=execution_request.execution_metadata
        )
        
        response = await adapter.execute_workflow(
            credentials=credential_data,
            request=request
        )
        
        # Update execution record
        db_execution.external_execution_id = response.execution_id
        db_execution.status = response.status.value
        db_execution.output_data = response.output_data
        db_execution.error_data = {"error": response.error} if response.error else None
        db_execution.completed_at = response.completed_at or datetime.utcnow()
        db_execution.duration_ms = response.duration_ms
        db_execution.estimated_cost = response.cost_estimate or 0
        
        db.commit()
        db.refresh(db_execution)
        
        return PlatformExecutionResponse.model_validate(db_execution)
        
    except Exception as e:
        # Update execution record with error
        db_execution.status = "failed"
        db_execution.error_data = {"error": str(e)}
        db_execution.completed_at = datetime.utcnow()
        db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Execution failed: {str(e)}"
        )


@router.get("/executions/{execution_id}", response_model=PlatformExecutionResponse)
async def get_execution_status(
    execution_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get status of a platform execution"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformExecution, PlatformCredential
    
    # Get execution
    execution = db.query(PlatformExecution).join(
        PlatformCredential,
        PlatformExecution.credential_id == PlatformCredential.id
    ).filter(
        PlatformExecution.id == execution_id,
        PlatformCredential.user_id == current_user.get("id")
    ).first()
    
    if not execution:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Execution not found"
        )
    
    return PlatformExecutionResponse.model_validate(execution)


@router.get("/health", response_model=List[PlatformHealthResponse])
async def get_platform_health(
    platform: Optional[PlatformType] = None,
    db: AsyncSession = Depends(get_db)
):
    """Get health status of platforms"""
    from models.platform_integration import PlatformHealth
    
    query = db.query(PlatformHealth)
    
    if platform:
        query = query.filter(PlatformHealth.platform == platform.value)
    
    health_records = query.all()
    
    return [PlatformHealthResponse.model_validate(h) for h in health_records]


@router.post("/health/{platform}/check")
async def check_platform_health(
    platform: PlatformType,
    credential_id: Optional[int] = None,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Perform a health check on a platform"""
    from models.platform_integration import PlatformHealth
    from datetime import datetime
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get adapter
    adapter = adapter_service.get_adapter_instance(platform)
    
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {platform.value}"
        )
    
    # Get credentials if provided
    credential_data = None
    if credential_id and current_user:
        credential_data = await service.get_credential_data(
            credential_id=credential_id,
            user_id=current_user.get("id")
        )
    
    # Perform health check
    health_result = await adapter.health_check(credentials=credential_data)
    
    # Update or create health record
    health_record = db.query(PlatformHealth).filter(
        PlatformHealth.platform == platform.value
    ).first()
    
    if health_record:
        health_record.is_healthy = health_result.is_healthy
        health_record.last_check_at = datetime.utcnow()
        health_record.response_time_ms = health_result.response_time_ms
        health_record.last_error = health_result.error
        health_record.health_metadata = health_result.details or {}
        
        if not health_result.is_healthy:
            health_record.consecutive_failures += 1
            health_record.failed_checks += 1
        else:
            health_record.consecutive_failures = 0
        
        health_record.total_checks += 1
        
        # Update uptime percentage
        if health_record.total_checks > 0:
            health_record.uptime_percentage = int(
                ((health_record.total_checks - health_record.failed_checks) / health_record.total_checks) * 100
            )
    else:
        health_record = PlatformHealth(
            platform=platform.value,
            is_healthy=health_result.is_healthy,
            last_check_at=datetime.utcnow(),
            response_time_ms=health_result.response_time_ms,
            last_error=health_result.error,
            health_metadata=health_result.details or {},
            consecutive_failures=0 if health_result.is_healthy else 1,
            total_checks=1,
            failed_checks=0 if health_result.is_healthy else 1,
            uptime_percentage=100 if health_result.is_healthy else 0
        )
        db.add(health_record)
    
    db.commit()
    
    return {
        "platform": platform.value,
        "healthy": health_result.is_healthy,
        "response_time_ms": health_result.response_time_ms,
        "error": health_result.error,
        "details": health_result.details
    }


@router.get("/usage/me")
async def get_my_usage(
    platform: Optional[PlatformType] = None,
    days: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get current user's platform usage statistics"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    tracker = PlatformUsageTracker(db)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    usage = await tracker.get_user_usage(
        user_id=current_user.get("id"),
        platform=platform,
        start_date=start_date
    )
    
    return usage


@router.get("/usage/platform/{platform}")
async def get_platform_usage(
    platform: PlatformType,
    days: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get platform usage statistics (admin only)"""
    if not current_user or not current_user.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    tracker = PlatformUsageTracker(db)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    usage = await tracker.get_platform_usage(
        platform=platform,
        start_date=start_date
    )
    
    return usage


@router.get("/usage/rate-limits")
async def check_rate_limits(
    platform: PlatformType,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Check if approaching rate limits for a platform"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    tracker = PlatformUsageTracker(db)
    
    limits = await tracker.check_rate_limits(
        user_id=current_user.get("id"),
        platform=platform
    )
    
    return limits


@router.get("/usage/costs")
async def get_cost_summary(
    days: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get cost summary for platform usage"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    tracker = PlatformUsageTracker(db)
    start_date = datetime.utcnow() - timedelta(days=days)
    
    costs = await tracker.get_cost_summary(
        user_id=current_user.get("id"),
        start_date=start_date
    )
    
    return costs


@router.get("/usage/trending")
async def get_trending_workflows(
    platform: Optional[PlatformType] = None,
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db)
):
    """Get trending workflows across platforms"""
    tracker = PlatformUsageTracker(db)
    
    trending = await tracker.get_trending_workflows(
        platform=platform,
        limit=limit
    )
    
    return {"workflows": trending}


@router.get("/cloud/info")
async def get_cloud_info():
    """Get information about the cloud deployment environment"""
    return {
        "provider": cloud_detector.provider.value,
        "environment": cloud_detector.environment.value,
        "is_cloud": cloud_detector.is_cloud,
        "is_serverless": cloud_detector.is_serverless,
        "resource_limits": cloud_detector.get_resource_limits(),
        "scaling_config": cloud_detector.get_scaling_config(),
        "networking_config": cloud_detector.get_networking_config(),
        "metadata": cloud_detector.metadata
    }


@router.get("/cloud/limits/{platform}")
async def get_platform_cloud_limits(
    platform: PlatformType,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get cloud-aware limits for a specific platform"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Get base platform limits adjusted for cloud
    platform_limits = platform_cloud_config.get_platform_limits(platform)
    
    # Get usage limiter for comprehensive check
    limiter = PlatformUsageLimiter(db)
    usage_summary = await limiter.get_usage_summary(current_user.get("id"))
    
    # Get platform compatibility info
    compatibility = platform_cloud_config.validate_platform_compatibility(platform)
    
    return {
        "platform": platform.value,
        "limits": platform_limits,
        "cloud_environment": usage_summary["cloud_environment"],
        "compatibility": compatibility,
        "retry_config": platform_cloud_config.get_retry_config(platform),
        "webhook_config": platform_cloud_config.get_webhook_config(platform),
        "should_use_async": platform_cloud_config.should_use_async(platform)
    }


@router.get("/cloud/config")
async def get_cloud_configuration():
    """Get cloud-specific configuration settings"""
    return {
        "cloud_provider": platform_cloud_config.cloud_provider.value,
        "environment": platform_cloud_config.environment.value,
        "is_serverless": platform_cloud_config.is_serverless,
        "configurations": {
            "caching": platform_cloud_config.get_caching_config(),
            "monitoring": platform_cloud_config.get_monitoring_config(),
            "security": platform_cloud_config.get_security_config(),
            "cost_tracking": platform_cloud_config.get_cost_config()
        }
    }


@router.post("/check-limits")
async def check_execution_limits(
    platform: PlatformType,
    estimated_duration_ms: Optional[int] = None,
    payload_size_mb: Optional[float] = None,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Check if a platform execution would be allowed under current limits"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    limiter = PlatformUsageLimiter(db)
    
    # Check all limits
    result = await limiter.check_execution_allowed(
        user_id=current_user.get("id"),
        platform=platform,
        estimated_duration_ms=estimated_duration_ms,
        payload_size_mb=payload_size_mb
    )
    
    return result


@router.get("/usage/limits")
async def get_usage_limits_summary(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive usage limits summary for current user"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    limiter = PlatformUsageLimiter(db)
    summary = await limiter.get_usage_summary(current_user.get("id"))
    
    return summary


@router.post("/workflows/browse", response_model=WorkflowBrowserResponse)
async def browse_platform_workflows(
    request: WorkflowBrowserRequest,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Browse workflows on a platform with pagination and search"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get credential data
    credential_data = await service.get_credential_data(
        credential_id=request.credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found or not accessible"
        )
    
    # Get adapter and list workflows
    adapter = adapter_service.get_adapter_instance(request.platform)
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {request.platform.value}"
        )
    
    # Configure adapter with credentials
    adapter.configure(credential_data)
    
    # Calculate offset
    offset = (request.page - 1) * request.page_size
    
    # Get workflows with pagination
    workflows = await adapter.list_workflows(
        limit=request.page_size,
        offset=offset,
        search_term=request.search_term,
        status_filter=request.status_filter
    )
    
    # Get total count
    total = await adapter.get_workflow_count(
        search_term=request.search_term,
        status_filter=request.status_filter
    )
    
    has_more = (offset + len(workflows)) < total
    
    return WorkflowBrowserResponse(
        workflows=workflows,
        total=total,
        page=request.page,
        page_size=request.page_size,
        has_more=has_more
    )


@router.post("/workflows/test", response_model=TestExecutionResponse)
async def test_platform_workflow(
    request: TestExecutionRequest,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Test execute a platform workflow with sample data"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get credential data
    credential_data = await service.get_credential_data(
        credential_id=request.credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found or not accessible"
        )
    
    # Get adapter
    adapter = adapter_service.get_adapter_instance(request.platform)
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {request.platform.value}"
        )
    
    # Configure adapter with credentials
    adapter.configure(credential_data)
    
    # Execute test
    import time
    start_time = time.time()
    
    try:
        result = await adapter.test_workflow(
            workflow_id=request.workflow_id,
            test_inputs=request.test_inputs,
            timeout=request.timeout
        )
        
        execution_time = int((time.time() - start_time) * 1000)
        
        return TestExecutionResponse(
            success=True,
            execution_time_ms=execution_time,
            output_data=result.get("output"),
            logs=result.get("logs", [])
        )
        
    except Exception as e:
        execution_time = int((time.time() - start_time) * 1000)
        
        return TestExecutionResponse(
            success=False,
            execution_time_ms=execution_time,
            error_message=str(e),
            logs=[]
        )


@router.get("/workflows/{platform}/{workflow_id}/ui-config", response_model=PlatformNodeUIConfig)
async def get_workflow_ui_config(
    platform: PlatformType,
    workflow_id: str,
    credential_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get UI configuration for a specific workflow"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    service = PlatformCredentialService(db)
    adapter_service = PlatformAdapterService(db)
    
    # Get credential
    credential = await service.get_credential(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    if not credential:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Credential not found or not accessible"
        )
    
    # Get credential data
    credential_data = await service.get_credential_data(
        credential_id=credential_id,
        user_id=current_user.get("id")
    )
    
    # Get adapter
    adapter = adapter_service.get_adapter_instance(platform)
    if not adapter:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Platform adapter not available for {platform.value}"
        )
    
    # Configure adapter with credentials
    adapter.configure(credential_data)
    
    # Get workflow details
    workflow_info = await adapter.get_workflow(workflow_id)
    
    if not workflow_info:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow not found on platform"
        )
    
    # Build UI config
    ui_config = PlatformNodeUIConfig(
        platform=platform,
        workflow_id=workflow_id,
        workflow_name=workflow_info.name,
        credential_id=credential_id,
        credential_name=credential.name,
        description=workflow_info.description,
        tags=workflow_info.tags,
        icon_url=f"/icons/{platform.value}-logo.svg",
        supports_test=True,
        test_data=workflow_info.metadata.get("test_data") if workflow_info.metadata else None,
        input_fields=workflow_info.input_schema,
        output_fields=workflow_info.output_schema
    )
    
    return ui_config


@router.get("/credentials/backend/health")
async def check_credential_backend_health(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Check health of the configured credential backend"""
    if not current_user or not current_user.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    import os
    backend_type = os.environ.get("CREDENTIAL_BACKEND", "database").lower()
    
    health_info = {
        "backend_type": backend_type,
        "healthy": False,
        "details": {}
    }
    
    try:
        if backend_type == "vault":
            # Check Vault health
            from services.vault_credential_backend import VaultCredentialBackend
            vault_backend = VaultCredentialBackend()
            vault_health = vault_backend.health_check()
            
            health_info["healthy"] = vault_health.get("healthy", False)
            health_info["details"] = vault_health
            
        elif backend_type == "database":
            # Check database connection
            try:
                # Simple query to check database
                await db.execute(text("SELECT 1"))
                health_info["healthy"] = True
                health_info["details"] = {
                    "connection": "ok",
                    "encryption_key_set": bool(os.environ.get("PLATFORM_CREDENTIAL_KEY"))
                }
            except Exception as e:
                health_info["details"] = {
                    "error": str(e)
                }
                
        elif backend_type == "file":
            # Check file access
            file_path = os.environ.get("CREDENTIAL_FILE_PATH", "/app/data/credentials.json")
            health_info["details"]["file_path"] = file_path
            health_info["details"]["file_exists"] = os.path.exists(file_path)
            health_info["details"]["file_writable"] = os.access(os.path.dirname(file_path), os.W_OK)
            health_info["details"]["encryption_key_set"] = bool(os.environ.get("CREDENTIAL_ENCRYPTION_KEY"))
            health_info["healthy"] = health_info["details"]["file_writable"]
            
        elif backend_type == "environment":
            # Check for any platform credentials in environment
            cred_count = sum(1 for k in os.environ if k.startswith("PLATFORM_CRED_"))
            health_info["healthy"] = True
            health_info["details"] = {
                "credential_count": cred_count,
                "read_only": True
            }
            
        else:
            health_info["details"] = {
                "error": f"Unknown backend type: {backend_type}"
            }
            
    except Exception as e:
        health_info["details"]["error"] = str(e)
    
    return health_info


@router.get("/credentials/backend/info")
async def get_credential_backend_info():
    """Get information about credential backend configuration"""
    import os
    
    backend_type = os.environ.get("CREDENTIAL_BACKEND", "database").lower()
    
    info = {
        "backend_type": backend_type,
        "configuration": {},
        "capabilities": {}
    }
    
    # Common capabilities
    capabilities = {
        "read": True,
        "write": True,
        "delete": True,
        "list": True,
        "rotate": False,
        "audit": False,
        "versioning": False
    }
    
    if backend_type == "environment":
        capabilities["write"] = False
        capabilities["delete"] = False
        info["configuration"] = {
            "read_only": True,
            "prefix": "PLATFORM_CRED_"
        }
        
    elif backend_type == "file":
        info["configuration"] = {
            "file_path": os.environ.get("CREDENTIAL_FILE_PATH", "/app/data/credentials.json"),
            "encrypted": True
        }
        
    elif backend_type == "database":
        info["configuration"] = {
            "encrypted": True,
            "multi_tenant": True,
            "soft_delete": True
        }
        
    elif backend_type == "vault":
        capabilities["rotate"] = True
        capabilities["audit"] = True
        capabilities["versioning"] = True
        info["configuration"] = {
            "vault_url": os.environ.get("VAULT_URL", "http://localhost:8200"),
            "mount_point": "secret",
            "path_prefix": "aictrlnet/platforms",
            "features": ["encryption", "versioning", "audit", "rotation", "access_control"]
        }
    
    info["capabilities"] = capabilities
    
    return info


@router.post("/webhooks", response_model=PlatformWebhookResponse)
async def register_webhook(
    webhook_data: PlatformWebhookCreate,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Register a new webhook endpoint"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from services.platform_webhook_service import PlatformWebhookService
    
    webhook_service = PlatformWebhookService(db)
    
    webhook = await webhook_service.register_webhook(
        platform=webhook_data.platform,
        webhook_url=webhook_data.webhook_url,
        events=webhook_data.events,
        secret=webhook_data.secret,
        user_id=current_user.get("id"),
        metadata=webhook_data.metadata
    )
    
    return PlatformWebhookResponse.model_validate(webhook)


@router.get("/webhooks", response_model=List[PlatformWebhookResponse])
async def list_webhooks(
    platform: Optional[PlatformType] = None,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """List registered webhooks"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformWebhook
    from sqlalchemy import select
    
    query = select(PlatformWebhook).where(
        PlatformWebhook.user_id == current_user.get("id")
    )
    
    if platform:
        query = query.where(PlatformWebhook.platform == platform.value)
    
    result = await db.execute(query)
    webhooks = result.scalars().all()
    
    return [PlatformWebhookResponse.model_validate(w) for w in webhooks]


@router.get("/webhooks/{webhook_id}", response_model=PlatformWebhookResponse)
async def get_webhook(
    webhook_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get webhook details"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformWebhook
    from sqlalchemy import select
    
    result = await db.execute(
        select(PlatformWebhook).where(
            PlatformWebhook.id == webhook_id,
            PlatformWebhook.user_id == current_user.get("id")
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    return PlatformWebhookResponse.model_validate(webhook)


@router.patch("/webhooks/{webhook_id}", response_model=PlatformWebhookResponse)
async def update_webhook(
    webhook_id: int,
    update_data: PlatformWebhookUpdate,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Update webhook configuration"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformWebhook
    from sqlalchemy import select
    
    result = await db.execute(
        select(PlatformWebhook).where(
            PlatformWebhook.id == webhook_id,
            PlatformWebhook.user_id == current_user.get("id")
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    # Update fields
    if update_data.webhook_url is not None:
        webhook.webhook_url = update_data.webhook_url
    if update_data.events is not None:
        webhook.events = [e.value for e in update_data.events]
    if update_data.is_active is not None:
        webhook.is_active = update_data.is_active
    if update_data.metadata is not None:
        webhook.webhook_metadata = update_data.metadata
    
    await db.commit()
    await db.refresh(webhook)
    
    return PlatformWebhookResponse.model_validate(webhook)


@router.delete("/webhooks/{webhook_id}")
async def delete_webhook(
    webhook_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Delete a webhook"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformWebhook
    from sqlalchemy import select
    
    result = await db.execute(
        select(PlatformWebhook).where(
            PlatformWebhook.id == webhook_id,
            PlatformWebhook.user_id == current_user.get("id")
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    await db.delete(webhook)
    await db.commit()
    
    return {"message": "Webhook deleted successfully"}


@router.post("/webhooks/{webhook_id}/verify", response_model=Dict[str, Any])
async def verify_webhook(
    webhook_id: int,
    verify_request: WebhookVerificationRequest,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Verify webhook endpoint is reachable"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformWebhook
    from services.platform_webhook_service import PlatformWebhookService
    
    webhook = db.query(PlatformWebhook).filter(
        PlatformWebhook.id == webhook_id,
        PlatformWebhook.user_id == current_user.get("id")
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    webhook_service = PlatformWebhookService(db)
    
    # Send test webhook
    test_payload = verify_request.test_payload or {
        "test": True,
        "timestamp": datetime.utcnow().isoformat(),
        "webhook_id": webhook_id
    }
    
    # Deliver test webhook
    delivery_id = await webhook_service._trigger_webhooks(
        platform=PlatformType(webhook.platform),
        event_type=WebhookEventType.OTHER,
        event_data=test_payload
    )
    
    webhook.verified = True
    await db.commit()
    
    return {
        "verified": True,
        "webhook_id": webhook_id,
        "test_sent": True
    }


@router.get("/webhooks/{webhook_id}/deliveries", response_model=List[WebhookDeliveryResponse])
async def get_webhook_deliveries(
    webhook_id: int,
    limit: int = Query(50, ge=1, le=200),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get webhook delivery history"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformWebhook, PlatformWebhookDelivery
    from sqlalchemy import select
    
    # Verify webhook ownership
    result = await db.execute(
        select(PlatformWebhook).where(
            PlatformWebhook.id == webhook_id,
            PlatformWebhook.user_id == current_user.get("id")
        )
    )
    webhook = result.scalar_one_or_none()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    result = await db.execute(
        select(PlatformWebhookDelivery)
        .where(PlatformWebhookDelivery.webhook_id == webhook_id)
        .order_by(PlatformWebhookDelivery.created_at.desc())
        .limit(limit)
    )
    deliveries = result.scalars().all()
    
    return [WebhookDeliveryResponse.model_validate(d) for d in deliveries]


@router.get("/webhooks/{webhook_id}/stats", response_model=Dict[str, Any])
async def get_webhook_stats(
    webhook_id: int,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get webhook statistics"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    from models.platform_integration import PlatformWebhook
    from services.platform_webhook_service import PlatformWebhookService
    
    # Verify webhook ownership
    webhook = db.query(PlatformWebhook).filter(
        PlatformWebhook.id == webhook_id,
        PlatformWebhook.user_id == current_user.get("id")
    ).first()
    
    if not webhook:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    webhook_service = PlatformWebhookService(db)
    stats = await webhook_service.get_webhook_stats(webhook_id)
    
    return stats


@router.post("/webhooks/incoming/{platform}", response_model=Dict[str, Any])
async def receive_platform_webhook(
    platform: PlatformType,
    request: WebhookEventRequest,
    db: AsyncSession = Depends(get_db)
):
    """Receive incoming webhook from platform"""
    from services.platform_webhook_service import PlatformWebhookService
    
    webhook_service = PlatformWebhookService(db)
    
    try:
        result = await webhook_service.process_webhook(
            platform=platform,
            headers=request.headers,
            body=request.body.encode() if isinstance(request.body, str) else json.dumps(request.body).encode()
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error processing webhook: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process webhook"
        )


# ==================== MONITORING ENDPOINTS ====================

@router.post("/monitoring/start")
async def start_real_time_monitoring(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Start real-time monitoring for current user"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    monitoring_service = PlatformMonitoringService(db)
    result = await monitoring_service.start_real_time_monitoring(
        user_id=current_user.get("id")
    )
    
    return result


@router.post("/monitoring/stop")
async def stop_real_time_monitoring(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Stop real-time monitoring for current user"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    monitoring_service = PlatformMonitoringService(db)
    result = await monitoring_service.stop_real_time_monitoring(
        user_id=current_user.get("id")
    )
    
    return result


@router.get("/monitoring/metrics/realtime")
async def get_real_time_metrics(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get current real-time metrics"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    # Try to get from cache first
    from core.cache import CacheService
    cache = CacheService()
    
    metrics = await cache.get(f"platform_metrics:{current_user.get('id')}")
    
    if not metrics:
        # Calculate fresh metrics
        monitoring_service = PlatformMonitoringService(db)
        executions = await monitoring_service._get_recent_executions(
            current_user.get("id"), 
            minutes=5
        )
        metrics = await monitoring_service._calculate_real_time_metrics(executions)
    
    return metrics


@router.get("/monitoring/analytics")
async def get_cross_platform_analytics(
    start_date: Optional[datetime] = Query(None),
    end_date: Optional[datetime] = Query(None),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get comprehensive cross-platform analytics"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    monitoring_service = PlatformMonitoringService(db)
    analytics = await monitoring_service.get_cross_platform_analytics(
        user_id=current_user.get("id"),
        start_date=start_date,
        end_date=end_date
    )
    
    return analytics


@router.get("/monitoring/recommendations")
async def get_performance_recommendations(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get performance optimization recommendations"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    monitoring_service = PlatformMonitoringService(db)
    recommendations = await monitoring_service.get_performance_recommendations(
        user_id=current_user.get("id")
    )
    
    return {
        "recommendations": recommendations,
        "generated_at": datetime.utcnow().isoformat()
    }


@router.get("/monitoring/anomalies")
async def detect_execution_anomalies(
    time_window_minutes: int = Query(60, ge=5, le=1440),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Detect anomalies in recent executions"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    monitoring_service = PlatformMonitoringService(db)
    anomalies = await monitoring_service.detect_anomalies(
        user_id=current_user.get("id"),
        time_window_minutes=time_window_minutes
    )
    
    return {
        "anomalies": anomalies,
        "time_window_minutes": time_window_minutes,
        "detected_at": datetime.utcnow().isoformat()
    }


@router.get("/monitoring/dashboard")
async def get_monitoring_dashboard_data(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get all monitoring data for dashboard display"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    monitoring_service = PlatformMonitoringService(db)
    
    # Get various monitoring data in parallel
    import asyncio
    
    # Create tasks
    tasks = {
        "real_time_metrics": monitoring_service._get_recent_executions(
            current_user.get("id"), minutes=5
        ),
        "analytics": monitoring_service.get_cross_platform_analytics(
            user_id=current_user.get("id")
        ),
        "anomalies": monitoring_service.detect_anomalies(
            user_id=current_user.get("id"),
            time_window_minutes=60
        ),
        "recommendations": monitoring_service.get_performance_recommendations(
            user_id=current_user.get("id")
        )
    }
    
    # Execute all tasks
    results = await asyncio.gather(*tasks.values(), return_exceptions=True)
    
    # Process results
    real_time_executions = results[0] if not isinstance(results[0], Exception) else []
    real_time_metrics = await monitoring_service._calculate_real_time_metrics(
        real_time_executions
    )
    
    dashboard_data = {
        "real_time_metrics": real_time_metrics,
        "analytics": results[1] if not isinstance(results[1], Exception) else {},
        "anomalies": results[2] if not isinstance(results[2], Exception) else [],
        "recommendations": results[3] if not isinstance(results[3], Exception) else [],
        "last_updated": datetime.utcnow().isoformat()
    }
    
    return dashboard_data


# ==================== COST OPTIMIZATION ENDPOINTS ====================

@router.post("/cost/predict")
async def predict_execution_cost(
    platform: PlatformType,
    workflow_id: str,
    estimated_duration_ms: Optional[int] = None,
    estimated_data_mb: Optional[float] = None,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Predict cost for a workflow execution"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cost_optimizer = PlatformCostOptimizer(db)
    prediction = await cost_optimizer.predict_execution_cost(
        platform=platform,
        workflow_id=workflow_id,
        estimated_duration_ms=estimated_duration_ms,
        estimated_data_mb=estimated_data_mb,
        user_id=current_user.get("id")
    )
    
    return prediction


@router.post("/cost/optimize-routing")
async def optimize_workflow_routing(
    workflow_type: str,
    requirements: Dict[str, Any],
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get optimal platform routing for workflow"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cost_optimizer = PlatformCostOptimizer(db)
    routing = await cost_optimizer.optimize_workflow_routing(
        workflow_type=workflow_type,
        requirements=requirements,
        user_id=current_user.get("id")
    )
    
    return routing


@router.post("/cost/budget-alert")
async def create_budget_alert(
    budget_config: Dict[str, Any],
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Create budget alert configuration"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cost_optimizer = PlatformCostOptimizer(db)
    result = await cost_optimizer.create_budget_alert(
        user_id=current_user.get("id"),
        budget_config=budget_config
    )
    
    return result


@router.get("/cost/budget-status")
async def get_budget_status(
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Check current budget status and alerts"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cost_optimizer = PlatformCostOptimizer(db)
    status = await cost_optimizer.check_budget_status(
        user_id=current_user.get("id")
    )
    
    return status


@router.get("/cost/optimization-opportunities")
async def get_cost_optimization_opportunities(
    lookback_days: int = Query(30, ge=1, le=90),
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Identify cost optimization opportunities"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cost_optimizer = PlatformCostOptimizer(db)
    opportunities = await cost_optimizer.get_cost_optimization_opportunities(
        user_id=current_user.get("id"),
        lookback_days=lookback_days
    )
    
    return {
        "opportunities": opportunities,
        "lookback_days": lookback_days,
        "generated_at": datetime.utcnow().isoformat()
    }


# ==================== EXECUTION CACHE ENDPOINTS ====================

@router.get("/cache/check")
async def check_cache(
    platform: PlatformType,
    workflow_id: str,
    input_data: Dict[str, Any],
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Check if execution result is cached"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cache_service = PlatformExecutionCache(db)
    cached_result = await cache_service.get_cached_result(
        platform=platform,
        workflow_id=workflow_id,
        input_data=input_data,
        user_id=current_user.get("id")
    )
    
    return {
        "cached": cached_result is not None,
        "result": cached_result
    }


@router.post("/cache/invalidate")
async def invalidate_cache(
    platform: PlatformType,
    workflow_id: str,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Invalidate cache entries for a workflow"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cache_service = PlatformExecutionCache(db)
    invalidated = await cache_service.invalidate_cache(
        platform=platform,
        workflow_id=workflow_id,
        user_id=current_user.get("id")
    )
    
    return {
        "invalidated": invalidated,
        "platform": platform.value,
        "workflow_id": workflow_id
    }


@router.get("/cache/statistics")
async def get_cache_statistics(
    platform: Optional[PlatformType] = None,
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get cache hit/miss statistics"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    
    cache_service = PlatformExecutionCache(db)
    stats = await cache_service.get_cache_statistics(
        platform=platform,
        user_id=current_user.get("id")
    )
    
    return stats


@router.put("/cache/config/{platform}")
async def configure_platform_cache(
    platform: PlatformType,
    config_updates: Dict[str, Any],
    current_user = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Update cache configuration for a platform"""
    if not current_user or not current_user.get("is_superuser"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    cache_service = PlatformExecutionCache(db)
    updated_config = await cache_service.configure_cache(
        platform=platform,
        config_updates=config_updates
    )
    
    return {
        "platform": platform.value,
        "config": updated_config
    }