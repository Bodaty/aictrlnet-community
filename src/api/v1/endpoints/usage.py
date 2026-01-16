"""Basic usage tracking endpoints for Community edition."""

from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from core.database import get_db
from core.dependencies import get_current_user_safe
from core.enforcement_simple import LicenseEnforcer, Edition
from core.usage_tracker import get_usage_tracker
from core.tenant_context import get_current_tenant_id
from schemas.usage_metrics import UsageStatusResponse
from schemas.license import CurrentUsageResponse, UsageHistoryResponse, UsageAlertsResponse, MetricInfo
from services.usage_service import UsageService
# Removed usage_metrics_service import - not needed anymore
from models import User, WorkflowDefinition

router = APIRouter()


@router.get("/status", response_model=UsageStatusResponse)
async def get_usage_status(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get current usage status and limits."""
    service = UsageService(db)
    
    # For Community, always use "community" tenant
    tenant_id = current_user.get("tenant_id", "community")
    
    return await service.get_usage_status(tenant_id)


@router.get("/check-limits")
async def check_limits(
    resource_type: str,
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Check if a specific resource limit has been reached."""
    service = UsageService(db)
    tenant_id = current_user.get("tenant_id", "community")
    
    status = await service.get_usage_status(tenant_id)
    
    # Check specific resource
    if resource_type == "workflows":
        if status.current_usage.workflow_count >= status.limits.max_workflows:
            raise HTTPException(
                status_code=402,  # Payment Required
                detail={
                    "error": "Workflow limit reached",
                    "message": f"Community edition is limited to {status.limits.max_workflows} workflows. Upgrade to Business for unlimited workflows.",
                    "current": status.current_usage.workflow_count,
                    "limit": status.limits.max_workflows,
                    "upgrade_url": "/pricing"
                }
            )
    elif resource_type == "adapters":
        if status.current_usage.adapter_count >= status.limits.max_adapters:
            raise HTTPException(
                status_code=402,
                detail={
                    "error": "Adapter limit reached",
                    "message": f"Community edition is limited to {status.limits.max_adapters} adapters. Upgrade to Business for more adapters.",
                    "current": status.current_usage.adapter_count,
                    "limit": status.limits.max_adapters,
                    "upgrade_url": "/pricing"
                }
            )
    elif resource_type == "api_calls":
        if status.current_usage.api_calls_month >= status.limits.max_api_calls_month:
            raise HTTPException(
                status_code=429,  # Too Many Requests
                detail={
                    "error": "API call limit reached",
                    "message": f"Monthly API call limit of {status.limits.max_api_calls_month} reached. Upgrade to Business for higher limits.",
                    "current": status.current_usage.api_calls_month,
                    "limit": status.limits.max_api_calls_month,
                    "upgrade_url": "/pricing"
                }
            )
    
    return {
        "resource": resource_type,
        "current": getattr(status.current_usage, f"{resource_type}_count", 0),
        "limit": getattr(status.limits, f"max_{resource_type}", 0),
        "percent_used": getattr(status, f"{resource_type}_percent", 0),
        "within_limits": True
    }


@router.get("/current", response_model=CurrentUsageResponse)
async def get_current_usage(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get current usage metrics for the billing period."""
    
    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
    
    # Get current period
    now = datetime.utcnow()
    period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    if now.month == 12:
        period_end = datetime(now.year + 1, 1, 1) - timedelta(seconds=1)
    else:
        period_end = datetime(now.year, now.month + 1, 1) - timedelta(seconds=1)
    
    # Get enforcer for limits
    enforcer = LicenseEnforcer(db)
    tenant_info = await enforcer._get_tenant_info(tenant_id)
    edition = Edition(tenant_info["edition"])
    limits = enforcer.EDITION_LIMITS.get(edition, {})
    
    # Get usage tracker for current metrics
    tracker = await get_usage_tracker(db)
    usage_summary = await tracker.get_usage_summary(
        tenant_id=tenant_id,
        start_date=period_start,
        end_date=now
    )
    
    # Count actual resources
    # Users
    user_count = await db.scalar(
        select(func.count(User.id)).where(User.tenant_id == tenant_id)
    )
    
    # Workflows
    workflow_count = await db.scalar(
        select(func.count(WorkflowDefinition.id)).where(WorkflowDefinition.tenant_id == tenant_id)
    )
    
    # Get metrics from usage summary
    executions = usage_summary.get("metrics", {}).get("executions", {}).get("total_value", 0)
    storage_gb = usage_summary.get("metrics", {}).get("storage_gb", {}).get("total_value", 0.0)
    
    # Build response
    metrics = {
        "users": MetricInfo(
            current=float(user_count or 0),
            limit=float(limits.get("USERS", 1)),
            percentage=float((user_count or 0) / limits.get("USERS", 1) * 100) if limits.get("USERS", 1) > 0 else 0
        ),
        "workflows": MetricInfo(
            current=float(workflow_count or 0),
            limit=float(limits.get("WORKFLOWS", 10)),
            percentage=float((workflow_count or 0) / limits.get("WORKFLOWS", 10) * 100) if limits.get("WORKFLOWS", 10) > 0 else 0
        ),
        "executions": MetricInfo(
            current=float(executions),
            limit=float(limits.get("EXECUTIONS", 1000)),
            percentage=float(executions / limits.get("EXECUTIONS", 1000) * 100) if limits.get("EXECUTIONS", 1000) > 0 else 0
        ),
        "storage": MetricInfo(
            current=float(storage_gb),
            limit=float(limits.get("STORAGE_GB", 1)),
            unit="GB",
            percentage=float(storage_gb / limits.get("STORAGE_GB", 1) * 100) if limits.get("STORAGE_GB", 1) > 0 else 0
        )
    }
    
    # Add platform_nodes for Business/Enterprise
    if edition.value in ["business_starter", "business_growth", "business_scale", "enterprise"]:
        # TODO: Count actual platform nodes when implemented
        metrics["platform_nodes"] = MetricInfo(
            current=0.0,
            limit=5.0,
            percentage=0.0
        )
    
    return CurrentUsageResponse(
        period={
            "start": period_start.isoformat() + "Z",
            "end": period_end.isoformat() + "Z"
        },
        metrics=metrics
    )


@router.get("/history", response_model=UsageHistoryResponse)
async def get_usage_history(
    period: str = Query(default="30d", regex="^\\d+d$"),
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get usage history for the specified period."""
    
    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
    
    # Parse period
    days = int(period[:-1])
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # Get usage tracker
    tracker = await get_usage_tracker(db)
    
    # Get timeline data
    timeline = await tracker.get_usage_timeline(
        tenant_id=tenant_id,
        metric_type="executions",
        granularity="day",
        days=days
    )
    
    # Build daily data points
    data_points = []
    current_date = start_date.date()
    
    while current_date <= end_date.date():
        # Find data for this date
        date_str = current_date.isoformat()
        executions = 0
        
        for point in timeline:
            if point["period"].startswith(date_str):
                executions = int(point["value"])
                break
        
        # Get actual counts from database
        # User count (active users in this period)
        user_count_result = await db.execute(
            select(func.count(func.distinct(User.id)))
            .where(User.tenant_id == tenant_id)
        )
        user_count = user_count_result.scalar() or 0
        
        # Workflow count
        workflow_count_result = await db.execute(
            select(func.count(WorkflowDefinition.id))
            .where(
                and_(
                    WorkflowDefinition.tenant_id == tenant_id,
                    WorkflowDefinition.created_at <= datetime.strptime(date_str, "%Y-%m-%d")
                )
            )
        )
        workflow_count = workflow_count_result.scalar() or 0
        
        # Storage estimate (workflows * avg size)
        storage_gb = round((workflow_count * 0.001), 2)  # Estimate 1MB per workflow
        
        data_points.append({
            "date": date_str,
            "users": user_count,
            "workflows": workflow_count,
            "executions": executions,
            "storage_gb": storage_gb
        })
        
        current_date += timedelta(days=1)
    
    return UsageHistoryResponse(
        period=period,
        data=data_points
    )


@router.get("/alerts", response_model=UsageAlertsResponse)
async def get_usage_alerts(
    current_user: dict = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get usage alerts for approaching or exceeded limits."""
    
    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
    
    # Get current usage
    usage_response = await get_current_usage(current_user, db)
    
    alerts = []
    alert_id = 1
    
    # Check each metric for alerts
    for metric_name, metric_info in usage_response.metrics.items():
        if metric_info.percentage >= 95:
            alerts.append({
                "id": f"alert_{alert_id}",
                "type": "limit_exceeded",
                "metric": metric_name,
                "threshold": 95,
                "current_percentage": metric_info.percentage,
                "message": f"You've exceeded 95% of your {metric_name} limit",
                "severity": "critical",
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            alert_id += 1
        elif metric_info.percentage >= 80:
            alerts.append({
                "id": f"alert_{alert_id}",
                "type": "approaching_limit",
                "metric": metric_name,
                "threshold": 80,
                "current_percentage": metric_info.percentage,
                "message": f"You've used {metric_info.percentage:.0f}% of your monthly {metric_name}",
                "severity": "warning",
                "created_at": datetime.utcnow().isoformat() + "Z"
            })
            alert_id += 1
    
    return UsageAlertsResponse(alerts=alerts)