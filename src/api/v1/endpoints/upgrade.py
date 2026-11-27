"""Upgrade flow API endpoints for license management."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_
try:
    import stripe
except ImportError:
    stripe = None
import logging

from core.database import get_db
from core.security import get_current_active_user as get_current_user
from core.dependencies import get_current_user_safe
from core.enforcement_simple import LicenseEnforcer, Edition, LimitType
from core.config import get_settings
from core.tenant_context import get_current_tenant_id
from api.v1.endpoints._auth_helpers import get_safe_user_id
from models.enforcement import FeatureTrial, UpgradePrompt, BillingEvent
from schemas.upgrade import (
    UpgradeOptionsResponse,
    TrialRequest,
    TrialResponse,
    UsageSummaryResponse,
    LimitOverrideRequest,
    LimitOverrideResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

TRIALABLE_EDITIONS = (Edition.BUSINESS_STARTER, Edition.BUSINESS_GROWTH, Edition.BUSINESS_SCALE)


@router.get("/options", response_model=UpgradeOptionsResponse)
async def get_upgrade_options(
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get available upgrade options for the current tenant."""
    
    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
    enforcer = LicenseEnforcer(db)
    
    # Get current tenant info
    tenant_info = await enforcer._get_tenant_info(tenant_id)
    current_edition = Edition(tenant_info["edition"])
    
    # Get usage summary
    usage_summary = await enforcer.get_usage_summary(tenant_id)
    
    # Get upgrade path
    upgrade_path = enforcer._get_upgrade_path(current_edition.value)
    
    # Skip tracking for now - would need proper UUID tenant_id
    # This would normally track that upgrade options were viewed
    
    return UpgradeOptionsResponse(
        current_edition=current_edition.value,
        current_usage=usage_summary.get("usage", {}),
        upgrade_options=upgrade_path["upgrade_options"],
        contact_sales=upgrade_path["contact_sales"],
        trial_available=False  # Simplified for community edition
    )


@router.get("/usage", response_model=UsageSummaryResponse)
async def get_usage_summary(
    period: Optional[str] = "current",  # current, previous, year
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed usage summary for billing period."""
    
    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()

    # Determine date range
    now = datetime.utcnow()
    if period == "current":
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = now
    elif period == "previous":
        start_date = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        end_date = now.replace(day=1) - timedelta(seconds=1)
    else:  # year
        start_date = now.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
        end_date = now
    
    # Get real usage data from database
    from sqlalchemy import select, func, and_
    from models import WorkflowDefinition, WorkflowExecution, Task
    from models.usage_metrics import UsageMetric
    
    # Get API calls from usage metrics
    usage_result = await db.execute(
        select(UsageMetric)
        .where(
            and_(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.period_start <= end_date,
                UsageMetric.period_end >= start_date
            )
        )
        .order_by(UsageMetric.last_updated.desc())
        .limit(1)
    )
    usage_metric = usage_result.scalar_one_or_none()
    api_calls_count = usage_metric.api_calls_month if usage_metric else 0
    
    # Count workflows
    workflows_result = await db.execute(
        select(func.count(WorkflowDefinition.id))
        .where(WorkflowDefinition.tenant_id == tenant_id)
    )
    workflows_count = workflows_result.scalar() or 0
    
    # Count executions
    executions_result = await db.execute(
        select(func.count(WorkflowExecution.id))
        .where(
            and_(
                WorkflowExecution.tenant_id == tenant_id,
                WorkflowExecution.created_at >= start_date,
                WorkflowExecution.created_at <= end_date
            )
        )
    )
    executions_count = executions_result.scalar() or 0
    
    # Calculate storage (simplified - count tasks)
    tasks_result = await db.execute(
        select(func.count(Task.id))
        .where(Task.tenant_id == tenant_id)
    )
    tasks_count = tasks_result.scalar() or 0
    # Estimate storage: 1KB per task
    storage_gb = (tasks_count * 1024) / (1024 * 1024 * 1024)
    
    # Community edition limits
    limits = {
        "api_calls": 10000,
        "workflows": 10,
        "storage_gb": 1,
        "executions": 1000
    }
    
    usage_data = {
        "tenant_id": tenant_id,
        "period": {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        },
        "metrics": {
            "api_calls": {
                "total_value": float(api_calls_count),
                "total_count": api_calls_count,
                "record_count": api_calls_count,
                "limit": limits["api_calls"],
                "percentage": (api_calls_count / limits["api_calls"] * 100) if limits["api_calls"] > 0 else 0
            },
            "workflows": {
                "total_value": float(workflows_count),
                "total_count": workflows_count,
                "record_count": workflows_count,
                "limit": limits["workflows"],
                "percentage": (workflows_count / limits["workflows"] * 100) if limits["workflows"] > 0 else 0
            },
            "storage_gb": {
                "total_value": round(storage_gb, 3),
                "total_count": tasks_count,
                "record_count": tasks_count,
                "limit": limits["storage_gb"],
                "percentage": (storage_gb / limits["storage_gb"] * 100) if limits["storage_gb"] > 0 else 0
            },
            "executions": {
                "total_value": float(executions_count),
                "total_count": executions_count,
                "record_count": executions_count,
                "limit": limits["executions"],
                "percentage": (executions_count / limits["executions"] * 100) if limits["executions"] > 0 else 0
            }
        }
    }
    
    return UsageSummaryResponse(**usage_data)


@router.post("/trial", response_model=TrialResponse)
async def start_feature_trial(
    request: TrialRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Start a trial for higher-tier features.

    If a trial already exists for the same edition and is still valid,
    returns the existing trial (idempotent behavior).
    """

    try:
        tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
        try:
            target_edition = Edition(request.target_edition)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown edition '{request.target_edition}'",
            )

        # Feature trials cover the self-serve Business tiers only. Enterprise
        # is contact-sales, and Community is free.
        if target_edition not in TRIALABLE_EDITIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Trial not available for the {target_edition.value} edition"
            )

        # One feature trial per tenant across all Business tiers, so trying
        # Starter, then Growth, then Scale cannot chain free periods.
        result = await db.execute(
            select(FeatureTrial).where(
                and_(
                    FeatureTrial.tenant_id == tenant_id,
                    FeatureTrial.edition_required.in_([e.value for e in TRIALABLE_EDITIONS])
                )
            ).order_by(FeatureTrial.started_at.desc()).limit(1)
        )
        existing_trial = result.scalar_one_or_none()

        # If trial exists and is still valid, return it (idempotent)
        if existing_trial and existing_trial.expires_at > datetime.utcnow():
            enforcer = LicenseEnforcer(db)
            return TrialResponse(
                trial_id=str(existing_trial.id),
                edition=existing_trial.edition_required,
                features=list(enforcer.EDITION_FEATURES.get(Edition(existing_trial.edition_required), set())),
                started_at=existing_trial.started_at,
                expires_at=existing_trial.expires_at,
                days_remaining=(existing_trial.expires_at - datetime.utcnow()).days
            )

        # If trial exists but expired, return error
        if existing_trial:
            raise HTTPException(
                status_code=400,
                detail="Your Business trial has already been used"
            )

        # The trial length is the server's, never the client's.
        trial_days = get_settings().TRIAL_DAYS

        # Create trial
        trial = FeatureTrial(
            tenant_id=tenant_id,
            feature_name=f"{request.target_edition}_trial",
            edition_required=request.target_edition,
            expires_at=datetime.utcnow() + timedelta(days=trial_days)
        )
        db.add(trial)

        # Track trial start
        prompt = UpgradePrompt(
            tenant_id=tenant_id,
            prompt_type="trial_started",
            prompt_message=f"Started {request.target_edition} trial for {trial_days} days",
            target_edition=request.target_edition
        )
        db.add(prompt)

        await db.commit()
        await db.refresh(trial)

        # Clear tenant cache to pick up new trial
        try:
            enforcer = LicenseEnforcer(db)
            cache_key = f"tenant_info:{tenant_id}"
            await enforcer.cache.delete(cache_key)
        except Exception as cache_err:
            logger.warning(f"Failed to clear tenant cache: {cache_err}")

        enforcer = LicenseEnforcer(db)
        return TrialResponse(
            trial_id=str(trial.id),
            edition=trial.edition_required,
            features=list(enforcer.EDITION_FEATURES.get(Edition(trial.edition_required), set())),
            started_at=trial.started_at,
            expires_at=trial.expires_at,
            days_remaining=(trial.expires_at - datetime.utcnow()).days
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start trial: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start trial: {str(e)}"
        )


@router.post("/cancel")
async def cancel_subscription(
    immediately: bool = False,
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Cancel current subscription."""
    from models.subscription import Subscription as SubModel, SubscriptionStatus as SubStatus

    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
    user_id = current_user.get("sub") or current_user.get("id")

    # Look up the user's actual subscription from the database
    sub_result = await db.execute(
        select(SubModel).where(
            SubModel.user_id == user_id,
            SubModel.status.in_([SubStatus.ACTIVE, SubStatus.TRIALING, SubStatus.PAST_DUE]),
        ).limit(1)
    )
    existing_sub = sub_result.scalar_one_or_none()

    if not existing_sub or not existing_sub.stripe_subscription_id:
        raise HTTPException(
            status_code=400,
            detail="No active subscription found"
        )

    tenant_info = {
        "id": tenant_id,
        "stripe_subscription_id": existing_sub.stripe_subscription_id,
        "edition": existing_sub.plan_id or Edition.COMMUNITY.value,
    }

    if stripe is None:
        raise HTTPException(status_code=503, detail="Payment processing not available")
    settings = get_settings()
    if settings.STRIPE_SECRET_KEY:
        stripe.api_key = settings.STRIPE_SECRET_KEY
    else:
        raise HTTPException(
            status_code=503,
            detail="Payment processing not configured"
        )
    
    try:
        if immediately:
            # Cancel immediately
            subscription = stripe.Subscription.delete(tenant_info["stripe_subscription_id"])
            existing_sub.status = SubStatus.CANCELED
            existing_sub.canceled_at = datetime.utcnow()
            existing_sub.stripe_subscription_id = None
            tenant_info["edition"] = Edition.COMMUNITY.value
            tenant_info["stripe_subscription_id"] = None
        else:
            # Cancel at period end
            subscription = stripe.Subscription.modify(
                tenant_info["stripe_subscription_id"],
                cancel_at_period_end=True
            )
            existing_sub.cancel_at_period_end = True
        
        # Create billing event
        billing_event = BillingEvent(
            tenant_id=tenant_id,
            event_type="cancelled",
            stripe_event_id=subscription.id,
            status="completed",
            event_data={
                "previous_edition": tenant_info["edition"],
                "new_edition": Edition.COMMUNITY.value if immediately else tenant_info["edition"],
                "immediate": immediately,
                "cancel_at": subscription.cancel_at
            }
        )
        db.add(billing_event)
        
        await db.commit()
        
        return {
            "status": "cancelled",
            "immediate": immediately,
            "cancel_at": datetime.fromtimestamp(subscription.cancel_at) if subscription.cancel_at else None
        }
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Error cancelling subscription: {str(e)}"
        )


@router.post("/override-limit", response_model=LimitOverrideResponse)
async def create_limit_override(
    request: LimitOverrideRequest,
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Create a custom limit override (admin only)."""
    
    # Check if user is admin
    if not current_user.get("is_admin"):
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    
    from models.community import TenantLimitOverride
    
    # Check if override exists
    result = await db.execute(
        select(TenantLimitOverride).where(
            and_(
                TenantLimitOverride.tenant_id == request.tenant_id,
                TenantLimitOverride.limit_type == request.limit_type
            )
        )
    )
    existing = result.scalar_one_or_none()
    
    if existing:
        # Update existing
        existing.limit_value = request.limit_value
        existing.reason = request.reason
        existing.expires_at = request.expires_at
        existing.created_by = get_safe_user_id(current_user)
    else:
        # Create new
        override = TenantLimitOverride(
            tenant_id=request.tenant_id,
            limit_type=request.limit_type,
            limit_value=request.limit_value,
            reason=request.reason,
            expires_at=request.expires_at,
            created_by=get_safe_user_id(current_user)
        )
        db.add(override)
    
    await db.commit()
    
    # Clear cache
    enforcer = LicenseEnforcer(db)
    cache_key = f"tenant_info:{request.tenant_id}"
    await enforcer.cache.delete(cache_key)
    
    return LimitOverrideResponse(
        tenant_id=request.tenant_id,
        limit_type=request.limit_type,
        limit_value=request.limit_value,
        reason=request.reason,
        expires_at=request.expires_at,
        created_at=datetime.utcnow()
    )


# Helper functions

async def _is_trial_available(
    db: AsyncSession,
    tenant_id: str,
    target_edition: Edition
) -> bool:
    """Check if trial is available for tenant."""
    
    # No trials for community edition
    if target_edition == Edition.COMMUNITY:
        return False
    
    # Check if already used trial for this edition
    result = await db.execute(
        select(FeatureTrial).where(
            and_(
                FeatureTrial.tenant_id == tenant_id,
                FeatureTrial.edition_required == target_edition.value
            )
        )
    )
    
    return result.scalar_one_or_none() is None


def _metric_to_limit_type(metric_type: str) -> Optional[LimitType]:
    """Convert metric type to limit type."""
    
    mapping = {
        "api_calls": LimitType.API_CALLS,
        "executions": LimitType.EXECUTIONS,
        "storage_gb": LimitType.STORAGE_GB,
    }
    
    return mapping.get(metric_type)
