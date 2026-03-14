"""Upgrade flow API endpoints for license management."""

from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
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
from core.usage_tracker import get_usage_tracker
from core.config import get_settings
from core.tenant_context import get_current_tenant_id
from models.enforcement import FeatureTrial, UpgradePrompt, BillingEvent
from schemas.upgrade import (
    UpgradeOptionsResponse,
    TrialRequest,
    TrialResponse,
    UsageSummaryResponse,
    SubscriptionRequest,
    SubscriptionResponse,
    LimitOverrideRequest,
    LimitOverrideResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()


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
    tracker = await get_usage_tracker(db)
    
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
        target_edition = Edition(request.target_edition)

        # No trials for community edition
        if target_edition == Edition.COMMUNITY:
            raise HTTPException(
                status_code=400,
                detail="Trial not available for Community edition"
            )

        # Check if trial already exists for this edition
        result = await db.execute(
            select(FeatureTrial).where(
                and_(
                    FeatureTrial.tenant_id == tenant_id,
                    FeatureTrial.edition_required == target_edition.value
                )
            )
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
                detail="Trial for this edition has already been used and expired"
            )

        # Create trial
        trial = FeatureTrial(
            tenant_id=tenant_id,
            feature_name=f"{request.target_edition}_trial",
            edition_required=request.target_edition,
            expires_at=datetime.utcnow() + timedelta(days=request.trial_days or get_settings().TRIAL_DAYS)
        )
        db.add(trial)

        # Track trial start
        prompt = UpgradePrompt(
            tenant_id=tenant_id,
            prompt_type="trial_started",
            prompt_message=f"Started {request.target_edition} trial for {request.trial_days or get_settings().TRIAL_DAYS} days",
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


@router.post("/subscribe", response_model=SubscriptionResponse)
async def create_subscription(
    request: SubscriptionRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user_safe),
    db: AsyncSession = Depends(get_db)
):
    """Create a new subscription or upgrade existing one."""
    
    from models.subscription import Subscription as SubModel, SubscriptionStatus as SubStatus, SubscriptionPlan as SubPlan

    tenant_id = current_user.get("tenant_id") or get_current_tenant_id()
    user_id = current_user.get("sub") or current_user.get("id")

    # Look up existing subscription from database
    existing_sub_result = await db.execute(
        select(SubModel).where(
            SubModel.user_id == user_id,
            SubModel.status.in_([SubStatus.ACTIVE, SubStatus.TRIALING, SubStatus.PAST_DUE]),
        ).limit(1)
    )
    existing_sub = existing_sub_result.scalar_one_or_none()

    # Resolve plan_id from DB instead of using raw edition string
    plan_lookup = await db.execute(
        select(SubPlan).where(SubPlan.name.ilike(f"%{request.target_edition}%")).limit(1)
    )
    plan_record = plan_lookup.scalar_one_or_none()
    resolved_plan_id = plan_record.id if plan_record else request.target_edition

    tenant_info = {
        "id": tenant_id,
        "stripe_customer_id": existing_sub.stripe_customer_id if existing_sub else None,
        "stripe_subscription_id": existing_sub.stripe_subscription_id if existing_sub else None,
        "billing_email": current_user.get("email"),
        "edition": Edition.COMMUNITY.value,
        "trial_ends_at": None
    }

    # Initialize Stripe if not already done
    if stripe is None:
        raise HTTPException(status_code=500, detail="Payment processing not available")
    settings = get_settings()
    if settings.STRIPE_SECRET_KEY:
        stripe.api_key = settings.STRIPE_SECRET_KEY
    else:
        raise HTTPException(
            status_code=500,
            detail="Payment processing not configured"
        )
    
    try:
        # Create or update Stripe customer
        if tenant_info["stripe_customer_id"]:
            customer = stripe.Customer.retrieve(tenant_info["stripe_customer_id"])
            # Update payment method if provided
            if request.payment_method_id:
                stripe.PaymentMethod.attach(
                    request.payment_method_id,
                    customer=tenant_info["stripe_customer_id"]
                )
                stripe.Customer.modify(
                    tenant_info["stripe_customer_id"],
                    invoice_settings={
                        "default_payment_method": request.payment_method_id
                    }
                )
        else:
            # Create new customer
            customer = stripe.Customer.create(
                email=request.billing_email or tenant_info["billing_email"],
                metadata={"tenant_id": str(tenant_id)}
            )
            tenant_info["stripe_customer_id"] = customer.id
            if request.billing_email:
                tenant_info["billing_email"] = request.billing_email
        
        # Get price ID for the edition
        price_id = _get_stripe_price_id(request.target_edition, request.billing_period)
        
        if not price_id:
            raise HTTPException(
                status_code=400,
                detail=f"No pricing available for {request.target_edition}"
            )
        
        # Create or update subscription
        if tenant_info["stripe_subscription_id"]:
            # Update existing subscription
            subscription = stripe.Subscription.retrieve(tenant_info["stripe_subscription_id"])
            
            # Update the subscription
            updated_subscription = stripe.Subscription.modify(
                tenant_info["stripe_subscription_id"],
                items=[{
                    "id": subscription["items"]["data"][0].id,
                    "price": price_id
                }],
                proration_behavior="always_invoice"
            )
            
            subscription = updated_subscription
        else:
            # Create new subscription
            subscription = stripe.Subscription.create(
                customer=customer.id,
                items=[{"price": price_id}],
                trial_period_days=get_settings().TRIAL_DAYS if request.start_trial else 0,
                metadata={"tenant_id": str(tenant_id)}
            )
            tenant_info["stripe_subscription_id"] = subscription.id
        
        # Update tenant edition
        old_edition = tenant_info["edition"]
        tenant_info["edition"] = request.target_edition

        if request.start_trial:
            tenant_info["trial_ends_at"] = datetime.utcnow() + timedelta(days=get_settings().TRIAL_DAYS)

        # Persist Stripe IDs to the subscription record in database
        if existing_sub:
            existing_sub.stripe_customer_id = tenant_info["stripe_customer_id"]
            existing_sub.stripe_subscription_id = tenant_info["stripe_subscription_id"]
            existing_sub.plan_id = resolved_plan_id
            existing_sub.status = SubStatus.TRIALING if request.start_trial else SubStatus.ACTIVE
        else:
            import uuid as _uuid
            from models.subscription import BillingPeriod as _BillingPeriod
            new_sub = SubModel(
                id=str(_uuid.uuid4()),
                user_id=user_id,
                tenant_id=tenant_id,
                plan_id=resolved_plan_id,
                status=SubStatus.TRIALING if request.start_trial else SubStatus.ACTIVE,
                billing_period=_BillingPeriod(request.billing_period) if request.billing_period in ("monthly", "annual", "quarterly") else _BillingPeriod.MONTHLY,
                started_at=datetime.utcnow(),
                current_period_start=datetime.fromtimestamp(subscription.current_period_start),
                current_period_end=datetime.fromtimestamp(subscription.current_period_end),
                trial_end=datetime.fromtimestamp(subscription.trial_end) if getattr(subscription, 'trial_end', None) else None,
                stripe_customer_id=tenant_info["stripe_customer_id"],
                stripe_subscription_id=tenant_info["stripe_subscription_id"],
            )
            db.add(new_sub)

        # Create billing event
        billing_event = BillingEvent(
            tenant_id=tenant_id,
            event_type="subscription_created" if not existing_sub else "upgraded",
            stripe_event_id=subscription.id,
            status="completed",
            event_data={
                "previous_edition": old_edition,
                "new_edition": request.target_edition,
                "price_id": price_id,
                "billing_period": request.billing_period
            }
        )
        db.add(billing_event)
        
        await db.commit()
        
        # Clear caches
        enforcer = LicenseEnforcer(db)
        cache_key = f"tenant_info:{tenant_id}"
        await enforcer.cache.delete(cache_key)
        
        # Send confirmation email in background
        background_tasks.add_task(
            _send_subscription_email,
            tenant_info["billing_email"],
            request.target_edition,
            subscription
        )
        
        return SubscriptionResponse(
            subscription_id=subscription.id,
            edition=request.target_edition,
            status=subscription.status,
            current_period_start=datetime.fromtimestamp(subscription.current_period_start),
            current_period_end=datetime.fromtimestamp(subscription.current_period_end),
            trial_end=datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None,
            cancel_at_period_end=subscription.cancel_at_period_end
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Payment processing error: {str(e)}"
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
        raise HTTPException(status_code=500, detail="Payment processing not available")
    settings = get_settings()
    if settings.STRIPE_SECRET_KEY:
        stripe.api_key = settings.STRIPE_SECRET_KEY
    else:
        raise HTTPException(
            status_code=500,
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
        existing.created_by = current_user["id"]
    else:
        # Create new
        override = TenantLimitOverride(
            tenant_id=request.tenant_id,
            limit_type=request.limit_type,
            limit_value=request.limit_value,
            reason=request.reason,
            expires_at=request.expires_at,
            created_by=current_user["id"]
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


def _get_stripe_price_id(edition: str, billing_period: str) -> Optional[str]:
    """Get Stripe price ID for edition and billing period from config."""

    settings = get_settings()

    # Map edition names to config price IDs
    # Note: Community is free, so no price ID needed
    if billing_period == "annual":
        price_mapping = {
            "business_starter": settings.STRIPE_PRICE_BUSINESS_STARTER_ANNUAL,
            "business_pro": settings.STRIPE_PRICE_BUSINESS_PRO_ANNUAL,
            "business_scale": settings.STRIPE_PRICE_BUSINESS_SCALE_ANNUAL,
            "enterprise": settings.STRIPE_PRICE_ENTERPRISE_ANNUAL,
            "business_growth": settings.STRIPE_PRICE_BUSINESS_PRO_ANNUAL,
        }
    else:
        price_mapping = {
            "business_starter": settings.STRIPE_PRICE_BUSINESS_STARTER,
            "business_pro": settings.STRIPE_PRICE_BUSINESS_PRO,
            "business_scale": settings.STRIPE_PRICE_BUSINESS_SCALE,
            "enterprise": settings.STRIPE_PRICE_ENTERPRISE,
            "business_growth": settings.STRIPE_PRICE_BUSINESS_PRO,
        }

    price_id = price_mapping.get(edition)

    # Return None if price ID is empty or not configured
    if not price_id or price_id == "":
        return None

    return price_id


async def _send_subscription_email(
    email: str,
    edition: str,
    subscription: Any
):
    """Send subscription confirmation email."""
    
    # This would integrate with email service
    logger.info(f"Sending subscription email to {email} for {edition}")
    # Implementation would go here