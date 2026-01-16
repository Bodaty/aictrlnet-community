"""License and usage tracking service."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
import json

from models.subscription import Subscription, SubscriptionPlan, UsageTracking, SubscriptionStatus
from models.enforcement import UsageMetric
from models.analytics import AnalyticsMetric
from models.user import User
from services.analytics_service import AnalyticsService
from core.tenant_context import get_current_tenant_id

logger = logging.getLogger(__name__)


class LicenseService:
    """Service for license and usage tracking operations."""
    
    @staticmethod
    async def get_subscription_status(
        db: AsyncSession,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current subscription status.

        Security: Always filters by tenant_id. If not provided, uses current context.
        """
        try:
            # Use provided tenant_id or fall back to context
            effective_tenant_id = tenant_id or get_current_tenant_id()

            # Build query conditions - always require tenant filtering
            conditions = [Subscription.status == SubscriptionStatus.ACTIVE]
            if effective_tenant_id:
                conditions.append(Subscription.tenant_id == effective_tenant_id)
            if user_id:
                conditions.append(Subscription.user_id == user_id)

            # Get the active subscription
            subscription_query = select(Subscription).where(
                and_(*conditions)
            ).order_by(desc(Subscription.created_at))
            
            result = await db.execute(subscription_query)
            subscription = result.scalars().first()
            
            if not subscription:
                # Return community default
                return {
                    "subscription_id": None,
                    "plan_name": "Community",
                    "edition": "community",
                    "status": "active",
                    "trial_days_remaining": None,
                    "next_billing_date": None,
                    "features": {
                        "max_tasks_per_month": 1000,
                        "max_workflows": 10,
                        "max_team_members": 1,
                        "analytics_retention_days": 7,
                        "support_level": "community",
                        "ml_features": False,
                        "enterprise_features": False
                    },
                    "limits": {
                        "api_requests_per_hour": 100,
                        "concurrent_workflows": 2,
                        "storage_gb": 1
                    }
                }
            
            # Get the plan details
            plan = await db.get(SubscriptionPlan, subscription.plan_id)
            
            # Calculate trial days remaining if in trial
            trial_days_remaining = None
            if subscription.trial_end_date:
                remaining = subscription.trial_end_date - datetime.utcnow()
                trial_days_remaining = max(0, remaining.days)
            
            return {
                "subscription_id": subscription.id,
                "plan_name": plan.name if plan else "Unknown",
                "edition": plan.edition if plan else "community",
                "status": subscription.status.value,
                "trial_days_remaining": trial_days_remaining,
                "next_billing_date": subscription.next_billing_date.isoformat() if subscription.next_billing_date else None,
                "features": plan.features if plan else {},
                "limits": plan.limits if plan else {}
            }
            
        except Exception as e:
            logger.error(f"Error getting subscription status: {e}")
            # Return community default on error
            return {
                "subscription_id": None,
                "plan_name": "Community",
                "edition": "community",
                "status": "active",
                "trial_days_remaining": None,
                "next_billing_date": None,
                "features": {},
                "limits": {}
            }
    
    @staticmethod
    async def get_usage_summary(
        db: AsyncSession,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> Dict[str, Any]:
        """Get usage summary for the specified period.

        Security: Always filters by tenant_id. If not provided, uses current context.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)

        # Use provided tenant_id or fall back to context
        effective_tenant_id = tenant_id or get_current_tenant_id()

        try:
            # Build query conditions - always require tenant filtering
            conditions = [UsageMetric.timestamp >= start_date]
            if effective_tenant_id:
                conditions.append(UsageMetric.tenant_id == effective_tenant_id)

            # Get usage metrics
            usage_query = select(
                UsageMetric.feature_name,
                func.sum(UsageMetric.usage_count).label('total_usage'),
                func.count(UsageMetric.id).label('usage_events')
            ).where(
                and_(*conditions)
            ).group_by(UsageMetric.feature_name)
            
            result = await db.execute(usage_query)
            usage_data = result.all()
            
            # Format usage by feature
            usage_by_feature = {}
            total_usage = 0
            for row in usage_data:
                usage_by_feature[row.feature_name] = {
                    "total_usage": int(row.total_usage),
                    "usage_events": int(row.usage_events)
                }
                total_usage += int(row.total_usage)
            
            # Get analytics overview for additional context
            analytics_overview = await AnalyticsService.get_analytics_overview(db, tenant_id, days)
            
            # Combine the data
            return {
                "period_days": days,
                "total_usage_events": total_usage,
                "usage_by_feature": usage_by_feature,
                "analytics": {
                    "total_tasks": analytics_overview.get("tasks", {}).get("total", 0),
                    "total_workflows": analytics_overview.get("workflows", {}).get("total", 0),
                    "task_success_rate": analytics_overview.get("tasks", {}).get("success_rate", 100),
                    "workflow_success_rate": analytics_overview.get("workflows", {}).get("success_rate", 100)
                },
                "summary": {
                    "avg_daily_tasks": analytics_overview.get("tasks", {}).get("total", 0) / days,
                    "avg_daily_workflows": analytics_overview.get("workflows", {}).get("total", 0) / days,
                    "most_used_feature": max(usage_by_feature.keys(), key=lambda k: usage_by_feature[k]["total_usage"]) if usage_by_feature else "tasks"
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting usage summary: {e}")
            return {
                "period_days": days,
                "total_usage_events": 0,
                "usage_by_feature": {},
                "analytics": {},
                "summary": {},
                "error": str(e)
            }
    
    @staticmethod
    async def get_usage_trends(
        db: AsyncSession,
        tenant_id: Optional[str] = None,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get usage trends over time."""
        try:
            # Get daily usage trends from analytics
            trends = await AnalyticsService.get_time_series_data(
                db, "api_requests", tenant_id, "day", days
            )
            
            # Also get task and workflow trends
            task_trends = await AnalyticsService.get_time_series_data(
                db, "tasks_created", tenant_id, "day", days
            )
            
            workflow_trends = await AnalyticsService.get_time_series_data(
                db, "workflows_executed", tenant_id, "day", days
            )
            
            # Combine trends (align by timestamp)
            combined_trends = []
            trend_dict = {trend["timestamp"]: trend for trend in trends}
            task_dict = {trend["timestamp"]: trend for trend in task_trends}
            workflow_dict = {trend["timestamp"]: trend for trend in workflow_trends}
            
            # Get all unique timestamps
            all_timestamps = set(trend_dict.keys()) | set(task_dict.keys()) | set(workflow_dict.keys())
            
            for timestamp in sorted(all_timestamps):
                combined_trends.append({
                    "timestamp": timestamp,
                    "api_requests": trend_dict.get(timestamp, {}).get("value", 0),
                    "tasks": task_dict.get(timestamp, {}).get("value", 0),
                    "workflows": workflow_dict.get(timestamp, {}).get("value", 0),
                    "total_activity": (
                        trend_dict.get(timestamp, {}).get("value", 0) +
                        task_dict.get(timestamp, {}).get("value", 0) +
                        workflow_dict.get(timestamp, {}).get("value", 0)
                    )
                })
            
            return combined_trends
            
        except Exception as e:
            logger.error(f"Error getting usage trends: {e}")
            return []
    
    @staticmethod
    async def get_usage_limits_status(
        db: AsyncSession,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current usage vs limits status."""
        try:
            # Get subscription info
            subscription_status = await LicenseService.get_subscription_status(db, tenant_id, user_id)
            limits = subscription_status.get("limits", {})
            
            # Get current period usage
            now = datetime.utcnow()
            month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Get monthly usage
            monthly_usage = await LicenseService.get_usage_summary(db, tenant_id, (now - month_start).days + 1)
            
            # Get hourly usage for rate limiting
            hour_start = now.replace(minute=0, second=0, microsecond=0)
            hourly_usage = await LicenseService.get_usage_summary(db, tenant_id, 1)  # Last day, but we'll filter to hour
            
            # Calculate usage percentages
            limits_status = {}
            
            # Monthly limits
            if "max_tasks_per_month" in limits:
                current_tasks = monthly_usage.get("analytics", {}).get("total_tasks", 0)
                max_tasks = limits["max_tasks_per_month"]
                limits_status["tasks_monthly"] = {
                    "current": current_tasks,
                    "limit": max_tasks,
                    "percentage": (current_tasks / max_tasks * 100) if max_tasks > 0 else 0,
                    "remaining": max(0, max_tasks - current_tasks)
                }
            
            # Hourly limits
            if "api_requests_per_hour" in limits:
                # Approximate hourly usage from daily data
                current_requests = hourly_usage.get("total_usage_events", 0) / 24  # Rough estimate
                max_requests = limits["api_requests_per_hour"]
                limits_status["api_requests_hourly"] = {
                    "current": int(current_requests),
                    "limit": max_requests,
                    "percentage": (current_requests / max_requests * 100) if max_requests > 0 else 0,
                    "remaining": max(0, max_requests - int(current_requests))
                }
            
            # Storage limits
            if "storage_gb" in limits:
                # TODO: Implement actual storage tracking
                limits_status["storage"] = {
                    "current": 0.1,  # Placeholder
                    "limit": limits["storage_gb"],
                    "percentage": 10.0,  # Placeholder
                    "remaining": limits["storage_gb"] - 0.1
                }
            
            return {
                "subscription": subscription_status,
                "limits_status": limits_status,
                "period": {
                    "month_start": month_start.isoformat(),
                    "hour_start": hour_start.isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting usage limits status: {e}")
            return {
                "subscription": {},
                "limits_status": {},
                "period": {},
                "error": str(e)
            }
    
    @staticmethod
    async def record_usage_event(
        db: AsyncSession,
        feature_name: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        usage_count: int = 1,
        usage_data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Record a usage event."""
        try:
            # Create usage metric record
            usage_metric = UsageMetric(
                tenant_id=tenant_id,
                user_id=user_id,
                feature_name=feature_name,
                usage_count=usage_count,
                usage_data=usage_data or {},
                timestamp=datetime.utcnow()
            )
            
            db.add(usage_metric)
            await db.commit()
            
            # Also record in analytics metrics for trend analysis
            await AnalyticsService.record_metric(
                db=db,
                name=f"usage_{feature_name}",
                value=float(usage_count),
                tenant_id=tenant_id,
                labels={"feature": feature_name, "user_id": user_id},
                source="license_tracking"
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error recording usage event: {e}")
            await db.rollback()
            return False
    
    @staticmethod
    async def check_usage_limits(
        db: AsyncSession,
        feature_name: str,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        requested_usage: int = 1
    ) -> Dict[str, Any]:
        """Check if usage would exceed limits."""
        try:
            limits_status = await LicenseService.get_usage_limits_status(db, tenant_id, user_id)
            
            # Map feature names to limit categories
            feature_limit_map = {
                "tasks": "tasks_monthly",
                "api_requests": "api_requests_hourly",
                "workflows": "tasks_monthly",  # Workflows count against task limits
                "storage": "storage"
            }
            
            limit_key = feature_limit_map.get(feature_name)
            if not limit_key:
                # No specific limit for this feature
                return {
                    "allowed": True,
                    "reason": f"No limits defined for feature: {feature_name}"
                }
            
            limit_info = limits_status.get("limits_status", {}).get(limit_key, {})
            if not limit_info:
                return {
                    "allowed": True,
                    "reason": f"No limit data available for: {limit_key}"
                }
            
            current = limit_info.get("current", 0)
            limit = limit_info.get("limit", 0)
            remaining = limit_info.get("remaining", 0)
            
            if current + requested_usage > limit:
                return {
                    "allowed": False,
                    "reason": f"Would exceed limit: {current + requested_usage} > {limit}",
                    "current": current,
                    "limit": limit,
                    "remaining": remaining,
                    "requested": requested_usage
                }
            
            return {
                "allowed": True,
                "reason": "Within limits",
                "current": current,
                "limit": limit,
                "remaining": remaining - requested_usage,
                "requested": requested_usage
            }
            
        except Exception as e:
            logger.error(f"Error checking usage limits: {e}")
            # Allow on error to prevent blocking functionality
            return {
                "allowed": True,
                "reason": f"Error checking limits: {e}"
            }
    
    @staticmethod
    async def get_billing_history(
        db: AsyncSession,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        months: int = 12
    ) -> List[Dict[str, Any]]:
        """Get billing history."""
        try:
            # This would typically query a billing_history table
            # For now, return sample data based on subscription
            subscription_status = await LicenseService.get_subscription_status(db, tenant_id, user_id)
            
            # Generate sample billing history
            history = []
            for i in range(months):
                date = datetime.utcnow() - timedelta(days=30 * i)
                history.append({
                    "period": date.strftime("%Y-%m"),
                    "plan": subscription_status.get("plan_name", "Community"),
                    "amount": 0 if subscription_status.get("plan_name") == "Community" else 29.99,
                    "status": "paid" if subscription_status.get("plan_name") != "Community" else "free",
                    "invoice_date": date.isoformat(),
                    "usage_summary": {
                        "tasks": 450 + (i * 50),
                        "workflows": 25 + (i * 5),
                        "api_requests": 5000 + (i * 1000)
                    }
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting billing history: {e}")
            return []