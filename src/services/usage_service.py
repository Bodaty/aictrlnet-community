"""Usage tracking service for Community edition."""

from typing import Optional
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from models.usage_metrics import UsageMetric, UsageLimit
from schemas.usage_metrics import (
    UsageMetricResponse, UsageLimitResponse, UsageStatusResponse
)


class UsageService:
    """Service for tracking basic usage in Community edition."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def get_current_usage(self, tenant_id: str = "community") -> UsageMetricResponse:
        """Get current usage metrics."""
        # Get current month's metrics
        now = datetime.now(timezone.utc)
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        result = await self.db.execute(
            select(UsageMetric).where(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.period_start == period_start
            )
        )
        
        metric = result.scalar_one_or_none()
        
        if not metric:
            # Create new metric for this period
            period_end = period_start + relativedelta(months=1) - relativedelta(seconds=1)
            metric = UsageMetric(
                tenant_id=tenant_id,
                period_start=period_start,
                period_end=period_end
            )
            self.db.add(metric)
            await self.db.commit()
            await self.db.refresh(metric)
        
        return UsageMetricResponse.from_orm(metric)
    
    async def get_usage_limits(self, edition: str = "community") -> UsageLimitResponse:
        """Get usage limits for edition."""
        result = await self.db.execute(
            select(UsageLimit).where(UsageLimit.edition == edition)
        )
        
        limit = result.scalar_one_or_none()
        
        if not limit:
            # Create default Community limits
            limit = UsageLimit(edition=edition)
            self.db.add(limit)
            await self.db.commit()
            await self.db.refresh(limit)
        
        return UsageLimitResponse.from_orm(limit)
    
    async def get_usage_status(self, tenant_id: str = "community") -> UsageStatusResponse:
        """Get combined usage status with upgrade prompts."""
        current = await self.get_current_usage(tenant_id)
        limits = await self.get_usage_limits("community")
        
        # Calculate percentages
        workflows_pct = (current.workflow_count / limits.max_workflows * 100) if limits.max_workflows > 0 else 0
        adapters_pct = (current.adapter_count / limits.max_adapters * 100) if limits.max_adapters > 0 else 0
        users_pct = (current.user_count / limits.max_users * 100) if limits.max_users > 0 else 0
        api_calls_pct = (current.api_calls_month / limits.max_api_calls_month * 100) if limits.max_api_calls_month > 0 else 0
        storage_pct = (current.storage_bytes / limits.max_storage_bytes * 100) if limits.max_storage_bytes > 0 else 0
        
        # Check for upgrade needs
        upgrade_reasons = []
        threshold = 80  # Alert at 80% usage
        
        if workflows_pct >= threshold:
            upgrade_reasons.append(f"Approaching workflow limit ({current.workflow_count}/{limits.max_workflows})")
        if adapters_pct >= threshold:
            upgrade_reasons.append(f"Approaching adapter limit ({current.adapter_count}/{limits.max_adapters})")
        if users_pct >= 100:  # Only 1 user allowed
            upgrade_reasons.append("Need multiple users? Upgrade to Business")
        if api_calls_pct >= threshold:
            upgrade_reasons.append(f"Approaching API call limit ({current.api_calls_month}/{limits.max_api_calls_month})")
        if storage_pct >= threshold:
            storage_gb = current.storage_bytes / 1073741824
            upgrade_reasons.append(f"Approaching storage limit ({storage_gb:.1f}GB/1GB)")
        
        return UsageStatusResponse(
            current_usage=current,
            limits=limits,
            workflows_percent=workflows_pct,
            adapters_percent=adapters_pct,
            users_percent=users_pct,
            api_calls_percent=api_calls_pct,
            storage_percent=storage_pct,
            needs_upgrade=len(upgrade_reasons) > 0,
            upgrade_reasons=upgrade_reasons
        )
    
    async def increment_api_calls(self, tenant_id: str = "community", count: int = 1) -> None:
        """Increment API call counter."""
        current = await self.get_current_usage(tenant_id)
        
        await self.db.execute(
            select(UsageMetric).where(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.period_start == current.period_start
            ).with_for_update()
        )
        
        result = await self.db.execute(
            select(UsageMetric).where(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.period_start == current.period_start
            )
        )
        
        metric = result.scalar_one()
        metric.api_calls_month += count
        metric.last_updated = datetime.now(timezone.utc)
        
        await self.db.commit()
    
    async def update_counts(
        self, 
        tenant_id: str = "community",
        workflows: Optional[int] = None,
        adapters: Optional[int] = None,
        users: Optional[int] = None,
        storage_bytes: Optional[int] = None
    ) -> None:
        """Update resource counts."""
        current = await self.get_current_usage(tenant_id)
        
        result = await self.db.execute(
            select(UsageMetric).where(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.period_start == current.period_start
            )
        )
        
        metric = result.scalar_one()
        
        if workflows is not None:
            metric.workflow_count = workflows
        if adapters is not None:
            metric.adapter_count = adapters
        if users is not None:
            metric.user_count = users
        if storage_bytes is not None:
            metric.storage_bytes = storage_bytes
        
        metric.last_updated = datetime.now(timezone.utc)
        await self.db.commit()