"""Usage tracking system for monitoring and recording usage metrics."""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from sqlalchemy.exc import IntegrityError

from core.config import get_settings
from core.cache import get_cache

logger = logging.getLogger(__name__)


class UsageTracker:
    """Track and record usage metrics for billing and limits."""
    
    def __init__(self, db: Optional[AsyncSession] = None):
        self.db = db
        self.cache = None  # Will be initialized on first use
        self.settings = get_settings()
        
        # In-memory buffer for batching metrics
        self._buffer = defaultdict(lambda: defaultdict(float))
        self._buffer_lock = asyncio.Lock()
        
        # Periodic flush task
        self._flush_task = None
        self._flush_interval = 30  # seconds
        
        # Start periodic flush
        if self.db:
            self._start_periodic_flush()
    
    def _start_periodic_flush(self):
        """Start background task to periodically flush metrics."""
        async def flush_loop():
            while True:
                try:
                    await asyncio.sleep(self._flush_interval)
                    await self.flush_buffer()
                except Exception as e:
                    logger.error(f"Error in periodic flush: {e}")
        
        self._flush_task = asyncio.create_task(flush_loop())
    
    async def track_usage(
        self,
        tenant_id: str,
        metric_type: str,
        value: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
        flush_immediately: bool = False
    ):
        """Track a usage metric."""
        
        # Quick increment in cache for rate limiting
        if not self.cache:
            self.cache = await get_cache()
        cache_key = f"usage:{tenant_id}:{metric_type}:{datetime.utcnow().strftime('%Y%m%d')}"
        await self.cache.increment(cache_key, delta=int(value))
        
        # Add to buffer for database persistence
        async with self._buffer_lock:
            buffer_key = f"{tenant_id}:{metric_type}"
            self._buffer[buffer_key]["value"] += value
            self._buffer[buffer_key]["count"] += 1
            
            if metadata:
                if "metadata" not in self._buffer[buffer_key]:
                    self._buffer[buffer_key]["metadata"] = []
                self._buffer[buffer_key]["metadata"].append(metadata)
        
        # Log high-value usage
        if value > 100:
            logger.info(f"High usage tracked: {tenant_id} - {metric_type}: {value}")
        
        # Flush if requested or buffer is large
        if flush_immediately or len(self._buffer) > 1000:
            await self.flush_buffer()
    
    async def flush_buffer(self):
        """Flush buffered metrics to database."""
        
        if not self.db:
            logger.warning("No database connection for flushing metrics")
            return
        
        async with self._buffer_lock:
            if not self._buffer:
                return
            
            # Copy and clear buffer
            buffer_copy = dict(self._buffer)
            self._buffer.clear()
        
        # Process buffered metrics
        from models.enforcement import UsageMetric
        import uuid
        
        for key, metrics in buffer_copy.items():
            tenant_id, metric_type = key.split(":", 1)
            
            try:
                # Validate tenant_id is a valid UUID
                try:
                    uuid.UUID(tenant_id)
                except ValueError:
                    # Skip metrics with invalid tenant_id (like 'default-tenant')
                    logger.debug(f"Skipping metric with invalid tenant_id: {tenant_id}")
                    continue
                
                # Aggregate metadata if present
                metadata = {}
                if "metadata" in metrics and metrics["metadata"]:
                    # Merge all metadata entries
                    for m in metrics["metadata"]:
                        metadata.update(m)
                
                # Create metric record
                metric = UsageMetric(
                    tenant_id=tenant_id,
                    metric_type=metric_type,
                    value=metrics["value"],
                    count=int(metrics["count"]),
                    meta_data=metadata,  # Note: changed from metadata to meta_data
                    timestamp=datetime.utcnow()
                )
                
                self.db.add(metric)
                await self.db.commit()
                
            except IntegrityError:
                await self.db.rollback()
                logger.error(f"Failed to save metric: {key}")
            except Exception as e:
                await self.db.rollback()
                logger.error(f"Error saving metric {key}: {e}")
    
    async def get_usage_summary(
        self,
        tenant_id: str,
        metric_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get usage summary for a tenant."""
        
        if not self.db:
            return {"error": "No database connection"}
        
        # Default to current month
        if not start_date:
            start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        if not end_date:
            end_date = datetime.utcnow()
        
        from models.enforcement import UsageMetric
        
        # Build query
        query = select(
            UsageMetric.metric_type,
            func.sum(UsageMetric.value).label("total_value"),
            func.sum(UsageMetric.count).label("total_count"),
            func.count(UsageMetric.id).label("record_count")
        ).where(
            and_(
                UsageMetric.tenant_id == tenant_id,
                UsageMetric.timestamp >= start_date,
                UsageMetric.timestamp <= end_date
            )
        ).group_by(UsageMetric.metric_type)
        
        if metric_types:
            query = query.where(UsageMetric.metric_type.in_(metric_types))
        
        result = await self.db.execute(query)
        rows = result.all()
        
        summary = {
            "tenant_id": tenant_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "metrics": {}
        }
        
        for row in rows:
            summary["metrics"][row.metric_type] = {
                "total_value": float(row.total_value or 0),
                "total_count": int(row.total_count or 0),
                "record_count": int(row.record_count or 0)
            }
        
        return summary
    
    async def get_usage_timeline(
        self,
        tenant_id: str,
        metric_type: str,
        granularity: str = "day",
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get usage timeline for visualization."""
        
        if not self.db:
            return []
        
        from models.enforcement import UsageMetric
        
        # Determine date truncation based on granularity
        if granularity == "hour":
            date_trunc = func.date_trunc("hour", UsageMetric.timestamp)
            interval = timedelta(hours=1)
        elif granularity == "day":
            date_trunc = func.date_trunc("day", UsageMetric.timestamp)
            interval = timedelta(days=1)
        elif granularity == "week":
            date_trunc = func.date_trunc("week", UsageMetric.timestamp)
            interval = timedelta(weeks=1)
        else:  # month
            date_trunc = func.date_trunc("month", UsageMetric.timestamp)
            interval = timedelta(days=30)
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Query grouped by time period
        result = await self.db.execute(
            select(
                date_trunc.label("period"),
                func.sum(UsageMetric.value).label("total_value"),
                func.sum(UsageMetric.count).label("total_count")
            ).where(
                and_(
                    UsageMetric.tenant_id == tenant_id,
                    UsageMetric.metric_type == metric_type,
                    UsageMetric.timestamp >= start_date
                )
            ).group_by("period").order_by("period")
        )
        
        timeline = []
        for row in result:
            timeline.append({
                "period": row.period.isoformat(),
                "value": float(row.total_value or 0),
                "count": int(row.total_count or 0)
            })
        
        return timeline
    
    async def get_top_users(
        self,
        metric_type: str,
        limit: int = 10,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get top users by usage for a metric type."""
        
        if not self.db:
            return []
        
        from models.enforcement import UsageMetric
        from models.community import Tenant
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        # Query top users
        result = await self.db.execute(
            select(
                UsageMetric.tenant_id,
                Tenant.name.label("tenant_name"),
                func.sum(UsageMetric.value).label("total_value"),
                func.sum(UsageMetric.count).label("total_count")
            ).select_from(
                UsageMetric
            ).join(
                Tenant, Tenant.id == UsageMetric.tenant_id
            ).where(
                and_(
                    UsageMetric.metric_type == metric_type,
                    UsageMetric.timestamp >= start_date
                )
            ).group_by(
                UsageMetric.tenant_id, Tenant.name
            ).order_by(
                func.sum(UsageMetric.value).desc()
            ).limit(limit)
        )
        
        top_users = []
        for row in result:
            top_users.append({
                "tenant_id": str(row.tenant_id),
                "tenant_name": row.tenant_name,
                "total_value": float(row.total_value or 0),
                "total_count": int(row.total_count or 0)
            })
        
        return top_users
    
    async def create_monthly_summary(self, month: Optional[datetime] = None):
        """Create monthly usage summaries for all tenants."""
        
        if not self.db:
            return
        
        from models.enforcement import UsageMetric, UsageSummary
        
        # Default to previous month
        if not month:
            now = datetime.utcnow()
            month = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        
        start_date = month
        end_date = (month + timedelta(days=32)).replace(day=1) - timedelta(seconds=1)
        
        # Get all usage for the month grouped by tenant and metric
        result = await self.db.execute(
            select(
                UsageMetric.tenant_id,
                UsageMetric.metric_type,
                func.sum(UsageMetric.value).label("total_value"),
                func.sum(UsageMetric.count).label("total_count"),
                func.date_trunc("day", UsageMetric.timestamp).label("day"),
                func.array_agg(UsageMetric.metadata).label("metadata_list")
            ).where(
                and_(
                    UsageMetric.timestamp >= start_date,
                    UsageMetric.timestamp <= end_date
                )
            ).group_by(
                UsageMetric.tenant_id,
                UsageMetric.metric_type,
                "day"
            )
        )
        
        # Process results into summaries
        summaries = defaultdict(lambda: defaultdict(lambda: {
            "total_value": 0,
            "total_count": 0,
            "daily_breakdown": {}
        }))
        
        for row in result:
            key = (str(row.tenant_id), row.metric_type)
            summaries[key]["total_value"] += float(row.total_value or 0)
            summaries[key]["total_count"] += int(row.total_count or 0)
            
            day_str = row.day.strftime("%Y-%m-%d")
            summaries[key]["daily_breakdown"][day_str] = {
                "value": float(row.total_value or 0),
                "count": int(row.total_count or 0)
            }
        
        # Save summaries
        for (tenant_id, metric_type), data in summaries.items():
            try:
                summary = UsageSummary(
                    tenant_id=tenant_id,
                    month=month.date(),
                    metric_type=metric_type,
                    total_value=data["total_value"],
                    total_count=data["total_count"],
                    daily_breakdown=data["daily_breakdown"]
                )
                
                # Upsert
                existing = await self.db.execute(
                    select(UsageSummary).where(
                        and_(
                            UsageSummary.tenant_id == tenant_id,
                            UsageSummary.month == month.date(),
                            UsageSummary.metric_type == metric_type
                        )
                    )
                )
                existing_summary = existing.scalar_one_or_none()
                
                if existing_summary:
                    existing_summary.total_value = data["total_value"]
                    existing_summary.total_count = data["total_count"]
                    existing_summary.daily_breakdown = data["daily_breakdown"]
                else:
                    self.db.add(summary)
                
                await self.db.commit()
                
            except Exception as e:
                await self.db.rollback()
                logger.error(f"Failed to create summary for {tenant_id}/{metric_type}: {e}")
    
    async def track_api_call(
        self,
        tenant_id: str,
        endpoint: str,
        method: str,
        response_time_ms: float,
        status_code: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track an API call with detailed metadata."""
        
        call_metadata = {
            "endpoint": endpoint,
            "method": method,
            "response_time_ms": response_time_ms,
            "status_code": status_code
        }
        
        if metadata:
            call_metadata.update(metadata)
        
        await self.track_usage(
            tenant_id=tenant_id,
            metric_type="api_calls",
            value=1,
            metadata=call_metadata,
            flush_immediately=(response_time_ms > 1000)  # Flush slow calls immediately
        )
    
    async def track_workflow_execution(
        self,
        tenant_id: str,
        workflow_id: str,
        duration_seconds: float,
        status: str,
        node_count: int = 0
    ):
        """Track workflow execution metrics."""
        
        await self.track_usage(
            tenant_id=tenant_id,
            metric_type="executions",
            value=1,
            metadata={
                "workflow_id": workflow_id,
                "duration_seconds": duration_seconds,
                "status": status,
                "node_count": node_count
            }
        )
    
    async def track_ai_agent_request(
        self,
        tenant_id: str,
        adapter_type: str,
        agent_id: str,
        request_type: str,
        duration_ms: float,
        success: bool,
        tokens_used: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track AI agent framework requests for usage and billing."""
        
        request_metadata = {
            "adapter_type": adapter_type,  # langchain, autogpt, autogen, etc.
            "agent_id": agent_id,
            "request_type": request_type,  # create_agent, execute, get_state, etc.
            "duration_ms": duration_ms,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if tokens_used is not None:
            request_metadata["tokens_used"] = tokens_used
        
        if error:
            request_metadata["error"] = error
        
        if metadata:
            request_metadata.update(metadata)
        
        # Track the AI agent request
        await self.track_usage(
            tenant_id=tenant_id,
            metric_type="ai_agent_requests",
            value=1,
            metadata=request_metadata,
            flush_immediately=(not success or duration_ms > 5000)  # Flush errors and slow requests
        )
        
        # Also track daily usage for Community Edition limit enforcement
        if adapter_type in ["langchain", "langchain-community"]:
            # Track daily usage specifically for rate limiting
            daily_key = f"ai_agent_daily:{tenant_id}:{datetime.utcnow().strftime('%Y%m%d')}"
            if not self.cache:
                self.cache = await get_cache()
            
            # Increment daily counter
            daily_count = await self.cache.increment(daily_key)
            
            # Set expiry to end of day if this is the first request
            if daily_count == 1:
                seconds_until_midnight = (
                    datetime.utcnow().replace(hour=23, minute=59, second=59) - 
                    datetime.utcnow()
                ).total_seconds()
                await self.cache.expire(daily_key, int(seconds_until_midnight))
        
        # Log significant AI agent requests
        if not success:
            logger.warning(
                f"AI agent request failed: {tenant_id} - {adapter_type}/{agent_id} - "
                f"{request_type} - Error: {error}"
            )
        elif duration_ms > 10000:  # Log slow requests (>10s)
            logger.info(
                f"Slow AI agent request: {tenant_id} - {adapter_type}/{agent_id} - "
                f"{request_type} - Duration: {duration_ms}ms"
            )
    
    async def get_ai_agent_daily_usage(
        self,
        tenant_id: str,
        date: Optional[datetime] = None
    ) -> int:
        """Get daily AI agent request count for a tenant."""
        
        if not date:
            date = datetime.utcnow()
        
        if not self.cache:
            self.cache = await get_cache()
        
        daily_key = f"ai_agent_daily:{tenant_id}:{date.strftime('%Y%m%d')}"
        count = await self.cache.get(daily_key)
        
        return int(count) if count else 0
    
    async def track_storage_usage(
        self,
        tenant_id: str,
        bytes_used: int,
        file_type: Optional[str] = None
    ):
        """Track storage usage in bytes."""
        
        gb_used = bytes_used / (1024 ** 3)  # Convert to GB
        
        await self.track_usage(
            tenant_id=tenant_id,
            metric_type="storage_gb",
            value=gb_used,
            metadata={
                "bytes": bytes_used,
                "file_type": file_type
            }
        )
    
    async def cleanup(self):
        """Clean up resources."""
        
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        
        # Final flush
        await self.flush_buffer()


# Global instance for dependency injection
_usage_tracker: Optional[UsageTracker] = None


async def get_usage_tracker(db: AsyncSession) -> UsageTracker:
    """Get usage tracker instance."""
    
    global _usage_tracker
    if not _usage_tracker:
        _usage_tracker = UsageTracker(db)
    
    return _usage_tracker


# Convenience functions for common tracking
async def track_api_request(
    tenant_id: str,
    request,
    response,
    start_time: float,
    db: AsyncSession
):
    """Track API request from middleware."""
    
    tracker = await get_usage_tracker(db)
    
    response_time = (datetime.utcnow().timestamp() - start_time) * 1000
    
    await tracker.track_api_call(
        tenant_id=tenant_id,
        endpoint=request.url.path,
        method=request.method,
        response_time_ms=response_time,
        status_code=response.status_code,
        metadata={
            "user_agent": request.headers.get("user-agent"),
            "ip": request.client.host if request.client else None
        }
    )