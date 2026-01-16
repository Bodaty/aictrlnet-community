"""Platform integration usage tracking service"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_

from models.platform_integration import PlatformExecution, PlatformCredential
from schemas.platform_integration import PlatformType, ExecutionStatus

logger = logging.getLogger(__name__)


class PlatformUsageTracker:
    """Tracks usage statistics for platform integrations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def track_execution(
        self,
        execution_id: int,
        user_id: str,
        platform: PlatformType,
        status: ExecutionStatus,
        duration_ms: Optional[int] = None,
        cost: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Track a platform execution for usage statistics"""
        try:
            # Update execution with tracking data
            result = await self.db.execute(
                select(PlatformExecution).where(
                    PlatformExecution.id == execution_id
                )
            )
            execution = result.scalar_one_or_none()
            
            if execution:
                execution.duration_ms = duration_ms
                execution.estimated_cost = cost
                
                # Add tracking metadata
                if not execution.execution_metadata:
                    execution.execution_metadata = {}
                
                execution.execution_metadata.update({
                    "tracked_at": datetime.utcnow().isoformat(),
                    "user_id": user_id,
                    "platform": platform.value,
                    "status": status.value,
                    **(metadata or {})
                })
                
                await self.db.commit()
                
        except Exception as e:
            logger.error(f"Error tracking execution: {e}")
    
    async def get_user_usage(
        self,
        user_id: str,
        platform: Optional[PlatformType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        # Default to last 30 days
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Build query
        query = select(PlatformExecution).join(
            PlatformCredential,
            PlatformExecution.credential_id == PlatformCredential.id
        ).where(
            and_(
                PlatformCredential.user_id == user_id,
                PlatformExecution.started_at >= start_date,
                PlatformExecution.started_at <= end_date
            )
        )
        
        if platform:
            query = query.where(PlatformExecution.platform == platform.value)
        
        result = await self.db.execute(query)
        executions = result.scalars().all()
        
        # Calculate statistics
        total_executions = len(executions)
        successful_executions = sum(1 for e in executions if e.status == ExecutionStatus.COMPLETED.value)
        failed_executions = sum(1 for e in executions if e.status == ExecutionStatus.FAILED.value)
        
        total_duration_ms = sum(e.duration_ms or 0 for e in executions)
        total_cost = sum(e.estimated_cost or 0 for e in executions)
        
        # Group by platform
        platform_stats = {}
        for execution in executions:
            platform = execution.platform
            if platform not in platform_stats:
                platform_stats[platform] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "duration_ms": 0,
                    "cost": 0
                }
            
            platform_stats[platform]["total"] += 1
            if execution.status == ExecutionStatus.COMPLETED.value:
                platform_stats[platform]["successful"] += 1
            elif execution.status == ExecutionStatus.FAILED.value:
                platform_stats[platform]["failed"] += 1
            
            platform_stats[platform]["duration_ms"] += execution.duration_ms or 0
            platform_stats[platform]["cost"] += execution.estimated_cost or 0
        
        return {
            "user_id": user_id,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_executions": total_executions,
                "successful_executions": successful_executions,
                "failed_executions": failed_executions,
                "success_rate": (successful_executions / total_executions * 100) if total_executions > 0 else 0,
                "total_duration_ms": total_duration_ms,
                "total_cost": total_cost,
                "avg_duration_ms": total_duration_ms / total_executions if total_executions > 0 else 0
            },
            "by_platform": platform_stats
        }
    
    async def get_platform_usage(
        self,
        platform: PlatformType,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get usage statistics for a platform across all users"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get execution counts
        result = await self.db.execute(
            select(
                func.count(PlatformExecution.id).label("total"),
                func.sum(func.cast(PlatformExecution.status == ExecutionStatus.COMPLETED.value, type_=int)).label("successful"),
                func.sum(func.cast(PlatformExecution.status == ExecutionStatus.FAILED.value, type_=int)).label("failed"),
                func.sum(PlatformExecution.duration_ms).label("total_duration"),
                func.sum(PlatformExecution.estimated_cost).label("total_cost"),
                func.count(func.distinct(PlatformCredential.user_id)).label("unique_users")
            ).join(
                PlatformCredential,
                PlatformExecution.credential_id == PlatformCredential.id
            ).where(
                and_(
                    PlatformExecution.platform == platform.value,
                    PlatformExecution.started_at >= start_date,
                    PlatformExecution.started_at <= end_date
                )
            )
        )
        
        stats = result.one()
        
        # Get hourly distribution
        hourly_result = await self.db.execute(
            select(
                func.date_trunc('hour', PlatformExecution.started_at).label("hour"),
                func.count(PlatformExecution.id).label("count")
            ).where(
                and_(
                    PlatformExecution.platform == platform.value,
                    PlatformExecution.started_at >= start_date,
                    PlatformExecution.started_at <= end_date
                )
            ).group_by("hour").order_by("hour")
        )
        
        hourly_distribution = [
            {
                "hour": row.hour.isoformat() if row.hour else None,
                "count": row.count
            }
            for row in hourly_result
        ]
        
        return {
            "platform": platform.value,
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_executions": stats.total or 0,
                "successful_executions": stats.successful or 0,
                "failed_executions": stats.failed or 0,
                "success_rate": ((stats.successful or 0) / stats.total * 100) if stats.total else 0,
                "total_duration_ms": stats.total_duration or 0,
                "total_cost": float(stats.total_cost or 0),
                "unique_users": stats.unique_users or 0,
                "avg_duration_ms": (stats.total_duration or 0) / stats.total if stats.total else 0
            },
            "hourly_distribution": hourly_distribution
        }
    
    async def check_rate_limits(
        self,
        user_id: str,
        platform: PlatformType,
        window_minutes: int = 5
    ) -> Dict[str, Any]:
        """Check if user is approaching rate limits"""
        start_time = datetime.utcnow() - timedelta(minutes=window_minutes)
        
        # Count recent executions
        result = await self.db.execute(
            select(func.count(PlatformExecution.id)).join(
                PlatformCredential,
                PlatformExecution.credential_id == PlatformCredential.id
            ).where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.platform == platform.value,
                    PlatformExecution.started_at >= start_time
                )
            )
        )
        
        execution_count = result.scalar() or 0
        
        # Get platform rate limits (would be from config or adapter)
        rate_limits = {
            PlatformType.N8N: {"per_minute": 100},
            PlatformType.ZAPIER: {"per_minute": 75},
            PlatformType.MAKE: {"per_minute": 60},
            PlatformType.POWER_AUTOMATE: {"per_5min": 100000},
            PlatformType.IFTTT: {"per_hour": 100}
        }
        
        platform_limits = rate_limits.get(platform, {})
        
        # Calculate usage percentage
        if platform == PlatformType.POWER_AUTOMATE and "per_5min" in platform_limits:
            limit = platform_limits["per_5min"]
            usage_percent = (execution_count / limit) * 100 if limit > 0 else 0
        elif platform == PlatformType.IFTTT and "per_hour" in platform_limits:
            # For IFTTT, check hourly window
            hour_start = datetime.utcnow() - timedelta(hours=1)
            hour_result = await self.db.execute(
                select(func.count(PlatformExecution.id)).join(
                    PlatformCredential,
                    PlatformExecution.credential_id == PlatformCredential.id
                ).where(
                    and_(
                        PlatformCredential.user_id == user_id,
                        PlatformExecution.platform == platform.value,
                        PlatformExecution.started_at >= hour_start
                    )
                )
            )
            hour_count = hour_result.scalar() or 0
            limit = platform_limits["per_hour"]
            usage_percent = (hour_count / limit) * 100 if limit > 0 else 0
            execution_count = hour_count
            window_minutes = 60
        else:
            # Per minute limits
            limit = platform_limits.get("per_minute", 100) * window_minutes
            usage_percent = (execution_count / limit) * 100 if limit > 0 else 0
        
        return {
            "platform": platform.value,
            "user_id": user_id,
            "window_minutes": window_minutes,
            "execution_count": execution_count,
            "limit": limit,
            "usage_percent": usage_percent,
            "approaching_limit": usage_percent > 80,
            "at_limit": usage_percent >= 100
        }
    
    async def get_cost_summary(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get cost summary for platform usage"""
        if not start_date:
            start_date = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Build query
        query = select(
            PlatformExecution.platform,
            func.count(PlatformExecution.id).label("execution_count"),
            func.sum(PlatformExecution.estimated_cost).label("total_cost")
        ).where(
            and_(
                PlatformExecution.started_at >= start_date,
                PlatformExecution.started_at <= end_date,
                PlatformExecution.estimated_cost > 0
            )
        )
        
        if user_id:
            query = query.join(
                PlatformCredential,
                PlatformExecution.credential_id == PlatformCredential.id
            ).where(PlatformCredential.user_id == user_id)
        
        query = query.group_by(PlatformExecution.platform)
        
        result = await self.db.execute(query)
        
        platform_costs = {}
        total_cost = 0
        total_executions = 0
        
        for row in result:
            platform_costs[row.platform] = {
                "execution_count": row.execution_count,
                "total_cost": float(row.total_cost or 0),
                "avg_cost_per_execution": float(row.total_cost or 0) / row.execution_count if row.execution_count > 0 else 0
            }
            total_cost += float(row.total_cost or 0)
            total_executions += row.execution_count
        
        return {
            "period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "user_id": user_id,
            "summary": {
                "total_cost": total_cost,
                "total_executions": total_executions,
                "avg_cost_per_execution": total_cost / total_executions if total_executions > 0 else 0
            },
            "by_platform": platform_costs
        }
    
    async def get_trending_workflows(
        self,
        platform: Optional[PlatformType] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending workflows based on execution count"""
        # Last 7 days
        start_date = datetime.utcnow() - timedelta(days=7)
        
        query = select(
            PlatformExecution.external_workflow_id,
            PlatformExecution.platform,
            func.count(PlatformExecution.id).label("execution_count"),
            func.sum(func.cast(PlatformExecution.status == ExecutionStatus.COMPLETED.value, type_=int)).label("success_count"),
            func.avg(PlatformExecution.duration_ms).label("avg_duration")
        ).where(
            and_(
                PlatformExecution.started_at >= start_date,
                PlatformExecution.external_workflow_id.isnot(None)
            )
        )
        
        if platform:
            query = query.where(PlatformExecution.platform == platform.value)
        
        query = query.group_by(
            PlatformExecution.external_workflow_id,
            PlatformExecution.platform
        ).order_by(func.count(PlatformExecution.id).desc()).limit(limit)
        
        result = await self.db.execute(query)
        
        trending = []
        for row in result:
            trending.append({
                "workflow_id": row.external_workflow_id,
                "platform": row.platform,
                "execution_count": row.execution_count,
                "success_count": row.success_count or 0,
                "success_rate": ((row.success_count or 0) / row.execution_count * 100) if row.execution_count > 0 else 0,
                "avg_duration_ms": float(row.avg_duration or 0)
            })
        
        return trending