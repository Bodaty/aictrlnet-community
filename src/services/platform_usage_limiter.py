"""Cloud-aware usage limiter for platform integrations"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_

from models.platform_integration import PlatformExecution, PlatformCredential
from schemas.platform_integration import PlatformType, ExecutionStatus
from services.cloud_detection import cloud_detector
from services.platform_cloud_config import platform_cloud_config
from services.usage_service import UsageService

logger = logging.getLogger(__name__)


class UsageLimitExceeded(Exception):
    """Raised when usage limit is exceeded"""
    def __init__(self, message: str, limit_type: str, current: int, limit: int):
        super().__init__(message)
        self.limit_type = limit_type
        self.current = current
        self.limit = limit


class PlatformUsageLimiter:
    """Enforces cloud-aware usage limits for platform integrations"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.cloud_config = platform_cloud_config
        self.usage_service = UsageService(db)
    
    async def check_execution_allowed(
        self,
        user_id: str,
        platform: PlatformType,
        estimated_duration_ms: Optional[int] = None,
        payload_size_mb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if platform execution is allowed based on all limits"""
        
        results = {
            "allowed": True,
            "checks": {},
            "warnings": [],
            "errors": []
        }
        
        # Check edition limits first
        edition_check = await self._check_edition_limits(user_id)
        results["checks"]["edition"] = edition_check
        if not edition_check["allowed"]:
            results["allowed"] = False
            results["errors"].append(edition_check["reason"])
        
        # Check cloud environment limits
        cloud_check = await self._check_cloud_limits(platform, estimated_duration_ms, payload_size_mb)
        results["checks"]["cloud"] = cloud_check
        if not cloud_check["allowed"]:
            results["allowed"] = False
            results["errors"].append(cloud_check["reason"])
        elif cloud_check.get("warnings"):
            results["warnings"].extend(cloud_check["warnings"])
        
        # Check platform-specific rate limits
        rate_check = await self._check_rate_limits(user_id, platform)
        results["checks"]["rate_limits"] = rate_check
        if not rate_check["allowed"]:
            results["allowed"] = False
            results["errors"].append(rate_check["reason"])
        
        # Check concurrent execution limits
        concurrent_check = await self._check_concurrent_limits(user_id, platform)
        results["checks"]["concurrent"] = concurrent_check
        if not concurrent_check["allowed"]:
            results["allowed"] = False
            results["errors"].append(concurrent_check["reason"])
        
        # Check cost limits if enabled
        if self.cloud_config.get_cost_config()["track_costs"]:
            cost_check = await self._check_cost_limits(user_id)
            results["checks"]["cost"] = cost_check
            if not cost_check["allowed"]:
                results["allowed"] = False
                results["errors"].append(cost_check["reason"])
        
        return results
    
    async def _check_edition_limits(self, user_id: str) -> Dict[str, Any]:
        """Check AICtrlNet edition-based limits"""
        try:
            # Get user's edition limits
            usage_status = await self.usage_service.get_usage_status(user_id)
            
            # Check API calls limit
            if usage_status.current_usage.api_calls_month >= usage_status.limits.max_api_calls_month:
                return {
                    "allowed": False,
                    "reason": f"Monthly API call limit reached ({usage_status.limits.max_api_calls_month})",
                    "limit_type": "api_calls",
                    "current": usage_status.current_usage.api_calls_month,
                    "limit": usage_status.limits.max_api_calls_month
                }
            
            # Check if platform integrations are allowed in edition
            if usage_status.edition == "community":
                # Community edition has limited platform integration features
                return {
                    "allowed": True,
                    "warnings": ["Platform integrations have limited features in Community edition"]
                }
            
            return {"allowed": True}
            
        except Exception as e:
            logger.error(f"Error checking edition limits: {e}")
            return {"allowed": True}  # Allow on error, log for monitoring
    
    async def _check_cloud_limits(
        self,
        platform: PlatformType,
        estimated_duration_ms: Optional[int] = None,
        payload_size_mb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check cloud environment limits"""
        
        platform_limits = self.cloud_config.get_platform_limits(platform)
        resource_limits = self.cloud_config.resource_limits
        
        # Check execution time
        if estimated_duration_ms:
            max_duration_ms = min(
                platform_limits["max_execution_time"] * 1000,
                resource_limits["max_execution_time_seconds"] * 1000
            )
            
            if estimated_duration_ms > max_duration_ms:
                if self.cloud_config.is_serverless:
                    return {
                        "allowed": False,
                        "reason": f"Execution time ({estimated_duration_ms}ms) exceeds serverless limit ({max_duration_ms}ms)",
                        "limit_type": "execution_time",
                        "current": estimated_duration_ms,
                        "limit": max_duration_ms
                    }
                else:
                    return {
                        "allowed": True,
                        "warnings": [f"Execution may timeout (estimated: {estimated_duration_ms}ms, limit: {max_duration_ms}ms)"]
                    }
        
        # Check payload size
        if payload_size_mb:
            max_payload_mb = min(
                platform_limits["max_payload_size_mb"],
                resource_limits["max_payload_size_mb"]
            )
            
            if payload_size_mb > max_payload_mb:
                return {
                    "allowed": False,
                    "reason": f"Payload size ({payload_size_mb}MB) exceeds limit ({max_payload_mb}MB)",
                    "limit_type": "payload_size",
                    "current": payload_size_mb,
                    "limit": max_payload_mb
                }
        
        # Validate platform compatibility
        compatibility = self.cloud_config.validate_platform_compatibility(platform)
        if compatibility["warnings"]:
            return {
                "allowed": True,
                "warnings": compatibility["warnings"],
                "recommendations": compatibility["recommendations"]
            }
        
        return {"allowed": True}
    
    async def _check_rate_limits(self, user_id: str, platform: PlatformType) -> Dict[str, Any]:
        """Check platform-specific rate limits"""
        
        platform_limits = self.cloud_config.get_platform_limits(platform)
        max_per_minute = platform_limits["max_executions_per_minute"]
        
        # Adjust for cloud environment
        if self.cloud_config.environment.value != "production":
            # Non-production environments get reduced rates
            max_per_minute = int(max_per_minute * 0.5)
        
        # Count recent executions
        one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
        
        result = await self.db.execute(
            select(func.count(PlatformExecution.id)).join(
                PlatformCredential,
                PlatformExecution.credential_id == PlatformCredential.id
            ).where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.platform == platform.value,
                    PlatformExecution.started_at >= one_minute_ago
                )
            )
        )
        
        current_count = result.scalar() or 0
        
        if current_count >= max_per_minute:
            return {
                "allowed": False,
                "reason": f"Rate limit exceeded ({current_count}/{max_per_minute} per minute)",
                "limit_type": "rate_limit",
                "current": current_count,
                "limit": max_per_minute,
                "retry_after": 60  # seconds
            }
        
        # Warn if approaching limit
        if current_count >= max_per_minute * 0.8:
            return {
                "allowed": True,
                "warnings": [f"Approaching rate limit ({current_count}/{max_per_minute} per minute)"]
            }
        
        return {"allowed": True}
    
    async def _check_concurrent_limits(self, user_id: str, platform: PlatformType) -> Dict[str, Any]:
        """Check concurrent execution limits"""
        
        platform_limits = self.cloud_config.get_platform_limits(platform)
        max_concurrent = platform_limits["max_concurrent_executions"]
        
        # Further limit in serverless
        if self.cloud_config.is_serverless:
            max_concurrent = min(max_concurrent, 5)
        
        # Count running executions
        result = await self.db.execute(
            select(func.count(PlatformExecution.id)).join(
                PlatformCredential,
                PlatformExecution.credential_id == PlatformCredential.id
            ).where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.platform == platform.value,
                    PlatformExecution.status == ExecutionStatus.RUNNING.value
                )
            )
        )
        
        current_running = result.scalar() or 0
        
        if current_running >= max_concurrent:
            return {
                "allowed": False,
                "reason": f"Concurrent execution limit reached ({current_running}/{max_concurrent})",
                "limit_type": "concurrent_executions",
                "current": current_running,
                "limit": max_concurrent
            }
        
        return {"allowed": True}
    
    async def _check_cost_limits(self, user_id: str) -> Dict[str, Any]:
        """Check cost-based limits"""
        
        cost_config = self.cloud_config.get_cost_config()
        if not cost_config["track_costs"] or not cost_config["monthly_budget"]:
            return {"allowed": True}
        
        # Get current month's cost
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
        
        result = await self.db.execute(
            select(func.sum(PlatformExecution.estimated_cost)).join(
                PlatformCredential,
                PlatformExecution.credential_id == PlatformCredential.id
            ).where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.started_at >= start_of_month,
                    PlatformExecution.estimated_cost.isnot(None)
                )
            )
        )
        
        current_cost = float(result.scalar() or 0)
        monthly_budget = cost_config["monthly_budget"]
        
        if current_cost >= monthly_budget:
            return {
                "allowed": False,
                "reason": f"Monthly cost budget exceeded (${current_cost:.2f}/${monthly_budget:.2f})",
                "limit_type": "cost_budget",
                "current": current_cost,
                "limit": monthly_budget
            }
        
        # Warn at 80% of budget
        if current_cost >= monthly_budget * 0.8:
            return {
                "allowed": True,
                "warnings": [f"Approaching monthly budget (${current_cost:.2f}/${monthly_budget:.2f})"]
            }
        
        return {"allowed": True}
    
    async def record_execution_start(
        self,
        user_id: str,
        platform: PlatformType,
        execution_id: int
    ):
        """Record execution start for limit tracking"""
        # Update last execution time for rate limiting
        await self.db.execute(
            f"""
            UPDATE platform_credentials 
            SET last_used_at = NOW() 
            WHERE user_id = %s AND platform = %s
            """,
            (user_id, platform.value)
        )
        
        # Log for monitoring
        logger.info(
            f"Platform execution started",
            extra={
                "user_id": user_id,
                "platform": platform.value,
                "execution_id": execution_id,
                "cloud_provider": self.cloud_config.cloud_provider.value,
                "environment": self.cloud_config.environment.value
            }
        )
    
    async def get_usage_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive usage summary with limits"""
        
        summary = {
            "cloud_environment": {
                "provider": self.cloud_config.cloud_provider.value,
                "environment": self.cloud_config.environment.value,
                "is_serverless": self.cloud_config.is_serverless,
                "resource_limits": self.cloud_config.resource_limits
            },
            "platforms": {}
        }
        
        # Get limits and usage for each platform
        for platform in PlatformType:
            if platform == PlatformType.CUSTOM:
                continue
                
            platform_limits = self.cloud_config.get_platform_limits(platform)
            
            # Get current usage
            one_minute_ago = datetime.utcnow() - timedelta(minutes=1)
            
            # Count executions
            result = await self.db.execute(
                select(
                    func.count(PlatformExecution.id).label("total"),
                    func.count(func.distinct(func.date(PlatformExecution.started_at))).label("active_days")
                ).join(
                    PlatformCredential,
                    PlatformExecution.credential_id == PlatformCredential.id
                ).where(
                    and_(
                        PlatformCredential.user_id == user_id,
                        PlatformExecution.platform == platform.value,
                        PlatformExecution.started_at >= datetime.utcnow() - timedelta(days=30)
                    )
                )
            )
            
            stats = result.one()
            
            # Count current rate
            rate_result = await self.db.execute(
                select(func.count(PlatformExecution.id)).join(
                    PlatformCredential,
                    PlatformExecution.credential_id == PlatformCredential.id
                ).where(
                    and_(
                        PlatformCredential.user_id == user_id,
                        PlatformExecution.platform == platform.value,
                        PlatformExecution.started_at >= one_minute_ago
                    )
                )
            )
            
            current_rate = rate_result.scalar() or 0
            
            summary["platforms"][platform.value] = {
                "limits": platform_limits,
                "usage": {
                    "total_30d": stats.total or 0,
                    "active_days": stats.active_days or 0,
                    "current_rate_per_minute": current_rate
                },
                "compatibility": self.cloud_config.validate_platform_compatibility(platform)
            }
        
        return summary