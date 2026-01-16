"""Platform execution caching service for cost optimization"""
import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, Any, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from models.platform_integration import PlatformExecution
from schemas.platform_integration import PlatformType
from core.cache import RedisCache
from core.events import event_bus


class PlatformExecutionCache:
    """Cache platform execution results to avoid duplicate runs"""
    
    def __init__(self, db: AsyncSession, cache: RedisCache = None):
        self.db = db
        self.cache = cache or RedisCache()
        
        # Cache configuration per platform
        self.cache_config = {
            PlatformType.N8N: {
                "enabled": True,
                "ttl": 3600,  # 1 hour
                "max_size_mb": 10,
                "cache_errors": False
            },
            PlatformType.ZAPIER: {
                "enabled": True,
                "ttl": 1800,  # 30 minutes
                "max_size_mb": 5,
                "cache_errors": False
            },
            PlatformType.MAKE: {
                "enabled": True,
                "ttl": 3600,
                "max_size_mb": 10,
                "cache_errors": False
            },
            PlatformType.POWER_AUTOMATE: {
                "enabled": True,
                "ttl": 7200,  # 2 hours
                "max_size_mb": 20,
                "cache_errors": False
            },
            PlatformType.IFTTT: {
                "enabled": True,
                "ttl": 86400,  # 24 hours (simple triggers)
                "max_size_mb": 1,
                "cache_errors": False
            }
        }
    
    def generate_cache_key(
        self,
        platform: PlatformType,
        workflow_id: str,
        input_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> str:
        """Generate cache key for execution"""
        # Create deterministic hash
        key_parts = [
            platform.value,
            workflow_id,
            json.dumps(input_data, sort_keys=True)
        ]
        
        if user_id:
            key_parts.append(user_id)
        
        key_string = "|".join(key_parts)
        hash_digest = hashlib.sha256(key_string.encode()).hexdigest()
        
        return f"exec_cache:{platform.value}:{hash_digest[:16]}"
    
    async def get_cached_result(
        self,
        platform: PlatformType,
        workflow_id: str,
        input_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached execution result if available"""
        
        config = self.cache_config.get(platform, {})
        if not config.get("enabled", False):
            return None
        
        cache_key = self.generate_cache_key(
            platform, workflow_id, input_data, user_id
        )
        
        # Try to get from cache
        cached_data = await self.cache.get(cache_key)
        
        if cached_data:
            # Update hit statistics
            await self._record_cache_hit(platform, workflow_id)
            
            # Check if still valid
            if self._is_cache_valid(cached_data):
                return {
                    "cached": True,
                    "cache_key": cache_key,
                    "cached_at": cached_data.get("cached_at"),
                    "output_data": cached_data.get("output_data"),
                    "execution_time_ms": cached_data.get("execution_time_ms"),
                    "cost_saved": cached_data.get("original_cost", 0)
                }
        
        # Also check database for recent executions
        db_result = await self._get_from_execution_history(
            platform, workflow_id, input_data, user_id
        )
        
        if db_result:
            # Store in cache for faster access
            await self.store_result(
                platform=platform,
                workflow_id=workflow_id,
                input_data=input_data,
                output_data=db_result["output_data"],
                execution_time_ms=db_result["execution_time_ms"],
                original_cost=db_result.get("cost", 0),
                user_id=user_id
            )
            
            return {
                "cached": True,
                "cache_key": cache_key,
                "cached_at": db_result["executed_at"],
                "output_data": db_result["output_data"],
                "execution_time_ms": db_result["execution_time_ms"],
                "cost_saved": db_result.get("cost", 0)
            }
        
        # Record cache miss
        await self._record_cache_miss(platform, workflow_id)
        
        return None
    
    async def store_result(
        self,
        platform: PlatformType,
        workflow_id: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        execution_time_ms: int,
        original_cost: float = 0,
        error_data: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None
    ) -> bool:
        """Store execution result in cache"""
        
        config = self.cache_config.get(platform, {})
        if not config.get("enabled", False):
            return False
        
        # Don't cache errors unless configured
        if error_data and not config.get("cache_errors", False):
            return False
        
        # Check size limits
        result_size_mb = len(json.dumps(output_data)) / (1024 * 1024)
        if result_size_mb > config.get("max_size_mb", 10):
            return False
        
        cache_key = self.generate_cache_key(
            platform, workflow_id, input_data, user_id
        )
        
        cache_data = {
            "platform": platform.value,
            "workflow_id": workflow_id,
            "input_data": input_data,
            "output_data": output_data,
            "error_data": error_data,
            "execution_time_ms": execution_time_ms,
            "original_cost": original_cost,
            "cached_at": datetime.utcnow().isoformat(),
            "expires_at": (
                datetime.utcnow() + timedelta(seconds=config.get("ttl", 3600))
            ).isoformat()
        }
        
        # Store in cache
        success = await self.cache.set(
            cache_key,
            cache_data,
            ttl=config.get("ttl", 3600)
        )
        
        if success:
            # Publish cache event
            await event_bus.publish(
                "platform.execution_cached",
                {
                    "platform": platform.value,
                    "workflow_id": workflow_id,
                    "cache_key": cache_key,
                    "size_mb": result_size_mb,
                    "ttl": config.get("ttl", 3600)
                }
            )
        
        return success
    
    async def invalidate_cache(
        self,
        platform: PlatformType,
        workflow_id: str,
        user_id: Optional[str] = None
    ) -> int:
        """Invalidate cache entries for a workflow"""
        
        # Pattern-based invalidation would be ideal, but for now
        # we'll track cache keys per workflow
        invalidated = 0
        
        # Get all cache keys for this workflow from tracking
        cache_keys = await self._get_workflow_cache_keys(
            platform, workflow_id, user_id
        )
        
        for key in cache_keys:
            if await self.cache.delete(key):
                invalidated += 1
        
        if invalidated > 0:
            await event_bus.publish(
                "platform.cache_invalidated",
                {
                    "platform": platform.value,
                    "workflow_id": workflow_id,
                    "keys_invalidated": invalidated
                }
            )
        
        return invalidated
    
    async def get_cache_statistics(
        self,
        platform: Optional[PlatformType] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get cache hit/miss statistics"""
        
        # In production, these would be stored in database
        # For now, use cache to track stats
        stats_key = f"cache_stats:{platform.value if platform else 'all'}"
        if user_id:
            stats_key += f":{user_id}"
        
        stats = await self.cache.get(stats_key) or {
            "hits": 0,
            "misses": 0,
            "total_cost_saved": 0,
            "total_time_saved_ms": 0,
            "last_hit": None,
            "last_miss": None
        }
        
        # Calculate hit rate
        total = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total if total > 0 else 0
        
        # Get size info
        if platform:
            config = self.cache_config.get(platform, {})
            stats["config"] = {
                "enabled": config.get("enabled", False),
                "ttl": config.get("ttl", 0),
                "max_size_mb": config.get("max_size_mb", 0)
            }
        
        return stats
    
    async def configure_cache(
        self,
        platform: PlatformType,
        config_updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update cache configuration for a platform"""
        
        current_config = self.cache_config.get(platform, {})
        
        # Update configuration
        for key, value in config_updates.items():
            if key in ["enabled", "ttl", "max_size_mb", "cache_errors"]:
                current_config[key] = value
        
        self.cache_config[platform] = current_config
        
        # Clear cache if disabled
        if not current_config.get("enabled", False):
            await self.invalidate_cache(platform, "*")
        
        return current_config
    
    def _is_cache_valid(self, cached_data: Dict[str, Any]) -> bool:
        """Check if cached data is still valid"""
        
        expires_at = cached_data.get("expires_at")
        if not expires_at:
            return False
        
        try:
            expiry_time = datetime.fromisoformat(expires_at)
            return datetime.utcnow() < expiry_time
        except:
            return False
    
    async def _get_from_execution_history(
        self,
        platform: PlatformType,
        workflow_id: str,
        input_data: Dict[str, Any],
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Get result from recent execution history"""
        
        # Look for recent successful execution with same inputs
        query = select(PlatformExecution).where(
            and_(
                PlatformExecution.platform == platform.value,
                PlatformExecution.external_workflow_id == workflow_id,
                PlatformExecution.status == "completed",
                PlatformExecution.started_at >= datetime.utcnow() - timedelta(hours=24)
            )
        )
        
        result = await self.db.execute(query)
        executions = result.scalars().all()
        
        # Find matching execution
        for exec in executions:
            if self._inputs_match(exec.input_data, input_data):
                return {
                    "output_data": exec.output_data,
                    "execution_time_ms": exec.duration_ms,
                    "executed_at": exec.started_at.isoformat(),
                    "cost": exec.estimated_cost / 100 if exec.estimated_cost else 0
                }
        
        return None
    
    def _inputs_match(self, input1: Dict[str, Any], input2: Dict[str, Any]) -> bool:
        """Check if two input data sets match"""
        # Simple comparison for now
        # In production, might need more sophisticated matching
        return json.dumps(input1, sort_keys=True) == json.dumps(input2, sort_keys=True)
    
    async def _record_cache_hit(self, platform: PlatformType, workflow_id: str):
        """Record cache hit statistics"""
        stats_key = f"cache_stats:{platform.value}"
        stats = await self.cache.get(stats_key) or {
            "hits": 0, "misses": 0, "total_cost_saved": 0,
            "total_time_saved_ms": 0
        }
        
        stats["hits"] += 1
        stats["last_hit"] = datetime.utcnow().isoformat()
        
        await self.cache.set(stats_key, stats, ttl=86400 * 7)  # 7 days
    
    async def _record_cache_miss(self, platform: PlatformType, workflow_id: str):
        """Record cache miss statistics"""
        stats_key = f"cache_stats:{platform.value}"
        stats = await self.cache.get(stats_key) or {
            "hits": 0, "misses": 0, "total_cost_saved": 0,
            "total_time_saved_ms": 0
        }
        
        stats["misses"] += 1
        stats["last_miss"] = datetime.utcnow().isoformat()
        
        await self.cache.set(stats_key, stats, ttl=86400 * 7)  # 7 days
    
    async def _get_workflow_cache_keys(
        self,
        platform: PlatformType,
        workflow_id: str,
        user_id: Optional[str] = None
    ) -> List[str]:
        """Get all cache keys for a workflow"""
        # In production, would maintain an index of cache keys
        # For now, return empty list
        return []