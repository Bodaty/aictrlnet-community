"""
Redis caching service for AICtrlNet FastAPI.

This module provides caching functionality using Redis for improved performance.
"""

import json
import logging
from typing import Any, Optional, Union
from functools import wraps
import asyncio

from redis import asyncio as aioredis
from redis.exceptions import RedisError
from aiocache import Cache
from aiocache.serializers import JsonSerializer

from .config import get_settings

logger = logging.getLogger(__name__)


class RedisCache:
    """Redis cache service for AICtrlNet."""
    
    def __init__(self):
        self.settings = get_settings()
        self._redis_client: Optional[aioredis.Redis] = None
        self._aiocache: Optional[Cache] = None
        
    async def connect(self):
        """Connect to Redis server."""
        try:
            # Connect to Redis
            self._redis_client = await aioredis.from_url(
                self.settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=10,
            )
            
            # Test connection
            await self._redis_client.ping()
            
            # Setup aiocache with Redis backend
            self._aiocache = Cache(
                Cache.REDIS,
                endpoint=self.settings.REDIS_HOST,
                port=self.settings.REDIS_PORT,
                password=self.settings.REDIS_PASSWORD if self.settings.REDIS_PASSWORD else None,
                serializer=JsonSerializer(),
                namespace=f"aictrlnet_{self.settings.EDITION}",
            )
            
            logger.info(f"Connected to Redis at {self.settings.REDIS_URL}")
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            # Don't fail the app if Redis is not available
            self._redis_client = None
            self._aiocache = None
    
    async def disconnect(self):
        """Disconnect from Redis server."""
        if self._redis_client:
            await self._redis_client.close()
            logger.info("Disconnected from Redis")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if not self._redis_client:
            return None
            
        try:
            value = await self._redis_client.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, expire: int = 300) -> bool:
        """Set value in cache with expiration."""
        if not self._redis_client:
            return False
            
        try:
            serialized = json.dumps(value)
            await self._redis_client.set(key, serialized, ex=expire)
            return True
        except Exception as e:
            logger.error(f"Redis set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if not self._redis_client:
            return False
            
        try:
            await self._redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        if not self._redis_client:
            return False
            
        try:
            return await self._redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Redis exists error for key {key}: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern."""
        if not self._redis_client:
            return 0
            
        try:
            keys = []
            async for key in self._redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                return await self._redis_client.delete(*keys)
            return 0
        except Exception as e:
            logger.error(f"Redis clear pattern error for {pattern}: {e}")
            return 0
    
    async def increment(self, key: str, delta: int = 1) -> int:
        """Increment a counter in cache."""
        if not self._redis_client:
            return 0
            
        try:
            return await self._redis_client.incrby(key, delta)
        except Exception as e:
            logger.error(f"Redis increment error for key {key}: {e}")
            return 0
    
    def cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate cache key from prefix and arguments."""
        parts = [prefix]
        parts.extend(str(arg) for arg in args)
        parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        return ":".join(parts)


# Global cache instance
_cache: Optional[RedisCache] = None


async def get_cache() -> RedisCache:
    """Get global cache instance."""
    global _cache
    if _cache is None:
        _cache = RedisCache()
        await _cache.connect()
    return _cache


def cache_result(
    prefix: str,
    expire: int = 300,
    key_builder: Optional[callable] = None,
):
    """
    Decorator to cache function results.
    
    Args:
        prefix: Cache key prefix
        expire: Expiration time in seconds
        key_builder: Optional function to build cache key
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache = await get_cache()
            
            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                # Skip 'self' for methods
                cache_args = args[1:] if args and hasattr(args[0], '__class__') else args
                cache_key = cache.cache_key(prefix, *cache_args, **kwargs)
            
            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                logger.debug(f"Cache hit for {cache_key}")
                return cached_value
            
            # Call original function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, expire)
            logger.debug(f"Cached result for {cache_key}")
            
            return result
        
        return wrapper
    return decorator


def invalidate_cache(patterns: Union[str, list[str]]):
    """
    Decorator to invalidate cache patterns after function execution.
    
    Args:
        patterns: Cache key pattern(s) to invalidate
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Call original function
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            cache = await get_cache()
            if isinstance(patterns, str):
                await cache.clear_pattern(patterns)
            else:
                for pattern in patterns:
                    await cache.clear_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator


# Cache key builders for common patterns
def task_cache_key(task_id: str = None, **kwargs) -> str:
    """Build cache key for task operations."""
    if task_id:
        return f"task:{task_id}"
    return f"tasks:{':'.join(f'{k}:{v}' for k, v in sorted(kwargs.items()))}"


def workflow_cache_key(workflow_id: str = None, **kwargs) -> str:
    """Build cache key for workflow operations."""
    if workflow_id:
        return f"workflow:{workflow_id}"
    return f"workflows:{':'.join(f'{k}:{v}' for k, v in sorted(kwargs.items()))}"


def user_cache_key(user_id: str = None, username: str = None, **kwargs) -> str:
    """Build cache key for user operations."""
    if user_id:
        return f"user:id:{user_id}"
    elif username:
        return f"user:username:{username}"
    return f"users:{':'.join(f'{k}:{v}' for k, v in sorted(kwargs.items()))}"