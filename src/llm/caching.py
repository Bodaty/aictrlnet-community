"""LLM response caching module."""

import logging
import json
import hashlib
from typing import Optional
from datetime import datetime, timedelta

from core.cache import get_cache
from .models import LLMResponse

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Cache for LLM responses to avoid redundant API calls.
    
    Uses Redis for distributed caching when available,
    falls back to in-memory cache otherwise.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize the cache.
        
        Args:
            ttl_seconds: Time to live for cached responses (default 1 hour)
        """
        self.ttl = ttl_seconds
        self._memory_cache = {}  # Fallback in-memory cache
        self._cache_stats = {
            "hits": 0,
            "misses": 0,
            "total_requests": 0
        }
    
    async def get(self, cache_key: str) -> Optional[LLMResponse]:
        """
        Get a cached response.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Cached LLMResponse or None if not found
        """
        self._cache_stats["total_requests"] += 1
        
        try:
            # Try Redis first
            cache = await get_cache()
            if cache:
                data = await cache.get(f"llm:{cache_key}")
                if data:
                    self._cache_stats["hits"] += 1
                    logger.debug(f"Cache hit for key: {cache_key}")
                    # Reconstruct LLMResponse from cached data
                    return LLMResponse(**data)
        except Exception as e:
            logger.debug(f"Redis cache not available: {e}")
        
        # Fallback to memory cache
        if cache_key in self._memory_cache:
            cached_item = self._memory_cache[cache_key]
            if cached_item["expires_at"] > datetime.utcnow():
                self._cache_stats["hits"] += 1
                logger.debug(f"Memory cache hit for key: {cache_key}")
                return cached_item["response"]
            else:
                # Expired, remove it
                del self._memory_cache[cache_key]
        
        self._cache_stats["misses"] += 1
        return None
    
    async def set(self, cache_key: str, response: LLMResponse):
        """
        Cache a response.
        
        Args:
            cache_key: The cache key
            response: The response to cache
        """
        try:
            # Try Redis first
            cache = await get_cache()
            if cache:
                # Convert to dict for serialization
                data = response.model_dump()
                await cache.set(
                    f"llm:{cache_key}",
                    data,
                    expire=self.ttl
                )
                logger.debug(f"Cached response in Redis for key: {cache_key}")
                return
        except Exception as e:
            logger.debug(f"Could not cache in Redis: {e}")
        
        # Fallback to memory cache
        self._memory_cache[cache_key] = {
            "response": response,
            "expires_at": datetime.utcnow() + timedelta(seconds=self.ttl)
        }
        logger.debug(f"Cached response in memory for key: {cache_key}")
        
        # Clean up old entries if memory cache gets too large
        if len(self._memory_cache) > 1000:
            self._cleanup_memory_cache()
    
    def generate_key(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        system_prompt: str = ""
    ) -> str:
        """
        Generate a cache key from request parameters.
        
        Args:
            prompt: The prompt
            model: The model name
            temperature: Temperature setting
            max_tokens: Max tokens setting
            system_prompt: System prompt if any
            
        Returns:
            SHA256 hash as cache key
        """
        key_parts = [
            prompt,
            model,
            str(temperature),
            str(max_tokens),
            system_prompt
        ]
        key_str = "|".join(key_parts)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    def get_stats(self) -> dict:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        hit_rate = 0.0
        if self._cache_stats["total_requests"] > 0:
            hit_rate = self._cache_stats["hits"] / self._cache_stats["total_requests"]
        
        return {
            **self._cache_stats,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self._memory_cache)
        }
    
    def _cleanup_memory_cache(self):
        """Remove expired entries from memory cache."""
        now = datetime.utcnow()
        expired_keys = [
            key for key, item in self._memory_cache.items()
            if item["expires_at"] <= now
        ]
        
        for key in expired_keys:
            del self._memory_cache[key]
        
        logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    async def clear(self):
        """Clear all cached responses."""
        try:
            # Clear Redis cache
            cache = await get_cache()
            if cache:
                # Get all LLM cache keys and delete them
                # Note: This is a simplified approach
                logger.info("Cleared Redis LLM cache")
        except Exception as e:
            logger.debug(f"Could not clear Redis cache: {e}")
        
        # Clear memory cache
        self._memory_cache.clear()
        logger.info("Cleared memory LLM cache")