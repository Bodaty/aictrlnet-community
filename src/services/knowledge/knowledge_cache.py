"""
Knowledge Service Cache Module

Provides a singleton cache for the KnowledgeRetrievalService to prevent
repeated initialization during workflow generation.

This addresses the performance issue where the Knowledge service was being
initialized 74 times during a single workflow generation, causing timeouts.
"""

import logging
import asyncio
from typing import Optional
from datetime import datetime, timedelta
from threading import Lock

logger = logging.getLogger(__name__)

class KnowledgeServiceCache:
    """
    Thread-safe singleton cache for Knowledge services.

    Caches initialized KnowledgeRetrievalService instances to avoid
    repeated rebuilding of the knowledge index (187 items).
    """

    _instance = None
    _lock = Lock()
    _services = {}  # Cache keyed by (service_class, db_id)
    _last_cleanup = datetime.now()
    _cache_ttl = timedelta(hours=1)  # Cache instances for 1 hour

    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize the cache (only runs once due to singleton)."""
        pass

    async def get_or_create(self, service_class, db, force_new=False):
        """
        Get or create a cached KnowledgeRetrievalService instance.

        Args:
            service_class: The class to instantiate (KnowledgeRetrievalService or subclass)
            db: Database session (used as part of cache key)
            force_new: Force creation of a new instance

        Returns:
            Initialized service instance
        """
        # Create cache key based on service class and a simple db identifier
        # We use the class name as we can't reliably hash the db session
        cache_key = (service_class.__name__, id(db) % 1000000)  # Mod to keep key reasonable

        # Check if we need to clean up old entries
        self._cleanup_if_needed()

        # Force new instance if requested
        if force_new and cache_key in self._services:
            logger.info(f"[KnowledgeCache] Forcing new instance for {service_class.__name__}")
            del self._services[cache_key]

        # Check cache
        if cache_key in self._services:
            service, created_at = self._services[cache_key]
            age = datetime.now() - created_at

            # Return cached instance if still valid
            if age < self._cache_ttl:
                logger.debug(f"[KnowledgeCache] Returning cached {service_class.__name__} (age: {age.total_seconds():.1f}s)")
                return service
            else:
                logger.info(f"[KnowledgeCache] Cached {service_class.__name__} expired (age: {age.total_seconds():.1f}s)")
                del self._services[cache_key]

        # Create new instance
        logger.info(f"[KnowledgeCache] Creating new {service_class.__name__} instance")
        service = service_class(db)

        # Initialize if it has an initialize method
        if hasattr(service, 'initialize'):
            await service.initialize()

        # Cache the instance
        self._services[cache_key] = (service, datetime.now())
        logger.info(f"[KnowledgeCache] Cached new {service_class.__name__} instance (total cached: {len(self._services)})")

        return service

    def _cleanup_if_needed(self):
        """Clean up expired cache entries periodically."""
        now = datetime.now()

        # Only cleanup every 10 minutes
        if now - self._last_cleanup < timedelta(minutes=10):
            return

        self._last_cleanup = now
        expired_keys = []

        # Find expired entries
        for key, (service, created_at) in self._services.items():
            if now - created_at > self._cache_ttl:
                expired_keys.append(key)

        # Remove expired entries
        for key in expired_keys:
            del self._services[key]

        if expired_keys:
            logger.info(f"[KnowledgeCache] Cleaned up {len(expired_keys)} expired entries")

    def clear(self):
        """Clear all cached instances."""
        count = len(self._services)
        self._services.clear()
        logger.info(f"[KnowledgeCache] Cleared {count} cached instances")

    def get_stats(self):
        """Get cache statistics."""
        return {
            "cached_instances": len(self._services),
            "cache_keys": list(self._services.keys()),
            "last_cleanup": self._last_cleanup.isoformat()
        }

# Global singleton instance
knowledge_cache = KnowledgeServiceCache()