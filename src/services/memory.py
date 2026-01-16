"""Basic Memory service for Community Edition.

Provides in-memory storage without database persistence.
Advanced features like distributed memory and persistence are
available in Business Edition.
"""

import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict
import asyncio

from core.config import settings


logger = logging.getLogger(__name__)


class MemoryService:
    """Basic memory service for Community Edition.
    
    Provides:
    - In-memory key-value storage
    - TTL support
    - Basic namespacing
    
    Advanced features available in Business Edition:
    - Persistent storage
    - Distributed memory across nodes
    - Advanced caching strategies
    - Memory analytics
    """
    
    def __init__(self):
        """Initialize the basic memory service."""
        # In-memory storage with namespace support
        self._memory: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._ttl: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self._locks: Dict[str, asyncio.Lock] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Cleanup task will be started on first use
    
    def _ensure_cleanup_task(self):
        """Ensure the cleanup task is running."""
        if self._cleanup_task is None:
            try:
                loop = asyncio.get_running_loop()
                self._cleanup_task = loop.create_task(self._cleanup_expired_entries())
            except RuntimeError:
                # No event loop running, skip cleanup task
                pass
    
    def _get_lock(self, namespace: str) -> asyncio.Lock:
        """Get or create a lock for a namespace."""
        if namespace not in self._locks:
            self._locks[namespace] = asyncio.Lock()
        return self._locks[namespace]
    
    async def set(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        ttl_seconds: Optional[int] = None
    ) -> bool:
        """Set a value in memory."""
        async with self._get_lock(namespace):
            # Store value
            self._memory[namespace][key] = {
                "value": value,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Set TTL if specified
            if ttl_seconds:
                self._ttl[namespace][key] = datetime.utcnow() + timedelta(seconds=ttl_seconds)
            elif key in self._ttl[namespace]:
                # Remove TTL if it was set before
                del self._ttl[namespace][key]
            
            return True
    
    async def get(
        self,
        key: str,
        namespace: str = "default",
        default: Any = None
    ) -> Any:
        """Get a value from memory."""
        async with self._get_lock(namespace):
            # Check if key exists
            if namespace not in self._memory or key not in self._memory[namespace]:
                return default
            
            # Check if expired
            if namespace in self._ttl and key in self._ttl[namespace]:
                if datetime.utcnow() > self._ttl[namespace][key]:
                    # Expired, remove it
                    del self._memory[namespace][key]
                    del self._ttl[namespace][key]
                    return default
            
            # Return value
            return self._memory[namespace][key]["value"]
    
    async def delete(
        self,
        key: str,
        namespace: str = "default"
    ) -> bool:
        """Delete a value from memory."""
        async with self._get_lock(namespace):
            if namespace in self._memory and key in self._memory[namespace]:
                del self._memory[namespace][key]
                
                # Remove TTL if exists
                if namespace in self._ttl and key in self._ttl[namespace]:
                    del self._ttl[namespace][key]
                
                return True
            return False
    
    async def exists(
        self,
        key: str,
        namespace: str = "default"
    ) -> bool:
        """Check if a key exists."""
        value = await self.get(key, namespace, None)
        return value is not None
    
    async def list_keys(
        self,
        namespace: str = "default",
        pattern: Optional[str] = None
    ) -> List[str]:
        """List keys in a namespace."""
        async with self._get_lock(namespace):
            if namespace not in self._memory:
                return []
            
            keys = list(self._memory[namespace].keys())
            
            # Filter expired keys
            valid_keys = []
            for key in keys:
                if namespace in self._ttl and key in self._ttl[namespace]:
                    if datetime.utcnow() <= self._ttl[namespace][key]:
                        valid_keys.append(key)
                else:
                    valid_keys.append(key)
            
            # Apply pattern filter if specified
            if pattern:
                import fnmatch
                valid_keys = [k for k in valid_keys if fnmatch.fnmatch(k, pattern)]
            
            return sorted(valid_keys)
    
    async def clear_namespace(
        self,
        namespace: str = "default"
    ) -> int:
        """Clear all keys in a namespace."""
        async with self._get_lock(namespace):
            count = len(self._memory.get(namespace, {}))
            
            if namespace in self._memory:
                del self._memory[namespace]
            if namespace in self._ttl:
                del self._ttl[namespace]
            
            return count
    
    async def get_stats(
        self,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get memory statistics."""
        if namespace:
            namespaces = [namespace] if namespace in self._memory else []
        else:
            namespaces = list(self._memory.keys())
        
        stats = {
            "namespaces": len(self._memory),
            "total_keys": sum(len(self._memory[ns]) for ns in self._memory),
            "namespace_details": {}
        }
        
        for ns in namespaces:
            if ns in self._memory:
                stats["namespace_details"][ns] = {
                    "keys": len(self._memory[ns]),
                    "ttl_keys": len(self._ttl.get(ns, {}))
                }
        
        stats["features"] = {
            "in_memory": True,
            "persistent": False,
            "distributed": False,
            "ttl_support": True,
            "namespacing": True
        }
        
        stats["upgrade_available"] = True
        stats["upgrade_benefits"] = [
            "Persistent storage",
            "Distributed memory across nodes",
            "Advanced caching strategies",
            "Memory analytics",
            "Larger storage capacity"
        ]
        
        return stats
    
    async def _cleanup_expired_entries(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Check all namespaces
                for namespace in list(self._ttl.keys()):
                    async with self._get_lock(namespace):
                        if namespace not in self._ttl:
                            continue
                        
                        # Find expired keys
                        expired_keys = []
                        now = datetime.utcnow()
                        
                        for key, expiry in self._ttl[namespace].items():
                            if now > expiry:
                                expired_keys.append(key)
                        
                        # Remove expired entries
                        for key in expired_keys:
                            if namespace in self._memory and key in self._memory[namespace]:
                                del self._memory[namespace][key]
                            del self._ttl[namespace][key]
                        
                        if expired_keys:
                            logger.debug(f"Cleaned {len(expired_keys)} expired entries from namespace '{namespace}'")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
    
    # Compatibility methods for easy migration to Business Edition
    
    async def increment(
        self,
        key: str,
        namespace: str = "default",
        amount: int = 1
    ) -> int:
        """Increment a numeric value."""
        current = await self.get(key, namespace, 0)
        if not isinstance(current, (int, float)):
            current = 0
        
        new_value = current + amount
        await self.set(key, new_value, namespace)
        return new_value
    
    async def push(
        self,
        key: str,
        value: Any,
        namespace: str = "default",
        max_size: Optional[int] = None
    ) -> int:
        """Push a value to a list."""
        current = await self.get(key, namespace, [])
        if not isinstance(current, list):
            current = []
        
        current.append(value)
        
        # Trim to max size if specified
        if max_size and len(current) > max_size:
            current = current[-max_size:]
        
        await self.set(key, current, namespace)
        return len(current)