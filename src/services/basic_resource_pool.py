"""Basic resource pool service for Community Edition - Async version."""

import asyncio
import logging
from typing import Optional, List, Dict, Any, Set
from datetime import datetime
from uuid import uuid4
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from models.community_complete import ResourcePoolConfig
from core.exceptions import NotFoundError, ValidationError
from schemas.resource_pool import ResourcePoolConfigCreate, ResourcePoolConfigUpdate

logger = logging.getLogger(__name__)


class BasicResourcePool:
    """Simple resource pool for Community Edition."""
    
    def __init__(self, name: str, max_size: int = 10):
        self.name = name
        self.max_size = max_size
        self.resources = asyncio.Queue(maxsize=max_size)
        self.in_use: Set[Any] = set()
        self.created_at = datetime.utcnow()
        self._lock = asyncio.Lock()
        
    async def acquire(self, timeout: float = 30.0) -> Any:
        """Get a resource from pool."""
        try:
            resource = await asyncio.wait_for(
                self.resources.get(), 
                timeout=timeout
            )
            async with self._lock:
                self.in_use.add(resource)
            return resource
        except asyncio.TimeoutError:
            raise TimeoutError(f"Timeout acquiring resource from pool {self.name}")
    
    async def release(self, resource: Any) -> None:
        """Return resource to pool."""
        async with self._lock:
            if resource in self.in_use:
                self.in_use.remove(resource)
                await self.resources.put(resource)
            else:
                logger.warning(f"Attempted to release unknown resource: {resource}")
    
    async def add_resource(self, resource: Any) -> None:
        """Add a new resource to the pool."""
        if self.resources.qsize() + len(self.in_use) < self.max_size:
            await self.resources.put(resource)
        else:
            raise ValueError(f"Pool {self.name} is at maximum capacity")
    
    @property
    def available_count(self) -> int:
        """Number of available resources."""
        return self.resources.qsize()
    
    @property
    def in_use_count(self) -> int:
        """Number of resources currently in use."""
        return len(self.in_use)
    
    @property
    def total_count(self) -> int:
        """Total number of resources."""
        return self.available_count + self.in_use_count


class BasicResourcePoolManager:
    """Manages pools with basic persistence."""
    
    def __init__(self):
        self.pools: Dict[str, BasicResourcePool] = {}
        self._initialized = False
        
    async def initialize(self, db: AsyncSession) -> None:
        """Load pool configs from database."""
        if self._initialized:
            return
            
        # Use async query
        stmt = select(ResourcePoolConfig).where(ResourcePoolConfig.enabled == True)
        result = await db.execute(stmt)
        configs = result.scalars().all()
        
        for config in configs:
            pool = BasicResourcePool(config.name, config.max_size)
            self.pools[config.name] = pool
            logger.info(f"Initialized pool: {config.name} with max_size: {config.max_size}")
        
        self._initialized = True
    
    def get_pool(self, name: str) -> Optional[BasicResourcePool]:
        """Get a pool by name."""
        return self.pools.get(name)
    
    async def create_pool(self, name: str, max_size: int = 10) -> BasicResourcePool:
        """Create a new pool."""
        if name in self.pools:
            raise ValueError(f"Pool {name} already exists")
        
        pool = BasicResourcePool(name, max_size)
        self.pools[name] = pool
        return pool
    
    def list_pools(self) -> List[Dict[str, Any]]:
        """List all pools with their status."""
        return [
            {
                "name": name,
                "max_size": pool.max_size,
                "available": pool.available_count,
                "in_use": pool.in_use_count,
                "total": pool.total_count,
                "created_at": pool.created_at.isoformat()
            }
            for name, pool in self.pools.items()
        ]


class BasicResourcePoolService:
    """Service for managing resource pool configurations in the database - Async version."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.pool_manager = BasicResourcePoolManager()
        
    async def initialize(self) -> None:
        """Initialize the pool manager with database configs."""
        await self.pool_manager.initialize(self.db)
    
    async def list_configs(
        self,
        resource_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ResourcePoolConfig]:
        """List resource pool configurations."""
        stmt = select(ResourcePoolConfig)
        
        if resource_type:
            stmt = stmt.where(ResourcePoolConfig.resource_type == resource_type)
        if is_active is not None:
            stmt = stmt.where(ResourcePoolConfig.enabled == is_active)
        
        stmt = stmt.offset(skip).limit(limit)
        
        result = await self.db.execute(stmt)
        return result.scalars().all()
    
    async def get_config(self, config_id: str) -> Optional[ResourcePoolConfig]:
        """Get a specific resource pool configuration."""
        stmt = select(ResourcePoolConfig).where(ResourcePoolConfig.id == config_id)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()
    
    async def create_config(self, config_data: ResourcePoolConfigCreate) -> ResourcePoolConfig:
        """Create a new resource pool configuration."""
        # Check if name already exists
        stmt = select(ResourcePoolConfig).where(ResourcePoolConfig.name == config_data.name)
        result = await self.db.execute(stmt)
        if result.scalar_one_or_none():
            raise ValidationError(f"Pool configuration with name '{config_data.name}' already exists")
        
        config = ResourcePoolConfig(
            id=str(uuid4()),
            **config_data.dict()
        )
        self.db.add(config)
        await self.db.commit()
        await self.db.refresh(config)
        
        # Create the actual pool if active
        if config.enabled:
            await self.pool_manager.create_pool(config.name, config.max_size)
        
        return config
    
    async def update_config(
        self, 
        config_id: str, 
        config_data: ResourcePoolConfigUpdate
    ) -> Optional[ResourcePoolConfig]:
        """Update a resource pool configuration."""
        config = await self.get_config(config_id)
        if not config:
            return None
        
        update_data = config_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(config, field, value)
        
        config.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(config)
        
        # Update the actual pool if needed
        pool = self.pool_manager.get_pool(config.name)
        if pool and "max_size" in update_data:
            pool.max_size = update_data["max_size"]
        
        return config
    
    async def delete_config(self, config_id: str) -> bool:
        """Delete a resource pool configuration."""
        config = await self.get_config(config_id)
        if not config:
            return False
        
        # Remove from pool manager
        if config.name in self.pool_manager.pools:
            del self.pool_manager.pools[config.name]
        
        await self.db.delete(config)
        await self.db.commit()
        return True
    
    def get_pool_status(self, pool_name: str) -> Optional[Dict[str, Any]]:
        """Get runtime status of a pool."""
        pool = self.pool_manager.get_pool(pool_name)
        if not pool:
            return None
        
        return {
            "name": pool.name,
            "max_size": pool.max_size,
            "available": pool.available_count,
            "in_use": pool.in_use_count,
            "total": pool.total_count,
            "created_at": pool.created_at.isoformat()
        }