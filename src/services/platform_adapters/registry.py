"""Platform adapter registry for managing platform integrations"""
import logging
from typing import Dict, Type, Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import datetime

from models.platform_integration import PlatformAdapter, PlatformType
from schemas.platform_integration import (
    PlatformAdapterCreate,
    PlatformAdapterResponse,
    PlatformCapabilities
)
from .base import BasePlatformAdapter

logger = logging.getLogger(__name__)


class PlatformAdapterRegistry:
    """Registry for platform adapters"""
    
    _instance = None
    _adapters: Dict[PlatformType, Type[BasePlatformAdapter]] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def register(
        cls,
        platform_type: PlatformType,
        adapter_class: Type[BasePlatformAdapter]
    ):
        """Register a platform adapter"""
        if not issubclass(adapter_class, BasePlatformAdapter):
            raise ValueError(f"{adapter_class} must inherit from BasePlatformAdapter")
        
        cls._adapters[platform_type] = adapter_class
        logger.info(f"Registered adapter for {platform_type.value}: {adapter_class.__name__}")
    
    @classmethod
    def get_adapter(
        cls,
        platform_type: PlatformType
    ) -> Optional[BasePlatformAdapter]:
        """Get an adapter instance for a platform"""
        adapter_class = cls._adapters.get(platform_type)
        if adapter_class:
            return adapter_class(platform_type)
        return None
    
    @classmethod
    def list_platforms(cls) -> List[PlatformType]:
        """List all registered platforms"""
        return list(cls._adapters.keys())
    
    @classmethod
    def is_registered(cls, platform_type: PlatformType) -> bool:
        """Check if a platform adapter is registered"""
        return platform_type in cls._adapters
    
    @classmethod
    def unregister(cls, platform_type: PlatformType):
        """Unregister a platform adapter"""
        if platform_type in cls._adapters:
            del cls._adapters[platform_type]
            logger.info(f"Unregistered adapter for {platform_type.value}")
    
    @classmethod
    def clear(cls):
        """Clear all registered adapters"""
        cls._adapters.clear()
        logger.info("Cleared all registered adapters")


class PlatformAdapterService:
    """Service for managing platform adapters in database"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.registry = PlatformAdapterRegistry()
    
    async def sync_adapters_to_db(self):
        """Sync registered adapters to database"""
        for platform_type in self.registry.list_platforms():
            adapter = self.registry.get_adapter(platform_type)
            if not adapter:
                continue
            
            # Check if adapter exists in DB
            result = await self.db.execute(
                select(PlatformAdapter).where(
                    PlatformAdapter.platform == platform_type.value
                )
            )
            db_adapter = result.scalar_one_or_none()
            
            # Get adapter info
            capabilities = adapter.get_capabilities()
            supported_auth = [method.value for method in adapter.get_supported_auth_methods()]
            
            if db_adapter:
                # Update existing
                db_adapter.capabilities = capabilities.model_dump()
                db_adapter.supported_auth_methods = supported_auth
                db_adapter.is_active = True
                db_adapter.updated_at = datetime.utcnow()
            else:
                # Create new
                db_adapter = PlatformAdapter(
                    platform=platform_type.value,
                    adapter_class=f"{adapter.__class__.__module__}.{adapter.__class__.__name__}",
                    version="1.0.0",
                    capabilities=capabilities.model_dump(),
                    supported_auth_methods=supported_auth,
                    is_active=True,
                    is_beta=False,
                    config_schema={},  # TODO: Add JSON schema for config
                    documentation_url=None,  # TODO: Add docs URL
                    icon_url=None  # TODO: Add icon URL
                )
                self.db.add(db_adapter)
        
        await self.db.commit()
        logger.info("Synced adapters to database")
    
    async def get_adapter_info(
        self,
        platform_type: PlatformType
    ) -> Optional[PlatformAdapterResponse]:
        """Get adapter information from database"""
        result = await self.db.execute(
            select(PlatformAdapter).where(
                PlatformAdapter.platform == platform_type.value
            )
        )
        db_adapter = result.scalar_one_or_none()
        
        if db_adapter:
            return PlatformAdapterResponse.model_validate(db_adapter)
        return None
    
    async def list_adapters(
        self,
        active_only: bool = True
    ) -> List[PlatformAdapterResponse]:
        """List all adapters from database"""
        query = select(PlatformAdapter)
        
        if active_only:
            query = query.where(PlatformAdapter.is_active == True)
        
        result = await self.db.execute(query)
        adapters = result.scalars().all()
        return [PlatformAdapterResponse.model_validate(a) for a in adapters]
    
    async def update_adapter(
        self,
        platform_type: PlatformType,
        update_data: Dict[str, Any]
    ) -> Optional[PlatformAdapterResponse]:
        """Update adapter configuration in database"""
        result = await self.db.execute(
            select(PlatformAdapter).where(
                PlatformAdapter.platform == platform_type.value
            )
        )
        db_adapter = result.scalar_one_or_none()
        
        if not db_adapter:
            return None
        
        # Update allowed fields
        allowed_fields = [
            "is_active", "is_beta", "config_schema",
            "documentation_url", "icon_url"
        ]
        
        for field, value in update_data.items():
            if field in allowed_fields:
                setattr(db_adapter, field, value)
        
        db_adapter.updated_at = datetime.utcnow()
        await self.db.commit()
        await self.db.refresh(db_adapter)
        
        return PlatformAdapterResponse.model_validate(db_adapter)
    
    def get_adapter_instance(
        self,
        platform_type: PlatformType
    ) -> Optional[BasePlatformAdapter]:
        """Get an adapter instance from registry"""
        return self.registry.get_adapter(platform_type)


# Decorator for auto-registration
def register_adapter(platform_type: PlatformType):
    """Decorator to register platform adapters"""
    def decorator(cls: Type[BasePlatformAdapter]):
        PlatformAdapterRegistry.register(platform_type, cls)
        return cls
    return decorator