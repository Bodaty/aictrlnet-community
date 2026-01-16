"""Service layer for adapter configuration management."""

import logging
import asyncio
from typing import List, Optional, Tuple, Dict, Any
from datetime import datetime
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from pydantic import UUID4
import time

from models.adapter_config import UserAdapterConfig
from schemas.adapter_config import (
    AdapterConfigCreate,
    AdapterConfigUpdate,
    AdapterConfigTestResponse,
    AdapterConfigBulkTestResponse,
    TestStatus
)
from adapters.registry import adapter_registry
from adapters.models import AdapterConfig, AdapterCategory
from core.crypto import encrypt_data, decrypt_data


logger = logging.getLogger(__name__)


class AdapterConfigService:
    """Service for managing adapter configurations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def list_user_configs(
        self,
        user_id: str,
        adapter_type: Optional[str] = None,
        enabled_only: bool = False,
        skip: int = 0,
        limit: int = 100
    ) -> Tuple[List[UserAdapterConfig], int]:
        """List adapter configurations for a user."""
        # Build base query
        query = select(UserAdapterConfig).where(
            UserAdapterConfig.user_id == user_id
        )
        
        # Apply filters
        if adapter_type:
            query = query.where(UserAdapterConfig.adapter_type == adapter_type)
        
        if enabled_only:
            query = query.where(UserAdapterConfig.enabled == True)
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        total = await self.db.scalar(count_query)
        
        # Apply pagination and ordering
        query = query.order_by(UserAdapterConfig.created_at.desc())
        query = query.offset(skip).limit(limit)
        
        # Execute query
        result = await self.db.execute(query)
        configs = result.scalars().all()
        
        return configs, total
    
    async def get_config(
        self,
        config_id: UUID4,
        user_id: str
    ) -> Optional[UserAdapterConfig]:
        """Get a specific adapter configuration."""
        query = select(UserAdapterConfig).where(
            and_(
                UserAdapterConfig.id == config_id,
                UserAdapterConfig.user_id == user_id
            )
        )
        
        result = await self.db.execute(query)
        return result.scalar_one_or_none()
    
    async def create_config(
        self,
        user_id: str,
        config_data: AdapterConfigCreate
    ) -> UserAdapterConfig:
        """Create a new adapter configuration."""
        # Encrypt credentials if provided
        encrypted_credentials = None
        if config_data.credentials:
            encrypted_credentials = encrypt_data(config_data.credentials)
        
        # Create configuration
        config = UserAdapterConfig(
            user_id=user_id,
            adapter_type=config_data.adapter_type,
            name=config_data.name or f"{config_data.adapter_type}_config",
            display_name=config_data.display_name,
            credentials=encrypted_credentials,
            settings=config_data.settings,
            enabled=config_data.enabled,
            test_status="untested"
        )
        
        self.db.add(config)
        await self.db.commit()
        await self.db.refresh(config)
        
        return config
    
    async def update_config(
        self,
        config_id: UUID4,
        user_id: str,
        update_data: AdapterConfigUpdate
    ) -> Optional[UserAdapterConfig]:
        """Update an adapter configuration."""
        # Get existing configuration
        config = await self.get_config(config_id, user_id)
        if not config:
            return None
        
        # Update fields
        if update_data.name is not None:
            config.name = update_data.name
        
        if update_data.display_name is not None:
            config.display_name = update_data.display_name
        
        if update_data.credentials is not None:
            config.credentials = encrypt_data(update_data.credentials)
            config.test_status = "untested"  # Reset test status
        
        if update_data.settings is not None:
            config.settings = update_data.settings
        
        if update_data.enabled is not None:
            config.enabled = update_data.enabled
        
        config.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(config)
        
        return config
    
    async def delete_config(
        self,
        config_id: UUID4,
        user_id: str
    ) -> bool:
        """Delete an adapter configuration."""
        config = await self.get_config(config_id, user_id)
        if not config:
            return False
        
        # Deactivate if active
        await self.deactivate_adapter(config)
        
        await self.db.delete(config)
        await self.db.commit()
        
        return True
    
    async def test_config(
        self,
        config: UserAdapterConfig,
        timeout: int = 30
    ) -> AdapterConfigTestResponse:
        """Test an adapter configuration."""
        start_time = time.time()
        
        try:
            # Decrypt credentials
            credentials = {}
            if config.credentials:
                credentials = decrypt_data(config.credentials)
            
            # Create test adapter instance
            adapter_config = AdapterConfig(
                name=f"test_{config.adapter_type}",
                category=AdapterCategory.AI,  # Will be overridden by adapter
                credentials=credentials,
                custom_config=config.settings or {},
                timeout_seconds=timeout
            )
            
            # Try to create and initialize adapter
            adapter = await adapter_registry.create_adapter(
                config.adapter_type,
                adapter_config
            )
            
            if adapter:
                # Test basic functionality
                if hasattr(adapter, 'health_check'):
                    health = await asyncio.wait_for(
                        adapter.health_check(),
                        timeout=timeout
                    )
                    
                    if health.get('status') == 'healthy':
                        status = TestStatus.SUCCESS
                        message = "Adapter configuration is valid and operational"
                    else:
                        status = TestStatus.FAILED
                        message = f"Health check failed: {health.get('error', 'Unknown error')}"
                else:
                    status = TestStatus.SUCCESS
                    message = "Adapter created successfully"
                
                # Clean up test adapter
                await adapter_registry.shutdown_adapter(config.adapter_type)
            else:
                status = TestStatus.FAILED
                message = "Failed to create adapter instance"
            
        except asyncio.TimeoutError:
            status = TestStatus.FAILED
            message = f"Test timed out after {timeout} seconds"
        except Exception as e:
            status = TestStatus.FAILED
            message = f"Test failed: {str(e)}"
        
        # Update configuration test status
        config.test_status = status.value
        config.test_message = message
        config.last_tested_at = datetime.utcnow()
        await self.db.commit()
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return AdapterConfigTestResponse(
            status=status,
            message=message,
            tested_at=config.last_tested_at,
            duration_ms=duration_ms
        )
    
    async def test_configs_bulk(
        self,
        configs: List[UserAdapterConfig],
        parallel: bool = True,
        timeout: int = 30
    ) -> AdapterConfigBulkTestResponse:
        """Test multiple adapter configurations."""
        start_time = time.time()
        results = {}
        
        if parallel:
            # Test in parallel
            tasks = []
            for config in configs:
                task = asyncio.create_task(
                    self.test_config(config, timeout)
                )
                tasks.append((str(config.id), task))
            
            for config_id, task in tasks:
                try:
                    result = await task
                    results[config_id] = result
                except Exception as e:
                    results[config_id] = AdapterConfigTestResponse(
                        status=TestStatus.FAILED,
                        message=f"Test failed: {str(e)}",
                        tested_at=datetime.utcnow()
                    )
        else:
            # Test sequentially
            for config in configs:
                try:
                    result = await self.test_config(config, timeout)
                    results[str(config.id)] = result
                except Exception as e:
                    results[str(config.id)] = AdapterConfigTestResponse(
                        status=TestStatus.FAILED,
                        message=f"Test failed: {str(e)}",
                        tested_at=datetime.utcnow()
                    )
        
        # Calculate statistics
        successful = sum(1 for r in results.values() if r.status == TestStatus.SUCCESS)
        failed = sum(1 for r in results.values() if r.status == TestStatus.FAILED)
        duration_ms = int((time.time() - start_time) * 1000)
        
        return AdapterConfigBulkTestResponse(
            results=results,
            total_tested=len(results),
            successful=successful,
            failed=failed,
            duration_ms=duration_ms
        )
    
    async def activate_adapter(
        self,
        config: UserAdapterConfig
    ) -> bool:
        """Activate an adapter by creating a runtime instance."""
        try:
            # Decrypt credentials
            credentials = {}
            if config.credentials:
                credentials = decrypt_data(config.credentials)
            
            # Create adapter configuration
            adapter_config = AdapterConfig(
                name=config.name or config.adapter_type,
                category=AdapterCategory.AI,  # Will be overridden
                credentials=credentials,
                custom_config=config.settings or {}
            )
            
            # Create and initialize adapter
            adapter = await adapter_registry.create_adapter(
                config.adapter_type,
                adapter_config
            )
            
            if adapter:
                config.enabled = True
                await self.db.commit()
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to activate adapter: {e}")
            return False
    
    async def deactivate_adapter(
        self,
        config: UserAdapterConfig
    ) -> bool:
        """Deactivate an adapter by removing its runtime instance."""
        try:
            # Shutdown adapter if it exists
            if config.adapter_type in adapter_registry.adapters:
                await adapter_registry.shutdown_adapter(config.adapter_type)
            
            config.enabled = False
            await self.db.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to deactivate adapter: {e}")
            return False