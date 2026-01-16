"""Resource pool service for Community Edition."""

from typing import List, Optional
from uuid import uuid4
from sqlalchemy.orm import Session
from sqlalchemy import select

from models.community_complete import ResourcePoolConfig
from schemas.resource_pool import ResourcePoolConfigCreate, ResourcePoolConfigUpdate


class ResourcePoolService:
    """Service for managing resource pool configurations."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def list_configs(
        self,
        resource_type: Optional[str] = None,
        is_active: Optional[bool] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[ResourcePoolConfig]:
        """List resource pool configurations."""
        query = select(ResourcePoolConfig)
        
        if resource_type:
            query = query.filter(ResourcePoolConfig.resource_type == resource_type)
        if is_active is not None:
            query = query.filter(ResourcePoolConfig.is_active == is_active)
        
        query = query.offset(skip).limit(limit)
        result = self.db.execute(query)
        return result.scalars().all()
    
    def get_config(self, config_id: str) -> Optional[ResourcePoolConfig]:
        """Get a specific resource pool configuration."""
        return self.db.query(ResourcePoolConfig).filter(ResourcePoolConfig.id == config_id).first()
    
    def create_config(self, config_data: ResourcePoolConfigCreate) -> ResourcePoolConfig:
        """Create a new resource pool configuration."""
        config = ResourcePoolConfig(
            id=str(uuid4()),
            **config_data.dict()
        )
        self.db.add(config)
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def update_config(self, config_id: str, config_data: ResourcePoolConfigUpdate) -> Optional[ResourcePoolConfig]:
        """Update a resource pool configuration."""
        config = self.get_config(config_id)
        if not config:
            return None
        
        update_data = config_data.dict(exclude_unset=True)
        for field, value in update_data.items():
            setattr(config, field, value)
        
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def delete_config(self, config_id: str) -> bool:
        """Delete a resource pool configuration."""
        config = self.get_config(config_id)
        if not config:
            return False
        
        self.db.delete(config)
        self.db.commit()
        return True