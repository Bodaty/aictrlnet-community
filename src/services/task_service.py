"""Task service for managing tasks."""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from datetime import datetime
import logging

from models.community_complete import Task
from core.exceptions import ValidationError, NotFoundError

logger = logging.getLogger(__name__)


class TaskService:
    """Service for managing tasks."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_task(
        self,
        name: str,
        description: str = "",
        task_type: str = "generic",
        destination: str = "ai",
        payload: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> Task:
        """Create a new task."""
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            task_type=task_type,
            destination=destination,
            payload=payload or {},
            metadata=metadata or {},
            status="pending",
            created_at=datetime.utcnow()
        )
        
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        
        logger.info(f"Created task {task.id}: {task.name}")
        return task
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        result = await self.db.execute(
            select(Task).where(Task.id == task_id)
        )
        return result.scalar_one_or_none()
    
    async def execute_task(self, task_id: str) -> Dict[str, Any]:
        """Execute a task (mock implementation)."""
        task = await self.get_task(task_id)
        if not task:
            raise NotFoundError(f"Task {task_id} not found")
        
        # Mock execution
        task.status = "completed"
        task.updated_at = datetime.utcnow()
        task.result = {
            "output": f"Task {task.name} completed successfully",
            "execution_time": 0.5
        }
        
        await self.db.commit()
        
        return {
            "status": task.status,
            "output": task.result.get("output"),
            "execution_time": task.result.get("execution_time")
        }
    
    async def list_tasks(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Task]:
        """List tasks with optional filters."""
        query = select(Task)
        
        # Apply filters if provided
        if filters:
            # Simple implementation - in production would need proper filtering
            if "metadata.orchestrated_via" in filters:
                # Filter by metadata field
                pass
        
        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()