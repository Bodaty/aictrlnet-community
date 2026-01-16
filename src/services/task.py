"""Task service for business logic."""

from typing import List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.community import Task, TaskStatus
from schemas.task import TaskCreate, TaskUpdate


class TaskService:
    """Service for task-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_task(self, task_data: TaskCreate, tenant_id: str) -> Task:
        """Create a new task."""
        # Convert the data, handling the metadata -> task_metadata mapping
        task_dict = task_data.model_dump(by_alias=False)
        task_dict['tenant_id'] = tenant_id
        task = Task(**task_dict)
        self.db.add(task)
        await self.db.commit()
        await self.db.refresh(task)
        return task
    
    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID."""
        result = await self.db.execute(
            select(Task).filter(Task.id == task_id)
        )
        return result.scalar_one_or_none()
    
    async def list_tasks(
        self,
        skip: int = 0,
        limit: int = 100,
        status: Optional[TaskStatus] = None,
    ) -> List[Task]:
        """List tasks with optional filtering."""
        query = select(Task)
        
        if status:
            query = query.filter(Task.status == status)
        
        query = query.offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def update_task(
        self,
        task_id: str,
        task_update: TaskUpdate,
    ) -> Optional[Task]:
        """Update a task."""
        task = await self.get_task(task_id)
        if not task:
            return None
        
        update_data = task_update.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            setattr(task, field, value)
        
        await self.db.commit()
        await self.db.refresh(task)
        return task
    
    async def delete_task(self, task_id: str) -> bool:
        """Delete a task."""
        task = await self.get_task(task_id)
        if not task:
            return False
        
        await self.db.delete(task)
        await self.db.commit()
        return True