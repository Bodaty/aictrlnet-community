"""Task service for managing tasks."""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
import logging

from models.community_complete import Task, TaskStatus
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
        metadata: Dict[str, Any] = None
    ) -> Task:
        """Create a new task.

        Args:
            name: Task name
            description: Task description
            metadata: Additional metadata (stored in task_metadata column)
        """
        task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            status=TaskStatus.PENDING,
            task_metadata=metadata or {}
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

        # Mock execution - update status and store result in metadata
        task.status = TaskStatus.COMPLETED
        task.task_metadata = task.task_metadata or {}
        task.task_metadata["result"] = {
            "output": f"Task {task.name} completed successfully",
            "execution_time": 0.5
        }

        await self.db.commit()

        result_data = task.task_metadata.get("result", {})
        return {
            "status": task.status.value if hasattr(task.status, 'value') else str(task.status),
            "output": result_data.get("output"),
            "execution_time": result_data.get("execution_time")
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
            if filters.get('status'):
                query = query.where(Task.status == filters['status'])

        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()

    async def update_task(
        self,
        task_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Update an existing task."""
        task = await self.get_task(task_id)
        if not task:
            raise NotFoundError(f"Task {task_id} not found")

        if name is not None:
            task.name = name
        if description is not None:
            task.description = description
        if status is not None:
            # Convert string status to TaskStatus enum
            if isinstance(status, str):
                task.status = TaskStatus(status)
            else:
                task.status = status
        if metadata is not None:
            # Merge metadata rather than replace
            existing_metadata = task.task_metadata or {}
            existing_metadata.update(metadata)
            task.task_metadata = existing_metadata

        await self.db.commit()
        await self.db.refresh(task)

        logger.info(f"Updated task {task.id}: {task.name}")
        return task