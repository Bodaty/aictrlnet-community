"""Task service for managing tasks."""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm.attributes import flag_modified
from datetime import datetime
import uuid
import logging

from models.community_complete import Task, TaskStatus
from core.exceptions import ValidationError, NotFoundError
from core.tenant_context import get_current_tenant_id

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
            task_metadata=metadata or {},
            tenant_id=get_current_tenant_id()
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
        """Start task execution — transitions to IN_PROGRESS.

        Tasks are status-tracked entities. This method transitions a task
        from PENDING to IN_PROGRESS and records the start time. The caller
        (workflow engine, node executor, etc.) is responsible for updating
        the task to COMPLETED or FAILED when actual work finishes.
        """
        task = await self.get_task(task_id)
        if not task:
            raise NotFoundError(f"Task {task_id} not found")

        if task.status == TaskStatus.COMPLETED:
            return {
                "status": TaskStatus.COMPLETED.value,
                "message": "Task is already completed",
            }

        if task.status == TaskStatus.FAILED:
            return {
                "status": TaskStatus.FAILED.value,
                "message": "Task previously failed — update status to retry",
            }

        # Transition to IN_PROGRESS
        started_at = datetime.utcnow().isoformat()
        task.status = TaskStatus.IN_PROGRESS
        task.task_metadata = dict(task.task_metadata or {})
        task.task_metadata["started_at"] = started_at
        flag_modified(task, "task_metadata")

        await self.db.commit()

        logger.info(f"Task {task_id} ({task.name}) transitioned to IN_PROGRESS")

        return {
            "status": TaskStatus.IN_PROGRESS.value,
            "output": f"Task {task.name} started",
            "execution_time": None,
            "started_at": started_at,
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
            existing_metadata = dict(task.task_metadata or {})
            existing_metadata.update(metadata)
            task.task_metadata = existing_metadata
            flag_modified(task, "task_metadata")

        await self.db.commit()
        await self.db.refresh(task)

        logger.info(f"Updated task {task.id}: {task.name}")
        return task