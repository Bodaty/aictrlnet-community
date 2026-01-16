"""Simple tasks endpoints for testing."""

from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.database import get_db
from core.security import get_current_active_user
from models.community import Task
from schemas.task import TaskCreate, TaskResponse

router = APIRouter()

@router.get("/")
async def list_tasks(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """List all tasks."""
    result = await db.execute(select(Task))
    tasks = result.scalars().all()
    return [
        {
            "id": str(task.id),
            "name": task.name,
            "status": task.status,
            "description": task.description,
            "metadata": task.task_metadata,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat()
        }
        for task in tasks
    ]

@router.post("/")
async def create_task(
    task_data: TaskCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a new task."""
    task = Task(
        name=task_data.name,
        description=task_data.description,
        status="pending",
        task_metadata=task_data.task_metadata
    )
    db.add(task)
    await db.commit()
    await db.refresh(task)
    
    return {
        "id": str(task.id),
        "name": task.name,
        "status": task.status,
        "description": task.description,
        "metadata": task.task_metadata,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat()
    }