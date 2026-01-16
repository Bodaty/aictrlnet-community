"""Task-related endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from core.database import get_db
from core.dependencies import PaginationParams, require_edition
from core.security import get_current_active_user
from core.cache import cache_result, invalidate_cache, task_cache_key
from models.community import Task, TaskStatus
from schemas.task import TaskCreate, TaskUpdate, TaskResponse, TaskListResponse
from services.task import TaskService
# from .websocket import notify_task_created

router = APIRouter()

import logging
logger = logging.getLogger(__name__)
logger.info("Tasks router initialized")

@router.post("/simple")
async def simple_post():
    """Simple POST endpoint."""
    return {"message": "Simple POST works!"}

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify routing works."""
    return {"message": "Test endpoint works!"}

@router.post("/test-post")
async def test_post_endpoint():
    """Test POST endpoint."""
    return {"message": "POST works!"}


@router.get("/")
async def list_tasks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    status: Optional[TaskStatus] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
) -> List[TaskResponse]:
    """List all tasks."""
    query = select(Task).where(Task.tenant_id == current_user.tenant_id)
    
    # Apply filters
    if status:
        query = query.filter(Task.status == status)
    if search:
        query = query.filter(Task.name.ilike(f"%{search}%"))
    
    # Apply pagination
    query = query.offset(skip).limit(limit)
    
    # Execute query
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    # Convert to Pydantic models with proper serialization
    return [
        TaskResponse(
            id=task.id,
            name=task.name,
            description=task.description,
            status=task.status.value if hasattr(task.status, 'value') else task.status,
            metadata=task.task_metadata,
            tenant_id=task.tenant_id,
            created_at=task.created_at,
            updated_at=task.updated_at
        )
        for task in tasks
    ]


@router.post("/", response_model=TaskResponse)
async def create_task(
    task_data: TaskCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a new task."""
    task_service = TaskService(db)
    task = await task_service.create_task(task_data, tenant_id=current_user.tenant_id)
    
    # Send WebSocket notification
    # await notify_task_created(
    #     task_id=task.id,
    #     user_id=current_user["id"],
    #     task_data={
    #         "name": task.name,
    #         "description": task.description,
    #         "status": task.status.value if hasattr(task.status, 'value') else task.status
    #     }
    # )
    
    return TaskResponse(
        id=task.id,
        name=task.name,
        description=task.description,
        status=task.status.value if hasattr(task.status, 'value') else task.status,
        metadata=task.task_metadata,
        tenant_id=task.tenant_id,
        created_at=task.created_at,
        updated_at=task.updated_at
    )


@router.get("/{task_id}", response_model=TaskResponse)
# @cache_result(prefix="task", expire=300, key_builder=lambda task_id, **kwargs: f"task:{task_id}")
async def get_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get a specific task."""
    result = await db.execute(
        select(Task).filter(
            Task.id == task_id,
            Task.tenant_id == current_user.tenant_id
        )
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return TaskResponse(
        id=task.id,
        name=task.name,
        description=task.description,
        status=task.status.value if hasattr(task.status, 'value') else task.status,
        metadata=task.task_metadata,
        tenant_id=task.tenant_id,
        created_at=task.created_at,
        updated_at=task.updated_at
    )


@router.put("/{task_id}", response_model=TaskResponse)
# # @invalidate_cache(patterns=["task:*", "tasks:*"])
async def update_task(
    task_id: str,
    task_update: TaskUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Update a task."""
    # Get existing task
    result = await db.execute(
        select(Task).filter(
            Task.id == task_id,
            Task.tenant_id == current_user.tenant_id
        )
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update fields
    update_data = task_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(task, field, value)
    
    await db.commit()
    await db.refresh(task)
    
    return TaskResponse.model_validate(task, from_attributes=True)


@router.delete("/{task_id}", status_code=204)
# @invalidate_cache(patterns=["task:*", "tasks:*"])
async def delete_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Delete a task."""
    # Get existing task
    result = await db.execute(
        select(Task).filter(
            Task.id == task_id,
            Task.tenant_id == current_user.tenant_id
        )
    )
    task = result.scalar_one_or_none()
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    await db.delete(task)
    await db.commit()
    
    return None