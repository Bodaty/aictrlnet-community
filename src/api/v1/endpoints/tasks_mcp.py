"""Task endpoints with MCP integration."""

from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
import json
import uuid
from datetime import datetime

from core.database import get_db
from core.security import get_current_active_user
from models import TaskMCP, User
from services.mcp_integration import MCPTaskIntegration
from schemas.task import TaskCreate, TaskResponse
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/route")
async def route_task(
    task_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Route a task, potentially to MCP servers"""
    try:
        # Ensure task has required fields
        if "task_id" not in task_data:
            task_data["task_id"] = str(uuid.uuid4())
        if "source_id" not in task_data:
            task_data["source_id"] = str(current_user.id)
        if "destination" not in task_data:
            task_data["destination"] = "ai"
        if "status" not in task_data:
            task_data["status"] = "pending"
        
        # Check if this is an MCP task
        if MCPTaskIntegration.is_mcp_task(task_data):
            # Route to MCP
            result, status_code = await MCPTaskIntegration.route_task(task_data)
            
            # Store task result in database
            task_mcp = TaskMCP(
                task_id=task_data["task_id"],
                source_id=task_data["source_id"],
                destination="mcp",
                payload=json.dumps(task_data.get("payload", {})),
                status=result.get("status", "completed"),
                mcp_enabled=True,
                mcp_metadata=json.dumps(result.get("mcp_metadata", {})),
                input_tokens=result.get("usage", {}).get("prompt_tokens"),
                output_tokens=result.get("usage", {}).get("completion_tokens"),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            db.add(task_mcp)
            await db.commit()
            
            if status_code != 200:
                raise HTTPException(status_code=status_code, detail=result.get("error"))
            
            return result
        else:
            # Handle non-MCP tasks (placeholder for now)
            return {
                "task_id": task_data["task_id"],
                "status": "pending",
                "message": "Non-MCP task routing not implemented"
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Task routing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Task routing failed: {str(e)}"
        )


@router.get("/mcp/{task_id}")
async def get_mcp_task(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get details of an MCP task"""
    task = await db.execute(
        db.query(TaskMCP).filter_by(task_id=task_id)
    )
    task = task.scalar_one_or_none()
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="MCP task not found"
        )
    
    return {
        "task_id": task.task_id,
        "source_id": task.source_id,
        "destination": task.destination,
        "payload": json.loads(task.payload) if task.payload else {},
        "status": task.status,
        "quality_score": task.quality_score,
        "auto_escalated": task.auto_escalated,
        "mcp_enabled": task.mcp_enabled,
        "mcp_metadata": json.loads(task.mcp_metadata) if task.mcp_metadata else {},
        "input_tokens": task.input_tokens,
        "output_tokens": task.output_tokens,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat()
    }


@router.get("/mcp")
async def list_mcp_tasks(
    status: str = None,
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List MCP tasks"""
    query = db.query(TaskMCP).filter_by(mcp_enabled=True)
    
    if status:
        query = query.filter_by(status=status)
    
    query = query.offset(offset).limit(limit)
    
    result = await db.execute(query)
    tasks = result.scalars().all()
    
    return {
        "tasks": [
            {
                "task_id": task.task_id,
                "source_id": task.source_id,
                "destination": task.destination,
                "status": task.status,
                "created_at": task.created_at.isoformat(),
                "updated_at": task.updated_at.isoformat()
            }
            for task in tasks
        ],
        "total": len(tasks),
        "offset": offset,
        "limit": limit
    }