"""MCP Orchestration Service for task orchestration via MCP protocol."""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging
import uuid
from datetime import datetime

from services.task_service import TaskService
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class MCPOrchestrationService:
    """MCP service for task orchestration."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.task_service = TaskService(db)
    
    async def process_orchestration_request(
        self,
        messages: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process orchestration request via MCP."""
        try:
            # Extract task definition from messages
            task_definition = self._extract_task_definition(messages)
            
            # Validate task definition
            if not task_definition.get("name"):
                raise ValidationError("Task name is required")
            
            # Create task in AICtrlNet
            task = await self.task_service.create_task(
                name=task_definition.get("name"),
                description=task_definition.get("description", "MCP orchestrated task"),
                task_type=task_definition.get("type", "mcp_orchestration"),
                destination=task_definition.get("destination", "ai"),
                payload=task_definition.get("payload", {}),
                metadata={
                    "mcp_context": context,
                    "mcp_messages": messages,
                    "orchestrated_via": "mcp"
                }
            )
            
            # Execute task if requested
            if context.get("execute_immediately", True):
                result = await self.task_service.execute_task(str(task.id))
                
                return {
                    "task_id": str(task.id),
                    "status": result.get("status", "completed"),
                    "result": result.get("output"),
                    "execution_time": result.get("execution_time"),
                    "metadata": {
                        "orchestrated_by": "AICtrlNet",
                        "mcp_version": "1.0",
                        "executed_at": datetime.utcnow().isoformat()
                    }
                }
            else:
                return {
                    "task_id": str(task.id),
                    "status": "created",
                    "result": None,
                    "metadata": {
                        "orchestrated_by": "AICtrlNet",
                        "mcp_version": "1.0",
                        "created_at": datetime.utcnow().isoformat()
                    }
                }
                
        except Exception as e:
            logger.error(f"MCP orchestration failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "metadata": {
                    "orchestrated_by": "AICtrlNet",
                    "mcp_version": "1.0",
                    "failed_at": datetime.utcnow().isoformat()
                }
            }
    
    def _extract_task_definition(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract task definition from MCP messages."""
        task_def = {
            "name": "",
            "description": "",
            "type": "mcp_orchestration",
            "destination": "ai",
            "payload": {}
        }
        
        # Look for task definition in messages
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            if role == "system":
                # System messages often contain instructions
                if "task:" in content.lower():
                    task_def["name"] = content.split("task:", 1)[1].strip()
                elif "orchestrate:" in content.lower():
                    task_def["name"] = content.split("orchestrate:", 1)[1].strip()
                
                task_def["description"] = content
                
            elif role == "user":
                # User messages contain the actual task content
                task_def["payload"]["user_request"] = content
                
                # Try to extract structured data if present
                if isinstance(content, dict):
                    task_def["payload"].update(content)
                    if "name" in content:
                        task_def["name"] = content["name"]
                    if "destination" in content:
                        task_def["destination"] = content["destination"]
        
        # Generate name if not found
        if not task_def["name"]:
            task_def["name"] = f"MCP Task {uuid.uuid4().hex[:8]}"
        
        # Add all messages to payload for context
        task_def["payload"]["messages"] = messages
        
        return task_def
    
    async def list_orchestrated_tasks(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List tasks orchestrated via MCP."""
        tasks = await self.task_service.list_tasks(
            filters={"metadata.orchestrated_via": "mcp"},
            limit=limit,
            offset=offset
        )
        
        return {
            "tasks": [
                {
                    "task_id": str(task.id),
                    "name": task.name,
                    "status": task.status,
                    "created_at": task.created_at.isoformat(),
                    "metadata": task.metadata
                }
                for task in tasks
            ],
            "total": len(tasks),
            "limit": limit,
            "offset": offset
        }
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of an MCP orchestrated task."""
        task = await self.task_service.get_task(task_id)
        
        if not task:
            return {
                "error": "Task not found",
                "task_id": task_id
            }
        
        return {
            "task_id": str(task.id),
            "name": task.name,
            "status": task.status,
            "created_at": task.created_at.isoformat(),
            "updated_at": task.updated_at.isoformat() if task.updated_at else None,
            "result": task.result,
            "metadata": task.metadata
        }