"""Workflow service for managing workflows."""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import uuid
from datetime import datetime
import logging

from models.community_complete import WorkflowDefinition as Workflow
from models.workflow_execution import WorkflowExecution
from core.exceptions import ValidationError, NotFoundError
from core.tenant_context import get_current_tenant_id

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for managing workflows."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_workflow(
        self,
        name: str,
        description: str = "",
        definition: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> Workflow:
        """Create a new workflow.

        Automatically assigns current tenant_id.
        """
        tenant_id = get_current_tenant_id()
        workflow = Workflow(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            definition=definition or {},
            metadata=metadata or {},
            status="active",
            tenant_id=tenant_id,
            created_at=datetime.utcnow()
        )
        
        self.db.add(workflow)
        await self.db.commit()
        await self.db.refresh(workflow)
        
        logger.info(f"Created workflow {workflow.id}: {workflow.name}")
        return workflow
    
    async def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get a workflow by ID.

        Applies tenant filtering for security.
        Note: RLS provides database-level protection as well.
        """
        tenant_id = get_current_tenant_id()
        result = await self.db.execute(
            select(Workflow).where(
                Workflow.id == workflow_id,
                Workflow.tenant_id == tenant_id
            )
        )
        return result.scalar_one_or_none()
    
    async def update_workflow(
        self,
        workflow_id: str,
        metadata: Dict[str, Any] = None
    ) -> Workflow:
        """Update workflow metadata."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            raise NotFoundError(f"Workflow {workflow_id} not found")
        
        if metadata:
            workflow.metadata = metadata
            workflow.updated_at = datetime.utcnow()
        
        await self.db.commit()
        await self.db.refresh(workflow)
        
        return workflow
    
    async def execute_workflow(
        self,
        workflow_id: str,
        input_data: Dict[str, Any] = None
    ) -> WorkflowExecution:
        """Execute a workflow via the real execution engine."""
        from services.workflow_execution import WorkflowExecutionService
        exec_service = WorkflowExecutionService(self.db)
        execution = await exec_service.create_execution(
            workflow_id=workflow_id,
            input_data=input_data or {},
            triggered_by="api"
        )
        result = await exec_service.start_execution(execution.id)
        logger.info(f"Executed workflow {workflow_id}, execution {result.id}")
        return result
    
    async def get_workflow_executions(
        self,
        workflow_id: str,
        limit: int = 10
    ) -> List[WorkflowExecution]:
        """Get workflow executions."""
        result = await self.db.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.workflow_id == workflow_id)
            .order_by(WorkflowExecution.started_at.desc())
            .limit(limit)
        )
        return result.scalars().all()
    
    async def list_workflows(
        self,
        filters: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Workflow]:
        """List workflows with optional filters.

        Applies tenant filtering for security.
        Note: RLS provides database-level protection as well.
        """
        tenant_id = get_current_tenant_id()
        query = select(Workflow).where(Workflow.tenant_id == tenant_id)

        # Apply filters if provided
        if filters:
            # Simple implementation - in production would need proper filtering
            if "metadata.created_via" in filters:
                # Filter by metadata field
                pass

        query = query.limit(limit).offset(offset)
        result = await self.db.execute(query)
        return result.scalars().all()