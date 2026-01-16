"""Workflow service for business logic."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from models.community import WorkflowDefinition, WorkflowInstance
from schemas.workflow import WorkflowCreate, WorkflowUpdate
from .workflow_template_service import create_workflow_template_service
from core.tenant_context import get_current_tenant_id


class WorkflowService:
    """Service for workflow-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_workflow(self, workflow_data: WorkflowCreate, tenant_id: Optional[str] = None) -> WorkflowDefinition:
        """Create a new workflow."""
        # Convert nodes and edges to dict format
        definition_dict = {
            "nodes": [node.model_dump() for node in workflow_data.definition.nodes],
            "edges": [edge.model_dump() for edge in workflow_data.definition.edges],
            "metadata": workflow_data.definition.metadata or {},
        }
        
        workflow = WorkflowDefinition(
            name=workflow_data.name,
            description=workflow_data.description,
            definition=definition_dict,
            tags=workflow_data.tags,
            tenant_id=tenant_id or get_current_tenant_id(),  # Ensure tenant_id is never None
            workflow_metadata={
                "category": workflow_data.category,
                "is_template": workflow_data.is_template,
                "template_id": workflow_data.template_id,
                "status": workflow_data.status,
            } if any([workflow_data.category, workflow_data.is_template, workflow_data.template_id, workflow_data.status]) else None
        )
        
        self.db.add(workflow)
        await self.db.commit()
        await self.db.refresh(workflow)
        return workflow
    
    async def create_workflow_from_template(
        self,
        template: Dict[str, Any],
        name: str,
        description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        tenant_id: Optional[str] = None,
    ) -> WorkflowDefinition:
        """Create a workflow from a template."""
        template_service = create_workflow_template_service()

        # Convert Pydantic model to dict if necessary
        if hasattr(template, 'model_dump'):
            template_dict = template.model_dump()
        else:
            template_dict = template

        # Generate workflow configuration
        workflow_config = await template_service.generate_workflow_for_template(template_dict)

        # Apply parameters if provided
        if parameters:
            workflow_config = self._apply_parameters(workflow_config, parameters)

        # Create workflow
        workflow = WorkflowDefinition(
            name=name,
            description=description or template_dict.get("description", ""),
            definition=workflow_config,
            tags=template_dict.get("tags", []),
            active=True,  # Templates create active workflows by default
            tenant_id=tenant_id or get_current_tenant_id(),  # Ensure tenant_id is never None
            workflow_metadata={
                "template_id": str(template_dict.get("id")),  # Convert UUID to string
                "category": template_dict.get("category"),
                "created_from_template": True,
            }
        )
        
        self.db.add(workflow)
        await self.db.commit()
        await self.db.refresh(workflow)
        return workflow
    
    def _apply_parameters(
        self,
        workflow_config: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply parameters to workflow configuration."""
        # This is a simplified version - in production, you'd have more sophisticated parameter substitution
        config_str = json.dumps(workflow_config)
        
        for key, value in parameters.items():
            placeholder = f"{{{{{key}}}}}"
            config_str = config_str.replace(placeholder, str(value))
        
        return json.loads(config_str)
    
    async def get_workflow(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get a workflow by ID."""
        result = await self.db.execute(
            select(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id)
        )
        return result.scalar_one_or_none()
    
    async def list_workflows(
        self,
        skip: int = 0,
        limit: int = 100,
        category: Optional[str] = None,
        is_template: Optional[bool] = None,
    ) -> List[WorkflowDefinition]:
        """List workflows with optional filtering."""
        query = select(WorkflowDefinition)
        
        if category:
            query = query.filter(WorkflowDefinition.category == category)
        
        if is_template is not None:
            query = query.filter(WorkflowDefinition.is_template == is_template)
        
        query = query.offset(skip).limit(limit).order_by(WorkflowDefinition.created_at.desc())
        
        result = await self.db.execute(query)
        return result.scalars().all()
    
    async def update_workflow(
        self,
        workflow_id: str,
        workflow_update: WorkflowUpdate,
    ) -> Optional[WorkflowDefinition]:
        """Update a workflow."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        update_data = workflow_update.model_dump(exclude_unset=True)
        
        # If updating definition, increment version
        if "definition" in update_data:
            workflow.version += 1
            # Convert to dict format
            definition_dict = {
                "nodes": [node.model_dump() for node in update_data["definition"].nodes],
                "edges": [edge.model_dump() for edge in update_data["definition"].edges],
                "metadata": update_data["definition"].metadata or {},
            }
            update_data["definition"] = definition_dict
        
        for field, value in update_data.items():
            setattr(workflow, field, value)
        
        await self.db.commit()
        await self.db.refresh(workflow)
        return workflow
    
    async def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return False
        
        await self.db.delete(workflow)
        await self.db.commit()
        return True
    
    async def execute_workflow(
        self,
        workflow_id: str,
        context: Optional[Dict[str, Any]] = None,
        trigger_source: str = "manual",
    ) -> WorkflowInstance:
        """Execute a workflow by creating a new instance."""
        import uuid
        from datetime import datetime
        
        # Get the workflow definition
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        # Create workflow instance
        instance = WorkflowInstance(
            id=str(uuid.uuid4()),
            definition_id=workflow_id,
            name=workflow.name,  # WorkflowInstance requires a name
            status="running",
            context=context or {},
            started_at=datetime.utcnow(),
        )
        
        self.db.add(instance)
        await self.db.commit()
        await self.db.refresh(instance)
        
        # In a real implementation, this would trigger the actual workflow execution
        # For now, we'll just return the instance
        return instance