"""Workflow service for business logic."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
import json

from models.community import WorkflowDefinition, WorkflowInstance
from schemas.workflow import WorkflowCreate, WorkflowUpdate
from .workflow_template_service import create_workflow_template_service
from core.tenant_context import get_current_tenant_id
from .managed_workflows import refuse_managed_change
from .workflow_identity import (  # noqa: F401
    DefinitionChangeForbidden,
    GrantedWorkflowChangeForbidden,
    ReservedWorkflowName,
    check_granted_workflow_change,
    check_identity_grant,
    check_workflow_name,
    grant_identity,
    is_granted,
    protected_identity,
)

# The dial: how much a workflow does without a person. Only a tenant's
# administrator moves it or its lock (the autonomy endpoints enforce the same).
AUTONOMY_FIELDS = ("autonomy_level", "autonomy_locked")


class AutonomyChangeForbidden(PermissionError):
    """A non-admin update would change a workflow's autonomy level or lock."""

    def __init__(self, fields: List[str]):
        super().__init__(f"Only an administrator can change {', '.join(fields)}")
        self.fields = fields


def changed_autonomy_fields(workflow: Any, update_data: Dict[str, Any]) -> List[str]:
    """The autonomy fields ``update_data`` would actually change; a value equal
    to the current one (a full-object save) is not a change."""
    changed = []
    for field in AUTONOMY_FIELDS:
        if field not in update_data:
            continue
        new, current = update_data[field], getattr(workflow, field, None)
        if field == "autonomy_locked":
            new, current = bool(new), bool(current)
        if new != current:
            changed.append(field)
    return changed


def check_autonomy_change(workflow: Any, update_data: Dict[str, Any], is_admin: bool) -> None:
    """Raise AutonomyChangeForbidden when a non-admin would move the dial."""
    changed = changed_autonomy_fields(workflow, update_data)
    if changed and not is_admin:
        raise AutonomyChangeForbidden(changed)


def _as_dict(definition: Any) -> Dict[str, Any]:
    if isinstance(definition, str):
        definition = json.loads(definition)
    return definition if isinstance(definition, dict) else {}


class PhiDefinitionChangeForbidden(ValueError):
    """An update would take `phi` off a PHI workflow or change its templateId."""


def merge_definition_update(current: Optional[Dict[str, Any]], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """The definition an update stores: the current one with the keys the update
    carries replaced, so a key it does not carry (``phi``, ``templateId`` and the
    like) is kept. A workflow whose definition has ``phi: true`` keeps it and
    keeps ``metadata.templateId``: an update that sets ``phi`` to anything but
    true, or names a different templateId, raises PhiDefinitionChangeForbidden."""
    current = dict(_as_dict(current))
    merged = {**current, **incoming}
    if "metadata" in incoming:
        merged["metadata"] = dict(incoming["metadata"] or {})
    if not bool(current.get("phi")):
        return merged
    if "phi" in incoming and incoming["phi"] is not True:
        raise PhiDefinitionChangeForbidden(
            "This workflow handles patient information; its phi flag cannot be removed or changed")
    template_id = (current.get("metadata") or {}).get("templateId")
    if "metadata" in incoming:
        if "templateId" in merged["metadata"] and merged["metadata"]["templateId"] != template_id:
            raise PhiDefinitionChangeForbidden(
                "This workflow handles patient information; its templateId cannot be changed")
        if template_id is not None:
            merged["metadata"]["templateId"] = template_id
    return merged


def check_definition_change(workflow: Any, update_data: Dict[str, Any], is_admin: bool) -> None:
    """The generic update's guards on what a definition says it is.

    - A PHI workflow keeps its name (PhiDefinitionChangeForbidden): other parts
      of the platform recognise its runs by name and template.
    - Only an administrator sets ``phi`` or a ``metadata.templateId`` the
      workflow doesn't already have (DefinitionChangeForbidden), as with the
      autonomy dial.
    - Only an administrator renames a workflow to a reserved name
      (ReservedWorkflowName).
    Nothing is written when any raises."""
    current = _as_dict(getattr(workflow, "definition", None))
    if bool(current.get("phi")) and "name" in update_data \
            and update_data["name"] != getattr(workflow, "name", None):
        raise PhiDefinitionChangeForbidden(
            "This workflow handles patient information; it cannot be renamed")
    if "name" in update_data:
        check_workflow_name(update_data["name"], is_admin, getattr(workflow, "name", None))
    incoming = update_data.get("definition")
    if hasattr(incoming, "model_dump"):
        incoming = incoming.model_dump(exclude_unset=True)
    if not isinstance(incoming, dict) or is_admin:
        return
    fields = []
    if "phi" in incoming and incoming["phi"] != current.get("phi"):
        fields.append("phi")
    if "metadata" in incoming:
        new_template = (incoming["metadata"] or {}).get("templateId")
        if new_template is not None and new_template != (current.get("metadata") or {}).get("templateId"):
            fields.append("templateId")
    if fields:
        raise DefinitionChangeForbidden(fields)


def regrant_on_admin_update(current: Any, merged: Dict[str, Any], is_admin: bool,
                            actor_id: Optional[str]) -> Dict[str, Any]:
    """An administrator's update that gives (or changes) a protected identity
    stamps a fresh grant; any other update carries the stored grant through
    (the model layer keeps it whatever the update says)."""
    fields = protected_identity(merged)
    if not (is_admin and actor_id and fields):
        return merged
    stored = (_as_dict(current).get("identityGrant") or {})
    if is_granted(current) and sorted(stored.get("fields") or []) == sorted(fields):
        return merged
    return grant_identity(merged, actor_id)


def refuse_clearing_phi_definition(current: Optional[Dict[str, Any]]) -> None:
    """A PHI workflow's definition cannot be cleared (that would drop ``phi``)."""
    if bool(_as_dict(current).get("phi")):
        raise PhiDefinitionChangeForbidden(
            "This workflow handles patient information; its definition cannot be removed")


class WorkflowService:
    """Service for workflow-related operations."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def create_workflow(self, workflow_data: WorkflowCreate, tenant_id: Optional[str] = None,
                              *, actor_is_admin: bool = False,
                              actor_id: Optional[str] = None) -> WorkflowDefinition:
        """Create a new workflow. A definition marked ``phi``, or carrying a
        protected template id, needs ``actor_is_admin`` (DefinitionChangeForbidden),
        as does a reserved name (ReservedWorkflowName); an administrator's
        (``actor_id``) such workflow is stamped with the grant."""
        incoming = workflow_data.definition.model_dump() if workflow_data.definition is not None else {}
        check_identity_grant(incoming, actor_is_admin)
        check_workflow_name(workflow_data.name, actor_is_admin)
        # Convert nodes and edges to dict format
        definition_dict = {
            "nodes": [node.model_dump() for node in workflow_data.definition.nodes],
            "edges": [edge.model_dump() for edge in workflow_data.definition.edges],
            "metadata": workflow_data.definition.metadata or {},
        }
        if actor_is_admin and actor_id and protected_identity(incoming):
            if incoming.get("phi") is True:
                definition_dict["phi"] = True
            definition_dict = grant_identity(definition_dict, actor_id)
        
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
        *,
        actor_is_admin: bool = False,
        actor_id: Optional[str] = None,
    ) -> Optional[WorkflowDefinition]:
        """Update a workflow. Changing its autonomy level or lock needs
        ``actor_is_admin`` (AutonomyChangeForbidden otherwise; nothing is written).
        A definition update keeps the keys it does not carry, and cannot take a
        PHI workflow's ``phi``, ``templateId`` or name (PhiDefinitionChangeForbidden);
        setting ``phi`` or a templateId needs ``actor_is_admin``
        (DefinitionChangeForbidden)."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return None
        
        update_data = workflow_update.model_dump(exclude_unset=True)
        refuse_managed_change(workflow, definition_change="definition" in update_data)
        check_granted_workflow_change(workflow, actor_is_admin)
        check_autonomy_change(workflow, update_data, actor_is_admin)
        check_definition_change(workflow, update_data, actor_is_admin)
        
        if "definition" in update_data:
            if workflow_update.definition is None:
                refuse_clearing_phi_definition(workflow.definition)
            else:
                update_data["definition"] = regrant_on_admin_update(
                    workflow.definition,
                    merge_definition_update(
                        workflow.definition, workflow_update.definition.model_dump(exclude_unset=True)),
                    actor_is_admin, actor_id)
            workflow.version += 1
        
        for field, value in update_data.items():
            setattr(workflow, field, value)
        
        await self.db.commit()
        await self.db.refresh(workflow)
        return workflow
    
    async def delete_workflow(self, workflow_id: str, *, actor_is_admin: bool = False) -> bool:
        """Delete a workflow. A granted one needs ``actor_is_admin``
        (GrantedWorkflowChangeForbidden otherwise; nothing is deleted)."""
        workflow = await self.get_workflow(workflow_id)
        if not workflow:
            return False
        check_granted_workflow_change(workflow, actor_is_admin)
        
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
            tenant_id=get_current_tenant_id(),
        )
        
        self.db.add(instance)
        await self.db.commit()
        await self.db.refresh(instance)
        
        # In a real implementation, this would trigger the actual workflow execution
        # For now, we'll just return the instance
        return instance