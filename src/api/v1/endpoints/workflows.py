"""Workflow-related endpoints."""

import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)

from core.database import get_db
from core.security import get_current_active_user
from core.enforcement import LicenseEnforcer, LimitType
from core.usage_tracker import get_usage_tracker
from core.tenant_context import get_current_tenant_id
from middleware.enforcement import require_feature
from core.upgrade_hints import attach_upgrade_hints
from models.community import WorkflowDefinition, WorkflowInstance
from models.community_complete import Adapter, MCPTool
from models.iam import IAMAgent
from schemas.workflow import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    WorkflowInstanceResponse,
)
from schemas.workflow_templates import (
    WorkflowTemplateResponse,
    WorkflowFromTemplate
)
from services.workflow import WorkflowService
from services.workflow_template_service import create_workflow_template_service
from services.workflow_execution import WorkflowExecutionService
from services.node_catalog import DynamicNodeCatalogService
from services.workflow_scheduler import WorkflowScheduler, TriggerType
from nodes.registry import node_registry
from schemas.workflow_execution import (
    WorkflowExecutionCreate,
    WorkflowExecutionResponse,
    WorkflowTriggerCreate,
    WorkflowScheduleCreate
)
from schemas.workflow_node import (
    WorkflowCatalog,
    WorkflowValidationResult
)

router = APIRouter()


@router.get("/create")
async def get_workflow_create_info(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get information needed to create a workflow (templates, node types, etc.)."""
    # Get available templates
    template_service = create_workflow_template_service()
    
    from schemas.workflow_templates import TemplateListRequest
    request = TemplateListRequest(
        edition=getattr(current_user, "edition", "community"),
        include_system=True,
        include_public=True,
        include_private=False,
        limit=100
    )
    
    templates, total = await template_service.list_templates(
        db, current_user.get("id", "unknown"), request
    )
    
    # Get available node types from registry
    node_types = node_registry.list_node_types()
    
    return {
        "templates": templates,
        "node_types": node_types,
        "categories": ["data_processing", "ai_ml", "integration", "control_flow", "quality", "governance"],
        "editions": {
            "community": ["basic nodes", "templates"],
            "business": ["AI governance", "quality checks", "advanced templates"],
            "enterprise": ["multi-tenant", "federation", "compliance"]
        }
    }


@router.get("/catalog", response_model=WorkflowCatalog)
async def get_workflow_catalog(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get the dynamic workflow node catalog with available components."""
    tenant_id = getattr(current_user, "tenant_id", None) or get_current_tenant_id()
    edition = getattr(current_user, "edition", "community")
    
    # Use dynamic catalog service
    catalog_service = DynamicNodeCatalogService(db)
    catalog = await catalog_service.get_catalog(tenant_id, edition)
    
    return catalog


@router.get("/", response_model=List[WorkflowResponse])
async def list_workflows(
    response: Response,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    category: Optional[str] = None,
    is_template: Optional[bool] = None,
    search: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """List all workflows."""
    attach_upgrade_hints(response, "workflows")
    query = select(WorkflowDefinition)
    
    # Apply filters
    if category:
        query = query.filter(WorkflowDefinition.category == category)
    if is_template is not None:
        query = query.filter(WorkflowDefinition.is_template == is_template)
    if search:
        query = query.filter(WorkflowDefinition.name.ilike(f"%{search}%"))
    
    # Apply pagination
    query = query.offset(skip).limit(limit).order_by(WorkflowDefinition.created_at.desc())
    
    # Execute query
    result = await db.execute(query)
    workflows = result.scalars().all()
    
    # Convert to response models with error handling
    valid_workflows = []
    for workflow in workflows:
        try:
            # Try to convert workflow definition positions if needed
            if workflow.definition and isinstance(workflow.definition, dict):
                workflow_def = workflow.definition
                # Fix node positions if they're arrays
                if 'nodes' in workflow_def and isinstance(workflow_def['nodes'], list):
                    for node in workflow_def['nodes']:
                        if 'position' in node and isinstance(node['position'], list):
                            if len(node['position']) >= 2:
                                # Convert array [x, y] to object {x: x, y: y}
                                node['position'] = {
                                    'x': float(node['position'][0]),
                                    'y': float(node['position'][1])
                                }
                                logger.warning(f"Fixed position format for workflow {workflow.id} node {node.get('id', 'unknown')}")
            
            # Validate and add to response
            valid_workflows.append(WorkflowResponse.model_validate(workflow))
            
        except Exception as e:
            # Log the error but don't fail the entire request
            logger.error(f"Failed to validate workflow {workflow.id}: {str(e)}")
            logger.debug(f"Workflow {workflow.id} definition causing error: {workflow.definition}")
            
            # Optionally, add a minimal version of the workflow with error info
            try:
                # Create a minimal response without the problematic definition
                minimal_workflow = {
                    'id': str(workflow.id),
                    'name': workflow.name or f"Workflow {workflow.id}",
                    'description': workflow.description or "Error loading workflow definition",
                    'created_at': workflow.created_at,
                    'updated_at': workflow.updated_at,
                    'active': workflow.active,
                    'tenant_id': workflow.tenant_id,
                    'definition': {'error': 'Invalid workflow definition format', 'nodes': [], 'edges': []},
                    'tags': workflow.tags or []
                }
                valid_workflows.append(WorkflowResponse.model_validate(minimal_workflow))
                logger.info(f"Added minimal version of workflow {workflow.id} due to validation errors")
            except Exception as e2:
                logger.error(f"Could not create minimal workflow for {workflow.id}: {str(e2)}")
                # Skip this workflow entirely
                continue
    
    return valid_workflows


@router.post("/", response_model=WorkflowResponse, status_code=201)
async def create_workflow(
    workflow_data: WorkflowCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a new workflow."""
    # Check workflow limit
    tenant_id = getattr(current_user, "tenant_id", None) or get_current_tenant_id()
    enforcer = LicenseEnforcer(db)
    await enforcer.check_limit(
        tenant_id=tenant_id,
        limit_type=LimitType.WORKFLOWS,
        increment=1
    )
    
    workflow_service = WorkflowService(db)
    
    # Check if creating from template
    if workflow_data.template_id:
        # Use the template service's instantiate method which properly handles everything
        template_service = create_workflow_template_service()

        from schemas.workflow_templates import InstantiateTemplateRequest
        instantiate_request = InstantiateTemplateRequest(
            name=workflow_data.name,
            description=workflow_data.description,
            parameters=workflow_data.parameters if hasattr(workflow_data, 'parameters') else {}
        )

        # Use the working instantiate method
        result = await template_service.instantiate_template(
            db=db,
            template_id=workflow_data.template_id,
            user_id=current_user.id,
            request=instantiate_request
        )

        # Get the created workflow
        workflow_id = result.get('workflow_id')
        if not workflow_id:
            raise HTTPException(status_code=500, detail="Failed to create workflow from template")

        # Load the created workflow to return it
        from sqlalchemy import select
        from models.community import WorkflowDefinition
        query = select(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id)
        db_result = await db.execute(query)
        workflow = db_result.scalar_one_or_none()

        if not workflow:
            raise HTTPException(status_code=500, detail="Workflow created but not found")
    else:
        # Create workflow from definition
        if not workflow_data.definition:
            raise HTTPException(
                status_code=400,
                detail="Either template_id or definition must be provided"
            )

        # Apply Q/G/S/M metadata before saving (Business/Enterprise)
        try:
            from aictrlnet_business.services.qgsm import apply_qgsm

            qgsm_result = apply_qgsm(
                {"definition": workflow_data.definition},
                context={"prompt": workflow_data.description or workflow_data.name},
                edition="business",
            )
            workflow_data.definition = qgsm_result.get("definition", workflow_data.definition)
        except ImportError:
            pass  # Community edition — Q/G/S/M not available
        except Exception as qgsm_err:
            logger.warning(f"Q/G/S/M pre-save failed (non-critical): {qgsm_err}")

        workflow = await workflow_service.create_workflow(workflow_data, tenant_id=tenant_id)
    
    # Track usage
    tracker = await get_usage_tracker(db)
    await tracker.track_usage(
        tenant_id=tenant_id,
        metric_type="workflows_created",
        metadata={"workflow_id": str(workflow.id), "name": workflow.name}
    )
    
    return WorkflowResponse.model_validate(workflow)


@router.post("/from-template", response_model=WorkflowResponse, status_code=201)
async def create_workflow_from_template(
    request: WorkflowFromTemplate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a workflow from a template."""
    # Check workflow limit
    tenant_id = getattr(current_user, "tenant_id", None) or get_current_tenant_id()
    enforcer = LicenseEnforcer(db)
    await enforcer.check_limit(
        tenant_id=tenant_id,
        limit_type=LimitType.WORKFLOWS,
        increment=1
    )
    
    workflow_service = WorkflowService(db)
    template_service = create_workflow_template_service()
    
    # Get template by ID (convert string to UUID if needed)
    import uuid
    try:
        template_uuid = uuid.UUID(request.template_id) if isinstance(request.template_id, str) else request.template_id
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid template ID format")

    # Use the template service's instantiate method - the ONE correct implementation
    from schemas.workflow_templates import InstantiateTemplateRequest
    instantiate_request = InstantiateTemplateRequest(
        name=request.name,
        description=request.description,
        parameters=request.parameters if hasattr(request, 'parameters') else {}
    )

    result = await template_service.instantiate_template(
        db=db,
        template_id=template_uuid,
        user_id=current_user.id if hasattr(current_user, 'id') else current_user.get('id'),
        request=instantiate_request
    )

    # Get the created workflow
    workflow_id = result.get('workflow_id')
    if not workflow_id:
        raise HTTPException(status_code=500, detail="Failed to create workflow from template")

    from models.community import WorkflowDefinition
    from sqlalchemy import select
    query = select(WorkflowDefinition).where(WorkflowDefinition.id == workflow_id)
    db_result = await db.execute(query)
    workflow = db_result.scalar_one_or_none()

    if not workflow:
        raise HTTPException(status_code=500, detail="Workflow created but not found")
    
    return WorkflowResponse.model_validate(workflow)


@router.get("/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get a specific workflow."""
    result = await db.execute(
        select(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id)
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return WorkflowResponse.model_validate(workflow)


@router.get("/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get the execution status of a workflow."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"Getting status for workflow: {workflow_id}")
    
    # First check if workflow exists
    result = await db.execute(
        select(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id)
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        logger.error(f"Workflow not found: {workflow_id}")
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Get the latest instance of this workflow
    instance_result = await db.execute(
        select(WorkflowInstance)
        .filter(WorkflowInstance.definition_id == workflow_id)
        .order_by(WorkflowInstance.created_at.desc())
        .limit(1)
    )
    instance = instance_result.scalar_one_or_none()
    
    if not instance:
        # No instance found, workflow has never been executed
        return {
            "workflow_id": workflow_id,
            "status": "not_started",
            "message": "Workflow has not been executed yet"
        }
    
    # Return the status of the latest instance
    return {
        "workflow_id": workflow_id,
        "instance_id": instance.id,
        "status": instance.status,
        "started_at": instance.started_at,
        "completed_at": instance.completed_at,
        "context": instance.context or {},
        "outputs": instance.outputs or {}
    }


@router.put("/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    workflow_update: WorkflowUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Update a workflow."""
    # Get existing workflow
    result = await db.execute(
        select(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id)
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Update fields
    update_data = workflow_update.model_dump(exclude_unset=True)
    
    # If updating definition, increment version
    if "definition" in update_data:
        workflow.version += 1
    
    for field, value in update_data.items():
        setattr(workflow, field, value)
    
    await db.commit()
    await db.refresh(workflow)
    
    return WorkflowResponse.model_validate(workflow)


@router.delete("/{workflow_id}", status_code=204)
async def delete_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Delete a workflow and all related records."""
    # Get existing workflow
    result = await db.execute(
        select(WorkflowDefinition).filter(WorkflowDefinition.id == workflow_id)
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Import models that have foreign keys to workflow_definitions
    from models.workflow_execution import WorkflowExecution, WorkflowTrigger, WorkflowSchedule
    from sqlalchemy import delete
    
    # Delete related records in correct order to avoid foreign key violations
    # Use proper delete statements without select
    
    # Delete workflow executions
    await db.execute(
        delete(WorkflowExecution).where(WorkflowExecution.workflow_id == workflow_id)
    )
    
    # Delete workflow instances
    await db.execute(
        delete(WorkflowInstance).where(WorkflowInstance.definition_id == workflow_id)
    )
    
    # Delete workflow triggers
    await db.execute(
        delete(WorkflowTrigger).where(WorkflowTrigger.workflow_id == workflow_id)
    )
    
    # Delete workflow schedules
    await db.execute(
        delete(WorkflowSchedule).where(WorkflowSchedule.workflow_id == workflow_id)
    )
    
    # Now delete the workflow itself
    await db.delete(workflow)
    await db.commit()
    
    return None


@router.get("/{workflow_id}/instances", response_model=List[WorkflowInstanceResponse])
async def list_workflow_instances(
    workflow_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """List instances of a workflow."""
    # Check workflow exists
    result = await db.execute(
        select(WorkflowDefinition).filter(WorkflowDefinition.id == str(workflow_id))
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Get instances
    query = (
        select(WorkflowInstance)
        .filter(WorkflowInstance.definition_id == workflow_id)
        .options(selectinload(WorkflowInstance.steps))
        .offset(skip)
        .limit(limit)
        .order_by(WorkflowInstance.created_at.desc())
    )
    
    result = await db.execute(query)
    instances = result.scalars().all()
    
    return [WorkflowInstanceResponse.model_validate(instance) for instance in instances]


@router.post("/{workflow_id}/execute", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_id: str,
    request: dict = {},
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Execute a workflow."""
    # Check execution limit
    tenant_id = getattr(current_user, "tenant_id", None) or get_current_tenant_id()
    enforcer = LicenseEnforcer(db)
    await enforcer.check_limit(
        tenant_id=tenant_id,
        limit_type=LimitType.EXECUTIONS,
        increment=1
    )
    
    # Check workflow exists
    result = await db.execute(
        select(WorkflowDefinition).filter(WorkflowDefinition.id == str(workflow_id))
    )
    workflow = result.scalar_one_or_none()
    
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Track execution start
    import time
    start_time = time.time()
    
    # Create workflow execution using new service
    execution_service = WorkflowExecutionService(db)
    import uuid
    
    # Create execution
    execution = await execution_service.create_execution(
        workflow_id=workflow_id,  # Already a string
        input_data=request.get("input_data", {}),
        triggered_by=request.get("trigger_source", "manual"),
        trigger_metadata=request.get("trigger_metadata", {})
    )
    
    # Start execution
    execution = await execution_service.start_execution(
        execution_id=execution.id,
        agent_id=None  # Local execution
    )
    
    # Track workflow execution
    tracker = await get_usage_tracker(db)
    await tracker.track_workflow_execution(
        tenant_id=tenant_id,
        workflow_id=workflow_id,
        duration_seconds=time.time() - start_time,
        status="running",  # Execution is now async
        node_count=len(workflow.definition.get("nodes", []))
    )
    
    # Track in learning loop (Business edition feature)
    try:
        from aictrlnet_business.services.learning_loop_service import learning_loop_service
        
        # Determine initiator type
        is_ai_agent = request.get("trigger_source") == "ai_agent"
        initiated_by = "ai_agent" if is_ai_agent else "human"
        
        # Get initiator ID safely
        if hasattr(current_user, "get"):
            user_id = current_user.get("id", "unknown")
        elif hasattr(current_user, "id"):
            user_id = str(current_user.id)
        else:
            user_id = "unknown"
        
        # Convert to UUID or use default
        if user_id == "unknown" or not user_id:
            initiator_id = uuid.UUID("00000000-0000-0000-0000-000000000003")  # Default for workflow execution
        else:
            try:
                initiator_id = uuid.UUID(user_id)
            except (ValueError, TypeError):
                initiator_id = uuid.UUID("00000000-0000-0000-0000-000000000003")
        
        # Track execution start in learning loop
        learning_execution = await learning_loop_service.track_workflow_execution(
            db=db,
            workflow_id=uuid.UUID(workflow_id),
            workflow_config=workflow.definition,
            initiated_by=initiated_by,
            initiator_id=initiator_id,
            modifications_made=request.get("modifications", None)
        )
        
        # Store learning execution ID for later update
        execution.metadata = execution.metadata or {}
        execution.metadata["learning_execution_id"] = str(learning_execution.id)
        await db.commit()
    except ImportError:
        # Learning loop is a Business edition feature
        pass
    except Exception as e:
        logger.warning(f"Failed to track execution in learning loop: {str(e)}")
    
    return execution


@router.post("/{workflow_id}/pause", status_code=200)
async def pause_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Pause a running workflow."""
    # Get the latest running instance
    result = await db.execute(
        select(WorkflowInstance)
        .filter(
            WorkflowInstance.definition_id == workflow_id,
            WorkflowInstance.status == "running"
        )
        .order_by(WorkflowInstance.created_at.desc())
        .limit(1)
    )
    instance = result.scalar_one_or_none()
    
    if not instance:
        raise HTTPException(status_code=404, detail="No running workflow instance found")
    
    # Pause the instance
    instance.status = "paused"
    await db.commit()
    
    return {"message": f"Workflow {workflow_id} paused successfully"}


@router.post("/{workflow_id}/resume", status_code=200)
async def resume_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Resume a paused workflow."""
    # Get the latest paused instance
    result = await db.execute(
        select(WorkflowInstance)
        .filter(
            WorkflowInstance.definition_id == workflow_id,
            WorkflowInstance.status == "paused"
        )
        .order_by(WorkflowInstance.created_at.desc())
        .limit(1)
    )
    instance = result.scalar_one_or_none()
    
    if not instance:
        raise HTTPException(status_code=404, detail="No paused workflow instance found")
    
    # Resume the instance
    instance.status = "running"
    await db.commit()
    
    return {"message": f"Workflow {workflow_id} resumed successfully"}


@router.post("/{workflow_id}/cancel", status_code=200)
async def cancel_workflow(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Cancel a running workflow."""
    # Get the latest active instance
    result = await db.execute(
        select(WorkflowInstance)
        .filter(
            WorkflowInstance.definition_id == workflow_id,
            WorkflowInstance.status.in_(["running", "paused"])
        )
        .order_by(WorkflowInstance.created_at.desc())
        .limit(1)
    )
    instance = result.scalar_one_or_none()
    
    if not instance:
        raise HTTPException(status_code=404, detail="No active workflow instance found")
    
    # Cancel the instance
    instance.status = "cancelled"
    await db.commit()
    
    return {"message": f"Workflow {workflow_id} cancelled successfully"}


@router.get("/{workflow_id}/executions", response_model=List[WorkflowExecutionResponse])
async def list_workflow_executions(
    workflow_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """List executions of a workflow."""
    import uuid
    execution_service = WorkflowExecutionService(db)
    executions = await execution_service.list_executions(
        workflow_id=uuid.UUID(workflow_id),
        skip=skip,
        limit=limit
    )
    return executions


@router.get("/executions/{execution_id}")
async def get_execution_details(
    execution_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get detailed execution information."""
    import uuid
    execution_service = WorkflowExecutionService(db)
    details = await execution_service.get_execution_details(
        execution_id=uuid.UUID(execution_id)
    )
    return details


@router.post("/{workflow_id}/triggers", response_model=dict)
async def create_workflow_trigger(
    workflow_id: str,
    trigger_data: WorkflowTriggerCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a trigger for a workflow."""
    import uuid
    execution_service = WorkflowExecutionService(db)
    trigger = await execution_service.create_trigger(
        workflow_id=uuid.UUID(workflow_id),
        trigger_data=trigger_data
    )
    return {
        "id": str(trigger.id),
        "workflow_id": str(trigger.workflow_id),
        "trigger_type": trigger.trigger_type,
        "config": trigger.config,
        "is_active": trigger.is_active
    }


@router.get("/{workflow_id}/schedules", response_model=dict)
async def get_workflow_schedules(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get all schedules for a workflow."""
    from sqlalchemy import select
    from models.workflow_execution import WorkflowSchedule

    # Query schedules for this workflow
    # Note: workflow_id is String(36) in both tables, no UUID conversion needed
    result = await db.execute(
        select(WorkflowSchedule).where(
            WorkflowSchedule.workflow_id == workflow_id
        )
    )
    schedules = result.scalars().all()

    # Format response
    schedule_list = []
    for schedule in schedules:
        schedule_list.append({
            "id": str(schedule.id),
            "workflow_id": str(schedule.workflow_id),
            "name": getattr(schedule, 'name', None),
            "schedule_expression": schedule.schedule_expression,
            "timezone": schedule.timezone,
            "is_active": schedule.is_active,
            "trigger_type": getattr(schedule, 'trigger_type', 'schedule'),
            "next_run_at": schedule.next_run.isoformat() if schedule.next_run else None,
            "last_run": schedule.last_run.isoformat() if getattr(schedule, 'last_run', None) else None,
            "run_count": getattr(schedule, 'run_count', 0),
            "created_at": schedule.created_at.isoformat() if schedule.created_at else None,
            "updated_at": schedule.updated_at.isoformat() if schedule.updated_at else None,
        })

    return {
        "schedules": schedule_list,
        "total": len(schedule_list)
    }


@router.post("/{workflow_id}/schedules", response_model=dict)
async def create_workflow_schedule(
    workflow_id: str,
    schedule_data: WorkflowScheduleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a schedule for a workflow."""
    import uuid
    execution_service = WorkflowExecutionService(db)
    schedule = await execution_service.create_schedule(
        workflow_id=uuid.UUID(workflow_id),
        schedule_data=schedule_data
    )
    return {
        "id": str(schedule.id),
        "workflow_id": str(schedule.workflow_id),
        "schedule_expression": schedule.schedule_expression,
        "timezone": schedule.timezone,
        "is_active": schedule.is_active,
        "next_run": schedule.next_run.isoformat() if schedule.next_run else None
    }


@router.post("/validate", response_model=WorkflowValidationResult)
async def validate_workflow(
    workflow_definition: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Validate a workflow definition before saving or executing."""
    edition = getattr(current_user, "edition", "community")
    
    nodes = workflow_definition.get("nodes", [])
    edges = workflow_definition.get("edges", [])
    
    # Use catalog service for validation
    catalog_service = DynamicNodeCatalogService(db)
    validation_result = await catalog_service.validate_workflow_definition(
        nodes=nodes,
        edges=edges,
        edition=edition
    )
    
    # Add resource estimation
    if validation_result["is_valid"]:
        # Estimate resources based on nodes
        ai_nodes = sum(1 for n in nodes if n.get("type", "").startswith("ai_"))
        mcp_nodes = sum(1 for n in nodes if n.get("type", "").startswith("mcp_"))
        
        validation_result["resource_requirements"] = {
            "compute": "1 vCPU" if ai_nodes == 0 else f"{ai_nodes * 2} vCPU",
            "memory": "2GB" if ai_nodes == 0 else f"{ai_nodes * 4}GB",
            "ml_models": ai_nodes,
            "external_apis": mcp_nodes
        }
        
        # Estimate execution time (rough estimate)
        validation_result["estimated_execution_time"] = (
            len(nodes) * 2 +  # Base time per node
            ai_nodes * 10 +   # AI nodes take longer
            mcp_nodes * 5     # MCP calls add time
        )
    
    return WorkflowValidationResult(**validation_result)


@router.post("/{workflow_id}/schedule")
async def create_workflow_schedule(
    workflow_id: str,
    schedule_data: WorkflowScheduleCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a schedule for workflow execution (Business/Enterprise only)."""
    scheduler = WorkflowScheduler(db)
    
    # Try to create schedule - will fail with upgrade message if Community
    result = await scheduler.create_schedule(
        workflow_id=workflow_id,
        name=schedule_data.name,
        cron_expression=schedule_data.cron_expression,
        timezone=schedule_data.timezone,
        config=schedule_data.config
    )
    
    # Check if it's an error (Community edition)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=403, detail=result["error"])
    
    return result


@router.post("/{workflow_id}/triggers/webhook")
async def create_webhook_trigger(
    workflow_id: str,
    trigger_data: WorkflowTriggerCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create a webhook trigger for workflow (Business/Enterprise only)."""
    scheduler = WorkflowScheduler(db)
    
    # Try to create webhook trigger
    result = await scheduler.create_webhook_trigger(
        workflow_id=workflow_id,
        name=trigger_data.name,
        webhook_path=trigger_data.config.get("webhook_path", ""),
        config=trigger_data.config
    )
    
    # Check if it's an error (Community edition)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=403, detail=result["error"])
    
    return result


@router.post("/{workflow_id}/triggers/event")
async def create_event_trigger(
    workflow_id: str,
    trigger_data: WorkflowTriggerCreate,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Create an event trigger for workflow (Business/Enterprise only)."""
    scheduler = WorkflowScheduler(db)
    
    # Try to create event trigger
    result = await scheduler.create_event_trigger(
        workflow_id=workflow_id,
        name=trigger_data.name,
        event_pattern=trigger_data.config.get("event_pattern", ""),
        config=trigger_data.config
    )
    
    # Check if it's an error (Community edition)
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=403, detail=result["error"])
    
    return result


@router.get("/triggers/available")
async def get_available_triggers(
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Get available trigger types for current edition."""
    scheduler = WorkflowScheduler(db)
    return await scheduler.list_available_triggers()


@router.post("/{workflow_id}/trigger")
async def trigger_workflow_manual(
    workflow_id: str,
    trigger_data: Optional[Dict[str, Any]] = None,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_active_user),
):
    """Manually trigger a workflow execution."""
    scheduler = WorkflowScheduler(db)
    
    import uuid
    execution = await scheduler.trigger_workflow(
        workflow_id=uuid.UUID(workflow_id),
        trigger_type=TriggerType.MANUAL,
        trigger_data=trigger_data
    )
    
    return {
        "execution_id": str(execution.id),
        "workflow_id": str(execution.workflow_id),
        "status": execution.status.value,
        "triggered_by": execution.triggered_by,
        "created_at": execution.created_at.isoformat()
    }


@router.post("/create-manual")
async def create_manual_workflow(
    request_data: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user = Depends(get_current_active_user)
):
    """
    Create a workflow manually with user-selected enhancements.
    
    This endpoint supports the manual workflow creation path where users:
    1. Define the basic workflow structure (nodes, edges)
    2. Choose which enhancements to apply
    3. Get the workflow enhanced through the unified pipeline
    """
    try:
        # Extract workflow definition
        workflow_name = request_data.get("name", "Manual Workflow")
        workflow_description = request_data.get("description", "")
        workflow_category = request_data.get("category", "custom")
        workflow_definition = request_data.get("definition", {})
        
        # Extract enhancement options
        enhancements = request_data.get("enhancements", {})
        enhancement_sub_options = request_data.get("enhancementSubOptions", {})
        
        # Check if unified pipeline is available
        try:
            from aictrlnet_business.services.unified_enhancement_pipeline import unified_enhancement_pipeline
            
            # Prepare workflow for enhancement
            workflow_dict = {
                "name": workflow_name,
                "description": workflow_description,
                "definition": workflow_definition
            }
            
            # Prepare enhancement config for manual path
            user_options = enhancements.copy() if enhancements else {}
            if enhancement_sub_options:
                user_options['sub_options'] = enhancement_sub_options
            
            enhancement_config = {
                'source': 'manual',
                'automatic': False,  # Manual creation requires user choice
                'user_options': user_options,
                'context': {
                    'category': workflow_category,
                    'user_id': getattr(current_user, 'id', 'anonymous'),
                    'edition': 'community'  # Will be overridden by pipeline based on actual edition
                }
            }
            
            # Apply unified enhancement pipeline
            enhanced_workflow = await unified_enhancement_pipeline.enhance(
                workflow_dict,
                enhancement_config,
                db
            )
            
            # Extract enhanced definition
            final_definition = enhanced_workflow.get("definition", workflow_definition)
            
        except ImportError:
            logger.info("Unified enhancement pipeline not available, using basic workflow")
            final_definition = workflow_definition

        # Ensure Q/G/S/M metadata on every node (Business/Enterprise)
        try:
            from aictrlnet_business.services.qgsm import apply_qgsm

            qgsm_workflow = apply_qgsm(
                {"definition": final_definition},
                context={
                    "prompt": workflow_description or workflow_name,
                    "category": workflow_category,
                },
                edition="business",
            )
            final_definition = qgsm_workflow.get("definition", final_definition)
        except ImportError:
            pass  # Community edition — Q/G/S/M not available
        except Exception as qgsm_err:
            logger.warning(f"Q/G/S/M enforcement failed (non-critical): {qgsm_err}")

        # Create the workflow in database
        workflow = WorkflowDefinition(
            name=workflow_name,
            description=workflow_description,
            definition=final_definition,
            active=True,
            tags=["manual_created", workflow_category],
            tenant_id=getattr(current_user, 'tenant_id', None) or get_current_tenant_id()
        )
        
        db.add(workflow)
        await db.commit()
        await db.refresh(workflow)
        
        # Track in learning loop for continuous improvement (Business/Enterprise only)
        try:
            from aictrlnet_business.services.learning_loop_service import LearningLoopService
            learning_service = LearningLoopService()
            
            await learning_service.track_workflow_execution(
                db=db,
                workflow_id=workflow.id,
                workflow_config=final_definition,
                initiated_by='human',  # Manual creation is human-initiated
                initiator_id=getattr(current_user, 'id', None),
                modifications_made={
                    'source': 'manual',
                    'category': workflow_category,
                    'enhancements': enhancements,
                    'enhancements_applied': final_definition.get('metadata', {}).get('enhancements_applied', [])
                }
            )
            logger.info("✅ Tracked manual workflow creation in learning loop")
        except ImportError:
            logger.info("Learning loop not available in Community edition")
        except Exception as e:
            logger.warning(f"Could not track in learning loop (non-critical): {e}")
        
        # Now that workflow exists with an ID, connect to backend services
        try:
            from aictrlnet_business.services.workflow_metadata_connector import WorkflowMetadataConnector
            
            connector = WorkflowMetadataConnector()
            workflow_dict = {
                "id": str(workflow.id),
                "name": workflow.name,
                "definition": workflow.definition
            }
            
            connections = await connector.connect_metadata_to_hub_features(workflow_dict, db)
            logger.info(f"✅ Manual workflow connected to services: {connections.get('services_connected', [])}")
            
            # Update workflow metadata with connections
            updated_definition = dict(workflow.definition)
            if 'metadata' not in updated_definition:
                updated_definition['metadata'] = {}
            updated_definition['metadata']['hub_connections'] = connections
            workflow.definition = updated_definition
            
            db.add(workflow)
            await db.commit()
            await db.refresh(workflow)
            
        except ImportError:
            logger.info("WorkflowMetadataConnector not available (Community edition)")
        except Exception as e:
            logger.warning(f"Could not connect workflow to services (non-critical): {e}")
        
        # Track usage
        usage_tracker = await get_usage_tracker(db)
        await usage_tracker.track_usage(
            tenant_id=getattr(current_user, 'tenant_id', None) or get_current_tenant_id(),
            metric_type='workflow_manual_creation',
            value=1.0,
            metadata={'workflow_id': str(workflow.id), 'user_id': getattr(current_user, 'id', 'anonymous')}
        )
        
        return WorkflowResponse.model_validate(workflow)
        
    except Exception as e:
        logger.error(f"Error creating manual workflow: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))