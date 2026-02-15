"""Workflow execution service for orchestrating workflow runs."""

from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
from datetime import datetime, timedelta
import uuid
import asyncio
import logging
import json

from models.workflow_execution import (
    WorkflowExecution, NodeExecution, WorkflowCheckpoint,
    WorkflowTrigger, WorkflowSchedule,
    WorkflowExecutionStatus, NodeExecutionStatus
)
from models.community import WorkflowDefinition, WorkflowInstance
from models.iam import IAMAgent, IAMMessage, IAMMessageType
from schemas.workflow_execution import (
    WorkflowExecutionCreate, WorkflowExecutionResponse,
    NodeExecutionResponse, WorkflowCheckpointCreate,
    WorkflowTriggerCreate, WorkflowScheduleCreate
)
from services.iam import IAMService
from nodes.executor import NodeExecutor
from nodes.registry import node_registry
from events.event_bus import event_bus
from core.database import get_session_maker

logger = logging.getLogger(__name__)


class WorkflowExecutionService:
    """Service for managing workflow execution."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.node_executor = NodeExecutor()
        self.iam_service = IAMService(db)
    
    async def create_execution(
        self,
        workflow_id: str,
        input_data: Optional[Dict[str, Any]] = None,
        triggered_by: str = "manual",
        trigger_metadata: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecutionResponse:
        """Create a new workflow execution."""
        # Get workflow definition
        result = await self.db.execute(
            select(WorkflowDefinition)
            .where(WorkflowDefinition.id == str(workflow_id))
            .options(selectinload(WorkflowDefinition.instances))
        )
        workflow = result.scalar_one_or_none()
        
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if not workflow.active:
            raise ValueError(f"Workflow {workflow_id} is not active")
        
        # Extract dry-run flag from trigger_metadata and store in context
        is_dry_run = (trigger_metadata or {}).get("is_dry_run", False)

        # Create execution record
        execution = WorkflowExecution(
            workflow_id=str(workflow_id),
            status=WorkflowExecutionStatus.PENDING,
            input_data=input_data or {},
            triggered_by=triggered_by,
            trigger_metadata=trigger_metadata or {},
            context={
                "workflow_name": workflow.name,
                "workflow_version": workflow.version,
                "is_dry_run": is_dry_run
            }
        )
        
        self.db.add(execution)
        await self.db.commit()
        await self.db.refresh(execution)
        
        # Publish event
        await event_bus.publish(
            "workflow.execution.created",
            {
                "execution_id": str(execution.id),
                "workflow_id": str(workflow_id),
                "triggered_by": triggered_by
            }
        )
        
        return WorkflowExecutionResponse.model_validate(execution)
    
    async def start_execution(
        self,
        execution_id: uuid.UUID,
        agent_id: Optional[uuid.UUID] = None
    ) -> WorkflowExecutionResponse:
        """Start a workflow execution."""
        # Get execution with workflow
        result = await self.db.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.id == execution_id)
            .options(selectinload(WorkflowExecution.workflow))
        )
        execution = result.scalar_one_or_none()
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        if execution.status != WorkflowExecutionStatus.PENDING:
            raise ValueError(f"Execution {execution_id} is not in pending state")
        
        # Update execution status
        execution.status = WorkflowExecutionStatus.RUNNING
        execution.started_at = datetime.utcnow()
        
        # If agent_id provided, assign to specific agent
        if agent_id:
            execution.context["assigned_agent_id"] = str(agent_id)
        
        await self.db.commit()
        
        # Create node executions for all nodes in workflow
        workflow_def = json.loads(execution.workflow.definition) if isinstance(execution.workflow.definition, str) else execution.workflow.definition
        nodes = workflow_def.get("nodes", [])
        
        for node in nodes:
            node_execution = NodeExecution(
                execution_id=execution_id,
                node_id=node["id"],
                node_type=node["type"],
                status=NodeExecutionStatus.PENDING,
                execution_context=node.get("data", {})
            )
            self.db.add(node_execution)
        
        await self.db.commit()
        
        # Publish event
        await event_bus.publish(
            "workflow.execution.started",
            {
                "execution_id": str(execution_id),
                "workflow_id": str(execution.workflow_id),
                "agent_id": str(agent_id) if agent_id else None
            }
        )
        
        # Determine if dry-run mode is active
        is_dry_run = execution.context.get("is_dry_run", False) if execution.context else False

        # If local execution (no agent) or dry-run (skip agent dispatch), execute locally
        if not agent_id or is_dry_run:
            asyncio.create_task(self._execute_workflow_locally(execution_id))
        else:
            # Send execution request to agent via IAM
            await self._send_execution_to_agent(execution_id, agent_id)
        
        await self.db.refresh(execution)
        return WorkflowExecutionResponse.model_validate(execution)
    
    async def _execute_workflow_locally(self, execution_id: uuid.UUID):
        """Execute workflow locally using a dedicated DB session.

        This runs as an asyncio.create_task background task, so it must use
        its own session — the caller's session may already be mid-operation.
        """
        session_maker = get_session_maker()
        async with session_maker() as db:
            execution = None
            try:
                # Get execution with all node executions
                result = await db.execute(
                    select(WorkflowExecution)
                    .where(WorkflowExecution.id == execution_id)
                    .options(
                        selectinload(WorkflowExecution.workflow),
                        selectinload(WorkflowExecution.node_executions)
                    )
                )
                execution = result.scalar_one_or_none()

                if not execution:
                    return

                # Get workflow definition
                workflow_def = json.loads(execution.workflow.definition) if isinstance(execution.workflow.definition, str) else execution.workflow.definition

                # Thread dry-run flag from execution context into workflow variables
                is_dry_run = execution.context.get("is_dry_run", False) if execution.context else False

                # Build node instances from workflow definition
                from nodes.models import (
                    WorkflowInstance as NodeWorkflowInstance,
                    NodeInstance, NodeConfig, NodeType
                )

                nodes = workflow_def.get("nodes", [])
                edges = workflow_def.get("edges", [])

                # Build edge lookup for prev/next relationships
                next_map = {}  # node_id -> [target_ids]
                prev_map = {}  # node_id -> [source_ids]
                for edge in edges:
                    src = edge.get("from") or edge.get("source")
                    tgt = edge.get("to") or edge.get("target")
                    if src and tgt:
                        next_map.setdefault(src, []).append(tgt)
                        prev_map.setdefault(tgt, []).append(src)

                node_instances = {}
                for node_def in nodes:
                    node_id = node_def.get("id", str(uuid.uuid4()))
                    raw_type = node_def.get("type", "task")
                    parameters = node_def.get("parameters") or node_def.get("data") or {}
                    if not isinstance(parameters, dict):
                        parameters = {}

                    # Resolve node type — "custom" nodes use custom_node_type param
                    try:
                        node_type = NodeType(raw_type)
                    except ValueError:
                        node_type = NodeType.TASK  # fallback for unknown types

                    node_config = NodeConfig(
                        id=node_id,
                        name=node_def.get("name") or node_id,
                        type=node_type,
                        parameters=parameters
                    )
                    node_instances[node_id] = NodeInstance(
                        id=node_id,
                        node_config=node_config,
                        workflow_instance_id=str(execution_id),
                        previous_nodes=prev_map.get(node_id, []),
                        next_nodes=next_map.get(node_id, [])
                    )

                # Create workflow instance for node executor
                workflow_instance = NodeWorkflowInstance(
                    id=str(execution_id),
                    template_id=str(execution.workflow_id),
                    name=execution.workflow.name,
                    input_data=execution.input_data or {},
                    variables={"_is_dry_run": is_dry_run},
                    node_instances=node_instances
                )

                # Execute workflow
                await self.node_executor.execute_workflow(workflow_instance)

                # Update execution status
                execution.status = WorkflowExecutionStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
                execution.duration_ms = int((execution.completed_at - execution.started_at).total_seconds() * 1000)
                execution.output_data = workflow_instance.output_data

                await db.commit()

                # Publish completion event
                await event_bus.publish(
                    "workflow.execution.completed",
                    {
                        "execution_id": str(execution_id),
                        "workflow_id": str(execution.workflow_id),
                        "duration_ms": execution.duration_ms,
                        "triggered_by": execution.triggered_by,
                        "trigger_metadata": execution.trigger_metadata or {},
                    }
                )

            except Exception as e:
                logger.error(f"Workflow execution failed: {e}")
                if execution:
                    execution.status = WorkflowExecutionStatus.FAILED
                    execution.completed_at = datetime.utcnow()
                    execution.error_details = {"error": str(e)}
                    await db.commit()

                # Publish failure event
                await event_bus.publish(
                    "workflow.execution.failed",
                    {
                        "execution_id": str(execution_id),
                        "error": str(e)
                    }
                )
    
    async def _send_execution_to_agent(self, execution_id: uuid.UUID, agent_id: uuid.UUID):
        """Send workflow execution request to agent via IAM."""
        # Create execution request message
        message_data = {
            "recipient_id": agent_id,
            "message_type": IAMMessageType.REQUEST,
            "content": {
                "action": "execute_workflow",
                "execution_id": str(execution_id),
                "priority": "normal"
            },
            "correlation_id": f"workflow_execution_{execution_id}",
            "ttl_seconds": 3600  # 1 hour timeout
        }
        
        # Get system agent ID (workflow coordinator)
        coordinator = await self._get_or_create_coordinator_agent()
        
        # Send message via IAM
        await self.iam_service.send_message(coordinator.id, message_data)
    
    async def _get_or_create_coordinator_agent(self) -> IAMAgent:
        """Get or create the workflow coordinator agent."""
        result = await self.db.execute(
            select(IAMAgent).where(IAMAgent.name == "workflow_coordinator")
        )
        agent = result.scalar_one_or_none()
        
        if not agent:
            from schemas.iam import IAMAgentCreate
            agent_data = IAMAgentCreate(
                name="workflow_coordinator",
                agent_type="workflow",
                description="System agent for coordinating workflow execution",
                capabilities=["workflow_coordination", "execution_dispatch"],
                config={"system_agent": True}
            )
            agent = await self.iam_service.create_agent(agent_data)
        
        return agent
    
    async def update_node_execution(
        self,
        execution_id: uuid.UUID,
        node_id: str,
        status: NodeExecutionStatus,
        outputs: Optional[Dict[str, Any]] = None,
        error_details: Optional[Dict[str, Any]] = None
    ) -> NodeExecutionResponse:
        """Update node execution status."""
        result = await self.db.execute(
            select(NodeExecution)
            .where(and_(
                NodeExecution.execution_id == execution_id,
                NodeExecution.node_id == node_id
            ))
        )
        node_execution = result.scalar_one_or_none()
        
        if not node_execution:
            raise ValueError(f"Node execution not found for {node_id} in {execution_id}")
        
        # Update node execution
        node_execution.status = status
        if outputs:
            node_execution.outputs = outputs
        if error_details:
            node_execution.error_details = error_details
        
        if status == NodeExecutionStatus.RUNNING and not node_execution.started_at:
            node_execution.started_at = datetime.utcnow()
        elif status in [NodeExecutionStatus.COMPLETED, NodeExecutionStatus.FAILED]:
            node_execution.completed_at = datetime.utcnow()
            if node_execution.started_at:
                node_execution.duration_ms = int(
                    (node_execution.completed_at - node_execution.started_at).total_seconds() * 1000
                )
        
        await self.db.commit()
        await self.db.refresh(node_execution)
        
        # Publish node execution event
        await event_bus.publish(
            f"workflow.node.{status.value}",
            {
                "execution_id": str(execution_id),
                "node_id": node_id,
                "node_type": node_execution.node_type,
                "status": status.value
            }
        )
        
        return NodeExecutionResponse.model_validate(node_execution)
    
    async def create_checkpoint(
        self,
        execution_id: uuid.UUID,
        checkpoint_data: WorkflowCheckpointCreate
    ) -> WorkflowCheckpoint:
        """Create a workflow checkpoint."""
        # Verify execution exists
        result = await self.db.execute(
            select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
        )
        execution = result.scalar_one_or_none()
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        # Create checkpoint
        checkpoint = WorkflowCheckpoint(
            execution_id=execution_id,
            **checkpoint_data.model_dump()
        )
        
        self.db.add(checkpoint)
        await self.db.commit()
        await self.db.refresh(checkpoint)
        
        return checkpoint
    
    async def pause_execution(self, execution_id: uuid.UUID) -> WorkflowExecutionResponse:
        """Pause a running workflow execution."""
        result = await self.db.execute(
            select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
        )
        execution = result.scalar_one_or_none()
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        if execution.status != WorkflowExecutionStatus.RUNNING:
            raise ValueError(f"Execution {execution_id} is not running")
        
        # Update status
        execution.status = WorkflowExecutionStatus.PAUSED
        
        # Create automatic checkpoint
        checkpoint_data = WorkflowCheckpointCreate(
            checkpoint_type="auto",
            state_data={"paused_at": datetime.utcnow().isoformat()},
            description="Automatic checkpoint on pause"
        )
        await self.create_checkpoint(execution_id, checkpoint_data)
        
        await self.db.commit()
        await self.db.refresh(execution)
        
        # Publish pause event
        await event_bus.publish(
            "workflow.execution.paused",
            {
                "execution_id": str(execution_id),
                "workflow_id": str(execution.workflow_id)
            }
        )
        
        return WorkflowExecutionResponse.model_validate(execution)
    
    async def resume_execution(
        self,
        execution_id: uuid.UUID,
        checkpoint_id: Optional[uuid.UUID] = None
    ) -> WorkflowExecutionResponse:
        """Resume a paused workflow execution."""
        result = await self.db.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.id == execution_id)
            .options(selectinload(WorkflowExecution.checkpoints))
        )
        execution = result.scalar_one_or_none()
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        if execution.status != WorkflowExecutionStatus.PAUSED:
            raise ValueError(f"Execution {execution_id} is not paused")
        
        # If checkpoint specified, restore from it
        if checkpoint_id:
            checkpoint = next(
                (cp for cp in execution.checkpoints if cp.id == checkpoint_id),
                None
            )
            if checkpoint:
                execution.context.update(checkpoint.state_data)
        
        # Update status
        execution.status = WorkflowExecutionStatus.RESUMING
        await self.db.commit()
        
        # Resume execution
        asyncio.create_task(self._execute_workflow_locally(execution_id))
        
        await self.db.refresh(execution)
        return WorkflowExecutionResponse.model_validate(execution)
    
    async def cancel_execution(self, execution_id: uuid.UUID) -> WorkflowExecutionResponse:
        """Cancel a workflow execution."""
        result = await self.db.execute(
            select(WorkflowExecution).where(WorkflowExecution.id == execution_id)
        )
        execution = result.scalar_one_or_none()
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        if execution.status in [WorkflowExecutionStatus.COMPLETED, WorkflowExecutionStatus.FAILED]:
            raise ValueError(f"Execution {execution_id} is already finished")
        
        # Update status
        execution.status = WorkflowExecutionStatus.CANCELLED
        execution.completed_at = datetime.utcnow()
        
        # Cancel all pending node executions
        await self.db.execute(
            select(NodeExecution)
            .where(and_(
                NodeExecution.execution_id == execution_id,
                NodeExecution.status.in_([NodeExecutionStatus.PENDING, NodeExecutionStatus.RUNNING])
            ))
            .execution_options(synchronize_session="fetch")
        )
        
        await self.db.commit()
        await self.db.refresh(execution)
        
        # Publish cancellation event
        await event_bus.publish(
            "workflow.execution.cancelled",
            {
                "execution_id": str(execution_id),
                "workflow_id": str(execution.workflow_id)
            }
        )
        
        return WorkflowExecutionResponse.model_validate(execution)
    
    async def list_executions(
        self,
        workflow_id: Optional[uuid.UUID] = None,
        status: Optional[WorkflowExecutionStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[WorkflowExecutionResponse]:
        """List workflow executions with filters."""
        query = select(WorkflowExecution).options(
            selectinload(WorkflowExecution.workflow)
        )
        
        if workflow_id:
            query = query.where(WorkflowExecution.workflow_id == workflow_id)
        if status:
            query = query.where(WorkflowExecution.status == status)
        
        query = query.order_by(WorkflowExecution.created_at.desc())
        query = query.offset(skip).limit(limit)
        
        result = await self.db.execute(query)
        executions = result.scalars().all()
        
        return [WorkflowExecutionResponse.model_validate(ex) for ex in executions]
    
    async def get_execution_details(
        self,
        execution_id: uuid.UUID
    ) -> Dict[str, Any]:
        """Get detailed execution information including all node executions."""
        result = await self.db.execute(
            select(WorkflowExecution)
            .where(WorkflowExecution.id == execution_id)
            .options(
                selectinload(WorkflowExecution.workflow),
                selectinload(WorkflowExecution.node_executions),
                selectinload(WorkflowExecution.checkpoints)
            )
        )
        execution = result.scalar_one_or_none()
        
        if not execution:
            raise ValueError(f"Execution {execution_id} not found")
        
        return {
            "execution": WorkflowExecutionResponse.model_validate(execution),
            "node_executions": [
                NodeExecutionResponse.model_validate(ne) 
                for ne in execution.node_executions
            ],
            "checkpoints": [
                {
                    "id": str(cp.id),
                    "type": cp.checkpoint_type,
                    "created_at": cp.created_at.isoformat(),
                    "description": cp.description
                }
                for cp in execution.checkpoints
            ]
        }
    
    async def create_trigger(
        self,
        workflow_id: uuid.UUID,
        trigger_data: WorkflowTriggerCreate
    ) -> WorkflowTrigger:
        """Create a workflow trigger."""
        trigger = WorkflowTrigger(
            workflow_id=str(workflow_id),
            **trigger_data.model_dump()
        )
        
        self.db.add(trigger)
        await self.db.commit()
        await self.db.refresh(trigger)
        
        return trigger
    
    async def create_schedule(
        self,
        workflow_id: uuid.UUID,
        schedule_data: WorkflowScheduleCreate
    ) -> WorkflowSchedule:
        """Create a workflow schedule."""
        schedule = WorkflowSchedule(
            workflow_id=str(workflow_id),
            **schedule_data.model_dump()
        )
        
        # Calculate next run time
        # This would use a cron parser in production
        schedule.next_run = datetime.utcnow() + timedelta(hours=1)
        
        self.db.add(schedule)
        await self.db.commit()
        await self.db.refresh(schedule)
        
        return schedule