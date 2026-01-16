"""
Progressive Executor Service for Intelligent Assistant.

Executes action plans step-by-step with checkpoints, user confirmations,
and rollback capabilities for safe and controlled execution.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from services.action_planner import ActionPlan, ActionStep, StepType

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Status of execution."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    WAITING_CONFIRMATION = "waiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    CANCELLED = "cancelled"


class CheckpointType(str, Enum):
    """Types of execution checkpoints."""
    USER_CONFIRMATION = "user_confirmation"
    AUTOMATIC = "automatic"
    ERROR_RECOVERY = "error_recovery"
    ROLLBACK_POINT = "rollback_point"


@dataclass
class ExecutionCheckpoint:
    """Represents a checkpoint during execution."""
    id: str = field(default_factory=lambda: str(uuid4()))
    step_id: str = ""
    type: CheckpointType = CheckpointType.AUTOMATIC
    timestamp: datetime = field(default_factory=datetime.utcnow)
    state: Dict[str, Any] = field(default_factory=dict)
    message: str = ""
    requires_input: bool = False
    can_rollback: bool = True


@dataclass
class StepResult:
    """Result of executing a single step."""
    step_id: str
    status: ExecutionStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    rollback_data: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionProgress:
    """Track overall execution progress."""
    plan_id: str
    current_step_index: int = 0
    total_steps: int = 0
    status: ExecutionStatus = ExecutionStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    checkpoints: List[ExecutionCheckpoint] = field(default_factory=list)
    results: List[StepResult] = field(default_factory=list)
    is_paused: bool = False
    can_resume: bool = True


class ProgressiveExecutor:
    """
    Executes action plans progressively with checkpoints and rollback.
    Provides safe, controlled execution with user confirmations.
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self._execution_handlers = self._initialize_handlers()
        self._rollback_handlers = self._initialize_rollback_handlers()
        self._active_executions: Dict[str, ExecutionProgress] = {}

    def _initialize_handlers(self) -> Dict[StepType, Callable]:
        """Initialize step execution handlers."""
        return {
            StepType.CREATE_RESOURCE: self._execute_create_resource,
            StepType.CONFIGURE: self._execute_configure,
            StepType.CONNECT: self._execute_connect,
            StepType.ENABLE_FEATURE: self._execute_enable_feature,
            StepType.VALIDATE: self._execute_validate,
            StepType.DEPLOY: self._execute_deploy,
        }

    def _initialize_rollback_handlers(self) -> Dict[StepType, Callable]:
        """Initialize rollback handlers."""
        return {
            StepType.CREATE_RESOURCE: self._rollback_create_resource,
            StepType.CONFIGURE: self._rollback_configure,
            StepType.CONNECT: self._rollback_connect,
            StepType.ENABLE_FEATURE: self._rollback_enable_feature,
            StepType.VALIDATE: self._rollback_validate,
            StepType.DEPLOY: self._rollback_deploy,
        }

    async def execute_plan(
        self,
        plan: ActionPlan,
        context: Dict[str, Any],
        confirm_callback: Optional[Callable] = None
    ) -> ExecutionProgress:
        """
        Execute an action plan progressively.

        Args:
            plan: The action plan to execute
            context: Execution context with user session info
            confirm_callback: Async callback for user confirmations

        Returns:
            ExecutionProgress with complete execution details
        """
        progress = ExecutionProgress(
            plan_id=plan.id,
            total_steps=len(plan.steps),
            status=ExecutionStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        self._active_executions[plan.id] = progress

        try:
            for i, step in enumerate(plan.steps):
                progress.current_step_index = i

                # Check if execution was cancelled
                if progress.status == ExecutionStatus.CANCELLED:
                    logger.info(f"Execution cancelled at step {i}: {step.name}")
                    break

                # Create checkpoint before step if needed
                if step.requires_confirmation:
                    checkpoint = await self._create_checkpoint(
                        step, CheckpointType.USER_CONFIRMATION, progress
                    )
                    
                    if confirm_callback:
                        confirmed = await confirm_callback({
                            "step": step.name,
                            "description": step.description,
                            "params": step.params,
                            "checkpoint_id": checkpoint.id
                        })
                        
                        if not confirmed:
                            progress.status = ExecutionStatus.CANCELLED
                            logger.info(f"User cancelled at step: {step.name}")
                            break

                # Execute the step
                result = await self._execute_step(step, context, progress)
                progress.results.append(result)

                if result.status == ExecutionStatus.COMPLETED:
                    progress.completed_steps.append(step.id)
                    logger.info(f"Completed step {i+1}/{len(plan.steps)}: {step.name}")
                    
                    # Create automatic checkpoint after successful step
                    await self._create_checkpoint(
                        step, CheckpointType.AUTOMATIC, progress,
                        message=f"Successfully completed: {step.name}"
                    )
                    
                elif result.status == ExecutionStatus.FAILED:
                    progress.failed_steps.append(step.id)
                    logger.error(f"Failed step {i+1}/{len(plan.steps)}: {step.name}")
                    
                    # Create error recovery checkpoint
                    checkpoint = await self._create_checkpoint(
                        step, CheckpointType.ERROR_RECOVERY, progress,
                        message=f"Error in {step.name}: {result.error}"
                    )
                    
                    # Ask user whether to retry, skip, or rollback
                    if confirm_callback:
                        action = await confirm_callback({
                            "type": "error_recovery",
                            "step": step.name,
                            "error": result.error,
                            "options": ["retry", "skip", "rollback"],
                            "checkpoint_id": checkpoint.id
                        })
                        
                        if action == "rollback":
                            await self._rollback_to_checkpoint(checkpoint, progress)
                            progress.status = ExecutionStatus.ROLLED_BACK
                            break
                        elif action == "skip":
                            continue
                        elif action == "retry":
                            # Retry the step
                            result = await self._execute_step(step, context, progress)
                            if result.status == ExecutionStatus.COMPLETED:
                                progress.completed_steps.append(step.id)
                            else:
                                progress.status = ExecutionStatus.FAILED
                                break
                    else:
                        # No callback, fail the execution
                        progress.status = ExecutionStatus.FAILED
                        break

                # Add delay between steps if configured
                if i < len(plan.steps) - 1:  # Not the last step
                    await asyncio.sleep(0.5)  # Small delay for better UX

            # Set final status if not already set
            if progress.status == ExecutionStatus.IN_PROGRESS:
                if len(progress.completed_steps) == len(plan.steps):
                    progress.status = ExecutionStatus.COMPLETED
                else:
                    progress.status = ExecutionStatus.FAILED

            progress.completed_at = datetime.utcnow()

        except Exception as e:
            logger.error(f"Unexpected error during execution: {str(e)}")
            progress.status = ExecutionStatus.FAILED
            progress.completed_at = datetime.utcnow()

        finally:
            # Clean up active execution
            if plan.id in self._active_executions:
                del self._active_executions[plan.id]

        return progress

    async def _execute_step(
        self,
        step: ActionStep,
        context: Dict[str, Any],
        progress: ExecutionProgress
    ) -> StepResult:
        """Execute a single step."""
        result = StepResult(
            step_id=step.id,
            status=ExecutionStatus.IN_PROGRESS,
            started_at=datetime.utcnow()
        )

        try:
            # Get appropriate handler for step type
            handler = self._execution_handlers.get(step.type)
            if not handler:
                raise ValueError(f"No handler for step type: {step.type}")

            # Execute with handler
            output = await handler(step, context)
            
            result.output = output
            result.status = ExecutionStatus.COMPLETED
            result.completed_at = datetime.utcnow()
            
            # Store rollback data if step is reversible
            if step.rollback_action:
                result.rollback_data = {
                    "action": step.rollback_action,
                    "original_params": step.params,
                    "output": output
                }

        except Exception as e:
            logger.error(f"Step execution failed: {str(e)}")
            result.status = ExecutionStatus.FAILED
            result.error = str(e)
            result.completed_at = datetime.utcnow()

        return result

    async def _create_checkpoint(
        self,
        step: ActionStep,
        checkpoint_type: CheckpointType,
        progress: ExecutionProgress,
        message: str = ""
    ) -> ExecutionCheckpoint:
        """Create an execution checkpoint."""
        checkpoint = ExecutionCheckpoint(
            step_id=step.id,
            type=checkpoint_type,
            message=message or f"Checkpoint for {step.name}",
            requires_input=checkpoint_type == CheckpointType.USER_CONFIRMATION,
            can_rollback=bool(step.rollback_action),
            state={
                "step_index": progress.current_step_index,
                "completed_steps": progress.completed_steps.copy(),
                "results": len(progress.results)
            }
        )
        
        progress.checkpoints.append(checkpoint)
        return checkpoint

    async def _rollback_to_checkpoint(
        self,
        checkpoint: ExecutionCheckpoint,
        progress: ExecutionProgress
    ) -> bool:
        """
        Rollback execution to a specific checkpoint.

        Args:
            checkpoint: The checkpoint to rollback to
            progress: Current execution progress

        Returns:
            True if rollback successful, False otherwise
        """
        logger.info(f"Rolling back to checkpoint: {checkpoint.id}")
        
        # Get steps to rollback (completed after checkpoint)
        steps_to_rollback = progress.completed_steps[
            checkpoint.state["completed_steps"]:
        ]
        
        rollback_success = True
        for step_id in reversed(steps_to_rollback):
            # Find the result for this step
            result = next((r for r in progress.results if r.step_id == step_id), None)
            if not result or not result.rollback_data:
                logger.warning(f"Cannot rollback step {step_id}: no rollback data")
                continue
                
            # Find the original step
            step = next((s for s in progress.plan.steps if s.id == step_id), None)
            if not step:
                continue
                
            try:
                # Get rollback handler
                handler = self._rollback_handlers.get(step.type)
                if handler:
                    await handler(step, result.rollback_data)
                    logger.info(f"Rolled back step: {step.name}")
                else:
                    logger.warning(f"No rollback handler for step type: {step.type}")
                    
            except Exception as e:
                logger.error(f"Rollback failed for step {step.name}: {str(e)}")
                rollback_success = False
                
        # Update progress state
        progress.completed_steps = checkpoint.state["completed_steps"].copy()
        progress.current_step_index = checkpoint.state["step_index"]
        
        return rollback_success

    async def pause_execution(self, plan_id: str) -> bool:
        """Pause an active execution."""
        if plan_id in self._active_executions:
            progress = self._active_executions[plan_id]
            progress.is_paused = True
            logger.info(f"Paused execution: {plan_id}")
            return True
        return False

    async def resume_execution(self, plan_id: str) -> bool:
        """Resume a paused execution."""
        if plan_id in self._active_executions:
            progress = self._active_executions[plan_id]
            if progress.is_paused and progress.can_resume:
                progress.is_paused = False
                logger.info(f"Resumed execution: {plan_id}")
                return True
        return False

    async def cancel_execution(self, plan_id: str) -> bool:
        """Cancel an active execution."""
        if plan_id in self._active_executions:
            progress = self._active_executions[plan_id]
            progress.status = ExecutionStatus.CANCELLED
            logger.info(f"Cancelled execution: {plan_id}")
            return True
        return False

    def get_execution_progress(self, plan_id: str) -> Optional[ExecutionProgress]:
        """Get current execution progress."""
        return self._active_executions.get(plan_id)

    # Step execution handlers
    async def _execute_create_resource(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute resource creation step."""
        # Try to get resource type from params, or infer from step name/context
        resource_type = step.params.get('type')

        # If not in params, infer from step name or context
        if not resource_type or resource_type == 'resource':
            step_name_lower = step.name.lower()
            if 'workflow' in step_name_lower:
                resource_type = 'workflow'
            elif 'agent' in step_name_lower:
                resource_type = 'agent'
            elif 'integration' in step_name_lower or 'connect' in step_name_lower:
                resource_type = 'integration'
            elif 'company' in step_name_lower or 'organization' in step_name_lower:
                resource_type = 'company_automation'
            else:
                # Check context for intent
                user_query = context.get('user_query', '').lower()
                if 'agent' in user_query:
                    resource_type = 'agent'
                elif 'workflow' in user_query:
                    resource_type = 'workflow'
                elif 'integrate' in user_query or 'connect' in user_query:
                    resource_type = 'integration'
                else:
                    resource_type = 'resource'  # Keep default

        logger.info(f"Creating resource: {resource_type} (inferred from step: {step.name})")

        # Route to appropriate handler based on resource type
        if resource_type == 'workflow':
            return await self._create_workflow(step, context)
        elif resource_type == 'agent':
            return await self._create_agent(step, context)
        elif resource_type == 'integration':
            return await self._create_integration(step, context)
        elif resource_type == 'company_automation':
            # Company automation requires Business edition or higher
            return await self._escalate_to_higher_edition(
                feature='company_automation',
                step=step,
                context=context
            )
        else:
            # Unknown resource type
            raise ValueError(f"Unsupported resource type: {resource_type}")

    async def _create_workflow(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create workflow using NLP service."""
        from services.nlp import NLPService
        nlp_service = NLPService(self.db)

        # Extract workflow parameters - prefer original user query for better context
        workflow_prompt = (
            step.params.get('description') or
            context.get('user_query') or
            step.description or
            f"Create a workflow for: {step.params.get('name', step.name)}"
        )
        user_id = context.get('user_id', 'anonymous')

        logger.info(f"Creating workflow via NLP service: {workflow_prompt}")

        # Use NLP service to create the actual workflow
        result = await nlp_service.process_natural_language(
            prompt=workflow_prompt,
            context=context,
            return_transparency=True,
            user_id=user_id
        )

        if result.get('workflow'):
            workflow = result['workflow']
            logger.info(f"Workflow created successfully: {workflow.get('id')}")
            return {
                "workflow_id": workflow.get('id'),
                "type": "workflow",
                "name": workflow.get('name', step.name),
                "created_at": datetime.utcnow().isoformat(),
                "workflow": workflow  # Include full workflow data for frontend navigation
            }
        else:
            logger.error(f"Failed to create workflow: {result.get('error')}")
            raise Exception(f"Workflow creation failed: {result.get('error', 'Unknown error')}")

    async def _create_agent(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create basic AI agent in Community Edition."""
        from services.basic_agent_service import BasicAgentService

        # Get user info
        user_id = context.get('user_id')
        if not user_id:
            # Try to get from session if available
            user = context.get('user')
            if user:
                user_id = user.get('id') if isinstance(user, dict) else getattr(user, 'id', None)

        if not user_id:
            logger.warning("No user_id in context, using anonymous")
            user_id = 'anonymous'

        agent_service = BasicAgentService(self.db)

        # Extract agent parameters
        agent_name = step.params.get('name', step.name)
        agent_type = step.params.get('agent_type', 'assistant')
        description = step.params.get('description', step.description)
        capabilities = step.params.get('capabilities', ['chat', 'assist'])

        logger.info(f"Creating basic agent: {agent_name} (type: {agent_type})")

        # Create agent using service
        result = await agent_service.create_agent_from_progressive_executor(
            user_id=user_id,
            name=agent_name,
            agent_type=agent_type,
            description=description,
            capabilities=capabilities
        )

        if result['success']:
            return {
                "agent_id": result['agent_id'],
                "type": "agent",
                "name": result['name'],
                "created_at": datetime.utcnow().isoformat(),
                "agent": {
                    "id": result['agent_id'],
                    "name": result['name'],
                    "type": result['type'],
                    "model": result['model'],
                    "capabilities": result['capabilities']
                },
                "message": result['message']
            }
        else:
            # Handle failure (could be limit reached)
            if result.get('requires_upgrade'):
                # Escalate to Business Edition
                raise Exception(
                    f"Agent creation limit reached. {result['message']} "
                    "Upgrade to Business Edition for up to 50 agents with enhanced capabilities."
                )
            else:
                raise Exception(f"Failed to create agent: {result['message']}")

    async def _create_integration(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create platform integration credential."""
        from models.platform_integration import PlatformCredential, PlatformType, AuthMethod
        from core.services.credential_service import CredentialService

        # Extract integration parameters
        platform_name = step.params.get('platform', 'n8n')
        integration_name = step.params.get('name', step.name)

        # Get user info
        user_id = context.get('user_id', 'anonymous')

        logger.info(f"Creating platform integration: {integration_name} ({platform_name})")

        # Create credential record
        credential = PlatformCredential(
            name=integration_name,
            platform=platform_name,
            auth_method=step.params.get('auth_method', 'api_key'),
            encrypted_data="",  # Will be encrypted by credential service
            user_id=user_id,
            is_shared=step.params.get('is_shared', False),
            config_metadata=step.params.get('config', {})
        )

        self.db.add(credential)
        await self.db.commit()
        await self.db.refresh(credential)

        return {
            "integration_id": str(credential.id),
            "type": "integration",
            "name": credential.name,
            "created_at": credential.created_at.isoformat() if hasattr(credential.created_at, 'isoformat') else datetime.utcnow().isoformat(),
            "integration": {
                "id": str(credential.id),
                "name": credential.name,
                "platform": credential.platform,
                "auth_method": credential.auth_method,
                "status": "active"
            }
        }

    async def _escalate_to_higher_edition(
        self,
        feature: str,
        step: ActionStep,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle edition escalation for features not available in Community."""
        from services.edition_boundary_detector import EditionBoundaryDetector, EditionTier

        detector = EditionBoundaryDetector()
        required_tier = detector.get_required_edition(feature)

        logger.info(f"Feature '{feature}' requires {required_tier.value} edition")

        # Return escalation message
        raise Exception(
            f"The '{feature}' feature requires {required_tier.value.title()} Edition or higher. "
            f"Please upgrade to access this functionality. "
            f"Contact sales@aictrlnet.com for more information."
        )

    async def _execute_configure(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute configuration step."""
        logger.info(f"Configuring: {step.name}")
        
        # Quick simulated delay for UX feedback (not the full duration)
        await asyncio.sleep(min(0.5, step.duration_seconds))
        
        return {
            "configuration": step.params,
            "applied_at": datetime.utcnow().isoformat()
        }

    async def _execute_connect(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute connection step."""
        logger.info(f"Connecting: {step.params.get('channels', [])}")
        
        # Quick simulated delay for UX feedback (not the full duration)
        await asyncio.sleep(min(0.5, step.duration_seconds))
        
        return {
            "connected_channels": step.params.get("channels", []),
            "connection_status": "active",
            "connected_at": datetime.utcnow().isoformat()
        }

    async def _execute_enable_feature(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute feature enablement step."""
        logger.info(f"Enabling feature: {step.params.get('feature')}")
        
        # Quick simulated delay for UX feedback (not the full duration)
        await asyncio.sleep(min(0.5, step.duration_seconds))
        
        return {
            "feature": step.params.get("feature"),
            "enabled": True,
            "settings": step.params,
            "enabled_at": datetime.utcnow().isoformat()
        }

    async def _execute_validate(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation step."""
        logger.info(f"Validating: {step.name}")
        
        # Quick simulated delay for UX feedback (not the full duration)
        await asyncio.sleep(min(0.5, step.duration_seconds))
        
        return {
            "validation_passed": True,
            "checks": step.params.get("checks", []),
            "validated_at": datetime.utcnow().isoformat()
        }

    async def _execute_deploy(self, step: ActionStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment step."""
        logger.info(f"Deploying: {step.name}")
        
        # Quick simulated delay for UX feedback (not the full duration)
        await asyncio.sleep(min(0.5, step.duration_seconds))
        
        return {
            "deployment_id": str(uuid4()),
            "environment": step.params.get("environment", "production"),
            "deployed_at": datetime.utcnow().isoformat()
        }

    # Rollback handlers
    async def _rollback_create_resource(self, step: ActionStep, rollback_data: Dict[str, Any]) -> None:
        """Rollback resource creation."""
        resource_id = rollback_data.get("output", {}).get("resource_id")
        logger.info(f"Deleting resource: {resource_id}")
        await asyncio.sleep(2)  # Simulate deletion

    async def _rollback_configure(self, step: ActionStep, rollback_data: Dict[str, Any]) -> None:
        """Rollback configuration."""
        logger.info(f"Resetting configuration for: {step.name}")
        await asyncio.sleep(2)  # Simulate reset

    async def _rollback_connect(self, step: ActionStep, rollback_data: Dict[str, Any]) -> None:
        """Rollback connection."""
        channels = rollback_data.get("output", {}).get("connected_channels", [])
        logger.info(f"Disconnecting channels: {channels}")
        await asyncio.sleep(2)  # Simulate disconnection

    async def _rollback_enable_feature(self, step: ActionStep, rollback_data: Dict[str, Any]) -> None:
        """Rollback feature enablement."""
        feature = rollback_data.get("output", {}).get("feature")
        logger.info(f"Disabling feature: {feature}")
        await asyncio.sleep(2)  # Simulate disabling

    async def _rollback_validate(self, step: ActionStep, rollback_data: Dict[str, Any]) -> None:
        """Rollback validation (usually no-op)."""
        logger.info(f"Validation rollback for: {step.name} (no-op)")

    async def _rollback_deploy(self, step: ActionStep, rollback_data: Dict[str, Any]) -> None:
        """Rollback deployment."""
        deployment_id = rollback_data.get("output", {}).get("deployment_id")
        logger.info(f"Rolling back deployment: {deployment_id}")
        await asyncio.sleep(3)  # Simulate rollback