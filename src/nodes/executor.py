"""Node executor for running workflow nodes."""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
import time

from .models import (
    NodeInstance, NodeStatus, NodeExecutionRequest, 
    NodeExecutionResult, WorkflowInstance
)
from .registry import node_registry
from .state_manager import NodeStateManager, state_manager as default_state_manager
from .error_handler import NodeErrorHandler, node_error_handler as default_error_handler
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class NodeExecutor:
    """Executor for running nodes in workflows."""
    
    def __init__(
        self, 
        max_parallel_nodes: int = 10,
        state_manager: Optional[NodeStateManager] = None,
        error_handler: Optional[NodeErrorHandler] = None
    ):
        self.max_parallel_nodes = max_parallel_nodes
        self._execution_semaphore = asyncio.Semaphore(max_parallel_nodes)
        self._running_nodes: Set[str] = set()
        self._lock = asyncio.Lock()
        
        # Use provided managers or defaults
        self.state_manager = state_manager or default_state_manager
        self.error_handler = error_handler or default_error_handler
    
    async def execute_workflow(
        self,
        workflow_instance: WorkflowInstance
    ) -> WorkflowInstance:
        """Execute an entire workflow."""
        start_time = time.time()
        
        try:
            # Load previous state if resuming
            previous_state = await self.state_manager.load_workflow_state(workflow_instance)
            if previous_state and workflow_instance.status == "resuming":
                logger.info(f"Resuming workflow {workflow_instance.id} from previous state")
                workflow_instance.variables.update(previous_state.get("variables", {}))
            
            # Update workflow status
            workflow_instance.status = "running"
            workflow_instance.started_at = datetime.utcnow()
            
            # Save initial workflow state
            await self.state_manager.save_workflow_state(
                workflow_instance,
                {
                    "status": "running",
                    "started_at": workflow_instance.started_at.isoformat(),
                    "variables": workflow_instance.variables
                }
            )
            
            # Publish workflow start event
            await event_bus.publish(
                "workflow.started",
                {
                    "workflow_id": workflow_instance.id,
                    "template_id": workflow_instance.template_id,
                    "name": workflow_instance.name
                },
                source_id=workflow_instance.id,
                source_type="workflow"
            )
            
            # Find start nodes
            start_nodes = self._find_start_nodes(workflow_instance)
            if not start_nodes:
                raise ValueError("No start nodes found in workflow")
            
            # Execute workflow
            await self._execute_from_nodes(
                workflow_instance,
                start_nodes,
                workflow_instance.input_data
            )
            
            # Collect output from end nodes
            output_data = self._collect_workflow_output(workflow_instance)
            workflow_instance.output_data = output_data
            
            # Update workflow status
            workflow_instance.status = "completed"
            workflow_instance.completed_at = datetime.utcnow()
            workflow_instance.duration_ms = (time.time() - start_time) * 1000
            
            # Save final workflow state
            await self.state_manager.save_workflow_state(
                workflow_instance,
                {
                    "status": "completed",
                    "completed_at": workflow_instance.completed_at.isoformat(),
                    "duration_ms": workflow_instance.duration_ms,
                    "output_data": output_data
                }
            )
            
            # Publish completion event
            await event_bus.publish(
                "workflow.completed",
                {
                    "workflow_id": workflow_instance.id,
                    "duration_ms": workflow_instance.duration_ms,
                    "nodes_executed": len(workflow_instance.node_instances)
                },
                source_id=workflow_instance.id,
                source_type="workflow"
            )
            
            return workflow_instance
            
        except Exception as e:
            # Handle workflow failure with error handler
            await self.error_handler.handle_workflow_error(
                e, workflow_instance
            )
            
            workflow_instance.status = "failed"
            workflow_instance.status_message = str(e)
            workflow_instance.completed_at = datetime.utcnow()
            workflow_instance.duration_ms = (time.time() - start_time) * 1000
            
            await event_bus.publish(
                "workflow.failed",
                {
                    "workflow_id": workflow_instance.id,
                    "error": str(e)
                },
                source_id=workflow_instance.id,
                source_type="workflow"
            )
            
            raise
    
    async def execute_node(
        self,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance
    ) -> NodeExecutionResult:
        """Execute a single node."""
        async with self._execution_semaphore:
            # Check if node is already running
            async with self._lock:
                if node_instance.id in self._running_nodes:
                    logger.warning(f"Node {node_instance.id} is already running")
                    return NodeExecutionResult(
                        node_instance_id=node_instance.id,
                        status=NodeStatus.RUNNING,
                        error="Node is already running"
                    )
                
                self._running_nodes.add(node_instance.id)
            
            try:
                # Load previous node state if exists
                previous_state = await self.state_manager.load_node_state(
                    node_instance, workflow_instance.id
                )
                if previous_state and node_instance.status == NodeStatus.PENDING:
                    logger.info(f"Loading previous state for node {node_instance.id}")
                    node_instance.state_data = previous_state
                
                # Update node status
                node_instance.status = NodeStatus.RUNNING
                node_instance.started_at = datetime.utcnow()
                
                # Save node running state
                await self.state_manager.save_node_state(
                    node_instance,
                    {
                        "status": "running",
                        "started_at": node_instance.started_at.isoformat(),
                        "input_data": node_instance.input_data
                    },
                    workflow_instance.id
                )
                
                # Create node from config
                node = node_registry.create_node(node_instance.node_config)
                
                # Run the node with error handling
                try:
                    result = await node.run(
                        node_instance,
                        workflow_instance.variables
                    )
                except Exception as node_error:
                    # Handle node execution error
                    result = await self.error_handler.handle_node_error(
                        node_error, node_instance, workflow_instance
                    )
                
                # Save node completion state
                await self.state_manager.save_node_state(
                    node_instance,
                    {
                        "status": result.status.value if hasattr(result.status, 'value') else str(result.status),
                        "completed_at": datetime.utcnow().isoformat(),
                        "output_data": node_instance.output_data,
                        "result": result.dict() if hasattr(result, 'dict') else str(result)
                    },
                    workflow_instance.id
                )
                
                # Update workflow variables if needed
                if hasattr(node_instance, 'config') and hasattr(node_instance.config, 'outputs'):
                    for output_var in node_instance.config.outputs:
                        if output_var in node_instance.output_data:
                            workflow_instance.variables[output_var] = (
                                node_instance.output_data[output_var]
                            )
                
                return result
                
            except Exception as e:
                # Handle unexpected errors
                logger.error(f"Unexpected error in node {node_instance.id}: {e}")
                return await self.error_handler.handle_node_error(
                    e, node_instance, workflow_instance
                )
                
            finally:
                # Remove from running nodes
                async with self._lock:
                    self._running_nodes.discard(node_instance.id)
    
    async def _execute_from_nodes(
        self,
        workflow_instance: WorkflowInstance,
        node_ids: List[str],
        initial_data: Dict[str, Any]
    ):
        """Execute workflow starting from given nodes."""
        # Queue of nodes to execute
        execution_queue = asyncio.Queue()
        
        # Add initial nodes to queue
        for node_id in node_ids:
            await execution_queue.put((node_id, initial_data))
        
        # Track completed nodes
        completed_nodes = set()
        
        # Process nodes
        while not execution_queue.empty():
            # Get batch of nodes to execute in parallel
            batch = []
            batch_size = min(self.max_parallel_nodes, execution_queue.qsize())
            
            for _ in range(batch_size):
                try:
                    node_id, input_data = execution_queue.get_nowait()
                    
                    # Skip if already completed
                    if node_id in completed_nodes:
                        continue
                    
                    node_instance = workflow_instance.node_instances.get(node_id)
                    if node_instance:
                        node_instance.input_data = input_data
                        batch.append(node_instance)
                        
                except asyncio.QueueEmpty:
                    break
            
            if not batch:
                continue
            
            # Execute batch in parallel
            tasks = [
                self.execute_node(node_instance, workflow_instance)
                for node_instance in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, result in enumerate(results):
                node_instance = batch[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Node {node_instance.id} failed with exception: {result}")
                    continue

                # Mark as completed
                completed_nodes.add(node_instance.id)

                # Add next nodes to queue
                if result.status == NodeStatus.COMPLETED:
                    for next_node_id in result.next_node_ids:
                        if next_node_id not in completed_nodes:
                            # Accumulate data: merge input with output so
                            # downstream nodes see all upstream data.
                            # Output keys take priority over input keys.
                            accumulated = {
                                **node_instance.input_data,
                                **node_instance.output_data,
                            }
                            await execution_queue.put(
                                (next_node_id, accumulated)
                            )
                
                # Handle retry
                elif result.status == NodeStatus.PENDING and node_instance.attempt_number < node_instance.node_config.retry_count:
                    # Wait before retry
                    await asyncio.sleep(node_instance.node_config.retry_delay_seconds)
                    await execution_queue.put(
                        (node_instance.id, node_instance.input_data)
                    )
    
    def _find_start_nodes(self, workflow_instance: WorkflowInstance) -> List[str]:
        """Find start nodes in the workflow."""
        start_nodes = []
        
        for node_id, node_instance in workflow_instance.node_instances.items():
            if node_instance.node_config.type == "start":
                start_nodes.append(node_id)
            elif not node_instance.previous_nodes:
                # Nodes with no predecessors can also be start nodes
                start_nodes.append(node_id)
        
        return start_nodes
    
    def _collect_workflow_output(self, workflow_instance: WorkflowInstance) -> Dict[str, Any]:
        """Collect output from end nodes."""
        output_data = {}
        
        for node_instance in workflow_instance.node_instances.values():
            if node_instance.node_config.type == "end" and node_instance.status == NodeStatus.COMPLETED:
                # Merge output from end nodes
                output_data.update(node_instance.output_data)
        
        # If no end nodes, use workflow variables as output
        if not output_data:
            output_data = workflow_instance.variables.copy()
        
        return output_data
    
    async def cancel_workflow(self, workflow_instance: WorkflowInstance):
        """Cancel a running workflow."""
        workflow_instance.status = "cancelled"
        workflow_instance.status_message = "Workflow cancelled by user"
        
        # Cancel all pending/running nodes
        for node_instance in workflow_instance.node_instances.values():
            if node_instance.status in [NodeStatus.PENDING, NodeStatus.RUNNING]:
                node_instance.status = NodeStatus.CANCELLED
        
        # Save cancellation state
        await self.state_manager.save_workflow_state(
            workflow_instance,
            {
                "status": "cancelled",
                "cancelled_at": datetime.utcnow().isoformat(),
                "cancelled_nodes": [
                    node.id for node in workflow_instance.node_instances.values()
                    if node.status == NodeStatus.CANCELLED
                ]
            }
        )
        
        await event_bus.publish(
            "workflow.cancelled",
            {"workflow_id": workflow_instance.id},
            source_id=workflow_instance.id,
            source_type="workflow"
        )
    
    async def pause_workflow(self, workflow_instance: WorkflowInstance) -> str:
        """Pause a running workflow and create checkpoint."""
        workflow_instance.status = "paused"
        workflow_instance.status_message = "Workflow paused by user"
        
        # Create checkpoint
        checkpoint_id = await self.state_manager.checkpoint_workflow(workflow_instance)
        
        # Save pause state
        await self.state_manager.save_workflow_state(
            workflow_instance,
            {
                "status": "paused",
                "paused_at": datetime.utcnow().isoformat(),
                "checkpoint_id": checkpoint_id
            }
        )
        
        await event_bus.publish(
            "workflow.paused",
            {
                "workflow_id": workflow_instance.id,
                "checkpoint_id": checkpoint_id
            },
            source_id=workflow_instance.id,
            source_type="workflow"
        )
        
        return checkpoint_id
    
    async def resume_workflow(
        self,
        workflow_instance: WorkflowInstance,
        checkpoint_id: Optional[str] = None
    ) -> WorkflowInstance:
        """Resume a paused workflow from checkpoint."""
        # Restore from checkpoint if provided
        if checkpoint_id:
            restored = await self.state_manager.restore_checkpoint(
                workflow_instance, checkpoint_id
            )
            if not restored:
                raise ValueError(f"Failed to restore checkpoint {checkpoint_id}")
        
        # Mark as resuming
        workflow_instance.status = "resuming"
        
        # Execute workflow (will handle resuming state)
        return await self.execute_workflow(workflow_instance)
    
    async def get_workflow_progress(
        self,
        workflow_instance: WorkflowInstance
    ) -> Dict[str, Any]:
        """Get current progress of workflow execution."""
        total_nodes = len(workflow_instance.node_instances)
        completed_nodes = sum(
            1 for node in workflow_instance.node_instances.values()
            if node.status == NodeStatus.COMPLETED
        )
        running_nodes = sum(
            1 for node in workflow_instance.node_instances.values()
            if node.status == NodeStatus.RUNNING
        )
        failed_nodes = sum(
            1 for node in workflow_instance.node_instances.values()
            if node.status == NodeStatus.FAILED
        )
        
        progress = {
            "workflow_id": workflow_instance.id,
            "status": workflow_instance.status,
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "running_nodes": running_nodes,
            "failed_nodes": failed_nodes,
            "progress_percentage": (completed_nodes / total_nodes * 100) if total_nodes > 0 else 0,
            "current_nodes": [
                {
                    "id": node.id,
                    "type": node.node_config.type,
                    "status": node.status.value if hasattr(node.status, 'value') else str(node.status)
                }
                for node in workflow_instance.node_instances.values()
                if node.status == NodeStatus.RUNNING
            ]
        }
        
        return progress