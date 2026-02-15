"""Base node class for workflow nodes."""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import time

from .models import NodeConfig, NodeStatus, NodeInstance, NodeExecutionResult
from events.event_bus import event_bus
from adapters.registry import adapter_registry
from adapters.models import AdapterRequest


logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """Base class for all workflow nodes."""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.status = NodeStatus.PENDING
        self._lock = asyncio.Lock()
    
    @abstractmethod
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute the node logic. Must be implemented by subclasses."""
        pass
    
    async def run(
        self,
        instance: NodeInstance,
        workflow_variables: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Run the node with full lifecycle management."""
        start_time = time.time()
        
        async with self._lock:
            try:
                # Update status
                instance.status = NodeStatus.RUNNING
                instance.started_at = datetime.utcnow()
                instance.attempt_number += 1
                
                # Check skip condition
                if self.config.skip_condition and await self._evaluate_condition(
                    self.config.skip_condition,
                    instance.input_data,
                    workflow_variables
                ):
                    instance.status = NodeStatus.SKIPPED
                    return NodeExecutionResult(
                        node_instance_id=instance.id,
                        status=NodeStatus.SKIPPED,
                        duration_ms=0,
                        next_node_ids=instance.next_nodes
                    )
                
                # Prepare context
                is_dry_run = workflow_variables.get("_is_dry_run", False)
                context = {
                    **instance.context,
                    "workflow_id": instance.workflow_instance_id,
                    "node_id": self.config.id,
                    "node_name": self.config.name,
                    "attempt": instance.attempt_number,
                    "workflow_variables": workflow_variables,
                    "is_dry_run": is_dry_run
                }

                # Dry-run interception (Business+ feature via try/except import)
                if is_dry_run:
                    try:
                        from aictrlnet_business.services.dry_run_interceptor import should_intercept, simulate_node
                        # Resolve effective node type: custom_node_type takes priority over enum type
                        effective_type = self.config.parameters.get("custom_node_type") if (
                            hasattr(self.config, 'parameters') and isinstance(self.config.parameters, dict)
                        ) else None
                        if not effective_type:
                            effective_type = self.config.type.value if hasattr(self.config.type, 'value') else str(self.config.type)
                        if should_intercept(effective_type, self.config):
                            logger.info(f"DRY-RUN INTERCEPTED node {self.config.id} (effective_type={effective_type})")
                            simulated = simulate_node(effective_type, self.config, instance.input_data)
                            instance.output_data = simulated
                            instance.status = NodeStatus.COMPLETED
                            instance.completed_at = datetime.utcnow()
                            instance.duration_ms = (time.time() - start_time) * 1000
                            return NodeExecutionResult(
                                node_instance_id=instance.id,
                                status=NodeStatus.COMPLETED,
                                output_data=simulated,
                                duration_ms=instance.duration_ms,
                                next_node_ids=instance.next_nodes
                            )
                    except ImportError:
                        pass  # Community edition — no interceptor, run normally

                # Execute with timeout
                output_data = await asyncio.wait_for(
                    self.execute(instance.input_data, context),
                    timeout=self.config.timeout_seconds
                )
                
                # Update instance
                instance.output_data = output_data
                instance.status = NodeStatus.COMPLETED
                instance.completed_at = datetime.utcnow()
                instance.duration_ms = (time.time() - start_time) * 1000
                
                # Determine next nodes
                next_node_ids = await self._determine_next_nodes(
                    instance,
                    workflow_variables
                )
                
                # Publish completion event
                await event_bus.publish(
                    "node.completed",
                    {
                        "node_id": self.config.id,
                        "node_name": self.config.name,
                        "node_type": self.config.type.value,
                        "workflow_id": instance.workflow_instance_id,
                        "duration_ms": instance.duration_ms
                    },
                    source_id=self.config.id,
                    source_type="node"
                )
                
                return NodeExecutionResult(
                    node_instance_id=instance.id,
                    status=NodeStatus.COMPLETED,
                    output_data=output_data,
                    duration_ms=instance.duration_ms,
                    next_node_ids=next_node_ids
                )
                
            except asyncio.TimeoutError:
                error = f"Node execution timed out after {self.config.timeout_seconds}s"
                return await self._handle_error(instance, error, start_time)
                
            except Exception as e:
                return await self._handle_error(instance, str(e), start_time)
    
    async def _handle_error(
        self,
        instance: NodeInstance,
        error: str,
        start_time: float
    ) -> NodeExecutionResult:
        """Handle node execution error."""
        instance.status = NodeStatus.FAILED
        instance.last_error = error
        instance.completed_at = datetime.utcnow()
        instance.duration_ms = (time.time() - start_time) * 1000
        
        logger.error(f"Node {self.config.name} failed: {error}")
        
        # Publish error event
        await event_bus.publish(
            "node.failed",
            {
                "node_id": self.config.id,
                "node_name": self.config.name,
                "node_type": self.config.type.value,
                "workflow_id": instance.workflow_instance_id,
                "error": error,
                "attempt": instance.attempt_number
            },
            source_id=self.config.id,
            source_type="node"
        )
        
        # Check if retry is needed
        if instance.attempt_number < self.config.retry_count:
            instance.status = NodeStatus.PENDING  # Will be retried
            return NodeExecutionResult(
                node_instance_id=instance.id,
                status=NodeStatus.PENDING,
                error=error,
                duration_ms=instance.duration_ms,
                next_node_ids=[]  # Don't proceed until retry
            )
        
        # Check for error handler
        next_nodes = []
        if self.config.error_handler_node_id:
            next_nodes = [self.config.error_handler_node_id]
        
        return NodeExecutionResult(
            node_instance_id=instance.id,
            status=NodeStatus.FAILED,
            error=error,
            duration_ms=instance.duration_ms,
            next_node_ids=next_nodes
        )
    
    async def _determine_next_nodes(
        self,
        instance: NodeInstance,
        workflow_variables: Dict[str, Any]
    ) -> List[str]:
        """Determine which nodes to execute next."""
        # For decision nodes, evaluate conditions
        if self.config.type == "decision":
            # This would be implemented in DecisionNode subclass
            pass
        
        # Default: return all next nodes
        return instance.next_nodes
    
    async def _evaluate_condition(
        self,
        condition: str,
        data: Dict[str, Any],
        variables: Dict[str, Any]
    ) -> bool:
        """Evaluate a condition expression."""
        # Simple evaluation (in production, use a safe expression evaluator)
        try:
            # Create evaluation context
            context = {
                **data,
                **variables,
                "true": True,
                "false": False,
                "null": None
            }
            
            # WARNING: eval is dangerous! Use a safe expression evaluator in production
            # This is just for demonstration
            result = eval(condition, {"__builtins__": {}}, context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Failed to evaluate condition '{condition}': {str(e)}")
            return False
    
    async def call_adapter(
        self,
        adapter_id: str,
        capability: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Helper method to call an adapter."""
        # Get adapter class from registry
        adapter_class = adapter_registry.get_adapter_class(adapter_id)
        if not adapter_class:
            raise ValueError(f"Adapter {adapter_id} not found")
        
        # Create adapter instance
        adapter = adapter_class({})
        
        request = AdapterRequest(
            capability=capability,
            parameters=parameters
        )
        
        response = await adapter.handle_request(request)
        
        if response.status == "error":
            raise RuntimeError(f"Adapter error: {response.error}")
        
        return response.data
    
    def get_info(self) -> Dict[str, Any]:
        """Get node information."""
        return {
            "id": self.config.id,
            "name": self.config.name,
            "type": self.config.type.value,
            "description": self.config.description,
            "parameters": self.config.parameters,
            "inputs": self.config.inputs,
            "outputs": self.config.outputs,
            "required_edition": self.config.required_edition
        }