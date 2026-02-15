"""Error handling and recovery for node execution."""

import asyncio
import logging
import traceback
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
import time
from enum import Enum
import json

from .models import (
    NodeInstance, NodeStatus, NodeExecutionResult, 
    WorkflowInstance, NodeType
)
from .state_manager import NodeStateManager, state_manager as default_state_manager
from events.event_bus import event_bus


logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Recovery strategies for errors."""
    RETRY = "retry"
    SKIP = "skip"
    FAIL = "fail"
    COMPENSATE = "compensate"
    FALLBACK = "fallback"
    MANUAL = "manual"


class NodeError(Exception):
    """Base exception for node execution errors."""
    
    def __init__(
        self,
        message: str,
        node_id: str,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        recovery_strategy: RecoveryStrategy = RecoveryStrategy.FAIL,
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message)
        self.node_id = node_id
        self.severity = severity
        self.recovery_strategy = recovery_strategy
        self.details = details or {}
        self.timestamp = time.time()


class NodeErrorHandler:
    """Handles errors and recovery for node execution."""
    
    def __init__(self, state_manager: Optional[NodeStateManager] = None):
        self.state_manager = state_manager or default_state_manager
        
        # Error handlers by node type
        self._error_handlers: Dict[str, Callable] = {}
        
        # Recovery strategies
        self._recovery_strategies: Dict[RecoveryStrategy, Callable] = {
            RecoveryStrategy.RETRY: self._retry_strategy,
            RecoveryStrategy.SKIP: self._skip_strategy,
            RecoveryStrategy.FAIL: self._fail_strategy,
            RecoveryStrategy.COMPENSATE: self._compensate_strategy,
            RecoveryStrategy.FALLBACK: self._fallback_strategy,
            RecoveryStrategy.MANUAL: self._manual_strategy
        }
        
        # Error history for pattern detection
        self._error_history: List[Dict[str, Any]] = []
        self._error_patterns: Dict[str, int] = {}
    
    async def handle_node_error(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        context: Optional[Dict[str, Any]] = None
    ) -> NodeExecutionResult:
        """Handle an error during node execution."""
        # Log the error
        error_details = {
            "node_id": node_instance.id,
            "node_type": node_instance.node_config.type,
            "workflow_id": workflow_instance.id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "timestamp": time.time(),
            "attempt": node_instance.attempt_number,
            "context": context or {}
        }
        
        logger.error(f"Node error in {node_instance.id}: {error}", extra=error_details)
        
        # Record error in history
        await self._record_error(error_details)
        
        # Determine error severity and recovery strategy
        if isinstance(error, NodeError):
            severity = error.severity
            recovery_strategy = error.recovery_strategy
            error_details.update(error.details)
        else:
            severity = self._determine_severity(error, node_instance)
            recovery_strategy = self._determine_recovery_strategy(
                error, node_instance, workflow_instance
            )
        
        # Save error state
        await self._save_error_state(
            node_instance, workflow_instance, error_details
        )
        
        # Publish error event
        await event_bus.publish(
            "node.error",
            {
                "node_id": node_instance.id,
                "workflow_id": workflow_instance.id,
                "severity": severity.value,
                "recovery_strategy": recovery_strategy.value,
                "error": str(error),
                "attempt": node_instance.attempt_number
            },
            source_id=node_instance.id,
            source_type="node"
        )
        
        # Execute recovery strategy
        result = await self._execute_recovery_strategy(
            recovery_strategy,
            error,
            node_instance,
            workflow_instance,
            error_details
        )
        
        return result
    
    async def handle_workflow_error(
        self,
        error: Exception,
        workflow_instance: WorkflowInstance,
        failed_node_id: Optional[str] = None
    ) -> None:
        """Handle an error at the workflow level."""
        error_details = {
            "workflow_id": workflow_instance.id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_node_id": failed_node_id,
            "timestamp": time.time()
        }
        
        logger.error(f"Workflow error in {workflow_instance.id}: {error}", extra=error_details)
        
        # Update workflow status
        workflow_instance.status = "failed"
        workflow_instance.status_message = str(error)
        
        # Save workflow error state
        await self.state_manager.save_workflow_state(
            workflow_instance,
            {
                "error": error_details,
                "last_error_time": time.time()
            }
        )
        
        # Publish workflow error event
        await event_bus.publish(
            "workflow.error",
            error_details,
            source_id=workflow_instance.id,
            source_type="workflow"
        )
        
        # Check if workflow can be recovered
        if await self._can_recover_workflow(workflow_instance, error):
            await self._initiate_workflow_recovery(workflow_instance)
    
    def register_error_handler(
        self,
        node_type: str,
        handler: Callable
    ) -> None:
        """Register a custom error handler for a node type."""
        self._error_handlers[node_type] = handler
    
    async def _record_error(self, error_details: Dict[str, Any]) -> None:
        """Record error in history for pattern detection."""
        self._error_history.append(error_details)
        
        # Keep only recent errors (last 1000)
        if len(self._error_history) > 1000:
            self._error_history = self._error_history[-1000:]
        
        # Update error patterns
        pattern_key = f"{error_details['node_type']}:{error_details['error_type']}"
        self._error_patterns[pattern_key] = self._error_patterns.get(pattern_key, 0) + 1
        
        # Check for error storm
        await self._check_error_storm(error_details)
    
    async def _check_error_storm(self, error_details: Dict[str, Any]) -> None:
        """Check if we're experiencing an error storm."""
        # Count recent errors (last 5 minutes)
        cutoff_time = time.time() - 300
        recent_errors = [
            e for e in self._error_history 
            if e["timestamp"] > cutoff_time
        ]
        
        if len(recent_errors) > 50:
            logger.warning(f"Error storm detected: {len(recent_errors)} errors in last 5 minutes")
            
            # Publish error storm event
            await event_bus.publish(
                "system.error_storm",
                {
                    "error_count": len(recent_errors),
                    "time_window": 300,
                    "most_common_errors": self._get_most_common_errors(recent_errors)
                },
                source_id="error_handler",
                source_type="system"
            )
    
    def _get_most_common_errors(
        self, 
        errors: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Get most common errors from a list."""
        error_counts = {}
        for error in errors:
            key = f"{error['node_type']}:{error['error_type']}"
            if key not in error_counts:
                error_counts[key] = {
                    "node_type": error["node_type"],
                    "error_type": error["error_type"],
                    "count": 0,
                    "last_message": error["error_message"]
                }
            error_counts[key]["count"] += 1
        
        # Sort by count and return top 5
        sorted_errors = sorted(
            error_counts.values(),
            key=lambda x: x["count"],
            reverse=True
        )
        return sorted_errors[:5]
    
    def _determine_severity(
        self,
        error: Exception,
        node_instance: NodeInstance
    ) -> ErrorSeverity:
        """Determine error severity based on error type and context."""
        # Critical errors
        if isinstance(error, (MemoryError, SystemError)):
            return ErrorSeverity.CRITICAL
        
        # High severity for certain node types
        critical_node_types = ["payment", "compliance", "security"]
        if any(ct in node_instance.node_config.type for ct in critical_node_types):
            return ErrorSeverity.HIGH
        
        # Medium severity for most errors
        if isinstance(error, (ValueError, TypeError, KeyError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity for expected errors
        if isinstance(error, (TimeoutError, ConnectionError)):
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _determine_recovery_strategy(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance
    ) -> RecoveryStrategy:
        """Determine recovery strategy based on error and context."""
        # Check node configuration for strategy
        if hasattr(node_instance.node_config, 'error_strategy'):
            try:
                return RecoveryStrategy(node_instance.node_config.error_strategy)
            except ValueError:
                pass
        
        # Retry for transient errors
        if isinstance(error, (TimeoutError, ConnectionError)):
            if node_instance.attempt_number < node_instance.node_config.retry_count:
                return RecoveryStrategy.RETRY
        
        # Skip for non-critical nodes
        if hasattr(node_instance.node_config, 'allow_failure') and node_instance.node_config.allow_failure:
            return RecoveryStrategy.SKIP
        
        # Compensate for nodes with compensation logic
        if hasattr(node_instance.node_config, 'compensation_node_id'):
            return RecoveryStrategy.COMPENSATE
        
        # Fallback if available
        if hasattr(node_instance.node_config, 'fallback_node_id'):
            return RecoveryStrategy.FALLBACK
        
        # Default to fail
        return RecoveryStrategy.FAIL
    
    async def _save_error_state(
        self,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> None:
        """Save error state for debugging and recovery."""
        error_state = {
            "error": error_details,
            "node_state": {
                "status": node_instance.status.value if hasattr(node_instance.status, 'value') else str(node_instance.status),
                "input_data": node_instance.input_data,
                "output_data": node_instance.output_data,
                "attempt_number": node_instance.attempt_number
            },
            "workflow_variables": workflow_instance.variables.copy()
        }
        
        await self.state_manager.save_node_state(
            node_instance,
            error_state,
            workflow_instance.id
        )
    
    async def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Execute the selected recovery strategy."""
        handler = self._recovery_strategies.get(strategy)
        if handler:
            return await handler(
                error, node_instance, workflow_instance, error_details
            )
        else:
            logger.error(f"Unknown recovery strategy: {strategy}")
            return await self._fail_strategy(
                error, node_instance, workflow_instance, error_details
            )
    
    async def _retry_strategy(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Retry the node execution."""
        node_instance.attempt_number += 1
        
        # Calculate backoff delay
        delay = min(
            node_instance.node_config.retry_delay_seconds * (2 ** (node_instance.attempt_number - 1)),
            300  # Max 5 minutes
        )
        
        logger.info(
            f"Retrying node {node_instance.id} after {delay}s "
            f"(attempt {node_instance.attempt_number}/{node_instance.node_config.retry_count})"
        )
        
        return NodeExecutionResult(
            node_instance_id=node_instance.id,
            status=NodeStatus.PENDING,
            retry_after=time.time() + delay,
            error=str(error),
            metadata={
                "recovery_strategy": "retry",
                "attempt": node_instance.attempt_number,
                "delay": delay
            }
        )
    
    async def _skip_strategy(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Skip the failed node and continue."""
        logger.warning(f"Skipping failed node {node_instance.id}: {error}")
        
        node_instance.status = NodeStatus.SKIPPED
        node_instance.status_message = f"Skipped due to error: {error}"
        
        # Determine next nodes
        next_nodes = []
        if hasattr(node_instance, 'next_nodes'):
            next_nodes = node_instance.next_nodes
        
        return NodeExecutionResult(
            node_instance_id=node_instance.id,
            status=NodeStatus.SKIPPED,
            next_node_ids=next_nodes,
            error=str(error),
            metadata={
                "recovery_strategy": "skip",
                "reason": "error_recovery"
            }
        )
    
    async def _fail_strategy(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Fail the node and stop workflow execution."""
        logger.error(f"Node {node_instance.id} failed permanently: {error}")
        
        node_instance.status = NodeStatus.FAILED
        node_instance.status_message = str(error)
        node_instance.failed_at = datetime.utcnow()
        
        return NodeExecutionResult(
            node_instance_id=node_instance.id,
            status=NodeStatus.FAILED,
            error=str(error),
            metadata={
                "recovery_strategy": "fail",
                "error_details": error_details
            }
        )
    
    async def _compensate_strategy(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Execute compensation logic for the failed node."""
        compensation_node_id = getattr(
            node_instance.node_config,
            'compensation_node_id',
            None
        )
        
        if not compensation_node_id:
            logger.warning(f"No compensation node found for {node_instance.id}")
            return await self._fail_strategy(
                error, node_instance, workflow_instance, error_details
            )
        
        logger.info(f"Executing compensation node {compensation_node_id} for {node_instance.id}")
        
        # Mark current node as compensated
        node_instance.status = NodeStatus.COMPENSATED
        node_instance.status_message = f"Compensated due to error: {error}"
        
        return NodeExecutionResult(
            node_instance_id=node_instance.id,
            status=NodeStatus.COMPENSATED,
            next_node_ids=[compensation_node_id],
            error=str(error),
            metadata={
                "recovery_strategy": "compensate",
                "compensation_node": compensation_node_id
            }
        )
    
    async def _fallback_strategy(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Execute fallback node for the failed node."""
        fallback_node_id = getattr(
            node_instance.node_config,
            'fallback_node_id',
            None
        )
        
        if not fallback_node_id:
            logger.warning(f"No fallback node found for {node_instance.id}")
            return await self._fail_strategy(
                error, node_instance, workflow_instance, error_details
            )
        
        logger.info(f"Executing fallback node {fallback_node_id} for {node_instance.id}")
        
        # Mark current node as using fallback
        node_instance.status = NodeStatus.FALLBACK
        node_instance.status_message = f"Using fallback due to error: {error}"
        
        return NodeExecutionResult(
            node_instance_id=node_instance.id,
            status=NodeStatus.FALLBACK,
            next_node_ids=[fallback_node_id],
            error=str(error),
            metadata={
                "recovery_strategy": "fallback",
                "fallback_node": fallback_node_id
            }
        )
    
    async def _manual_strategy(
        self,
        error: Exception,
        node_instance: NodeInstance,
        workflow_instance: WorkflowInstance,
        error_details: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Pause workflow for manual intervention."""
        logger.warning(f"Manual intervention required for node {node_instance.id}: {error}")
        
        node_instance.status = NodeStatus.WAITING_MANUAL
        node_instance.status_message = f"Manual intervention required: {error}"
        
        # Create manual intervention task
        await event_bus.publish(
            "workflow.manual_intervention_required",
            {
                "workflow_id": workflow_instance.id,
                "node_id": node_instance.id,
                "error": str(error),
                "error_details": error_details
            },
            source_id=workflow_instance.id,
            source_type="workflow"
        )
        
        return NodeExecutionResult(
            node_instance_id=node_instance.id,
            status=NodeStatus.WAITING_MANUAL,
            error=str(error),
            metadata={
                "recovery_strategy": "manual",
                "intervention_required": True
            }
        )
    
    async def _can_recover_workflow(
        self,
        workflow_instance: WorkflowInstance,
        error: Exception
    ) -> bool:
        """Check if workflow can be recovered from error."""
        # Check workflow recovery policy
        if hasattr(workflow_instance, 'recovery_policy'):
            if workflow_instance.recovery_policy == "never":
                return False
            elif workflow_instance.recovery_policy == "always":
                return True
        
        # Check error type
        if isinstance(error, (SystemError, MemoryError)):
            return False
        
        # Check failed node count
        failed_nodes = sum(
            1 for node in workflow_instance.node_instances.values()
            if node.status == NodeStatus.FAILED
        )
        
        # Too many failures
        if failed_nodes > len(workflow_instance.node_instances) * 0.5:
            return False
        
        return True
    
    async def _initiate_workflow_recovery(
        self,
        workflow_instance: WorkflowInstance
    ) -> None:
        """Initiate workflow recovery process."""
        logger.info(f"Initiating recovery for workflow {workflow_instance.id}")
        
        # Create checkpoint before recovery
        checkpoint_id = await self.state_manager.checkpoint_workflow(workflow_instance)
        
        # Mark workflow as recovering
        workflow_instance.status = "recovering"
        workflow_instance.status_message = "Automatic recovery in progress"
        
        # Publish recovery event
        await event_bus.publish(
            "workflow.recovery_started",
            {
                "workflow_id": workflow_instance.id,
                "checkpoint_id": checkpoint_id
            },
            source_id=workflow_instance.id,
            source_type="workflow"
        )


# Note: NodeStatus values are defined in models.py and cannot be modified at runtime


# Global error handler instance
node_error_handler = NodeErrorHandler()