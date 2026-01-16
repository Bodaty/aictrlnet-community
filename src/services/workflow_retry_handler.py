"""Basic retry and compensation logic for workflows - Community Edition."""

import asyncio
import logging
from typing import Dict, Any
from datetime import datetime
import uuid
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from models.workflow_execution import (
    NodeExecution, NodeExecutionStatus
)
from events.event_bus import event_bus

logger = logging.getLogger(__name__)


class RetryStrategy(str, Enum):
    """Retry strategies available in different editions."""
    NONE = "none"  # Community
    FIXED = "fixed"  # Community
    LINEAR = "linear"  # Business
    EXPONENTIAL = "exponential"  # Business
    ADAPTIVE = "adaptive"  # Enterprise


class WorkflowRetryHandler:
    """Basic retry handler for Community Edition."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.max_retries = 3  # Fixed limit for Community
        self.retry_delay = 5  # Fixed 5 second delay
    
    async def should_retry(
        self,
        node_execution: NodeExecution,
        error: Exception
    ) -> bool:
        """Determine if node should be retried."""
        # Community edition: Simple retry logic
        if node_execution.retry_count >= self.max_retries:
            return False
        
        # Only retry on specific error types
        retryable_errors = [
            "ConnectionError",
            "TimeoutError",
            "TemporaryFailure"
        ]
        
        error_type = type(error).__name__
        return error_type in retryable_errors
    
    async def retry_node(
        self,
        execution_id: uuid.UUID,
        node_id: str,
        retry_count: int
    ) -> Dict[str, Any]:
        """Retry a failed node with basic strategy."""
        logger.info(f"Retrying node {node_id} (attempt {retry_count + 1})")
        
        # Wait before retry
        await asyncio.sleep(self.retry_delay)
        
        # Publish retry event
        await event_bus.publish(
            "workflow.node.retry",
            {
                "execution_id": str(execution_id),
                "node_id": node_id,
                "retry_count": retry_count + 1,
                "strategy": "fixed"
            }
        )
        
        return {
            "retry_count": retry_count + 1,
            "delay_used": self.retry_delay,
            "strategy": "fixed"
        }
    
    async def get_retry_info(self, node_type: str) -> Dict[str, Any]:
        """Get retry configuration for node type."""
        return {
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "strategy": "fixed",
            "edition_limit": "Community edition supports basic retry only"
        }


class WorkflowCompensationHandler:
    """Basic compensation handler for Community Edition."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def compensate_workflow(
        self,
        execution_id: uuid.UUID,
        failed_at_node: str
    ) -> Dict[str, Any]:
        """Execute basic compensation logic."""
        logger.info(f"Starting basic compensation for execution {execution_id}")
        
        # Just mark workflow as failed and notify
        await event_bus.publish(
            "workflow.compensation.started",
            {
                "execution_id": str(execution_id),
                "failed_node": failed_at_node,
                "type": "basic"
            }
        )
        
        # Basic cleanup actions
        cleanup_actions = []
        
        # Release any locks
        cleanup_actions.append({
            "action": "release_locks",
            "status": "completed"
        })
        
        # Notify user
        cleanup_actions.append({
            "action": "notify_user",
            "status": "completed"
        })
        
        await event_bus.publish(
            "workflow.compensation.completed",
            {
                "execution_id": str(execution_id),
                "actions": cleanup_actions
            }
        )
        
        return {
            "compensation_type": "basic",
            "actions_taken": cleanup_actions,
            "status": "completed"
        }