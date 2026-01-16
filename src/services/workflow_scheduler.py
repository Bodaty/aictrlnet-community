"""Workflow scheduling service - Community Edition (manual triggers only)."""

import logging
from typing import List, Dict, Any, Optional
import uuid
from enum import Enum

from sqlalchemy.ext.asyncio import AsyncSession

from models.workflow_execution import WorkflowExecution
from services.workflow_execution import WorkflowExecutionService

logger = logging.getLogger(__name__)


class TriggerType(str, Enum):
    """Trigger types available in different editions."""
    MANUAL = "manual"  # Community
    WEBHOOK = "webhook"  # Business+
    EVENT = "event"  # Business+
    SCHEDULE = "schedule"  # Business+
    CONDITION = "condition"  # Enterprise


class WorkflowScheduler:
    """Basic scheduler for Community Edition - manual triggers only."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.execution_service = WorkflowExecutionService(db)
        self.supported_triggers = [TriggerType.MANUAL]
    
    async def trigger_workflow(
        self,
        workflow_id: uuid.UUID,
        trigger_type: TriggerType,
        trigger_data: Optional[Dict[str, Any]] = None
    ) -> WorkflowExecution:
        """Trigger workflow execution - Community supports manual only."""
        
        # Community edition only supports manual triggers
        if trigger_type != TriggerType.MANUAL:
            logger.warning(
                f"Trigger type {trigger_type} not supported in Community edition. "
                f"Using manual trigger instead."
            )
            trigger_type = TriggerType.MANUAL
        
        # Create execution
        execution = await self.execution_service.create_execution(
            workflow_id=workflow_id,
            input_data=trigger_data.get("input_data", {}) if trigger_data else {},
            triggered_by=trigger_type.value,
            trigger_metadata=trigger_data or {}
        )
        
        # Start execution
        await self.execution_service.start_execution(execution.id)
        
        return execution
    
    async def list_available_triggers(self) -> List[Dict[str, Any]]:
        """List trigger types available in Community edition."""
        return [
            {
                "type": TriggerType.MANUAL.value,
                "description": "Manual trigger via UI or API",
                "available": True,
                "configuration": {}
            }
        ]
    
    async def validate_schedule(self, cron_expression: str) -> Dict[str, Any]:
        """Validate schedule - not supported in Community."""
        return {
            "valid": False,
            "error": "Scheduled workflows require Business or Enterprise edition",
            "upgrade_info": "Upgrade to Business edition for scheduling support"
        }
    
    async def create_schedule(self, *args, **kwargs) -> Dict[str, Any]:
        """Create schedule - not supported in Community."""
        return {
            "error": "Scheduled workflows require Business or Enterprise edition",
            "upgrade_info": "Upgrade to Business edition for scheduling support"
        }
    
    async def create_webhook_trigger(self, *args, **kwargs) -> Dict[str, Any]:
        """Create webhook trigger - not supported in Community."""
        return {
            "error": "Webhook triggers require Business or Enterprise edition",
            "upgrade_info": "Upgrade to Business edition for webhook trigger support"
        }
    
    async def create_event_trigger(self, *args, **kwargs) -> Dict[str, Any]:
        """Create event trigger - not supported in Community."""
        return {
            "error": "Event triggers require Business or Enterprise edition",
            "upgrade_info": "Upgrade to Business edition for event trigger support"
        }