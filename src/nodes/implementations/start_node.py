"""Start node implementation."""

import logging
from typing import Any, Dict

from nodes.base_node import BaseNode


logger = logging.getLogger(__name__)


class StartNode(BaseNode):
    """Start node for workflows."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Initialize workflow execution."""
        logger.info(f"Starting workflow with node: {self.config.name}")
        
        # Validate required inputs
        required_inputs = self.config.parameters.get("required_inputs", [])
        missing_inputs = []
        
        for input_name in required_inputs:
            if input_name not in input_data:
                missing_inputs.append(input_name)
        
        if missing_inputs:
            raise ValueError(f"Missing required inputs: {missing_inputs}")
        
        # Apply default values
        defaults = self.config.parameters.get("defaults", {})
        output_data = {**defaults, **input_data}
        
        # Add workflow metadata
        output_data["_workflow_started_at"] = context.get("workflow_started_at")
        output_data["_workflow_id"] = context.get("workflow_id")
        
        return output_data