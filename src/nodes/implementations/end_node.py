"""End node implementation."""

import logging
from typing import Any, Dict

from nodes.base_node import BaseNode


logger = logging.getLogger(__name__)


class EndNode(BaseNode):
    """End node for workflows."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Finalize workflow execution."""
        logger.info(f"Ending workflow with node: {self.config.name}")
        
        # Extract output fields if specified
        output_fields = self.config.parameters.get("output_fields", [])
        
        if output_fields:
            # Return only specified fields
            output_data = {}
            for field in output_fields:
                if field in input_data:
                    output_data[field] = input_data[field]
        else:
            # Return all data
            output_data = input_data.copy()
        
        # Add completion metadata
        output_data["_workflow_completed"] = True
        output_data["_completion_node"] = self.config.name
        
        # Apply any output transformations
        transformations = self.config.parameters.get("output_transformations", {})
        for field, transform in transformations.items():
            if field in output_data:
                if transform == "stringify":
                    output_data[field] = str(output_data[field])
                elif transform == "parse_json":
                    try:
                        import json
                        output_data[field] = json.loads(output_data[field])
                    except:
                        pass
        
        return output_data