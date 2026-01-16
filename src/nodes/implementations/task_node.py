"""Basic task node implementation."""

import logging
from typing import Any, Dict

from nodes.base_node import BaseNode


logger = logging.getLogger(__name__)


class TaskNode(BaseNode):
    """Basic task execution node."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a basic task."""
        logger.info(f"Executing task node: {self.config.name}")
        
        # Get task parameters
        task_type = self.config.parameters.get("task_type", "process")
        
        # Process based on task type
        if task_type == "process":
            return await self._process_data(input_data, context)
        elif task_type == "validate":
            return await self._validate_data(input_data, context)
        elif task_type == "calculate":
            return await self._calculate(input_data, context)
        else:
            # Custom task execution
            return await self._execute_custom_task(input_data, context)
    
    async def _process_data(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process input data."""
        # Example: Simple data processing
        processed_data = {}
        
        # Apply any transformations specified in parameters
        transformations = self.config.parameters.get("transformations", {})
        
        for key, value in input_data.items():
            if key in transformations:
                # Apply transformation
                transform = transformations[key]
                if transform == "uppercase":
                    processed_data[key] = str(value).upper()
                elif transform == "lowercase":
                    processed_data[key] = str(value).lower()
                else:
                    processed_data[key] = value
            else:
                processed_data[key] = value
        
        return processed_data
    
    async def _validate_data(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate input data."""
        validations = self.config.parameters.get("validations", {})
        errors = []
        
        for field, rules in validations.items():
            value = input_data.get(field)
            
            # Required check
            if rules.get("required") and value is None:
                errors.append(f"{field} is required")
                continue
            
            # Type check
            if "type" in rules and value is not None:
                expected_type = rules["type"]
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"{field} must be a string")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"{field} must be a number")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"{field} must be a boolean")
            
            # Min/max checks for numbers
            if isinstance(value, (int, float)):
                if "min" in rules and value < rules["min"]:
                    errors.append(f"{field} must be >= {rules['min']}")
                if "max" in rules and value > rules["max"]:
                    errors.append(f"{field} must be <= {rules['max']}")
            
            # Length checks for strings
            if isinstance(value, str):
                if "min_length" in rules and len(value) < rules["min_length"]:
                    errors.append(f"{field} must be at least {rules['min_length']} characters")
                if "max_length" in rules and len(value) > rules["max_length"]:
                    errors.append(f"{field} must be at most {rules['max_length']} characters")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "data": input_data
        }
    
    async def _calculate(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform calculations."""
        expression = self.config.parameters.get("expression")
        if not expression:
            return {"error": "No expression provided"}
        
        try:
            # Create calculation context
            calc_context = {
                **input_data,
                "sum": sum,
                "min": min,
                "max": max,
                "len": len,
                "abs": abs,
                "round": round
            }
            
            # WARNING: eval is dangerous! Use a safe expression evaluator in production
            result = eval(expression, {"__builtins__": {}}, calc_context)
            
            return {
                "result": result,
                "expression": expression
            }
            
        except Exception as e:
            return {
                "error": f"Calculation failed: {str(e)}",
                "expression": expression
            }
    
    async def _execute_custom_task(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute custom task logic."""
        # This would be extended based on specific task requirements
        task_name = self.config.parameters.get("task_name", "unknown")
        
        logger.info(f"Executing custom task: {task_name}")
        
        # Return input data with task metadata
        return {
            **input_data,
            "_task_executed": task_name,
            "_node_name": self.config.name
        }