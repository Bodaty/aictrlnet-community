"""Transform node for data manipulation."""

import logging
import json
from typing import Any, Dict, List

from nodes.base_node import BaseNode


logger = logging.getLogger(__name__)


class TransformNode(BaseNode):
    """Node for data transformation."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute data transformation."""
        logger.info(f"Executing transform node: {self.config.name}")
        
        # Get transformation type
        transform_type = self.config.parameters.get("transform_type", "mapping")
        
        if transform_type == "mapping":
            return await self._apply_mapping(input_data)
        elif transform_type == "template":
            return await self._apply_template(input_data, context)
        elif transform_type == "aggregation":
            return await self._apply_aggregation(input_data)
        elif transform_type == "filter":
            return await self._apply_filter(input_data)
        elif transform_type == "custom":
            return await self._apply_custom_transform(input_data, context)
        else:
            return input_data
    
    async def _apply_mapping(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply field mapping transformation."""
        mapping = self.config.parameters.get("mapping", {})
        output_data = {}
        
        for output_field, input_field in mapping.items():
            if isinstance(input_field, str):
                # Simple field mapping
                value = self._get_nested_value(input_data, input_field)
                if value is not None:
                    output_data[output_field] = value
            elif isinstance(input_field, dict):
                # Complex mapping with transformations
                source = input_field.get("source")
                transform = input_field.get("transform")
                default = input_field.get("default")
                
                value = self._get_nested_value(input_data, source) if source else None
                
                if value is not None:
                    # Apply transformation
                    if transform == "uppercase":
                        value = str(value).upper()
                    elif transform == "lowercase":
                        value = str(value).lower()
                    elif transform == "stringify":
                        value = str(value)
                    elif transform == "parse_int":
                        try:
                            value = int(value)
                        except:
                            value = default
                    elif transform == "parse_float":
                        try:
                            value = float(value)
                        except:
                            value = default
                    
                    output_data[output_field] = value
                elif default is not None:
                    output_data[output_field] = default
        
        return output_data
    
    async def _apply_template(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply template-based transformation."""
        templates = self.config.parameters.get("templates", {})
        output_data = {}
        
        # Merge context for template rendering
        template_context = {
            **input_data,
            **context.get("workflow_variables", {}),
            "context": context
        }
        
        for field, template in templates.items():
            try:
                # Simple template rendering (replace with Jinja2 in production)
                rendered = template
                for key, value in template_context.items():
                    if isinstance(value, (str, int, float, bool)):
                        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
                
                output_data[field] = rendered
            except Exception as e:
                logger.error(f"Template rendering failed for {field}: {str(e)}")
                output_data[field] = template
        
        return output_data
    
    async def _apply_aggregation(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply aggregation transformation."""
        aggregations = self.config.parameters.get("aggregations", {})
        output_data = {}
        
        for output_field, agg_config in aggregations.items():
            source_field = agg_config.get("source")
            agg_type = agg_config.get("type", "sum")
            
            values = input_data.get(source_field, [])
            if not isinstance(values, list):
                values = [values]
            
            try:
                if agg_type == "sum":
                    result = sum(float(v) for v in values if v is not None)
                elif agg_type == "count":
                    result = len(values)
                elif agg_type == "min":
                    result = min(float(v) for v in values if v is not None)
                elif agg_type == "max":
                    result = max(float(v) for v in values if v is not None)
                elif agg_type == "avg":
                    numeric_values = [float(v) for v in values if v is not None]
                    result = sum(numeric_values) / len(numeric_values) if numeric_values else 0
                elif agg_type == "concat":
                    separator = agg_config.get("separator", "")
                    result = separator.join(str(v) for v in values if v is not None)
                else:
                    result = None
                
                if result is not None:
                    output_data[output_field] = result
                    
            except Exception as e:
                logger.error(f"Aggregation failed for {output_field}: {str(e)}")
        
        return output_data
    
    async def _apply_filter(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply filtering transformation."""
        filters = self.config.parameters.get("filters", [])
        
        # Start with all data
        output_data = input_data.copy()
        
        for filter_config in filters:
            filter_type = filter_config.get("type", "include")
            fields = filter_config.get("fields", [])
            
            if filter_type == "include":
                # Keep only specified fields
                output_data = {k: v for k, v in output_data.items() if k in fields}
            elif filter_type == "exclude":
                # Remove specified fields
                for field in fields:
                    output_data.pop(field, None)
            elif filter_type == "condition":
                # Filter based on condition
                condition = filter_config.get("condition")
                if condition:
                    # Simple condition evaluation
                    try:
                        if not eval(condition, {"__builtins__": {}}, output_data):
                            # Condition not met, return empty or default
                            return filter_config.get("default", {})
                    except:
                        pass
        
        return output_data
    
    async def _apply_custom_transform(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply custom transformation logic."""
        # This would be extended based on specific requirements
        transform_name = self.config.parameters.get("transform_name", "unknown")
        
        logger.info(f"Applying custom transform: {transform_name}")
        
        # Example custom transformations
        if transform_name == "flatten":
            # Flatten nested structure
            output_data = {}
            self._flatten_dict(input_data, output_data)
            return output_data
        elif transform_name == "nest":
            # Create nested structure
            nesting_config = self.config.parameters.get("nesting", {})
            return self._create_nested_structure(input_data, nesting_config)
        else:
            return input_data
    
    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get value from nested dictionary using dot notation."""
        parts = path.split(".")
        current = data
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _flatten_dict(
        self,
        nested_dict: Dict[str, Any],
        output_dict: Dict[str, Any],
        prefix: str = ""
    ):
        """Flatten a nested dictionary."""
        for key, value in nested_dict.items():
            new_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                self._flatten_dict(value, output_dict, new_key)
            else:
                output_dict[new_key] = value
    
    def _create_nested_structure(
        self,
        flat_data: Dict[str, Any],
        nesting_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create nested structure from flat data."""
        output_data = {}
        
        for key, value in flat_data.items():
            # Check if this key should be nested
            for nest_key, nest_fields in nesting_config.items():
                if key in nest_fields:
                    if nest_key not in output_data:
                        output_data[nest_key] = {}
                    output_data[nest_key][key] = value
                    break
            else:
                # Not nested, add at root level
                output_data[key] = value
        
        return output_data