"""Decision node for conditional branching."""

import logging
from typing import Any, Dict, List

from nodes.base_node import BaseNode
from nodes.models import NodeInstance


logger = logging.getLogger(__name__)


class DecisionNode(BaseNode):
    """Node that makes decisions for conditional branching."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute decision logic."""
        logger.info(f"Executing decision node: {self.config.name}")
        
        # Get decision configuration
        decision_type = self.config.parameters.get("decision_type", "condition")
        
        if decision_type == "condition":
            result = await self._evaluate_conditions(input_data, context)
        elif decision_type == "switch":
            result = await self._evaluate_switch(input_data, context)
        elif decision_type == "rules":
            result = await self._evaluate_rules(input_data, context)
        else:
            result = {"selected_branch": "default", "reason": "Unknown decision type"}
        
        return result
    
    async def _evaluate_conditions(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate conditional branches."""
        conditions = self.config.parameters.get("conditions", [])
        
        for i, condition in enumerate(conditions):
            expression = condition.get("expression")
            branch_id = condition.get("branch_id", f"branch_{i}")
            
            if expression and await self._evaluate_condition(
                expression,
                input_data,
                context.get("workflow_variables", {})
            ):
                return {
                    "selected_branch": branch_id,
                    "condition_met": expression,
                    "branch_data": condition.get("data", {})
                }
        
        # No condition met, use default
        default_branch = self.config.parameters.get("default_branch", "default")
        return {
            "selected_branch": default_branch,
            "condition_met": None,
            "reason": "No conditions met"
        }
    
    async def _evaluate_switch(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate switch-case style branching."""
        switch_on = self.config.parameters.get("switch_on")
        cases = self.config.parameters.get("cases", {})
        
        if not switch_on:
            return {"error": "No switch_on field specified"}
        
        # Get the value to switch on
        switch_value = self._get_nested_value(input_data, switch_on)
        
        # Find matching case
        for case_value, branch_info in cases.items():
            if str(switch_value) == str(case_value):
                return {
                    "selected_branch": branch_info.get("branch_id", case_value),
                    "switch_value": switch_value,
                    "case_matched": case_value,
                    "branch_data": branch_info.get("data", {})
                }
        
        # No case matched, use default
        default_branch = self.config.parameters.get("default_branch", "default")
        return {
            "selected_branch": default_branch,
            "switch_value": switch_value,
            "case_matched": None,
            "reason": "No case matched"
        }
    
    async def _evaluate_rules(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Evaluate business rules."""
        rules = self.config.parameters.get("rules", [])
        matched_rules = []
        
        for rule in rules:
            rule_name = rule.get("name", "unnamed")
            conditions = rule.get("conditions", [])
            
            # Check if all conditions in the rule are met
            all_met = True
            for condition in conditions:
                field = condition.get("field")
                operator = condition.get("operator")
                value = condition.get("value")
                
                if not await self._check_rule_condition(
                    input_data.get(field),
                    operator,
                    value
                ):
                    all_met = False
                    break
            
            if all_met:
                matched_rules.append({
                    "rule_name": rule_name,
                    "branch_id": rule.get("branch_id", rule_name),
                    "actions": rule.get("actions", [])
                })
        
        if matched_rules:
            # Use first matched rule (or implement priority)
            selected_rule = matched_rules[0]
            return {
                "selected_branch": selected_rule["branch_id"],
                "matched_rules": [r["rule_name"] for r in matched_rules],
                "actions": selected_rule["actions"]
            }
        else:
            default_branch = self.config.parameters.get("default_branch", "default")
            return {
                "selected_branch": default_branch,
                "matched_rules": [],
                "reason": "No rules matched"
            }
    
    async def _check_rule_condition(
        self,
        field_value: Any,
        operator: str,
        compare_value: Any
    ) -> bool:
        """Check a single rule condition."""
        try:
            if operator == "equals":
                return field_value == compare_value
            elif operator == "not_equals":
                return field_value != compare_value
            elif operator == "greater_than":
                return float(field_value) > float(compare_value)
            elif operator == "less_than":
                return float(field_value) < float(compare_value)
            elif operator == "contains":
                return str(compare_value) in str(field_value)
            elif operator == "in":
                return field_value in compare_value
            elif operator == "not_in":
                return field_value not in compare_value
            else:
                return False
        except:
            return False
    
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
    
    async def _determine_next_nodes(
        self,
        instance: NodeInstance,
        workflow_variables: Dict[str, Any]
    ) -> List[str]:
        """Determine next nodes based on decision result."""
        # Get the selected branch from output
        selected_branch = instance.output_data.get("selected_branch", "default")
        
        # Get branch mapping
        branch_mapping = self.config.parameters.get("branch_mapping", {})
        
        # Find next nodes for selected branch
        if selected_branch in branch_mapping:
            next_nodes = branch_mapping[selected_branch]
            if isinstance(next_nodes, str):
                return [next_nodes]
            elif isinstance(next_nodes, list):
                return next_nodes
            else:
                return []
        else:
            # Return default next nodes
            return instance.next_nodes