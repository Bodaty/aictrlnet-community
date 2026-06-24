"""Adapter node for calling external adapters."""

import logging
from typing import Any, Dict

from nodes.base_node import BaseNode


logger = logging.getLogger(__name__)


class AdapterNode(BaseNode):
    """Node that calls an adapter."""
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute adapter call."""
        # Get adapter configuration
        adapter_id = self.config.adapter_id or self.config.parameters.get("adapter_id")
        capability = self.config.adapter_capability or self.config.parameters.get("capability")

        # Dynamic per-item adapter routing: when `adapter_id_from` names an input
        # path (dot notation), resolve the adapter from the current input data.
        # This lets a loop fan a query out across engines (GEO multi-engine):
        # each iteration's item carries its engine, picked up here.
        adapter_id_from = self.config.parameters.get("adapter_id_from")
        if adapter_id_from:
            resolved = self._get_nested_value(input_data, adapter_id_from)
            if resolved:
                adapter_id = resolved

        if not adapter_id:
            raise ValueError("No adapter_id specified")
        if not capability:
            raise ValueError("No capability specified")
        
        logger.info(f"Calling adapter {adapter_id} with capability {capability}")

        # Prepare adapter parameters
        adapter_params = self._prepare_adapter_params(input_data, context)

        # Dry-run: simulate without touching the external system. Honors the workflow-level
        # is_dry_run (the runtime threads it into every node) and a node-level is_dry_run/
        # dry_run. The starters run this way until a real integration is connected.
        node_dry_run = self.config.parameters.get("is_dry_run")
        if node_dry_run is None:
            node_dry_run = self.config.parameters.get("dry_run", False)
        if (context or {}).get("is_dry_run") or node_dry_run:
            logger.info(
                "Adapter node %s dry-run: would call %s/%s", self.config.id, adapter_id, capability
            )
            return {
                "_dry_run": True,
                "_adapter_called": adapter_id,
                "_capability_used": capability,
                "would_send": adapter_params,
            }

        try:
            # Call adapter
            result = await self.call_adapter(
                adapter_id=adapter_id,
                capability=capability,
                parameters=adapter_params,
                context=context,
            )

            # Process adapter response
            output = self._process_adapter_response(result)

            # Add metadata
            output["_adapter_called"] = adapter_id
            output["_capability_used"] = capability

            return output

        except Exception as e:
            # "Real when connected, else dry-run": when an opt-in node points at an OAuth
            # integration the user hasn't connected, simulate instead of failing so the
            # workflow still completes (the curriculum hands-on before the user connects).
            from nodes.template_utils import CredentialsUnavailable
            if (
                isinstance(e, CredentialsUnavailable)
                and not getattr(e, "connected", False)
                and self.config.parameters.get("dry_run_if_unconnected")
            ):
                logger.info(
                    "Adapter node %s: %s not connected -> dry-run fallback",
                    self.config.id, adapter_id,
                )
                return {
                    "_dry_run": True,
                    "_not_connected": True,
                    "_adapter_called": adapter_id,
                    "_capability_used": capability,
                    "would_send": adapter_params,
                }

            logger.error(f"Adapter call failed: {str(e)}")
            # Check if we should fail or continue
            if self.config.parameters.get("fail_on_error", True):
                raise
            else:
                return {
                    "error": str(e),
                    "_adapter_called": adapter_id,
                    "_capability_used": capability,
                    "_failed": True
                }
    
    def _prepare_adapter_params(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare parameters for adapter call."""
        # Get parameter mapping
        param_mapping = self.config.parameters.get("parameter_mapping", {})
        
        if param_mapping:
            # Map input data to adapter parameters
            adapter_params = {}
            
            for adapter_param, source_path in param_mapping.items():
                # Support simple dot notation for nested access
                value = self._get_nested_value(input_data, source_path)
                if value is not None:
                    adapter_params[adapter_param] = value
            
            return adapter_params
        else:
            # Use input data directly
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
    
    def _process_adapter_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Process adapter response."""
        # Get response mapping
        response_mapping = self.config.parameters.get("response_mapping", {})
        
        if response_mapping:
            # Map adapter response to output
            output = {}
            
            for output_key, response_path in response_mapping.items():
                value = self._get_nested_value(response, response_path)
                if value is not None:
                    output[output_key] = value
            
            return output
        else:
            # Return response as-is
            return response