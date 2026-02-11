"""Browser automation node — calls browser-service for headless browser actions.

Sends action sequences to the browser microservice at http://browser-service:8005
and returns structured results to the workflow.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List

import httpx

from ..base_node import BaseNode
from ..models import NodeConfig, NodeExecutionResult, NodeStatus
from events.event_bus import event_bus

logger = logging.getLogger(__name__)

BROWSER_SERVICE_URL = "http://browser-service:8005"


class BrowserAutomationNode(BaseNode):
    """Node for browser automation via the browser microservice.

    Executes action sequences (navigate, click, fill, screenshot,
    extract_text, run_script, wait_for, download) in an isolated
    headless browser context.
    """

    async def execute(self, input_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute browser automation actions."""
        start_time = datetime.utcnow()

        try:
            # Build action sequence from config or input
            actions = self.config.parameters.get("actions") or input_data.get("actions", [])
            timeout_ms = self.config.parameters.get("timeout_ms", 30000)
            viewport = self.config.parameters.get("viewport")

            if not actions:
                raise ValueError("No browser actions specified")

            # Template substitution — replace {{variable}} in action values
            actions = self._substitute_variables(actions, input_data, context)

            # Call browser service
            payload = {
                "actions": actions,
                "timeout_ms": timeout_ms,
            }
            if viewport:
                payload["viewport"] = viewport

            async with httpx.AsyncClient(timeout=timeout_ms / 1000 + 10) as client:
                response = await client.post(
                    f"{BROWSER_SERVICE_URL}/browser/execute",
                    json=payload,
                )
                response.raise_for_status()
                result = response.json()

            duration_ms = (datetime.utcnow() - start_time).total_seconds() * 1000

            await event_bus.publish(
                "node.executed",
                {
                    "node_id": self.config.id,
                    "node_type": "browserAutomation",
                    "actions_count": len(actions),
                    "success": result.get("success", False),
                    "duration_ms": duration_ms,
                },
            )

            return {
                "success": result.get("success", False),
                "results": result.get("results", []),
                "page_url": result.get("page_url"),
                "page_title": result.get("page_title"),
                "total_duration_ms": result.get("total_duration_ms", 0),
                # Extract text results for easy downstream consumption
                "extracted_texts": [
                    r.get("data", {}).get("text", "")
                    for r in result.get("results", [])
                    if r.get("action_type") == "extract_text" and r.get("success")
                ],
            }

        except httpx.ConnectError:
            logger.error(f"BrowserAutomationNode {self.config.id}: browser-service not reachable")
            raise ConnectionError(
                "Browser service is not running. Start it with: docker-compose up browser-service"
            )
        except Exception as e:
            logger.error(f"BrowserAutomationNode {self.config.id} failed: {e}")
            raise

    def _substitute_variables(
        self, actions: List[Dict[str, Any]], input_data: Dict[str, Any], context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Replace {{variable}} placeholders in action values."""
        variables = {**context.get("workflow_variables", {}), **input_data}
        result = []

        for action in actions:
            new_action = dict(action)
            for key in ("url", "value", "selector", "script"):
                val = new_action.get(key)
                if isinstance(val, str) and "{{" in val:
                    for var_name, var_val in variables.items():
                        val = val.replace(f"{{{{{var_name}}}}}", str(var_val))
                    new_action[key] = val
            result.append(new_action)

        return result

    def validate_config(self) -> bool:
        """Validate node configuration."""
        actions = self.config.parameters.get("actions")
        if actions is not None and not isinstance(actions, list):
            raise ValueError("actions must be a list")
        return True
