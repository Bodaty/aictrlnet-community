"""
DEPRECATED: This file is redundant and not used anywhere.

DO NOT USE THIS FILE - Use services.mcp_unified.UnifiedMCPService instead.

This file should be removed as it duplicates functionality already available
in the unified MCP service.

============================================

MCP Task Integration Service for AICtrlNet."""

from typing import Dict, Any, Tuple, Optional
import logging

from adapters.mcp.factory import create_mcp_dispatcher
from core.exceptions import InternalServerError

logger = logging.getLogger(__name__)


class MCPTaskIntegration:
    """Integration between AICtrlNet tasks and MCP servers"""
    
    @staticmethod
    def is_mcp_task(task: Dict[str, Any]) -> bool:
        """Determine if a task should be handled by an MCP server"""
        # Check task destination
        destination = task.get("destination", "")
        if destination == "mcp":
            return True
        
        # Check payload for MCP flag
        payload = task.get("payload", {})
        if payload.get("use_mcp", False):
            return True
        
        # Check for MCP-specific destinations
        mcp_destinations = ["mcp-message", "mcp-embedding", "mcp-tool"]
        if destination in mcp_destinations:
            return True
            
        return False
    
    @staticmethod
    async def route_task(
        task: Dict[str, Any],
        control_plane_url: Optional[str] = None,
        *,
        caller,
        db,
    ) -> Tuple[Dict[str, Any], int]:
        """Route a task to a server the caller may use.

        `caller` and `db` are required so the dispatcher can apply the per-owner
        rule on the caller's own session.
        """
        try:
            logger.debug(f"MCPTaskIntegration.route_task received task: {task}")

            dispatcher = create_mcp_dispatcher(control_plane_url)

            # Dispatch the task
            result = await dispatcher.dispatch_task(task, caller=caller, db=db)
            
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                logger.error(f"MCP task failed: {error_msg}")
                return {
                    "task_id": task.get("task_id", ""),
                    "status": "failed",
                    "error": error_msg,
                    "destination": "mcp"
                }, 400
            
            # Format successful response
            # Extract the response but keep it as a dict for MCPTaskResponse
            result_data = result.get("response")
            if isinstance(result_data, str):
                # Wrap string response in a dict
                result_data = {"response": result_data}
            elif result_data is None:
                # Use the full result if no specific response field
                result_data = result
            
            response = {
                "task_id": task.get("task_id", ""),
                "source_id": task.get("source_id", ""),
                "destination": "mcp",
                "status": "completed",
                "result": result_data,
                "mcp_metadata": result.get("mcp_metadata", {}),
                "usage": result.get("usage", {})
            }
            
            return response, 200
            
        except Exception as e:
            logger.error(f"MCP task routing failed: {str(e)}")
            return {
                "task_id": task.get("task_id", ""),
                "status": "failed",
                "error": f"MCP task routing failed: {str(e)}",
                "destination": "mcp"
            }, 500
    
    @staticmethod
    def prepare_mcp_task(
        task_data: Dict[str, Any],
        api_type: str = "message"
    ) -> Dict[str, Any]:
        """Prepare a task for MCP processing"""
        # Ensure proper structure
        task = {
            "task_id": task_data.get("task_id", ""),
            "source_id": task_data.get("source_id", ""),
            "destination": "mcp",
            "payload": {
                "api_type": api_type,
                **task_data.get("payload", {})
            }
        }
        
        # Add MCP-specific fields if not present
        if "use_mcp" not in task["payload"]:
            task["payload"]["use_mcp"] = True
        
        return task
    
    @staticmethod
    async def list_mcp_capabilities(*, caller, db) -> Dict[str, Any]:
        """List the MCP capabilities the caller can see."""
        try:
            dispatcher = create_mcp_dispatcher()

            servers = await dispatcher.list_available_servers(caller=caller, db=db)
            
            # Aggregate capabilities
            all_capabilities = set()
            capability_servers = {}
            
            for server in servers:
                for capability in server.get("capabilities", []):
                    all_capabilities.add(capability)
                    if capability not in capability_servers:
                        capability_servers[capability] = []
                    capability_servers[capability].append(server["name"])
            
            return {
                "total_servers": len(servers),
                "capabilities": list(all_capabilities),
                "capability_providers": capability_servers,
                "servers": servers
            }
            
        except Exception as e:
            logger.error(f"Failed to list MCP capabilities: {str(e)}")
            raise InternalServerError(f"Failed to list capabilities: {str(e)}")
