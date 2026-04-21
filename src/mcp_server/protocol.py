"""MCP JSON-RPC 2.0 protocol handler (server side).

Implements the Streamable HTTP transport per MCP protocol version 2025-03-26.
Mirrors what mcp_client_adapter.py sends — this is the server counterpart.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .tool_executor import (
    ComplianceError,
    ScopeError,
    ToolExecutionError,
    execute_tool,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "AICtrlNet"
SERVER_VERSION = "1.0.0"


def _jsonrpc_result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _jsonrpc_error(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


class MCPProtocolHandler:
    """Server-side MCP JSON-RPC 2.0 handler."""

    def __init__(
        self,
        tools_registry: List[dict],
        db: AsyncSession,
        user_id: str,
        api_key: Optional[Any] = None,
        tenant_id: Optional[str] = None,
    ):
        self.tools_registry = tools_registry
        self.db = db
        self.user_id = user_id
        self.api_key = api_key
        self.tenant_id = tenant_id

    async def handle(self, message: dict) -> Optional[dict]:
        """Dispatch a JSON-RPC message. Returns None for notifications."""
        method = message.get("method", "")
        msg_id = message.get("id")  # None for notifications

        handlers = {
            "initialize": self._handle_initialize,
            "ping": self._handle_ping,
            "notifications/initialized": self._handle_notification,
            "notifications/cancelled": self._handle_notification,
            "tools/list": self._handle_tools_list,
            "tools/call": self._handle_tools_call,
        }

        handler = handlers.get(method)
        if handler is None:
            if msg_id is None:
                return None
            return _jsonrpc_error(msg_id, -32601, f"Method not found: {method}")

        try:
            return await handler(message)
        except Exception as e:
            logger.exception(f"Unhandled error in MCP handler for {method}")
            if msg_id is None:
                return None
            return _jsonrpc_error(msg_id, -32603, f"Internal error: {e}")

    async def _handle_initialize(self, message: dict) -> dict:
        msg_id = message.get("id")
        return _jsonrpc_result(msg_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name": SERVER_NAME,
                "version": SERVER_VERSION,
            },
        })

    async def _handle_ping(self, message: dict) -> dict:
        return _jsonrpc_result(message.get("id"), {})

    async def _handle_notification(self, message: dict) -> Optional[dict]:
        return None

    async def _handle_tools_list(self, message: dict) -> dict:
        msg_id = message.get("id")
        tools = [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in self.tools_registry
        ]
        return _jsonrpc_result(msg_id, {"tools": tools})

    async def _handle_tools_call(self, message: dict) -> dict:
        msg_id = message.get("id")
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return _jsonrpc_error(msg_id, -32602, "Missing 'name' in params")

        known_names = {t["name"] for t in self.tools_registry}
        if tool_name not in known_names:
            return _jsonrpc_error(msg_id, -32602, f"Unknown tool: {tool_name}")

        try:
            result = await execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                db=self.db,
                user_id=self.user_id,
                api_key=self.api_key,
                tenant_id=self.tenant_id,
            )
            return _jsonrpc_result(msg_id, {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                "isError": False,
            })

        except ScopeError as e:
            return _jsonrpc_result(msg_id, {
                "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                "isError": True,
            })

        except ComplianceError as e:
            return _jsonrpc_result(msg_id, {
                "content": [{"type": "text", "text": json.dumps({"error": f"Compliance: {e}"})}],
                "isError": True,
            })

        except ToolExecutionError as e:
            return _jsonrpc_result(msg_id, {
                "content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                "isError": True,
            })

        except Exception as e:
            logger.exception(f"Unexpected error executing tool {tool_name}")
            return _jsonrpc_result(msg_id, {
                "content": [{"type": "text", "text": json.dumps({"error": f"Internal error: {e}"})}],
                "isError": True,
            })
