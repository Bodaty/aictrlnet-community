"""MCP JSON-RPC 2.0 protocol handler (server side).

Implements the Streamable HTTP transport per MCP protocol version
2025-03-26. Mirrors what ``mcp_client_adapter.py`` sends — this is the
server counterpart.

Changes vs. v1:
- ``tools/list`` filters by the caller's effective plan tier so Claude
  only sees tools it can actually call (prevents leaking plan-gated
  tool names to lower tiers).
- ``capabilities.tools.listChanged`` is now ``True``: server will emit
  ``notifications/tools/list_changed`` when the plan changes mid-session.
- All gate errors (plan / scope / rate / quota / compliance / timeout)
  get structured payloads via ``err.to_payload()`` when available — the
  transport can surface ``upgrade_url`` / ``retry_after_seconds`` to the
  client as clickable CTAs.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from .plan_gate import PlanService, TOOL_MIN_PLAN
from .tool_executor import (
    ComplianceError,
    PlanError,
    QuotaError,
    RateError,
    ScopeError,
    ToolExecutionError,
    ToolTimeoutError,
    execute_tool,
)

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-03-26"
SERVER_NAME = "AICtrlNet"
SERVER_VERSION = "1.1.0"

_TIER_RANK = {"community": 0, "business": 1, "enterprise": 2}


def _jsonrpc_result(msg_id: Any, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def _jsonrpc_error(msg_id: Any, code: int, message: str, data: Any = None) -> dict:
    err: Dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": msg_id, "error": err}


def _structured_tool_error(
    msg_id: Any,
    payload: dict,
    human_message: str,
) -> dict:
    """Return an MCP tool error with both stringified text content (so
    existing clients render something readable) and a ``data`` field on
    the JSON-RPC response for clients that parse structured errors.

    Per MCP 2025-03-26, errors inside successful tool dispatch use
    ``isError: true`` on the result, not the JSON-RPC ``error`` field.
    """
    return {
        "jsonrpc": "2.0",
        "id": msg_id,
        "result": {
            "content": [
                {"type": "text", "text": human_message},
            ],
            "isError": True,
            "_meta": {"error": payload},
        },
    }


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
        # Per-request PlanService — cache is scoped to this JSON-RPC
        # HTTP request / batch so plan lookups are not repeated.
        self.plan_service = PlanService(db)

    async def handle(self, message: dict) -> Optional[dict]:
        """Dispatch a JSON-RPC message. Returns None for notifications."""
        method = message.get("method", "")
        msg_id = message.get("id")

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
                # Flipped True: server emits list_changed on plan mutation.
                "tools": {"listChanged": True},
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

        current_plan = await self.plan_service.get_effective_edition(self.tenant_id)
        current_rank = _TIER_RANK.get(current_plan, 0)

        tools = []
        for t in self.tools_registry:
            required = TOOL_MIN_PLAN.get(t["name"], "community")
            if _TIER_RANK.get(required, 0) <= current_rank:
                tools.append(
                    {
                        "name": t["name"],
                        "description": t["description"],
                        "inputSchema": t["inputSchema"],
                    }
                )
        return _jsonrpc_result(msg_id, {"tools": tools})

    async def _handle_tools_call(self, message: dict) -> dict:
        msg_id = message.get("id")
        params = message.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if not tool_name:
            return _jsonrpc_error(msg_id, -32602, "Missing 'name' in params")

        # Unknown / plan-hidden tools return identical errors — we do NOT
        # reveal to a lower-tier caller that a higher-tier tool exists.
        current_plan = await self.plan_service.get_effective_edition(self.tenant_id)
        current_rank = _TIER_RANK.get(current_plan, 0)
        required = TOOL_MIN_PLAN.get(tool_name)
        known_names = {t["name"] for t in self.tools_registry}

        if tool_name not in known_names:
            return _jsonrpc_error(msg_id, -32602, f"Unknown tool: {tool_name}")
        if required is not None and _TIER_RANK.get(required, 0) > current_rank:
            # Same wire shape as "unknown tool" — prevents tier-taxonomy leak
            return _jsonrpc_error(msg_id, -32602, f"Unknown tool: {tool_name}")

        try:
            result = await execute_tool(
                tool_name=tool_name,
                arguments=arguments,
                db=self.db,
                user_id=self.user_id,
                api_key=self.api_key,
                tenant_id=self.tenant_id,
                plan_service=self.plan_service,
            )
            return _jsonrpc_result(msg_id, {
                "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                "isError": False,
            })

        except PlanError as e:
            return _structured_tool_error(
                msg_id, e.to_payload(), f"Plan upgrade required: {e}"
            )

        except ScopeError as e:
            return _structured_tool_error(
                msg_id,
                {"error": "scope_denied", "message": str(e)},
                str(e),
            )

        except RateError as e:
            return _structured_tool_error(
                msg_id, e.to_payload(), str(e)
            )

        except QuotaError as e:
            return _structured_tool_error(
                msg_id, e.to_payload(), str(e)
            )

        except ComplianceError as e:
            return _structured_tool_error(
                msg_id,
                {"error": "compliance_denied", "message": str(e)},
                f"Compliance: {e}",
            )

        except ToolTimeoutError as e:
            return _structured_tool_error(
                msg_id,
                {"error": "timeout", "message": str(e)},
                str(e),
            )

        except ToolExecutionError as e:
            return _structured_tool_error(
                msg_id,
                {"error": "tool_error", "message": str(e)},
                str(e),
            )

        except Exception as e:
            logger.exception(f"Unexpected error executing tool {tool_name}")
            return _structured_tool_error(
                msg_id,
                {"error": "internal_error", "message": str(e)},
                f"Internal error: {e}",
            )
