"""MCP Streamable HTTP transport endpoint.

Single POST route that accepts JSON-RPC 2.0 requests from MCP clients
(e.g. Claude Code) and returns JSON responses.

Claude Code config (~/.claude/mcp.json):
  {"mcpServers": {"aictrlnet": {"type": "http", "url": "...api/v1/mcp-transport", "headers": {"Authorization": "Bearer ..."}}}}
"""

import logging

from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from core.tenant_context import get_current_tenant_id

from .protocol import MCPProtocolHandler
from .tools import get_tools_for_edition

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/mcp-transport")
async def handle_mcp_request(
    request: Request,
    body: dict = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """MCP Streamable HTTP endpoint (protocol 2025-03-26).

    Accepts JSON-RPC 2.0 requests. Returns JSON responses.
    Notifications (no 'id') return 202.
    """
    user_id = str(getattr(current_user, "id", "anonymous"))
    api_key = getattr(request.state, "api_key", None)
    tenant_id = get_current_tenant_id()

    handler = MCPProtocolHandler(
        tools_registry=get_tools_for_edition(),
        db=db,
        user_id=user_id,
        api_key=api_key,
        tenant_id=tenant_id,
    )

    response = await handler.handle(body)

    if response is None:
        return Response(status_code=202)

    return JSONResponse(content=response)
