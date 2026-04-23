"""MCP Streamable HTTP transport endpoint.

Single POST route that accepts JSON-RPC 2.0 requests from MCP clients
(e.g. Claude Code) and returns JSON responses.

Claude Code config (~/.claude/mcp.json):
  {"mcpServers": {"aictrlnet": {"type": "http", "url": ".../api/v1/mcp-transport", "headers": {"Authorization": "Bearer ..."}}}}

Wave 7 hardening:
- **A9**: ``MCP_REQUIRE_BEARER_OR_APIKEY`` (default true) — hard-reject
  any request that doesn't carry ``Authorization: Bearer`` OR
  ``X-API-Key``. Prevents cookie-only browser sessions from hitting the
  MCP surface (CSRF mitigation given the globally permissive CORS).
- **A8**: CORS policy for this endpoint is documented in this module
  (see ``_MCP_CORS_NOTE`` below). Application-wide CORS middleware
  should be configured with strict MCP-specific rules.
"""

import logging
import os

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from core.tenant_context import get_current_tenant_id

from .protocol import MCPProtocolHandler
from .tools import get_tools_for_edition

logger = logging.getLogger(__name__)

router = APIRouter()


# Documentation note — operators configuring CORS for this endpoint
# should set STRICT values on the application-level middleware:
#   allow_credentials=False, allow_methods=["POST","OPTIONS"],
#   allow_headers=["Authorization","X-API-Key","Content-Type"]
# Cookie-based auth MUST NOT be used for this endpoint.
_MCP_CORS_NOTE = "see http_transport.py docstring"


def _require_bearer_or_apikey(request: Request) -> None:
    """A9: hard-reject requests without an explicit MCP auth header.

    Applies before ``get_current_active_user`` so cookie-session users
    can't accidentally reach the endpoint. Disabled by setting
    ``MCP_REQUIRE_BEARER_OR_APIKEY=false`` (not recommended — see A9).
    """
    if os.environ.get("MCP_REQUIRE_BEARER_OR_APIKEY", "true").lower() != "true":
        return
    auth = request.headers.get("Authorization", "")
    apikey = request.headers.get("X-API-Key", "")
    if auth.startswith("Bearer ") or apikey:
        return
    raise HTTPException(
        status_code=401,
        detail=(
            "MCP endpoint requires an Authorization: Bearer header or an "
            "X-API-Key header. Cookie-only sessions are not accepted."
        ),
    )


@router.post("/mcp-transport")
async def handle_mcp_request(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_active_user),
):
    """MCP Streamable HTTP endpoint (protocol 2025-03-26).

    Accepts JSON-RPC 2.0 requests (single or batch array).
    Returns JSON responses. Notifications (no 'id') return 202.
    """
    # A9 gate runs before any processing. get_current_active_user already
    # validates identity; this guard is a defense-in-depth for the
    # pre-auth path (CORS preflight won't hit here, but cookie-based
    # attempts to the POST will).
    _require_bearer_or_apikey(request)

    body = await request.json()

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

    # JSON-RPC 2.0 batch support — array of requests
    if isinstance(body, list):
        responses = []
        for msg in body:
            resp = await handler.handle(msg)
            if resp is not None:
                responses.append(resp)
        if not responses:
            return Response(status_code=202)
        return JSONResponse(content=responses)

    response = await handler.handle(body)

    if response is None:
        return Response(status_code=202)

    return JSONResponse(content=response)
