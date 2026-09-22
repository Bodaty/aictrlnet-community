"""MCP Dispatcher for routing tasks to appropriate MCP servers."""

from typing import Dict, Any, List, Optional
import random
import logging
from sqlalchemy import select

from .server_adapter import MCPServerAdapter
from core.mcp_access import can_use, readable, usable
from models import MCPServer, MCPServerCapability

logger = logging.getLogger(__name__)


class MCPDispatcher:
    """Routes tasks to appropriate MCP servers based on capabilities"""
    
    def __init__(self, control_plane_url: str):
        self.control_plane_url = control_plane_url
        self.servers: Dict[str, MCPServerAdapter] = {}
        self._initialized = False
        
    async def initialize(self):
        """Kept for subclasses that call it; it no longer loads anything.

        This used to hold every active server — with its ``api_key`` — in a
        process-wide dict keyed by name AND id, shared by every caller, loaded
        once and never refreshed. Rows are resolved per call now, on the
        caller's own session, so there is nothing to preload and nothing for
        one caller's request to find of another's.
        """
        self._initialized = True

    async def _resolve_servers(
        self,
        db,
        caller,
        *,
        capability: Optional[str] = None,
        server_ref: Optional[str] = None,
        rule: str = "use",
    ) -> List[MCPServer]:
        """Rows this caller may act on, newest rule applied in the query."""
        clause = usable(MCPServer, caller) if rule == "use" else readable(MCPServer, caller)
        query = select(MCPServer).filter(clause)

        if server_ref:
            # By id only: names are not unique across owners.
            query = query.filter(MCPServer.id == str(server_ref))
        else:
            # Implicit selection never reaches beyond the caller's own rows
            # (plus shared ones for a superuser), so no request silently spends
            # another owner's credentials, and only a server declared active
            # and reachable over HTTP is a candidate.
            query = query.filter(
                MCPServer.status == "active", MCPServer.url.isnot(None)
            )
            if not caller.is_superuser:
                query = query.filter(MCPServer.owner_user_id == caller.user_id)
            if capability:
                query = query.join(MCPServerCapability).filter(
                    MCPServerCapability.capability == capability,
                    MCPServerCapability.supported.is_(True),
                )
        result = await db.execute(query)
        return list(result.scalars().unique().all())

    def _adapter_for(self, server: MCPServer) -> MCPServerAdapter:
        """A client for one authorized row. Callers close it when finished."""
        adapter = MCPServerAdapter(
            control_plane_url=self.control_plane_url,
            mcp_server_url=server.url or "",
            api_key=server.api_key,
            server_name=server.name,
        )
        adapter.server_id = server.id
        return adapter

    async def dispatch_task(
        self, task: Dict[str, Any], *, caller, db
    ) -> Dict[str, Any]:
        """Route a task to a server this caller may use.

        `caller` and `db` are required: authorization is per row, and the rows
        are read on the caller's own session so a request never resolves
        servers through a connection with someone else's tenant context.
        """
        payload = task.get("payload", {}) or {}
        server_ref = payload.get("server_name")
        api_type = payload.get("api_type", "message")

        servers = await self._resolve_servers(
            db, caller, capability=None if server_ref else api_type,
            server_ref=server_ref,
        )
        if not servers:
            if server_ref:
                # Same answer for "not yours" and "no such server".
                return {
                    "success": False,
                    "error": f"MCP server '{server_ref}' is not available",
                }
            return {"success": False, "error": "No MCP server available for this task"}

        server_row = servers[0] if server_ref else random.choice(servers)
        adapter = self._adapter_for(server_row)
        try:
            return await adapter.process_task(task)
        except Exception as e:
            logger.error(f"Task processing failed: {str(e)}")
            return {"success": False, "error": str(e), "server_name": server_row.name}
        finally:
            await adapter.close()

    async def list_available_servers(
        self, capability: Optional[str] = None, *, caller, db
    ) -> List[Dict[str, Any]]:
        """Servers this caller may see: shared ones, their own, all for an admin."""
        servers = await self._resolve_servers(db, caller, rule="read")
        listed = []
        for server in servers:
            caps = (
                await db.execute(
                    select(MCPServerCapability).filter_by(
                        server_id=server.id, supported=True
                    )
                )
            ).scalars().all()
            capability_names = [c.capability for c in caps]
            if capability and capability not in capability_names:
                continue
            listed.append({
                "name": server.name,
                "url": server.url if can_use(server, caller) else None,
                "capabilities": capability_names,
                "status": server.status,
                "server_id": server.id,
            })
        return listed

    async def check_health(self, server: MCPServer) -> Dict[str, Any]:
        """Health of one already-authorized row."""
        adapter = self._adapter_for(server)
        try:
            return await adapter.check_server_health()
        finally:
            await adapter.close()

    async def close_all(self):
        """Nothing is held open between calls; kept for callers that call it."""
        self.servers.clear()
        self._initialized = False