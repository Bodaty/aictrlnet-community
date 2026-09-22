"""Per-caller access rules for MCP server rows.

One place for the rule every MCP surface applies, so the REST endpoints, the
MCP tool handlers and the dispatcher cannot drift apart again:

- READ  : a shared row (``owner_user_id IS NULL``), your own row, or superuser.
- USE   : your own row, or superuser. Using a row connects with the secrets
          stored on it (``api_key``, ``env_vars``), so shared rows are
          superuser-only rather than mirroring the read rule.
- stdio : superuser only. Starting a stdio server runs a command on the
          platform host with the platform's own process environment.

Refusals are the caller's problem to distinguish: every surface answers an
unauthorized row exactly as it answers an unknown id, so existence is not
disclosed.

Lives in ``core/`` because ``adapters/``, ``mcp_server/`` and ``api/v1`` all
import it, and ``core/user_utils`` has no import chain of its own.
"""

from dataclasses import dataclass
from typing import Any, Optional

from core.user_utils import get_safe_attr, get_safe_user_id

# Ids the auth layer substitutes when it cannot resolve a real user
# (core/dependencies.py, business/.../core/dependencies.py, mcp_server/http_transport.py).
# They must never become a shared owner bucket, so they resolve to "no identity".
_SENTINEL_IDS = {"unknown", "anonymous", "none", ""}

# Fields describing how to reach a server. They carry credentials often enough
# (`--token=…` in args, `?key=…` in a url, the whole of env_vars) that only a
# caller who may USE the row sees them.
CONNECTION_FIELDS = ("command", "args", "env_vars", "url")


@dataclass(frozen=True)
class MCPCaller:
    """The principal an MCP surface authorizes against."""

    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    is_superuser: bool = False


def mcp_caller(current_user: Any) -> MCPCaller:
    """Build a caller from a User object or the dict shape Business/Enterprise pass."""
    raw = get_safe_user_id(current_user) or get_safe_attr(current_user, "sub")
    user_id = str(raw) if raw is not None else None
    if user_id is not None and user_id.strip().lower() in _SENTINEL_IDS:
        user_id = None

    tenant = get_safe_attr(current_user, "tenant_id")
    # is_admin is the dict shape's alias; mirrors _auth_helpers.is_superuser.
    is_super = bool(
        get_safe_attr(current_user, "is_superuser", False)
        or get_safe_attr(current_user, "is_admin", False)
    )
    return MCPCaller(
        user_id=user_id,
        tenant_id=str(tenant) if tenant else None,
        is_superuser=is_super,
    )


def _owner(server: Any) -> Optional[str]:
    """Owner id of a row, or the id itself when a raw-SQL caller passes one."""
    if server is None:
        return None
    if isinstance(server, str):
        return server or None
    owner = get_safe_attr(server, "owner_user_id")
    return str(owner) if owner is not None else None


def can_read(server: Any, caller: MCPCaller) -> bool:
    """Shared row, own row, or superuser. ``server`` may be a row or an owner id."""
    if caller.is_superuser:
        return True
    owner = _owner(server)
    return owner is None or (caller.user_id is not None and owner == caller.user_id)


def can_use(server: Any, caller: MCPCaller) -> bool:
    """Own row, or superuser — shared rows are superuser-only (they hold secrets)."""
    if caller.is_superuser:
        return True
    owner = _owner(server)
    return owner is not None and caller.user_id is not None and owner == caller.user_id


def can_run_stdio(caller: MCPCaller) -> bool:
    """Starting a stdio server is running code on the platform host."""
    return caller.is_superuser


def readable(model: Any, caller: MCPCaller):
    """SQLAlchemy clause for the READ rule over ``model.owner_user_id``."""
    from sqlalchemy import or_, true

    if caller.is_superuser:
        return true()
    shared = model.owner_user_id.is_(None)
    if caller.user_id is None:
        return shared
    return or_(shared, model.owner_user_id == caller.user_id)


def usable(model: Any, caller: MCPCaller):
    """SQLAlchemy clause for the USE rule over ``model.owner_user_id``."""
    from sqlalchemy import false, true

    if caller.is_superuser:
        return true()
    if caller.user_id is None:
        return false()
    return model.owner_user_id == caller.user_id


def visible_config(server: Any, caller: MCPCaller) -> bool:
    """Whether the connection configuration may be returned to this caller."""
    return can_use(server, caller)
