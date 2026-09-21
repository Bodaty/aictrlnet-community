"""Self MCP server id resolution + the admin transaction MCP audit writes run on.

Two structural facts drive this module's shape — don't "simplify" them away:

1. The request's own DB session can be unusable by the time the audit write
   runs in ``execute_tool``'s ``finally``: ``metering.py`` runs the handler
   under ``asyncio.wait_for``, and a timeout cancels asyncpg mid-statement.
   The audit write for ``status="timeout"`` must not reuse that session —
   it may be sitting on an aborted transaction, and its own ``commit()``
   would also commit/rollback whatever the handler left pending.
2. ``audit_logs`` is FORCE ROW LEVEL SECURITY (``mcp_servers`` is not —
   RLS there is deferred to a separate follow-up). Editions connect as the
   non-superuser role ``app``, and this engine's connection pins
   ``app.current_tenant_id`` to ``DEFAULT_TENANT_ID`` at connect (below),
   so a plain insert into ``audit_logs`` for any OTHER tenant — exactly
   what the SOC dual-write does for a non-default-tenant caller — is
   rejected by ``tenant_isolation_audit_logs``. ``SET LOCAL app.is_admin =
   'true'`` lets that insert through ``admin_bypass_audit_logs`` for the
   rest of that transaction — but once the transaction ends, the GUC reads
   ``''`` and every later query against an RLS table on the SAME
   CONNECTION raises ``invalid input syntax for type boolean: ""``
   (policies cast ``current_setting('app.is_admin', true)::boolean``). A
   transaction-local admin flag must therefore NEVER touch a pooled
   connection — this module uses a dedicated NullPool engine for exactly
   that reason.
"""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from core.config import get_settings
from core.tenant_context import DEFAULT_TENANT_ID

from . import observability

logger = logging.getLogger(__name__)

SELF_SERVER_ID = "aictrlnet-mcp-transport"
AUDIT_MAX_CONCURRENCY = 4
AUDIT_ACQUIRE_TIMEOUT_S = 5.0

# How often to log the "self server row missing" warning, per process.
_UNRESOLVED_LOG_INTERVAL_S = 60.0
_last_unresolved_log_at = 0.0


class AuditSessionUnavailable(RuntimeError):
    """Raised when the audit connection semaphore can't be acquired in time."""


_audit_engine = None
_audit_session_maker = None

# Per-event-loop semaphores. pytest-asyncio runs each test on its own loop,
# and a module-level asyncio.Semaphore that has once blocked stays bound to
# the loop it blocked on — a WeakKeyDictionary keyed by the running loop
# avoids that, and lets loops (and their semaphores) be garbage collected.
_semaphores: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def get_audit_engine():
    """Lazily build the ONE module-level engine audit writes run on.

    NullPool is a correctness requirement (see module docstring point 2),
    not a performance choice — it is also why this does not reuse
    ``core.database.get_session_maker()``: borrowing a second connection
    from the request's pooled engine while the request still holds one is
    a self-deadlock under load (pool 10+10 per worker, pool_timeout=30).
    """
    global _audit_engine
    if _audit_engine is None:
        settings = get_settings()
        _audit_engine = create_async_engine(
            str(settings.DATABASE_URL),
            echo=False,
            future=True,
            poolclass=NullPool,
            connect_args={
                "server_settings": {
                    "app.current_tenant_id": DEFAULT_TENANT_ID,
                }
            },
        )
    return _audit_engine


def _get_session_maker():
    global _audit_session_maker
    if _audit_session_maker is None:
        _audit_session_maker = async_sessionmaker(
            get_audit_engine(), class_=AsyncSession, expire_on_commit=False
        )
    return _audit_session_maker


def _semaphore_for_running_loop() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _semaphores.get(loop)
    if sem is None:
        sem = asyncio.Semaphore(AUDIT_MAX_CONCURRENCY)
        _semaphores[loop] = sem
    return sem


@asynccontextmanager
async def audit_session() -> AsyncIterator[AsyncSession]:
    """Yield an admin, RLS-bypassing session on the dedicated audit engine.

    Never commits — the caller (``audit_mcp_operation``) owns the commit.
    """
    sem = _semaphore_for_running_loop()
    try:
        await asyncio.wait_for(sem.acquire(), timeout=AUDIT_ACQUIRE_TIMEOUT_S)
    except asyncio.TimeoutError as e:
        raise AuditSessionUnavailable(
            f"could not acquire an audit connection within "
            f"{AUDIT_ACQUIRE_TIMEOUT_S}s ({AUDIT_MAX_CONCURRENCY} already in flight)"
        ) from e
    try:
        async with _get_session_maker()() as session:
            await session.execute(text("SET LOCAL app.is_admin = 'true'"))
            yield session
    finally:
        sem.release()


async def resolve_self_server_id(db: AsyncSession) -> Optional[str]:
    """Resolve the self MCP server row's id for FK-safe audit inserts.

    Called from inside ``audit_session()`` in practice, but does not itself
    require the admin transaction: RLS on ``mcp_servers`` is not enabled by
    the current migration set (deferred to a separate follow-up), so a
    plain SELECT for the self row succeeds on any session with SELECT on
    the table. The admin transaction this module builds is for the write
    side — ``audit_logs`` inserts, which ARE RLS-protected (see module
    docstring point 2) — not for this read.

    No positive cache: the row is small and rarely missing, and caching a
    stale "resolved" answer across a deploy that dropped the row would be
    worse than the extra SELECT.
    """
    global _last_unresolved_log_at

    row = (
        await db.execute(
            text("SELECT 1 FROM mcp_servers WHERE id = :id"),
            {"id": SELF_SERVER_ID},
        )
    ).first()
    if row is not None:
        return SELF_SERVER_ID

    observability.record_audit_server_unresolved()
    now = time.monotonic()
    if now - _last_unresolved_log_at >= _UNRESOLVED_LOG_INTERVAL_S:
        _last_unresolved_log_at = now
        logger.warning(
            "MCP self server row %r not found in mcp_servers — audit rows "
            "will be written with server_id=NULL",
            SELF_SERVER_ID,
        )
    return None


__all__ = [
    "AUDIT_ACQUIRE_TIMEOUT_S",
    "AUDIT_MAX_CONCURRENCY",
    "AuditSessionUnavailable",
    "SELF_SERVER_ID",
    "audit_session",
    "get_audit_engine",
    "resolve_self_server_id",
]
