"""Hooks that run before a workflow run is created and pinned to a definition version.

A hook gets the tenant and workflow ids, never the caller's session: it opens its own, so a
database error inside a hook is logged and skipped and cannot abort the run being created.
"""

import logging
from typing import Awaitable, Callable, List, Optional

logger = logging.getLogger(__name__)

PreExecutionHook = Callable[[Optional[str], str], Awaitable[None]]

_hooks: List[PreExecutionHook] = []


def register_pre_execution_hook(fn: PreExecutionHook) -> None:
    if fn not in _hooks:
        _hooks.append(fn)


async def run_pre_execution_hooks(tenant_id: Optional[str], workflow_id: str) -> None:
    for hook in list(_hooks):
        try:
            await hook(tenant_id, workflow_id)
        except Exception:
            logger.exception("Pre-execution hook %s failed for workflow %s", getattr(hook, "__name__", hook), workflow_id)
