"""Hooks that run before a workflow run is created and pinned to a definition version."""

import logging
from typing import Awaitable, Callable, List

from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

PreExecutionHook = Callable[[AsyncSession, str, str], Awaitable[None]]

_hooks: List[PreExecutionHook] = []


def register_pre_execution_hook(fn: PreExecutionHook) -> None:
    if fn not in _hooks:
        _hooks.append(fn)


async def run_pre_execution_hooks(db: AsyncSession, tenant_id: str, workflow_id: str) -> None:
    for hook in list(_hooks):
        try:
            await hook(db, tenant_id, workflow_id)
        except Exception:
            logger.exception("Pre-execution hook %s failed for workflow %s", getattr(hook, "__name__", hook), workflow_id)
