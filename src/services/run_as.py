"""Whose connections an unattended run uses (the workflow's run-as owner).

A run started with no user (a schedule, a work item, a trigger) runs as "system", so a
step that needs a person's connection had no one to resolve it for. The workflow's
``run_as_user_id`` names that person; it is honoured only while they are an active user
of the workflow's own tenant, and is used only to resolve credentials (node context
``owner_id`` / ``workflow_owner_id``), never as the run's user.
"""

import logging
from typing import Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

DEFAULT_TENANT = "default-tenant"
CONTEXT_KEY = "workflow_owner_id"


def owner_for(user_id: Optional[str]) -> Optional[str]:
    """The run-as owner a new workflow gets from the person creating it ("system" is no one)."""
    return str(user_id) if user_id and str(user_id) != "system" else None


def _same_tenant(user_tenant: Optional[str], workflow_tenant: Optional[str]) -> bool:
    # Community and Business users carry NULL for the one default tenant.
    return (user_tenant or DEFAULT_TENANT) == (workflow_tenant or DEFAULT_TENANT)


async def run_as_owner(db, workflow) -> Optional[str]:
    """The workflow's owner if they may stand behind its unattended runs, else None."""
    owner = getattr(workflow, "run_as_user_id", None)
    if not owner:
        return None
    row = (await db.execute(text("SELECT tenant_id, is_active FROM users WHERE id = :u"), {"u": owner})).first()
    if row is None or not row[1] or not _same_tenant(row[0], getattr(workflow, "tenant_id", None)):
        logger.warning("Workflow %s: its run-as owner is not an active user of its tenant; "
                       "unattended runs carry no owner", getattr(workflow, "id", "?"))
        return None
    return str(owner)


def node_context_owner(execution_context: Optional[dict]) -> Optional[str]:
    """The owner a run recorded at its start, for every node (first run and resume)."""
    return (execution_context or {}).get(CONTEXT_KEY)
