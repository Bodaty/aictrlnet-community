"""Version history of workflow definitions.

Every ORM write that changes a workflow's ``definition`` records the new
definition as a ``WorkflowDefinitionVersion`` row, on the same connection and
in the same transaction as the write (the listener lives in
``models.workflow_definition_version``).
"""

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from models.workflow_definition_version import WorkflowDefinitionVersion, version_context  # noqa: F401


async def current_version_id(db: AsyncSession, workflow_id: str) -> Optional[str]:
    result = await db.execute(
        select(WorkflowDefinitionVersion.id)
        .where(WorkflowDefinitionVersion.workflow_id == workflow_id)
        .order_by(WorkflowDefinitionVersion.version.desc())
        .limit(1)
    )
    return result.scalar_one_or_none()


async def get_version(db: AsyncSession, version_id: str) -> Optional[WorkflowDefinitionVersion]:
    return await db.get(WorkflowDefinitionVersion, version_id)
