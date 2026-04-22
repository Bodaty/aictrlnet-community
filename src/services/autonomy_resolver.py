"""Autonomy hierarchy resolver — Community stub.

Community has no Organization or Department concept, so this stub resolves:

    workflow.autonomy_locked (if set) →
    workflow.autonomy_level →
    user PersonalAgentConfig.user_context["autonomy_level"] →
    SYSTEM_DEFAULT_LEVEL

Business edition overrides by registering its own resolver that handles the
full 6-tier precedence including agent / department / organization scopes.

The `ResolvedAutonomy` dataclass and `AutonomyGate` protocol live here (in
Community) so that Business can import them. Community must NOT import
anything from Business at module load time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from services.autonomy_taxonomy import (
    SYSTEM_DEFAULT_LEVEL,
    level_to_auto_approve_threshold,
    level_to_phase,
)

logger = logging.getLogger(__name__)


@dataclass
class ResolvedAutonomy:
    """Resolved autonomy for one workflow execution.

    Populated once at execution start and stored in the node execution
    context as `context["autonomy"]`. Never mutated after creation.
    """

    level: int
    phase: str
    source: str  # which tier won: workflow_locked/agent/workflow/user/department/organization/system_default
    auto_approve_threshold: float
    approver_ids: List[str] = field(default_factory=list)
    trace: List[Tuple[str, Optional[int]]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "phase": self.phase,
            "source": self.source,
            "auto_approve_threshold": self.auto_approve_threshold,
            "approver_ids": list(self.approver_ids),
            "trace": [[src, lvl] for src, lvl in self.trace],
        }


class AutonomyGate(Protocol):
    """Protocol implemented by the Business `RuntimeAutonomyGate` and the
    Community `NoopAutonomyGate`. Kept as a Protocol to avoid inheritance
    across the edition boundary.
    """

    async def check(
        self,
        node_instance: Any,
        workflow_instance: Any,
        context: dict,
    ) -> None:
        """Decide whether to allow, pause-for-approval, or reject this node.

        Raises `RuntimeError` on rejection (matches manual ApprovalNode
        semantics). Returns normally when the node is allowed to proceed.
        """
        ...


class CommunityAutonomyResolver:
    """Stub resolver for pure-Community deploys.

    Resolves workflow → user-comfort → system default. Returns an empty
    `approver_ids` list because Community has no organization/department
    admin roles. Callers that need approver resolution should run Business.
    """

    def __init__(self, db: AsyncSession):
        self.db = db

    async def resolve(
        self,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        department_id: Optional[str] = None,
    ) -> ResolvedAutonomy:
        trace: List[Tuple[str, Optional[int]]] = []
        approver_ids: List[str] = []

        # Step 1: workflow.autonomy_locked short-circuits
        workflow_level: Optional[int] = None
        workflow_locked: bool = False
        workflow_created_by: Optional[str] = None
        if workflow_id:
            workflow_level, workflow_locked, workflow_created_by = await self._load_workflow(workflow_id)
            trace.append(("workflow", workflow_level))
            if workflow_locked and workflow_level is not None:
                if workflow_created_by:
                    approver_ids.append(workflow_created_by)
                return self._build(workflow_level, "workflow_locked", approver_ids, trace)

        # Community has no agent-level lookup (EnhancedAgent lives in Business).

        # Step 3: workflow.autonomy_level
        if workflow_level is not None:
            if workflow_created_by:
                approver_ids.append(workflow_created_by)
            return self._build(workflow_level, "workflow", approver_ids, trace)

        # Step 4: user PersonalAgentConfig.user_context["autonomy_level"]
        if user_id and user_id != "system":
            user_level = await self._load_user_autonomy(user_id)
            trace.append(("user", user_level))
            if user_level is not None:
                approver_ids.append(user_id)
                return self._build(user_level, "user", approver_ids, trace)

        # Community has no department/organization lookup.

        # Step 7: system default
        trace.append(("system_default", SYSTEM_DEFAULT_LEVEL))
        if workflow_created_by:
            approver_ids.append(workflow_created_by)
        elif user_id and user_id != "system":
            approver_ids.append(user_id)
        return self._build(SYSTEM_DEFAULT_LEVEL, "system_default", approver_ids, trace)

    async def _load_workflow(self, workflow_id: str) -> Tuple[Optional[int], bool, Optional[str]]:
        """Return (autonomy_level, autonomy_locked, created_by) for a workflow."""
        from models.community_complete import WorkflowDefinition  # local import avoids boot-time cycles

        try:
            result = await self.db.execute(
                select(
                    WorkflowDefinition.autonomy_level,
                    WorkflowDefinition.autonomy_locked,
                ).where(WorkflowDefinition.id == workflow_id)
            )
            row = result.one_or_none()
        except Exception:
            logger.debug("autonomy resolver: workflow lookup failed for %s", workflow_id, exc_info=True)
            return None, False, None
        if row is None:
            return None, False, None
        autonomy_level, autonomy_locked = row
        # WorkflowDefinition in Community has no explicit created_by column today;
        # Business enriches this via its own resolver. Return None for creator here.
        return autonomy_level, bool(autonomy_locked), None

    async def _load_user_autonomy(self, user_id: str) -> Optional[int]:
        from models.personal_agent import PersonalAgentConfig  # local import

        try:
            result = await self.db.execute(
                select(PersonalAgentConfig.user_context).where(PersonalAgentConfig.user_id == user_id)
            )
            row = result.scalar_one_or_none()
        except Exception:
            logger.debug("autonomy resolver: user config lookup failed for %s", user_id, exc_info=True)
            return None
        if not row or not isinstance(row, dict):
            return None
        level = row.get("autonomy_level")
        if isinstance(level, int):
            return level
        if isinstance(level, str) and level.isdigit():
            return int(level)
        return None

    @staticmethod
    def _build(
        level: int,
        source: str,
        approver_ids: List[str],
        trace: List[Tuple[str, Optional[int]]],
    ) -> ResolvedAutonomy:
        level = max(0, min(100, int(level)))
        # De-dupe approver ids while preserving order.
        seen: set = set()
        deduped: List[str] = []
        for uid in approver_ids:
            if uid and uid not in seen:
                seen.add(uid)
                deduped.append(uid)
        return ResolvedAutonomy(
            level=level,
            phase=level_to_phase(level),
            source=source,
            auto_approve_threshold=level_to_auto_approve_threshold(level),
            approver_ids=deduped,
            trace=trace,
        )


# Default export used by Community code paths. Business's plugin hook swaps
# the global resolver for its extended `BusinessAutonomyResolver`.
get_resolver = CommunityAutonomyResolver


class NoopAutonomyGate:
    """Gate implementation for pure-Community deploys. Always allows."""

    async def check(self, node_instance: Any, workflow_instance: Any, context: dict) -> None:
        return None


__all__ = [
    "ResolvedAutonomy",
    "AutonomyGate",
    "CommunityAutonomyResolver",
    "NoopAutonomyGate",
    "get_resolver",
]
