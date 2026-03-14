"""FastAPI dependencies for license enforcement.

Provides reusable Depends() wrappers around LicenseEnforcer
so endpoints can enforce usage limits declaratively.
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, Response
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.dependencies import get_current_user_safe
from core.config import get_settings
from core.enforcement_simple import (
    LicenseEnforcer,
    LimitType,
    EnforcementMode,
    LimitExceededException,
)
from core.user_utils import get_safe_attr

logger = logging.getLogger(__name__)

# Default enforcement mode per edition
_EDITION_ENFORCEMENT = {
    "community": EnforcementMode.SOFT,
    "business": EnforcementMode.STRICT,
    "enterprise": EnforcementMode.STRICT,
}


def _get_enforcement_mode() -> EnforcementMode:
    settings = get_settings()
    edition = getattr(settings, "EDITION", "community").lower()
    return _EDITION_ENFORCEMENT.get(edition, EnforcementMode.SOFT)


class EnforceLimitDep:
    """Callable dependency that checks a specific limit type.

    Usage in an endpoint:
        @router.post("/agents")
        async def create_agent(
            ...,
            _limit=Depends(EnforceLimitDep(LimitType.AGENTS)),
        ):
    """

    def __init__(self, limit_type: LimitType, increment: int = 1):
        self.limit_type = limit_type
        self.increment = increment

    async def __call__(
        self,
        response: Response,
        db: AsyncSession = Depends(get_db),
        current_user=Depends(get_current_user_safe),
    ) -> dict:
        mode = _get_enforcement_mode()
        if mode == EnforcementMode.NONE:
            return {"allowed": True}

        from core.tenant_context import get_current_tenant_id

        tenant_id = get_safe_attr(current_user, "tenant_id", None) or get_current_tenant_id()
        enforcer = LicenseEnforcer(db, mode=mode)

        try:
            result = await enforcer.check_limit(
                tenant_id=tenant_id,
                limit_type=self.limit_type,
                increment=self.increment,
            )
        except LimitExceededException as exc:
            raise HTTPException(status_code=402, detail=exc.detail)

        if result.get("warning"):
            response.headers["X-Usage-Warning"] = result["warning"]

        return result


# Pre-built dependencies for common resource types
enforce_workflow_limit = EnforceLimitDep(LimitType.WORKFLOWS)
enforce_agent_limit = EnforceLimitDep(LimitType.AGENTS)
enforce_user_limit = EnforceLimitDep(LimitType.USERS)
enforce_api_call_limit = EnforceLimitDep(LimitType.API_CALLS)
enforce_execution_limit = EnforceLimitDep(LimitType.EXECUTIONS)
