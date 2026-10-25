"""The 402 for an exceeded limit names the tier whose limit was hit.

It always said "community": check_limit raised without passing the edition, so
a Business deployment refusing a 6th user (Business Starter's cap) told the
caller they were on Community.
"""

from types import SimpleNamespace

import pytest

from core.enforcement_simple import EnforcementMode, LicenseEnforcer, LimitExceededException, LimitType


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment,tier", [("business", "business_starter"), ("enterprise", "enterprise")])
async def test_the_402_names_the_deployments_tier(deployment, tier):
    enforcer = LicenseEnforcer(db=None, mode=EnforcementMode.STRICT)
    enforcer.settings = SimpleNamespace(EDITION=deployment)
    limit = enforcer.EDITION_LIMITS[next(e for e in enforcer.EDITION_LIMITS if e.value == tier)][LimitType.USERS]

    async def at_the_limit(_tenant_id, _limit_type):
        return limit

    enforcer._get_current_usage = at_the_limit
    with pytest.raises(LimitExceededException) as exc:
        await enforcer.check_limit("t-1", LimitType.USERS, increment=1)
    assert exc.value.detail["edition"] == tier
    assert exc.value.detail["limit"] == limit
