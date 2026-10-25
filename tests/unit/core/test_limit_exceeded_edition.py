"""The 402 for an exceeded limit names the tier whose limit was hit.

It always said "community": check_limit raised without passing the edition, so
a Business deployment refusing a 6th user (Business Starter's cap) told the
caller they were on Community. Since 25 Oct the user cap is the tenant's
subscription's, and the deployment's only for a tenant with none.
"""

from types import SimpleNamespace

import pytest

from core.enforcement_simple import Edition, EnforcementMode, LicenseEnforcer, LimitExceededException, LimitType


def _enforcer_at_the_limit(deployment, tier, subscribed_tier):
    enforcer = LicenseEnforcer(db=None, mode=EnforcementMode.STRICT)
    enforcer.settings = SimpleNamespace(EDITION=deployment)
    limit = enforcer.EDITION_LIMITS[Edition(tier)][LimitType.USERS]

    async def at_the_limit(_tenant_id, _limit_type):
        return limit

    async def subscription(_tenant_id):
        return subscribed_tier

    enforcer._get_current_usage = at_the_limit
    enforcer._user_cap_tier = subscription
    return enforcer, limit


@pytest.mark.asyncio
@pytest.mark.parametrize("deployment,tier", [("business", "business_starter"), ("enterprise", "enterprise")])
async def test_the_402_names_the_deployments_tier_without_a_subscription(deployment, tier):
    enforcer, limit = _enforcer_at_the_limit(deployment, tier, subscribed_tier=None)
    with pytest.raises(LimitExceededException) as exc:
        await enforcer.check_limit("t-1", LimitType.USERS, increment=1)
    assert exc.value.detail["edition"] == tier
    assert exc.value.detail["limit"] == limit


@pytest.mark.asyncio
async def test_the_402_names_the_subscriptions_tier():
    enforcer, limit = _enforcer_at_the_limit("enterprise", "business_growth", Edition.BUSINESS_GROWTH)
    with pytest.raises(LimitExceededException) as exc:
        await enforcer.check_limit("t-1", LimitType.USERS, increment=1)
    assert exc.value.detail["edition"] == "business_growth"
    assert exc.value.detail["limit"] == limit
