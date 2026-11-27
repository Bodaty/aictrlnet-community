"""POST /upgrade/subscribe must not exist.

It created a Stripe subscription server-side with no card and wrote the plan
locally as ACTIVE whatever status Stripe returned (an unpaid subscription is
"incomplete"), or as a card-less TRIALING subscription with start_trial=true,
for any Business tier. No frontend screen called it; the one purchase path is
StripeService.create_plan_checkout.
"""

from api.v1.endpoints import upgrade


def test_upgrade_subscribe_route_removed():
    paths = {route.path for route in upgrade.router.routes}
    assert not any(path.endswith("/subscribe") for path in paths), paths
