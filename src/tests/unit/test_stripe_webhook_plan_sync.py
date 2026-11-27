"""customer.subscription.updated must sync plan_id from the price Stripe applied.

Before this fix the handler updated status and period dates only, so a plan
change made in Stripe (an API update, the Customer Portal or the Dashboard)
never reached the platform: the customer paid for one tier and was entitled
to another. The applied price is now the only source of the paid plan_id, and
a change still waiting on payment (`pending_update`) leaves it alone because
the items keep the old price until the invoice is paid.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.subscription import SubscriptionStatus
from services import stripe_service as stripe_module
from services.stripe_service import StripeService


PRICES = {
    "STRIPE_PRICE_BUSINESS_STARTER": "price_starter_m",
    "STRIPE_PRICE_BUSINESS_GROWTH": "price_growth_m",
    "STRIPE_PRICE_BUSINESS_SCALE": "price_scale_m",
    "STRIPE_PRICE_BUSINESS_STARTER_ANNUAL": "price_starter_y",
    "STRIPE_PRICE_BUSINESS_GROWTH_ANNUAL": "price_growth_y",
    "STRIPE_PRICE_BUSINESS_SCALE_ANNUAL": "price_scale_y",
    "STRIPE_PRICE_ENTERPRISE": "price_ent_m",
    "STRIPE_PRICE_ENTERPRISE_ANNUAL": "price_ent_y",
}


def _result(value):
    res = MagicMock()
    res.scalar_one_or_none.return_value = value
    return res


def _sub():
    return SimpleNamespace(
        id="sub-local-1",
        user_id="user-1",
        plan_id="plan-starter",
        status=SubscriptionStatus.ACTIVE,
        stripe_subscription_id="sub_stripe_1",
        cancel_at_period_end=False,
        current_period_end=datetime.utcnow() + timedelta(days=20),
    )


def _service(db):
    settings = SimpleNamespace(**PRICES, STRIPE_SECRET_KEY="sk_test_x", FRONTEND_URL="http://app.test")
    with patch.object(stripe_module, "get_settings", return_value=settings):
        return StripeService(db)


@pytest.mark.asyncio
async def test_subscription_updated_syncs_plan_id_from_applied_price():
    sub = _sub()
    scale_row = SimpleNamespace(id="plan-scale", name="business_scale", edition="business")
    db = MagicMock(
        execute=AsyncMock(side_effect=[_result(sub), _result(scale_row), _result(scale_row)]),
        get=AsyncMock(return_value=SimpleNamespace(edition="business")),
        commit=AsyncMock(),
    )

    await _service(db)._handle_subscription_updated({
        "id": "sub_stripe_1",
        "status": "active",
        "cancel_at_period_end": True,
        "items": {"data": [{"id": "si_1", "price": {"id": "price_scale_y"}}]},
    })

    assert sub.plan_id == "plan-scale"
    assert sub.cancel_at_period_end is True
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_subscription_updated_reads_the_legacy_plan_field():
    sub = _sub()
    growth_row = SimpleNamespace(id="plan-growth", name="business_growth", edition="business")
    db = MagicMock(
        execute=AsyncMock(side_effect=[_result(sub), _result(growth_row), _result(growth_row)]),
        get=AsyncMock(return_value=SimpleNamespace(edition="business")),
        commit=AsyncMock(),
    )

    await _service(db)._handle_subscription_updated({
        "id": "sub_stripe_1",
        "status": "active",
        "items": {"data": [{"id": "si_1", "plan": {"id": "price_growth_m"}}]},
    })

    assert sub.plan_id == "plan-growth"


@pytest.mark.asyncio
async def test_subscription_updated_leaves_plan_id_for_an_unknown_price():
    sub = _sub()
    starter_row = SimpleNamespace(id="plan-starter", name="business_starter", edition="business")
    db = MagicMock(
        execute=AsyncMock(side_effect=[_result(sub), _result(starter_row)]),
        get=AsyncMock(return_value=SimpleNamespace(edition="business")),
        commit=AsyncMock(),
    )

    await _service(db)._handle_subscription_updated({
        "id": "sub_stripe_1",
        "status": "active",
        "items": {"data": [{"id": "si_1", "price": {"id": "price_not_ours"}}]},
    })

    assert sub.plan_id == "plan-starter"
