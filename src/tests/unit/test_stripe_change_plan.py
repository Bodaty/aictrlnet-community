"""A paid plan change must go through Stripe; the webhook writes plan_id.

Before this fix POST /subscription/change set subscription.plan_id directly
with a $0 PENDING billing row: any signed-in trial user could move to any
plan without paying. StripeService.change_plan now asks Stripe to swap the
price with proration invoiced at once and payment_behavior=pending_if_incomplete,
so the new price applies only once paid, and never touches plan_id itself.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from models.subscription import SubscriptionStatus
from services import stripe_service as stripe_module
from services.stripe_service import CheckoutRequiredError, StripeService


def _result(value):
    res = MagicMock()
    res.scalar_one_or_none.return_value = value
    return res


def _sub(**kw):
    base = dict(
        id="sub-local-1",
        user_id="user-1",
        plan_id="plan-starter",
        status=SubscriptionStatus.ACTIVE,
        stripe_subscription_id="sub_stripe_1",
        trial_end=None,
        current_period_end=datetime.utcnow() + timedelta(days=20),
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _service(db):
    settings = SimpleNamespace(
        STRIPE_SECRET_KEY="sk_test_x",
        FRONTEND_URL="http://app.test",
        STRIPE_PRICE_BUSINESS_STARTER="price_starter_m",
        STRIPE_PRICE_BUSINESS_SCALE="price_scale_m",
    )
    with patch.object(stripe_module, "get_settings", return_value=settings):
        svc = StripeService(db)
    svc._stripe = MagicMock()
    return svc


@pytest.mark.asyncio
async def test_change_plan_without_stripe_subscription_requires_checkout():
    trial = _sub(status=SubscriptionStatus.TRIALING, stripe_subscription_id=None)
    db = MagicMock(execute=AsyncMock(return_value=_result(trial)), commit=AsyncMock())
    svc = _service(db)

    with pytest.raises(CheckoutRequiredError):
        await svc.change_plan("user-1", "business_scale", "monthly")

    svc._stripe.Subscription.modify.assert_not_called()
    assert trial.plan_id == "plan-starter"
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_change_plan_asks_stripe_and_leaves_plan_id_to_the_webhook():
    sub = _sub()
    db = MagicMock(execute=AsyncMock(return_value=_result(sub)), commit=AsyncMock())
    svc = _service(db)
    svc._stripe.Subscription.retrieve.return_value = {
        "items": {"data": [{"id": "si_1", "price": {"id": "price_starter_m"}}]}
    }
    svc._stripe.Subscription.modify.return_value = {"pending_update": {"expires_at": 1}}

    result = await svc.change_plan("user-1", "business_scale", "monthly")

    svc._stripe.Subscription.modify.assert_called_once_with(
        "sub_stripe_1",
        items=[{"id": "si_1", "price": "price_scale_m"}],
        proration_behavior="always_invoice",
        payment_behavior="pending_if_incomplete",
    )
    assert result["status"] == "pending_payment"
    assert sub.plan_id == "plan-starter"
    db.commit.assert_not_awaited()


@pytest.mark.asyncio
async def test_change_plan_refuses_contact_sales_plans():
    db = MagicMock(execute=AsyncMock(return_value=_result(_sub())))
    svc = _service(db)
    with pytest.raises(ValueError):
        await svc.change_plan("user-1", "enterprise", "monthly")
    svc._stripe.Subscription.modify.assert_not_called()
