"""One self-serve purchase path: StripeService.create_plan_checkout.

Before this fix /subscription/checkout/session and /license/upgrade each built
their own Stripe checkout: a customer already paying through Stripe could open
a second subscription and be billed twice, Enterprise (contact sales) could be
bought by card when a price was configured, and /license/upgrade returned a
fake /mock-checkout URL outside production. Both endpoints now delegate to
create_plan_checkout, which sells Business tiers only, refuses a second
subscription, and carries remaining in-app trial days with card entry required.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.v1.endpoints import license as license_endpoint
from models.subscription import SubscriptionStatus
from services import stripe_service as stripe_module
from services.stripe_service import StripeService


def _settings(**extra):
    values = dict(STRIPE_SECRET_KEY="sk_test_x", FRONTEND_URL="http://app.test", TRIAL_DAYS=14)
    values.update(extra)
    return SimpleNamespace(**values)


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
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _service(db):
    with patch.object(stripe_module, "get_settings", return_value=_settings()):
        svc = StripeService(db)
    svc.create_checkout_session = AsyncMock(return_value={"checkout_url": "u", "session_id": "s"})
    return svc


@pytest.mark.asyncio
async def test_checkout_refuses_a_second_subscription_for_a_paying_customer():
    svc = _service(MagicMock(execute=AsyncMock(return_value=_result(_sub()))))
    with pytest.raises(ValueError, match="already have a paid subscription"):
        await svc.create_plan_checkout("user-1", "business_growth", "monthly")
    svc.create_checkout_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkout_refuses_enterprise_card_purchase():
    svc = _service(MagicMock(execute=AsyncMock(return_value=_result(None))))
    with pytest.raises(ValueError, match="contact-sales"):
        await svc.create_plan_checkout("user-1", "enterprise", "monthly")
    svc.create_checkout_session.assert_not_awaited()


@pytest.mark.asyncio
async def test_checkout_carries_remaining_in_app_trial_with_card_required():
    trial = _sub(
        status=SubscriptionStatus.TRIALING,
        stripe_subscription_id=None,
        trial_end=datetime.utcnow() + timedelta(days=5, hours=2),
    )
    svc = _service(MagicMock(execute=AsyncMock(return_value=_result(trial))))

    await svc.create_plan_checkout("user-1", "business_starter", "yearly")

    svc.create_checkout_session.assert_awaited_once_with(
        user_id="user-1",
        plan="business_starter",
        billing_period="yearly",
        trial_days=6,
        collect_payment_method=True,
    )


@pytest.mark.asyncio
async def test_checkout_after_a_cancelled_subscription_is_allowed():
    ended = _sub(status=SubscriptionStatus.CANCELED)
    svc = _service(MagicMock(execute=AsyncMock(return_value=_result(ended))))

    await svc.create_plan_checkout("user-1", "business_scale", "monthly")

    svc.create_checkout_session.assert_awaited_once_with(
        user_id="user-1",
        plan="business_scale",
        billing_period="monthly",
        trial_days=0,
        collect_payment_method=False,
    )


@pytest.mark.asyncio
async def test_license_upgrade_never_returns_a_mock_checkout(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    request = SimpleNamespace(target_plan="business_scale", billing_period="monthly")
    with patch("core.config.get_settings", return_value=_settings(STRIPE_SECRET_KEY="")):
        with pytest.raises(HTTPException) as exc:
            await license_endpoint.upgrade_license(request=request, current_user={"sub": "user-1"}, db=MagicMock())
    assert exc.value.status_code == 503


@pytest.mark.asyncio
async def test_license_upgrade_uses_the_single_checkout_path():
    request = SimpleNamespace(target_plan="business_scale", billing_period="monthly")
    service_cls = MagicMock()
    service_cls.return_value.create_plan_checkout = AsyncMock(
        return_value={"checkout_url": "https://checkout.stripe.test/s", "session_id": "cs_1"}
    )
    with patch("core.config.get_settings", return_value=_settings()), \
         patch("services.stripe_service.StripeService", service_cls):
        resp = await license_endpoint.upgrade_license(request=request, current_user={"sub": "user-1"}, db=MagicMock())

    service_cls.return_value.create_plan_checkout.assert_awaited_once_with(
        user_id="user-1", plan="business_scale", billing_period="monthly"
    )
    assert resp.checkout_url == "https://checkout.stripe.test/s"
