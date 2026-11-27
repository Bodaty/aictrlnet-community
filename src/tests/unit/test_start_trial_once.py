"""POST /billing/start-trial: one free trial per user.

Each call opens a new card-less Stripe trial checkout. Before this fix nothing
checked history, so a user whose trial had ended (or who had cancelled a paid
subscription) could start another free trial, indefinitely. It is now allowed
only for a user with no subscription yet, or still inside the in-app trial
signup created.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.v1.endpoints import billing
from models.subscription import SubscriptionStatus


def _sub(**kw):
    base = dict(status=SubscriptionStatus.TRIALING, stripe_subscription_id=None, trial_end=None)
    base.update(kw)
    return SimpleNamespace(**base)


def _db(history):
    res = MagicMock()
    res.scalars.return_value.all.return_value = history
    res.scalars.return_value.first.return_value = history[0] if history else None
    return MagicMock(execute=AsyncMock(return_value=res))


def _stripe_cls():
    service_cls = MagicMock()
    service_cls.return_value.create_checkout_session = AsyncMock(
        return_value={"checkout_url": "u", "session_id": "s"}
    )
    return service_cls


@pytest.mark.parametrize("history", [
    [_sub(trial_end=datetime.utcnow() - timedelta(days=1))],
    [_sub(status=SubscriptionStatus.CANCELED, stripe_subscription_id="sub_old")],
    [_sub(status=SubscriptionStatus.EXPIRED)],
    [_sub(status=SubscriptionStatus.TRIALING, stripe_subscription_id="sub_trial",
          trial_end=datetime.utcnow() + timedelta(days=3))],
])
@pytest.mark.asyncio
async def test_start_trial_refuses_a_second_trial(history):
    service_cls = _stripe_cls()
    with patch.object(billing, "StripeService", service_cls), \
         patch.object(billing, "get_settings", return_value=SimpleNamespace(TRIAL_DAYS=14)):
        with pytest.raises(HTTPException) as exc:
            await billing.start_trial(
                request=SimpleNamespace(plan="business_starter"),
                current_user={"sub": "user-1"},
                db=_db(history),
            )
    assert exc.value.status_code == 409
    service_cls.return_value.create_checkout_session.assert_not_awaited()


@pytest.mark.parametrize("history", [
    [],
    [_sub(trial_end=datetime.utcnow() + timedelta(days=9, hours=1))],
])
@pytest.mark.asyncio
async def test_start_trial_allowed_once(history):
    service_cls = _stripe_cls()
    with patch.object(billing, "StripeService", service_cls), \
         patch.object(billing, "get_settings", return_value=SimpleNamespace(TRIAL_DAYS=14)):
        resp = await billing.start_trial(
            request=SimpleNamespace(plan="business_starter"),
            current_user={"sub": "user-1"},
            db=_db(history),
        )
    assert resp.trial_days == 14
    service_cls.return_value.create_checkout_session.assert_awaited_once()


@pytest.mark.asyncio
async def test_start_trial_refuses_plans_not_sold_by_card():
    service_cls = _stripe_cls()
    with patch.object(billing, "StripeService", service_cls):
        with pytest.raises(HTTPException) as exc:
            await billing.start_trial(
                request=SimpleNamespace(plan="business_onprem"),
                current_user={"sub": "user-1"},
                db=_db([]),
            )
    assert exc.value.status_code == 400
