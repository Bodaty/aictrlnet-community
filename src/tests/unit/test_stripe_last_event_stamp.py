"""The webhook records the last Stripe event it processed on the subscription.

The admin billing view shows it, so "did Stripe tell us?" is answerable without
the Stripe Dashboard. Recording it must never fail the webhook.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from services import stripe_service as stripe_module
from services.stripe_service import StripeService


def _service(db):
    with patch.object(stripe_module, "get_settings", return_value=SimpleNamespace(STRIPE_SECRET_KEY="sk_test_x")):
        return StripeService(db)


def _result(value):
    res = MagicMock()
    res.scalars.return_value.first.return_value = value
    return res


@pytest.mark.asyncio
async def test_invoice_event_is_stamped_on_its_subscription():
    sub = SimpleNamespace(meta_data={"keep": 1})
    db = MagicMock(execute=AsyncMock(return_value=_result(sub)), commit=AsyncMock())
    svc = _service(db)
    svc._handle_invoice_paid = AsyncMock()

    await svc.handle_webhook("invoice.paid", {"subscription": "sub_1"})

    assert sub.meta_data["keep"] == 1
    assert sub.meta_data["last_stripe_event"]["type"] == "invoice.paid"
    db.commit.assert_awaited()


@pytest.mark.asyncio
async def test_stamping_never_fails_the_webhook():
    db = MagicMock(execute=AsyncMock(side_effect=RuntimeError("db down")), commit=AsyncMock())
    svc = _service(db)
    svc._handle_subscription_updated = AsyncMock()

    await svc.handle_webhook("customer.subscription.updated", {"id": "sub_1"})

    svc._handle_subscription_updated.assert_awaited_once()
