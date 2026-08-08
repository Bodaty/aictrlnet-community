"""Stripe webhook security regressions.

- /billing/webhook previously accepted UNSIGNED JSON whenever
  STRIPE_WEBHOOK_SECRET was unset — including in production — and mutated
  subscription state from it. It must fail closed in deploy environments.
- /license/webhook/stripe previously passed a missing Stripe-Signature
  header (None) into stripe.Webhook.construct_event → AttributeError → 500.
- Neither endpoint checked event.livemode, so test-mode events could drive
  a live-configured deployment.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.v1.endpoints import billing
from api.v1.endpoints import license as license_endpoint


def _request(body: bytes = b'{"type": "ping", "data": {}}'):
    req = MagicMock()
    req.body = AsyncMock(return_value=body)
    return req


def _settings(environment="test", stripe_key="sk_test_dummy", webhook_secret="whsec_dummy"):
    return MagicMock(
        ENVIRONMENT=environment,
        STRIPE_SECRET_KEY=stripe_key,
        STRIPE_WEBHOOK_SECRET=webhook_secret,
    )


def _stripe_service_mock():
    service_cls = MagicMock()
    service_cls.return_value.handle_webhook = AsyncMock()
    return service_cls


# --- /billing/webhook ---


@pytest.mark.asyncio
async def test_billing_no_secret_deploy_env_fails_closed(monkeypatch):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    service_cls = _stripe_service_mock()
    with patch.object(billing, "get_settings", return_value=_settings(environment="production")), \
         patch.object(billing, "StripeService", service_cls):
        with pytest.raises(HTTPException) as exc:
            await billing.stripe_webhook(request=_request(), stripe_signature=None, db=MagicMock())
    assert exc.value.status_code == 503
    service_cls.assert_not_called()


@pytest.mark.asyncio
async def test_billing_no_secret_dev_env_still_parses(monkeypatch):
    monkeypatch.delenv("STRIPE_WEBHOOK_SECRET", raising=False)
    service_cls = _stripe_service_mock()
    with patch.object(billing, "get_settings", return_value=_settings(environment="development")), \
         patch.object(billing, "StripeService", service_cls):
        result = await billing.stripe_webhook(request=_request(), stripe_signature=None, db=MagicMock())
    assert result == {"received": True}
    service_cls.return_value.handle_webhook.assert_awaited_once()


@pytest.mark.asyncio
async def test_billing_missing_signature_header_is_400(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_testsecret")
    with pytest.raises(HTTPException) as exc:
        await billing.stripe_webhook(request=_request(), stripe_signature=None, db=MagicMock())
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_billing_bad_signature_is_400(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_testsecret")
    with pytest.raises(HTTPException) as exc:
        await billing.stripe_webhook(
            request=_request(), stripe_signature="t=1,v1=deadbeef", db=MagicMock()
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_billing_live_key_ignores_test_mode_event(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_testsecret")
    service_cls = _stripe_service_mock()
    event = {"id": "evt_test_1", "livemode": False, "type": "invoice.paid", "data": {"object": {}}}
    with patch.object(billing, "get_settings", return_value=_settings(stripe_key="sk_live_x")), \
         patch.object(billing, "StripeService", service_cls), \
         patch("stripe.Webhook.construct_event", return_value=event):
        result = await billing.stripe_webhook(
            request=_request(), stripe_signature="t=1,v1=ok", db=MagicMock()
        )
    assert result == {"received": True, "ignored": "livemode_mismatch"}
    service_cls.assert_not_called()


@pytest.mark.asyncio
async def test_billing_test_key_processes_test_mode_event(monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_testsecret")
    service_cls = _stripe_service_mock()
    event = {"id": "evt_test_2", "livemode": False, "type": "invoice.paid", "data": {"object": {}}}
    with patch.object(billing, "get_settings", return_value=_settings(stripe_key="sk_test_x")), \
         patch.object(billing, "StripeService", service_cls), \
         patch("stripe.Webhook.construct_event", return_value=event):
        result = await billing.stripe_webhook(
            request=_request(), stripe_signature="t=1,v1=ok", db=MagicMock()
        )
    assert result == {"received": True}
    service_cls.return_value.handle_webhook.assert_awaited_once_with("invoice.paid", {})


# --- /license/webhook/stripe ---


@pytest.mark.asyncio
async def test_license_missing_signature_header_is_400_not_500():
    with pytest.raises(HTTPException) as exc:
        await license_endpoint.stripe_webhook(
            request=_request(), stripe_signature=None, db=MagicMock()
        )
    assert exc.value.status_code == 400


@pytest.mark.asyncio
async def test_license_live_key_ignores_test_mode_event():
    service_cls = _stripe_service_mock()
    event = {"id": "evt_test_3", "livemode": False, "type": "invoice.paid", "data": {"object": {}}}
    with patch("core.config.get_settings", return_value=_settings(stripe_key="sk_live_x", webhook_secret="whsec_x")), \
         patch("services.stripe_service.StripeService", service_cls), \
         patch("stripe.Webhook.construct_event", return_value=event):
        result = await license_endpoint.stripe_webhook(
            request=_request(), stripe_signature="t=1,v1=ok", db=MagicMock()
        )
    assert result == {"status": "success", "ignored": "livemode_mismatch"}
    service_cls.assert_not_called()
