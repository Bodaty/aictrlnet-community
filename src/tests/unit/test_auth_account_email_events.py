"""The auth endpoints publish the reset and verification links for sending.

Before this fix they only logged "In production, send email". A missing
business_starter plan at signup was a warning; it is now an error naming the
fix, since the user silently got no trial.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from api.v1.endpoints import auth


@pytest.mark.asyncio
async def test_publish_account_email_sends_link_on_the_bus():
    bus = SimpleNamespace(publish=AsyncMock())
    user = SimpleNamespace(id="u1", email="ana@example.com", full_name="Ana")
    with patch("events.event_bus.event_bus", bus):
        await auth._publish_account_email("auth.password_reset_requested", user, "https://x/reset?token=t")
    bus.publish.assert_awaited_once_with("auth.password_reset_requested", {
        "user_id": "u1",
        "email": "ana@example.com",
        "full_name": "Ana",
        "link": "https://x/reset?token=t",
    })


@pytest.mark.asyncio
async def test_publish_account_email_never_raises():
    bus = SimpleNamespace(publish=AsyncMock(side_effect=RuntimeError("bus down")))
    user = SimpleNamespace(id="u1", email="ana@example.com", full_name="Ana")
    with patch("events.event_bus.event_bus", bus):
        await auth._publish_account_email("auth.email_verification_requested", user, "https://x/v/t")


def test_reset_and_verify_endpoints_publish_instead_of_only_logging():
    import inspect
    source = inspect.getsource(auth)
    assert "In production, send email" not in source
    assert source.count('"auth.email_verification_requested"') >= 2  # signup and resend
    assert '"auth.password_reset_requested"' in source
