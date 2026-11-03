"""Under PHI mode only a de-identified nudge leaves the product (platform gap G37)."""

from types import SimpleNamespace

import pytest

from core.phi_outbound import DeidentifiedNudge, channel_allowed, refusal_for, scrub_approval_notification

OFF = SimpleNamespace(AICTRLNET_PHI_MODE=False, AICTRLNET_PHI_OUTBOUND_CHANNELS="")
ON = SimpleNamespace(AICTRLNET_PHI_MODE=True, AICTRLNET_PHI_OUTBOUND_CHANNELS="email")
NUDGE = DeidentifiedNudge("items_waiting", "https://hitlai.example.test", {"count": 3})


def test_without_phi_mode_everything_goes():
    assert refusal_for("slack", "Patient Jane Roe is overdue", OFF) is None
    assert channel_allowed("telegram", OFF)


def test_under_phi_mode_free_text_never_goes():
    assert "only a de-identified nudge" in refusal_for("email", "Patient Jane Roe is overdue", ON)


def test_a_nudge_goes_only_on_an_allowlisted_channel():
    assert refusal_for("email", NUDGE, ON) is None
    assert "not in AICTRLNET_PHI_OUTBOUND_CHANNELS" in refusal_for("slack", NUDGE, ON)
    assert not channel_allowed("slack", ON) and channel_allowed("email", ON)


def test_a_nudge_is_counts_and_an_https_link_from_a_fixed_template():
    assert NUDGE.render() == "3 item(s) need you in HitLai today. Sign in: https://hitlai.example.test"
    with pytest.raises(ValueError):
        DeidentifiedNudge("free_text", "https://x").render()
    with pytest.raises(ValueError):
        DeidentifiedNudge("items_waiting", "http://x").render()


def test_approval_notifications_name_nothing_under_phi_mode():
    data = {"title": "Closures for Jane Roe", "context": {"patient": "Jane Roe"}, "decision_comment": "ok",
            "request_id": "r1"}
    scrubbed = scrub_approval_notification(data, ON)
    assert scrubbed["context"] is None and scrubbed["decision_comment"] is None
    assert scrubbed["title"] == "An approval is waiting in HitLai" and scrubbed["request_id"] == "r1"
    assert scrub_approval_notification(data, OFF) == data


@pytest.mark.asyncio
async def test_send_text_refuses_free_text_under_phi_mode(monkeypatch):
    from core import phi_egress
    from services.channel_outbound import send_text

    monkeypatch.setattr(phi_egress, "_settings", lambda settings=None: ON)
    delivered = await send_text("slack", "C1", "Patient Jane Roe is overdue")
    assert delivered.delivered is False and "only a de-identified nudge" in delivered.reason
