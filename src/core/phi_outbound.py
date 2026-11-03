"""What may leave the product under PHI mode: a de-identified nudge, nothing else.

Under ``AICTRLNET_PHI_MODE`` anything that names a patient or carries clinical
content stays inside HitLai (the briefing on the Today page, in-app
notifications). An outside channel may carry only a ``DeidentifiedNudge``:
counts and a sign-in link from a fixed template, on a channel the deployment
names in ``AICTRLNET_PHI_OUTBOUND_CHANNELS`` (those covered by a BAA). Every
outbound send path calls ``refusal_for`` (platform gap G37); the regression
test ``test_phi_outbound_chokepoint.py`` fails when a new send path appears
outside the chokepoints.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

TEMPLATES = {
    "items_waiting": "{count} item(s) need you in HitLai today. Sign in: {link}",
    "briefing_ready": "Your briefing is ready in HitLai. Sign in: {link}",
}


@dataclass(frozen=True)
class DeidentifiedNudge:
    template: str
    link: str
    counts: Dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        if self.template not in TEMPLATES:
            raise ValueError(f"unknown nudge template {self.template!r}")
        if not isinstance(self.link, str) or not self.link.startswith("https://"):
            raise ValueError("a nudge links to HitLai over https")
        counts = {k: int(v) for k, v in (self.counts or {}).items()}
        return TEMPLATES[self.template].format(count=counts.get("count", 0), link=self.link)


def _allowed_channels(settings) -> set:
    raw = str(getattr(settings, "AICTRLNET_PHI_OUTBOUND_CHANNELS", "") or "")
    return {c.strip().lower() for c in raw.split(",") if c.strip()}


def channel_allowed(channel_type: str, settings=None) -> bool:
    """Whether anything at all may go out on this channel (always, without PHI mode)."""
    from core.phi_egress import _settings, phi_mode_on

    settings = _settings(settings)
    return not phi_mode_on(settings) or str(channel_type or "").lower() in _allowed_channels(settings)


def refusal_for(channel_type: str, message: Any, settings=None) -> Optional[str]:
    """None when the send may go; otherwise the reason it may not."""
    from core.phi_egress import _settings, phi_mode_on

    settings = _settings(settings)
    if not phi_mode_on(settings):
        return None
    if not isinstance(message, DeidentifiedNudge):
        return "refused under PHI mode: only a de-identified nudge may leave HitLai"
    if str(channel_type or "").lower() not in _allowed_channels(settings):
        return f"refused under PHI mode: {channel_type} is not in AICTRLNET_PHI_OUTBOUND_CHANNELS"
    return None


def scrub_approval_notification(data: Dict[str, Any], settings=None) -> Dict[str, Any]:
    """Under PHI mode an approval notification names nothing: no context, no
    comment, a generic title. The approver reads the request in HitLai."""
    from core.phi_egress import _settings, phi_mode_on

    if not phi_mode_on(_settings(settings)):
        return data
    scrubbed = dict(data)
    for key in ("context", "decision_comment", "decision_reason", "requester_name"):
        scrubbed[key] = None
    scrubbed["title"] = "An approval is waiting in HitLai"
    return scrubbed
