"""Which tool calls ask for an in-chat "yes" before they run (spec §7.3 R4, T4b).

The user's effective autonomy level decides — the AI Control Spectrum, never a
mandatory approval. Rulings 1 Oct 2026
(.claude/plans/conversation-reliability-t4b-confirmation.md):

- Foundation, Assistance, Automation (0-50): every `requires_confirmation` tool.
- Optimization, Intelligence (51-83): only HIGH_RISK_TOOLS (the system default,
  55, lands here).
- Autonomy (84-100): none; the assistant acts and reports.
"""

from typing import FrozenSet

from services.autonomy_taxonomy import AutonomyPhase, level_to_phase

# Destructive, money, access/security, governance decisions, and actions outside
# the platform. Everything else that is flagged acts and reports at 51+.
HIGH_RISK_TOOLS: FrozenSet[str] = frozenset({
    # destructive
    "cancel_subscription", "delete_agent", "delete_role", "delete_schedule", "delete_webhook",
    "disable_mfa", "revoke_api_key", "revoke_oauth_connection", "revoke_permission", "suspend_user",
    # money
    "upgrade_plan", "downgrade_plan", "add_subscription_addon", "update_payment_method", "manage_license",
    # access and security
    "change_password", "enable_mfa", "generate_mfa_recovery_codes", "grant_permission", "assign_role",
    "create_role", "update_role", "create_api_key", "rotate_api_key", "revoke_session", "invite_user",
    # governance decisions
    "approve_request", "reject_request", "approve_generated_adapter", "set_approval_policy",
    "delegate_approval",
    # acts outside the platform
    "browser_execute", "execute_integration", "execute_mcp_tool", "configure_mcp_server",
    "configure_integration",
})

_CONFIRM_ALL = frozenset({AutonomyPhase.FOUNDATION, AutonomyPhase.ASSISTANCE, AutonomyPhase.AUTOMATION})
_CONFIRM_HIGH_RISK = frozenset({AutonomyPhase.OPTIMIZATION, AutonomyPhase.INTELLIGENCE})


def needs_confirmation(tool, autonomy_level: int) -> bool:
    """True when this call must become a proposal instead of running."""
    if not getattr(tool, "requires_confirmation", False):
        return False
    phase = level_to_phase(autonomy_level)
    if phase in _CONFIRM_ALL:
        return True
    if phase in _CONFIRM_HIGH_RISK:
        return bool(getattr(tool, "is_destructive", False)) or getattr(tool, "name", None) in HIGH_RISK_TOOLS
    return False
