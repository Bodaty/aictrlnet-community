"""Unit tests for mcp_server.scopes.SCOPE_GROUPS + helpers."""

from mcp_server.scopes import (
    ALL_SCOPES,
    SCOPE_GROUPS,
    READ,
    WRITE,
    scope_action,
    scope_group,
)


def test_every_scope_has_a_group():
    """Every scope in ALL_SCOPES must be mapped (no silent 'other')."""
    unmapped = [s for s in ALL_SCOPES if s not in SCOPE_GROUPS]
    assert not unmapped, f"Unmapped scopes: {unmapped}"


def test_groups_are_known_set():
    allowed = {
        "workflows", "adapters", "agents", "approvals", "knowledge",
        "governance", "org", "analytics", "admin",
    }
    for s, g in SCOPE_GROUPS.items():
        assert g in allowed, f"{s} mapped to unknown group {g}"


def test_scope_action_for_reads():
    for r in READ:
        assert scope_action(r) == "read"


def test_scope_action_for_writes():
    for w in WRITE:
        assert scope_action(w) == "write"


def test_scope_group_falls_back_to_other():
    assert scope_group("read:definitely-not-a-scope") == "other"


def test_scope_group_resolves_known():
    assert scope_group("read:workflows") == "workflows"
    assert scope_group("write:browser") == "adapters"
    assert scope_group("read:cost") == "analytics"
    assert scope_group("read:compliance") == "governance"
