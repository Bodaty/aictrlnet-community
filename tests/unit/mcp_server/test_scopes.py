"""Unit tests for mcp_server.scopes."""

import pytest

from mcp_server.scopes import (
    ALL_SCOPES,
    LEGACY_SCOPE_MAP,
    READ,
    WRITE,
    describe_scope,
    expand_legacy,
    scopes_satisfy,
    validate_scope,
)


def test_read_write_disjoint():
    assert READ.isdisjoint(WRITE), "read and write scope sets must not overlap"


def test_all_scopes_union():
    assert ALL_SCOPES == READ | WRITE


def test_validate_scope_accepts_known():
    for s in ALL_SCOPES:
        assert validate_scope(s), f"{s} should be valid"


def test_validate_scope_rejects_unknown():
    assert not validate_scope("read:all")
    assert not validate_scope("write:all")
    assert not validate_scope("read:nonsense")
    assert not validate_scope("")


def test_describe_scope_has_every_scope():
    for s in ALL_SCOPES:
        desc = describe_scope(s)
        assert desc and desc != s, f"missing description for {s}"


def test_legacy_read_all_expands_to_read_set():
    assert expand_legacy(["read:all"]) == READ


def test_legacy_write_all_expands_to_write_set():
    assert expand_legacy(["write:all"]) == WRITE


def test_legacy_combined_expands_to_all():
    assert expand_legacy(["read:all", "write:all"]) == ALL_SCOPES


def test_legacy_read_workflows_keeps_templates_grant():
    # read:workflows historically implied template access;
    # expansion preserves that so existing keys don't lose capability.
    assert "read:templates" in expand_legacy(["read:workflows"])


def test_expand_drops_unknown_scopes():
    assert expand_legacy(["read:not_a_scope", "gibberish"]) == set()


def test_expand_passes_through_new_scopes():
    assert expand_legacy(["read:policies", "write:memory"]) == {
        "read:policies",
        "write:memory",
    }


def test_scopes_satisfy_phase_b_rejects_legacy():
    """Phase B (2026-04-23): legacy read:all / write:all no longer
    satisfy new-taxonomy requirements. Callers must present the
    specific per-resource scope."""
    assert not scopes_satisfy(["read:all"], ["read:policies"])
    assert not scopes_satisfy(["write:all"], ["write:knowledge"])
    # But ``expand_legacy`` still works if a caller opts in explicitly
    # (used only by the one-time migration now).
    from mcp_server.scopes import expand_legacy

    assert "read:policies" in expand_legacy(["read:all"])


def test_scopes_satisfy_rejects_missing():
    assert not scopes_satisfy(["read:workflows"], ["write:workflows"])
    assert not scopes_satisfy([], ["read:policies"])


def test_scopes_satisfy_multi_required():
    assert scopes_satisfy(
        ["read:policies", "write:workflows"],
        ["read:policies", "write:workflows"],
    )
    assert not scopes_satisfy(
        ["read:policies"],
        ["read:policies", "write:workflows"],
    )


def test_legacy_map_keys_cover_historical_scopes():
    # If we ever retire a legacy scope, the migration must move first.
    assert "read:all" in LEGACY_SCOPE_MAP
    assert "write:all" in LEGACY_SCOPE_MAP
