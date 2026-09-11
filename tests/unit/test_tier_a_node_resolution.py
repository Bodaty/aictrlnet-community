"""Tier A-1: every node type this edition advertises resolves to ITS OWN
implementation through the production registry; every NodeType enum member
either resolves or is declared dead.

Ledger semantics (no skips, no xfail): every alias is exercised on every run,
and the set of unresolved aliases must equal KNOWN_UNRESOLVED exactly. A new
failure fails the gate; a fix that is not removed from the ledger also fails
the gate. The ledger mirrors .claude/plans/review-functional-findings.md.

Runs in dev-community-1 under `make test-unit-community`.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))  # find tier_a_* helpers without shadowing src packages

import pytest

import tier_a_support as _tier_a_support


@pytest.fixture(autouse=True)
def _edition_nodes_registered():
    _tier_a_support.ensure_edition_nodes_registered()

from nodes.models import NodeConfig, NodeType
from nodes.registry import get_node_registry
from tier_a_support import (EXPECTED_ALIASES, UNIMPLEMENTED_NODE_TYPES,
                            ALIAS_ONLY_NODE_TYPES, make_config)

EDITION = "community"

# alias -> (kind, finding). kind: "ctor" = class raises at construction,
# "fallback" = unregistered, silently resolves to TaskNode, "unregistered" = ValueError.
KNOWN_UNRESOLVED = {}

# A-12: registry.create_node + workflow_execution.py:289 turn ANY unknown type
# into TaskNode. Declared open; flip to False when the class fix lands.
TASK_FALLBACK_IS_OPEN = True


def _resolve(alias):
    try:
        node = get_node_registry().create_node(make_config(alias))
    except ValueError:
        return "unregistered"
    except TypeError as e:
        return f"ctor: {e}"
    if type(node).__name__ == "TaskNode" and alias not in ("task", "process"):
        return "fallback"
    return "ok"


def test_every_advertised_alias_resolves_or_is_in_the_ledger():
    results = {alias: _resolve(alias) for alias in sorted(EXPECTED_ALIASES[EDITION])}
    unresolved = {a: r for a, r in results.items() if r != "ok"}
    new = {a: r for a, r in unresolved.items() if a not in KNOWN_UNRESOLVED}
    fixed = {a: KNOWN_UNRESOLVED[a] for a in KNOWN_UNRESOLVED if a not in unresolved}
    wrong_kind = {a: (r, KNOWN_UNRESOLVED[a][0]) for a, r in unresolved.items()
                  if a in KNOWN_UNRESOLVED and not r.startswith(KNOWN_UNRESOLVED[a][0])}
    assert not new, f"NEW unresolved node types (file them): {new}"
    assert not fixed, f"FIXED - remove from KNOWN_UNRESOLVED and close the finding: {fixed}"
    assert not wrong_kind, f"ledger kind mismatch (actual, declared): {wrong_kind}"


def test_every_enum_member_resolves_or_is_declared_dead():
    registry = get_node_registry()
    dead = set()
    for member in NodeType:
        cfg = NodeConfig(id=f"enum-{member.value}", name=member.value, type=member, parameters={})
        try:
            registry.create_node(cfg)
        except ValueError:
            dead.add(member)
        except TypeError:
            pass  # registered but broken at construction - covered by the alias ledger
    undeclared = dead - UNIMPLEMENTED_NODE_TYPES - ALIAS_ONLY_NODE_TYPES
    resurrected = UNIMPLEMENTED_NODE_TYPES - dead
    assert not undeclared, f"enum members with no implementation and not declared dead: {sorted(m.value for m in undeclared)}"
    assert not resurrected, f"now implemented - remove from UNIMPLEMENTED_NODE_TYPES: {sorted(m.value for m in resurrected)}"


@pytest.mark.parametrize("member", sorted(ALIAS_ONLY_NODE_TYPES, key=lambda m: m.value))
def test_alias_only_members_resolve_with_custom_node_type(member):
    if EDITION == "community" and member in (NodeType.LOOP, NodeType.PARALLEL, NodeType.WEBHOOK):
        pytest.skip("registered by Business edition_nodes")
    alias = {NodeType.LOOP: "loop", NodeType.PARALLEL: "parallel", NodeType.HUMAN_TASK: "humanTask",
             NodeType.API_CALL: "apiCall", NodeType.WEBHOOK: "webhook"}[member]
    assert _resolve(alias) == "ok"


def test_unknown_alias_behaviour_matches_the_ledger():
    """A-12. While open, an unknown type silently becomes TaskNode. When the
    class fix lands, create_node raises ValueError and this test demands the
    flag be flipped - the gate fails in either direction if reality and ledger disagree."""
    kind = _resolve("no-such-node-type-tier-a")
    if TASK_FALLBACK_IS_OPEN:
        assert kind == "fallback", f"A-12 is declared open but an unknown alias now gives {kind!r} - flip TASK_FALLBACK_IS_OPEN and close the finding"
    else:
        assert kind == "unregistered", f"A-12 is declared fixed but an unknown alias gives {kind!r}"


def test_community_approval_is_a_stub_that_refuses():
    import asyncio
    node = get_node_registry().create_node(make_config("approval"))
    with pytest.raises(RuntimeError, match="Business Edition"):
        asyncio.run(node.execute({}, {}))
