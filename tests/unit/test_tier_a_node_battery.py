"""Tier A-3: execute every node that needs no external service once, with a
minimal valid config, against the real container DB, and assert a real result.

`assert_real` rejects the placeholder shapes this codebase has shipped as
success (skipped / timeout / feature_pending / _dry_run / empty). A row may
allow a specific marker only where that output is an honest statement about
the config, not a hidden failure. Rows encode the CORRECT expectation, not
today's behaviour - a failing row is a finding.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))  # find tier_a_* helpers without shadowing src packages

import pytest
from nodes.registry import get_node_registry
from tier_a_support import Row, _ctx, make_config, make_instance, assert_real

COMMUNITY_ROWS = [
    Row("start"),
    Row("end"),
    Row("task", input={"x": 1}, note="pass-through with input"),
    Row("process", input={"x": 1}, note="pass-through with input"),
    Row("task", {"task_type": "calculate", "expression": "2 + 2"}, required_keys=("result",),
        note="calculate; baa2d32b: a throwing calculation used to complete with an error dict"),
    Row("decision", required_keys=("selected_branch",),
        note="default branch is a real decision and must name the branch"),
    Row("decision", {"decision_type": "switch"}, raises=ValueError, match="switch_on", note="misconfigured switch raises"),
    Row("decision", {"decision_type": "not-a-type"}, raises=ValueError, note="unknown decision_type raises"),
    Row("transform", {"mapping": {"y": "x"}}, input={"x": 1}, required_keys=("y",), note="mapping y<-x"),
    Row("contextTransformation", {"mapping": {"y": "x"}}, input={"x": 1}, required_keys=("y",), note="mapping y<-x"),
    Row("transform", input={"x": 1}, raises=ValueError,
        note="A-21: no mapping configured completes with {} and drops every input silently"),
    Row("dataSource", {"source_type": "static", "data": {"a": 1}}, required_keys=("data",), note="static with data"),
    Row("dataSource", allow=(("note", "no static data configured"),),
        note="static without data: honest degrade; question filed in A-3"),
    Row("docGeneration"),
    Row("platform", allow=(("status", "skipped"),),
        note="documented: no PlatformNodeConfig -> skipped, nothing fabricated"),
    Row("mcp", {"operation": "discover_tools"}, required_keys=("tools",), note="discover_tools"),
    Row("iam", {"operation": "discover_agents"}, required_keys=("agents",),
        note="A-1: raises NameError today - iam_service is a local of execute()"),
]

COMMUNITY_ONLY_ROWS = [
    Row("approval", raises=RuntimeError, match="Business Edition", note="community stub refuses"),
]

BUSINESS_ROWS = [
    Row("approval", {"approval_type": "single", "approvers": ["sweep@example.com"],
                     "amount_threshold": 1000, "threshold_field": "amount"},
        input={"amount": 10}, required_keys=("approval_status", "approval_result"),
        note="auto-approve threshold path returns immediately; any other config polls up to 24h"),
    Row("code", {"code": "result = 1 + 1", "language": "python"}, note="sandboxed python"),
    Row("parallel", {"branches": [{"node": {"type": "task", "name": "t"}}]}, note="one task branch"),
    Row("loop", {"loop_type": "count", "count": 1, "body": {"type": "task", "name": "t"}}, note="count=1"),
    Row("careGapEngine", {"operation": "refresh_worklist"}, raises=ValueError, match="CARE_GAPS_ENABLED",
        note="dark by default; the flag-on row is in Tier A-4"),
]

ENTERPRISE_ROWS = [
    Row("compliance", note="A-13: cannot even be constructed today"),
]

ROWS = COMMUNITY_ROWS + COMMUNITY_ONLY_ROWS


def _ids(r):
    return f"{r.alias}:{r.note[:32] or 'default'}"


# Ledger: row id -> finding. Must match the failing set exactly (a new failure
# and an unrecorded fix both fail the gate). Mirrors review-functional-findings.md.
KNOWN_FAILING_ROWS = {
    "transform:A-21: no mapping configured comp": "A-21",
    "mcp:discover_tools": "A-18",
    "iam:A-1: raises NameError today - ia": "A-1",
}


async def _run(row: Row, db, monkeypatch):
    if row.alias == "careGapEngine":
        from core.config import settings
        monkeypatch.setattr(settings, "CARE_GAPS_ENABLED", False, raising=False)
    node = get_node_registry().create_node(make_config(row.alias, **row.params))
    instance = make_instance(node.config, row.input, _ctx(db))
    if row.raises:
        # Not pytest.raises: its "DID NOT RAISE" is a BaseException outcome that would
        # abort the ledger loop instead of being classified as one row's failure.
        try:
            await node.execute(row.input, {**instance.context, "node_id": node.config.id})
        except row.raises as e:
            if row.match:
                import re as _re
                assert _re.search(row.match, str(e)), f"raised {type(e).__name__}: {e} (no match for {row.match!r})"
            return
        raise AssertionError(f"expected {row.raises} but the node completed")
    result = await node.run(instance, workflow_variables={})
    assert_real(result, required_keys=row.required_keys, allow=row.allow)
    if row.alias == "approval":
        assert result.output_data["approval_result"].get("decided_by") == "system", result.output_data["approval_result"]
    if row.alias == "compliance":
        for name, fw in (result.output_data.get("results") or result.output_data.get("frameworks") or {}).items():
            assert not (isinstance(fw, dict) and fw.get("error")), f"{name} degraded: {fw}"


@pytest.mark.asyncio
async def test_every_row_executes_for_real_or_is_in_the_ledger(db, monkeypatch):
    failures = {}
    for row in ROWS:
        try:
            await _run(row, db, monkeypatch)
        except Exception as e:  # noqa: BLE001 - classified against the ledger
            failures[_ids(row)] = f"{type(e).__name__}: {str(e)[:160]}"
    print(f"\ntier-a battery: {len(failures)} failing of {len(ROWS)}: {failures}")
    new = {k: v for k, v in failures.items() if k not in KNOWN_FAILING_ROWS}
    fixed = {k: v for k, v in KNOWN_FAILING_ROWS.items() if k not in failures}
    assert not new, f"NEW battery failures (file them): {new}"
    assert not fixed, f"FIXED - remove from KNOWN_FAILING_ROWS and close: {fixed}"


# --- Tier A: order-independent edition-node registration (see tier_a_support) ---
import pytest as _pytest_taf  # noqa: E402
import tier_a_support as _tier_a_support  # noqa: E402


@_pytest_taf.fixture(autouse=True)
def _edition_nodes_registered():
    _tier_a_support.ensure_edition_nodes_registered()
