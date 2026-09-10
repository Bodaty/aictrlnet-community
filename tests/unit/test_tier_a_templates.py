"""Tier A-2: every template JSON this edition ships loads, every node in it
resolves to its own implementation through the production registry using the
engine's own mapping (workflow_execution.py:271-309), and every edge references
a real node.

Ledger semantics (no skips, no xfail): every template is exercised on every run.
Each failure must match a KNOWN_FAILURE_PATTERNS entry (else NEW) and each
entry must match at least one failure (else FIXED - remove it). The ledger
mirrors .claude/plans/review-functional-findings.md.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parent))  # community runs importlib mode

import json
import re
from pathlib import Path
from nodes.registry import get_node_registry
from tier_a_support import make_config

TEMPLATE_ROOTS = [
    Path("/app/workflow-templates/system"),
]
MIN_TEMPLATES = 1

KNOWN_FAILURE_PATTERNS = {
    # A-16: enterprise-only node types in a template silently run as no-op tasks here
    "A-16": r"(orchestrationPattern|orchestration|enhancedAdapter|compliance|federation) silently resolved to the TaskNode fallback",
}


def _templates():
    for root in TEMPLATE_ROOTS:
        if not root.exists():
            raise AssertionError(f"template root missing in container: {root}")
        for p in sorted(root.rglob("*.json")):
            if p.name.endswith(".metadata.json"):
                continue
            yield p


def _check(path):
    doc = json.loads(path.read_text())
    wf = doc.get("workflow", doc)
    assert isinstance(wf.get("nodes"), list) and wf["nodes"], "no nodes"
    assert isinstance(wf.get("edges"), list), "no edges list"
    registry = get_node_registry()
    ids = set()
    for node in wf["nodes"]:
        assert "id" in node and "type" in node, f"node missing id/type: {node}"
        ids.add(node["id"])
        params = dict(node.get("parameters") or node.get("config") or node.get("data") or {})
        cfg = make_config(node["type"], **params)
        cfg.id, cfg.name = node["id"], node.get("name") or node["id"]
        inst = registry.create_node(cfg)  # ValueError = unregistered; TypeError = broken ctor
        if type(inst).__name__ == "TaskNode" and node["type"] not in ("task", "process"):
            raise AssertionError(f"{node['type']} silently resolved to the TaskNode fallback")
    for edge in wf["edges"]:
        src = edge.get("source", edge.get("from"))
        dst = edge.get("target", edge.get("to"))
        assert src in ids, f"edge {edge.get('id')} source {src!r} is not a node"
        assert dst in ids, f"edge {edge.get('id')} target {dst!r} is not a node"


def test_every_template_resolves_or_is_in_the_ledger():
    failures = {}
    for p in _templates():
        try:
            _check(p)
        except Exception as e:  # noqa: BLE001 - classified below
            failures["/".join(p.parts[-3:])] = f"{type(e).__name__}: {e}"
    matched = {fid: [t for t, err in failures.items() if re.search(rx, err)] for fid, rx in KNOWN_FAILURE_PATTERNS.items()}
    accounted = {t for ts in matched.values() for t in ts}
    new = {t: err for t, err in failures.items() if t not in accounted}
    fixed = [fid for fid, ts in matched.items() if not ts]
    counts = {fid: len(ts) for fid, ts in matched.items()}
    print(f"\\ntier-a templates: {len(failures)} failing of {sum(1 for _ in _templates())}; by finding: {counts}")
    assert not new, f"NEW template failures (file them): {json.dumps(new, indent=1)[:4000]}"
    assert not fixed, f"FIXED - remove from KNOWN_FAILURE_PATTERNS and close: {fixed}"


def test_template_count_matches_inventory():
    n = sum(1 for _ in _templates())
    assert n >= MIN_TEMPLATES, f"expected at least {MIN_TEMPLATES} system templates, found {n}"
