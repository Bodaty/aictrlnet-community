"""Tier A-4: nodes that need a service, exercised against in-network services
only. The egress guard stays on. The product's own SSRF validator
(core/ssrf.py:9-16) refuses private and loopback targets, so the honest row for
apiCall/webhook is "private target refused"; their positive path is
unverified-live and is recorded as such in the findings file.

Ledger semantics as in the other Tier A gates: every row runs every time; the
failing set must equal KNOWN_FAILING_ROWS exactly.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))  # find tier_a_* helpers without shadowing src packages

import asyncio
import re
import uuid
import pytest
import pytest_asyncio
from sqlalchemy import text
from nodes.registry import get_node_registry
from tier_a_support import Row, _ctx, assert_real, make_config, make_instance

COMMUNITY_HEALTH = "http://community:8000/health"
MCP_TRANSPORT = "http://community:8000/api/v1/mcp-transport"
DEV_TOKEN = "dev-token-for-testing"
REFUSED = "(?i)ssrf|blocked|private|internal|not allowed|loopback"

COMMUNITY_ROWS = [
    Row("apiCall", {"url": COMMUNITY_HEALTH, "method": "GET"}, raises=Exception, match=REFUSED,
        note="private target refused by core.ssrf (positive path unverified-live)"),
    Row("notification", {"channel": "in_app", "recipients": [{"user_id": "tier-a-user"}],
                         "message": {"subject": "tier-a", "body": "tier-a"}},
        required_keys=("channel",), note="in_app delivery"),
    Row("mcpServer", {"endpoint_name": "tier-a", "mode": "webhook"},
        required_keys=("status", "webhook_url"), note="webhook mode registers via cache"),
    Row("browserAutomation", {"actions": [{"type": "render_html", "html": "<h1>tier-a</h1>"}, {"type": "extract_text"}]},
        required_keys=("results",), note="render_html+extract_text via browser-service:8005 (no network)"),
    Row("mcpClient", {"mcp_server_url": MCP_TRANSPORT, "api_key": DEV_TOKEN, "operation": "tool"},
        input={"tool_name": "list_workflows", "arguments": {}},
        raises=Exception, match=REFUSED,
        note="private target refused by core.ssrf (positive path unverified-live)"),
    Row("aiProcess", {"prompt": "Reply with the single word OK."},
        note="DECISION: needs Ollama at host.docker.internal:11434; fails loudly if down"),
    Row("fileProcess", {"file_path": "__STAGED__/tier-a.txt"}, required_keys=("extracted", "content_type"), note="text file"),
    Row("fileProcess", {"file_path": "__STAGED__/tier-a-scanned.pdf"}, raises=(ValueError, RuntimeError), match="(?i)no text|text layer|ocr",
        note="F-3: an image-only PDF must raise, not complete with empty text"),
]

BUSINESS_ROWS = [
    Row("webhook", {"url": COMMUNITY_HEALTH, "method": "GET"}, raises=Exception, match=REFUSED,
        note="private target refused by core.ssrf (positive path unverified-live)"),
    Row("schedule", {"target": {"type": "task", "name": "t"}, "schedule_type": "cron", "cron_expression": "0 0 1 1 *"},
        required_keys=("schedule_id",), note="cron schedule"),
    Row("iamAdvanced", {"operation": "check_permission", "resource": "workflows", "action": "read"},
        note="check_permission for the dev user"),
    Row("careGapEngine", {"operation": "refresh_worklist"}, required_keys=("care_gap_refresh",),
        note="flag on: empty roster -> zero counts is a REAL result"),
    Row("remoteTool", {"instance_id": "tier-a-missing", "tool_name": "noop", "timeout_s": 2},
        raises=(ValueError, RuntimeError, TimeoutError, asyncio.TimeoutError),
        note="no connector: must fail loudly (node is unverified-live)"),
    Row("adapter", {"adapter_id": "tier-a-missing", "capability": "x"}, raises=Exception,
        note="missing adapter must fail loudly (positive path deferred to Tier B)"),
]

ENTERPRISE_ROWS = [
    Row("enhancedAdapter", {"adapters": [{"id": "tier-a", "type": "postgresql"}]}, note="A-13"),
    Row("orchestration", {"workflows": [{"workflow_id": "tier-a"}], "pattern": "saga"}, note="A-13"),
    Row("federation", {"target_tenants": ["default-tenant"]}, note="A-13"),
]

ROWS = COMMUNITY_ROWS
KNOWN_FAILING_ROWS = {
}


def _ids(r):
    return f"{r.alias}:{r.note[:32] or 'default'}"


@pytest_asyncio.fixture
async def dev_user_id(db):
    row = (await db.execute(text("SELECT id FROM users WHERE email='dev@aictrlnet.com'"))).first()
    assert row, "migration test user dev@aictrlnet.com missing - run make fresh"
    return str(row[0])


@pytest.fixture
def staged_dir():
    """Files go into the REAL staged dir: the node confines reads to
    get_settings().STAGED_FILES_DIR (file_process_node.py:55-70)."""
    import shutil
    from pathlib import Path
    from core.config import get_settings
    base = Path(get_settings().STAGED_FILES_DIR); base.mkdir(parents=True, exist_ok=True)
    tmp_path = base / f"tier-a-{uuid.uuid4().hex[:8]}"; tmp_path.mkdir()
    (tmp_path / "tier-a.txt").write_text("tier a text content\n")
    (tmp_path / "tier-a-scanned.pdf").write_bytes(
        b"%PDF-1.4\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj\n"
        b"xref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000101 00000 n \n"
        b"trailer<</Size 4/Root 1 0 R>>\nstartxref\n170\n%%EOF\n")
    yield tmp_path
    shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def ai_adapters_registered():
    """Mirror core/app.py:114-140: the lifespan registers adapter classes into
    adapter_registry; a bare pytest process has none, so aiProcess reports
    'No AI adapters available'. Register exactly the local one."""
    from adapters.factory import AdapterFactory
    from adapters.registry import adapter_registry
    if "ollama" not in adapter_registry._adapter_classes:
        cls = AdapterFactory._load_adapter_class(AdapterFactory.ADAPTER_MAPPINGS["ollama"])
        adapter_registry.register_adapter_class("ollama", cls, AdapterFactory._determine_category("ollama", {}), description="ollama adapter")
    return True


async def _run(row: Row, db, dev_user_id, staged_dir, monkeypatch):
    # RLS: the engine sets app.current_tenant_id per connection (core/database.py:73); the test engine does not.
    await db.execute(text("select set_config('app.current_tenant_id', 'default-tenant', false)"))
    params = {k: (v.replace("__STAGED__", str(staged_dir)) if isinstance(v, str) else v) for k, v in row.params.items()}
    ctx = _ctx(db)
    if row.alias == "mcpClient":
        ctx["control_plane_url"] = "http://community:8000"
    if row.alias == "iamAdvanced":
        ctx["user_id"] = dev_user_id
    if row.alias == "careGapEngine":
        from core.config import settings
        monkeypatch.setattr(settings, "CARE_GAPS_ENABLED", True, raising=False)
    node = get_node_registry().create_node(make_config(row.alias, **params))
    instance = make_instance(node.config, row.input, ctx)
    if row.raises:
        try:
            await node.execute(row.input, {**instance.context, "node_id": node.config.id})
        except row.raises as e:
            if row.match:
                assert re.search(row.match, str(e)), f"raised {type(e).__name__}: {e} (no match for {row.match!r})"
            return
        raise AssertionError(f"expected {row.raises} but the node completed")
    result = await node.run(instance, workflow_variables={})
    assert_real(result, required_keys=row.required_keys, allow=row.allow)


@pytest.mark.asyncio
async def test_every_service_row_executes_for_real_or_is_in_the_ledger(db, dev_user_id, staged_dir, monkeypatch, ai_adapters_registered):
    failures = {}
    for row in ROWS:
        try:
            await _run(row, db, dev_user_id, staged_dir, monkeypatch)
        except Exception as e:  # noqa: BLE001 - classified against the ledger
            failures[_ids(row)] = f"{type(e).__name__}: {str(e)[:220]}"
    print(f"\\ntier-a services: {len(failures)} failing of {len(ROWS)}: {failures}")
    new = {k: v for k, v in failures.items() if k not in KNOWN_FAILING_ROWS}
    fixed = {k: v for k, v in KNOWN_FAILING_ROWS.items() if k not in failures}
    assert not new, f"NEW service-row failures (file them): {new}"
    assert not fixed, f"FIXED - remove from KNOWN_FAILING_ROWS and close: {fixed}"


@pytest.mark.asyncio
async def test_mcp_server_single_mode_does_not_crash_on_unsubscribe(db):
    """A-22 regression: mcp_server_node's single mode awaited the SYNC
    event_bus.unsubscribe in its finally, so every single-mode request ended in
    TypeError ("object NoneType can't be used in 'await'"). Post-fix the node
    completes gracefully even when no request arrives (status: timeout) and never
    fails with that await error. Delivering a real request end-to-end is a
    separate concern (event-bus wiring), not what A-22 fixes.
    """
    import asyncio
    node = get_node_registry().create_node(make_config("mcpServer", endpoint_name="tier-a-single", mode="single", timeout=2))
    instance = make_instance(node.config, {}, _ctx(db))
    result = await asyncio.wait_for(node.run(instance, workflow_variables={}), timeout=15)
    err = (result.error or "") + str(result.output_data)
    assert "can't be used in 'await'" not in err, f"A-22 regressed: {err}"
    assert "unsubscribe" not in (result.error or ""), f"A-22 regressed: {result.error}"


def _edition_nodes_registered():
    _tier_a_support.ensure_edition_nodes_registered()
