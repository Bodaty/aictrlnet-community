"""Tier A-7: dispatch every conversation tool this edition advertises once with
schema-derived minimal arguments. DEAD = success=True with no substantive data:
None/empty, or a bare {"message": "<string>"} - the shape the four Enterprise
stubs return today (business tool_dispatcher.py:2893-2902).

Ledger semantics as in the other Tier A gates.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))  # find tier_a_* helpers without shadowing src packages

import pytest
import pytest_asyncio
from sqlalchemy import text
from services.tool_dispatcher import ToolDispatcher, Edition
from tier_a_egress import synthesize_value, SmokeEgressBlocked

# A-32: DocumentationKnowledgeService.__init__ calls os.environ.get(..., headers=...) which
# raises TypeError unconditionally; the business dispatcher builds it in _ensure_services() at the
# top of every invoke(). While open, EVERY business/enterprise conversation tool raises before
# routing (since 2026-08-10, commit 48030272). Flip when fixed; the A-8 ledger then applies.
A32_IS_OPEN = False
EDITION = Edition.COMMUNITY
KNOWN_DEAD = {
    # A-33: browser-backed conversation tools report success=False with no error string.
    "browser_extract": "A-33",
    "browser_screenshot": "A-33",
}


@pytest_asyncio.fixture
async def dev_user_id(db):
    await db.execute(text("select set_config('app.current_tenant_id', 'default-tenant', false)"))
    row = (await db.execute(text("SELECT id FROM users WHERE email='dev@aictrlnet.com'"))).first()
    assert row, "dev@aictrlnet.com missing - make fresh"
    return str(row[0])


def _dispatcher(db):
    return ToolDispatcher(db, edition=EDITION)


def _tools(dispatcher):
    return {t.name: t for t in dispatcher.get_available_tools()}


def _chain_has(exc, cls):
    while exc is not None:
        if isinstance(exc, cls):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _is_bare_message(data):
    return isinstance(data, dict) and set(data) <= {"message"} and isinstance(data.get("message"), str)


async def _classify(tool, dispatcher, db, dev_user_id):
    args = synthesize_value(tool.parameters or {}, components={})
    try:
        result = await dispatcher.invoke(tool.name, args, dev_user_id, context={"db": db})
    except Exception as e:  # noqa: BLE001
        if _chain_has(e, SmokeEgressBlocked):
            return "NEEDS_LIVE", type(e).__name__
        if "Mapping.get() got an unexpected keyword argument 'headers'" in str(e):
            return "BLOCKED_A32", str(e)[:80]
        return "RAISED", f"{type(e).__name__}: {str(e)[:100]}"
    finally:
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
    if not result.success:
        return ("SURFACED" if (result.error or "").strip() else "DEAD"), (result.error or "success=False, no error")[:100]
    data = result.data
    if data is None or data == {} or data == []:
        return "DEAD", "success with empty data"
    if _is_bare_message(data):
        return "DEAD", f"canned: {data['message'][:80]}"
    return "REAL", type(data).__name__


@pytest.mark.asyncio
async def test_every_conversation_tool_is_not_dead_or_is_in_the_ledger(db, dev_user_id):
    dispatcher = _dispatcher(db)
    tools = _tools(dispatcher)
    from collections import Counter
    classes, detail = {}, {}
    for name in sorted(tools):
        k, d = await _classify(tools[name], dispatcher, db, dev_user_id)
        classes[name] = k; detail[name] = d
    counts = Counter(classes.values())
    dead = {n: detail[n] for n, k in classes.items() if k == "DEAD"}
    raised = {n: detail[n] for n, k in classes.items() if k == "RAISED"}
    print(f"\\ntier-a conversation tools ({len(tools)}): {dict(counts)}")
    print(f"tier-a conv DEAD: {dead}")
    print(f"tier-a conv RAISED: {raised}")
    blocked = [n for n, k in classes.items() if k == "BLOCKED_A32"]
    if A32_IS_OPEN:
        assert blocked, "A-32 declared open but no tool hit the _ensure_services TypeError - flip A32_IS_OPEN and the A-8 ledger applies"
        assert not dead, f"tools reached routing and died despite A-32 - unexpected: {dead}"
        return
    new = {n: d for n, d in dead.items() if n not in KNOWN_DEAD}
    fixed = {n: KNOWN_DEAD[n] for n in KNOWN_DEAD if n in tools and n not in dead}
    assert not new, f"NEW dead conversation tools (file them): {new}"
    assert not fixed, f"FIXED - remove from KNOWN_DEAD and close: {fixed}"
