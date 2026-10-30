"""Tier A-8 (make-it-real): call execute() for every capability of every adapter
the factory maps, and health_check() on each, and reject success that touched
nothing. test_tier_a_adapters stops at construction, initialize() and
health-check ownership; it never calls execute(), which is how fully
fabricated adapters passed it (.claude/plans/make-it-real-ledger.md §1).

Each call runs inside io_probe.probe() with the egress guard on:

  EGRESS   SmokeEgressBlocked raised            - reached for a third party: real
  IO       socket / SQL / subprocess observed   - real
  REFUSED  raised with a message, or status=error with an error string - real
  BROKEN   AttributeError/TypeError/NameError/ImportError/KeyError/ValidationError
           - the code cannot run as written; ledgered in KNOWN_BROKEN (fix-queue)
  PURE     success with no I/O                  - a fabrication unless DECLARED_PURE

PURE rows must be in KNOWN_FABRICATED (shrinks as the ledger is worked) or in
DECLARED_PURE (genuine deterministic work over the caller's input, D-1: an
exact set, each entry justified, never auto-grown). Exact-set semantics as in the
other Tier A gates: a new failure fails, and a fixed entry left in a ledger fails.
Aliases of one class are exercised once, under the alphabetically first alias.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))  # find tier_a_* helpers without shadowing src packages

import asyncio
import re
import pytest
from adapters.factory import AdapterFactory
from adapters.models import AdapterConfig, AdapterCategory, AdapterRequest
from tier_a_egress import SmokeEgressBlocked, synthesize_value  # noqa: F401  (puts smoke_common on sys.path)
from smoke_common.io_probe import probe, chain_has, is_broken

MAPPINGS = dict(AdapterFactory.ADAPTER_MAPPINGS)
PER_CALL_TIMEOUT = 10

# Keys are "<adapter_type>:<capability>" for execute rows and "<adapter_type>:health" for health rows.
KNOWN_FABRICATED = {
    'autogen:health': 'MR-155',
    'autogpt:health': 'MR-155',
    'crewai:health': 'MR-155',
    'mcp-client:health': 'MR-100',
    'openclaw:health': 'MR-100',
}
DECLARED_PURE = {
    'webhook:batch_webhooks': 'the synthesized batch is empty, so it honestly reports zero sends',
    'webhook:health_check': 'the synthesized URL fails validation; it reports healthy=False with the error, no claim',
    'webhook:verify_webhook_signature': "computes the HMAC over the caller's payload - deterministic, no I/O needed",
}
KNOWN_BROKEN = {
    'autogen:create_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'autogen:execute_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'autogpt:create_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'autogpt:execute_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'crewai:create_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'crewai:execute_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'langchain:chain_operations': 'FQ: agentic execute reads request.metadata',
    'langchain:create_agent': 'FQ: agentic execute reads request.metadata',
    'langchain:execute_task': 'FQ: agentic execute reads request.metadata',
    'langchain:manage_conversation': 'FQ: agentic execute reads request.metadata',
    'langchain:use_tools': 'FQ: agentic execute reads request.metadata',
    'mcp-client:mcp.call_tool': 'FQ: execute treats AdapterRequest as a dict',
    'mcp-client:mcp.list_resources': 'FQ: execute treats AdapterRequest as a dict',
    'mcp-client:mcp.list_tools': 'FQ: execute treats AdapterRequest as a dict',
    'mcp-client:mcp.read_resource': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:a_b_testing': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:fine_tuning': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:model.anomaly_detection': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:model.ml.classification': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:model.ml.regression': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:model.timeseries.forecasting': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:quality_verification': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:risk_assessment': 'FQ: execute treats AdapterRequest as a dict',
    'ml-service:training': 'FQ: execute treats AdapterRequest as a dict',
    'ollama:delete_model': 'FQ: httpx delete() takes no json',
    'openclaw:delegate_to_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:evaluate_action': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:execute_skill': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:list_channels': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:list_skills': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:register_runtime': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:report_action': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:send_message': 'FQ adapter contract: AdapterResponse(success=...)',
    'openclaw:spawn_sub_agent': 'FQ adapter contract: AdapterResponse(success=...)',
    'semantic-kernel:*': 'A-27',
}

# An error status whose message is a Python exception text is broken code that
# caught itself, not a refusal (e.g. "'AdapterRequest' object has no attribute 'get'").
_PYTHON_ERROR_TEXT = re.compile(
    r"object has no attribute|missing \d+ required|unexpected keyword argument|not subscriptable"
    r"|is not defined|takes \d+ positional|validation errors? for|NoneType"
)


def _canonical_types():
    by_class = {}
    for adapter_type in sorted(MAPPINGS):
        by_class.setdefault(MAPPINGS[adapter_type], adapter_type)
    return sorted(by_class.values())


def _params(capability):
    schema = {"type": "object", "properties": dict(capability.parameters or {}),
              "required": list(capability.required_parameters or [])}
    try:
        return synthesize_value(schema, components={}) or {}
    except Exception:  # noqa: BLE001 - an exotic schema is not the adapter's fault; required keys only
        return {k: "tier-a" for k in schema["required"]}


def _classify_exc(e):
    if chain_has(e, SmokeEgressBlocked):
        return "EGRESS", type(e).__name__
    if is_broken(e):
        return "BROKEN", f"{type(e).__name__}: {str(e)[:100]}"
    if str(e).strip():
        return ("BROKEN" if _PYTHON_ERROR_TEXT.search(str(e)) else "REFUSED"), f"{type(e).__name__}: {str(e)[:100]}"
    return "BROKEN", f"{type(e).__name__} with no message"


def _classify_result(result, io):
    if io.egress_blocked:
        return "EGRESS", io.summary()
    status = getattr(result, "status", None)
    error = getattr(result, "error", None)
    if isinstance(result, dict):
        status = result.get("status", status)
        error = result.get("error", error)
        if result.get("success") is False:
            status = "error"
    if status in ("error", "failed", "failure"):
        text = str(error or "").strip()
        if not text:
            return "BROKEN", "error status with no message"
        return ("BROKEN" if _PYTHON_ERROR_TEXT.search(text) else "REFUSED"), text[:100]
    if io.touched:
        return "IO", io.summary()
    return "PURE", f"status={status!r}"


async def _build(adapter_type):
    cls = AdapterFactory._load_adapter_class(MAPPINGS[adapter_type])
    adapter = cls(AdapterConfig(name=adapter_type, category=AdapterCategory.UTILITY, credentials={}, base_url=None))
    await asyncio.wait_for(adapter.start(), timeout=PER_CALL_TIMEOUT)
    return adapter


async def _sweep():
    rows = {}
    for adapter_type in _canonical_types():
        try:
            adapter = await _build(adapter_type)
        except Exception as e:  # noqa: BLE001 - construction/start refusal covers every capability
            rows[f"{adapter_type}:*"] = _classify_exc(e)
            continue
        try:
            capabilities = adapter.get_capabilities()
        except Exception as e:  # noqa: BLE001
            rows[f"{adapter_type}:*"] = _classify_exc(e)
            capabilities = []
        for capability in capabilities:
            request = AdapterRequest(capability=capability.name, parameters=_params(capability))
            with probe() as io:
                try:
                    result = await asyncio.wait_for(adapter.execute(request), timeout=PER_CALL_TIMEOUT)
                except asyncio.TimeoutError:
                    rows[f"{adapter_type}:{capability.name}"] = ("IO", "timeout") if io.touched else ("BROKEN", "timeout with no I/O")
                    continue
                except Exception as e:  # noqa: BLE001
                    rows[f"{adapter_type}:{capability.name}"] = _classify_exc(e)
                    continue
            rows[f"{adapter_type}:{capability.name}"] = _classify_result(result, io)
        with probe() as io:
            try:
                health = await asyncio.wait_for(adapter.health_check(), timeout=PER_CALL_TIMEOUT)
            except Exception as e:  # noqa: BLE001
                rows[f"{adapter_type}:health"] = _classify_exc(e)
                health = None
        if health is not None:
            healthy = isinstance(health, dict) and str(health.get("status", "")).lower() in ("healthy", "ok", "connected", "active")
            if healthy and not io.touched:
                rows[f"{adapter_type}:health"] = ("PURE", f"status={health.get('status')!r} with no probe")
            else:
                rows[f"{adapter_type}:health"] = ("IO" if io.touched else "HONEST", io.summary())
        try:
            await adapter.stop()
        except Exception:  # noqa: BLE001 - teardown is not under test
            pass
    return rows


@pytest.mark.asyncio
async def test_every_adapter_call_does_real_work_or_is_in_the_ledger():
    rows = await _sweep()
    from collections import Counter
    print(f"\\ntier-a execute ({len(rows)} rows): {dict(Counter(k for k, _ in rows.values()))}")
    pure = {r: d for r, (k, d) in rows.items() if k == "PURE"}
    broken = {r: d for r, (k, d) in rows.items() if k == "BROKEN"}
    print(f"tier-a execute PURE: {pure}")
    print(f"tier-a execute BROKEN: {broken}")
    new_pure = {r: d for r, d in pure.items() if r not in KNOWN_FABRICATED and r not in DECLARED_PURE}
    new_broken = {r: d for r, d in broken.items() if r not in KNOWN_BROKEN}
    fixed = sorted(r for r in KNOWN_FABRICATED if r not in pure)
    stale_pure = sorted(r for r in DECLARED_PURE if r not in pure)
    fixed_broken = sorted(r for r in KNOWN_BROKEN if r not in broken)
    assert not new_pure, f"NEW success with no I/O - fabricated, or declare it (D-1): {new_pure}"
    assert not new_broken, f"NEW broken adapter calls (file in fix-queue.md): {new_broken}"
    assert not fixed, f"FIXED - remove from KNOWN_FABRICATED and tick the ledger: {fixed}"
    assert not stale_pure, f"DECLARED_PURE entry no longer pure - remove it: {stale_pure}"
    assert not fixed_broken, f"no longer broken - remove from KNOWN_BROKEN: {fixed_broken}"
