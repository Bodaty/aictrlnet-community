"""Tier A-5: every adapter_type the factory advertises resolves to a BaseAdapter
subclass, instantiates from a minimal AdapterConfig, initialises without silently
reaching the network, and owns a real health check. The factory mapping is the
source of truth, as in production (core/app.py:114-140); dead files never appear.
Ledger semantics as in the other Tier A gates.
"""
import sys as _sys, pathlib as _pathlib
_sys.path.append(str(_pathlib.Path(__file__).resolve().parent))  # find tier_a_* helpers without shadowing src packages

import inspect
import pytest
from adapters.base_adapter import BaseAdapter
from adapters.factory import AdapterFactory
from adapters.models import AdapterConfig, AdapterCategory

from tier_a_egress import SmokeEgressBlocked

MAPPINGS = dict(AdapterFactory.ADAPTER_MAPPINGS)

# A-5b: adapters with no health check of their own (inherit {"status": "ok"}). Must shrink.
KNOWN_NO_HEALTH_CHECK = {"declarative-http", "perplexity", "taskrabbit"}
# Adapters whose initialize() legitimately opens a connection to an in-network service.
IN_NETWORK_ON_INIT = {"postgresql", "mysql", "redis"}
# Ledger by failure pattern (aliases map to the same class): every failure must match
# an entry and every entry must match at least one failure.
KNOWN_FAILING_PATTERNS = {
    # A-27: SemanticKernelAdapter reports initialize/get_agent_state abstract although the file defines both
    "A-27": r"abstract class SemanticKernelAdapter",
}


def _cls(adapter_type):
    return AdapterFactory._load_adapter_class(MAPPINGS[adapter_type])


@pytest.mark.parametrize("adapter_type", sorted(MAPPINGS))
def test_adapter_type_resolves_to_a_base_adapter(adapter_type):
    cls = _cls(adapter_type)
    assert inspect.isclass(cls) and issubclass(cls, BaseAdapter), (adapter_type, cls)


def test_health_check_ownership_matches_the_ledger():
    missing = set()
    for adapter_type in sorted(MAPPINGS):
        cls = _cls(adapter_type)
        own = any(("_perform_health_check" in vars(k) or "health_check" in vars(k)) for k in cls.__mro__ if k is not BaseAdapter and k is not object)
        if not own:
            missing.add(adapter_type)
    new = missing - KNOWN_NO_HEALTH_CHECK
    fixed = KNOWN_NO_HEALTH_CHECK - missing
    assert not new, f"adapters inheriting BaseAdapter's hardcoded status=ok (file them): {sorted(new)}"
    assert not fixed, f"now own a health check - remove from KNOWN_NO_HEALTH_CHECK: {sorted(fixed)}"


@pytest.mark.asyncio
async def test_every_adapter_initialises_without_silent_egress_or_is_in_the_ledger():
    failures = {}
    for adapter_type in sorted(MAPPINGS):
        cls = _cls(adapter_type)
        try:
            adapter = cls(AdapterConfig(name=adapter_type, category=AdapterCategory.UTILITY, credentials={}, base_url=None))
        except ValueError as e:  # a clear "credentials required" refusal at construction is a REAL result
            if not str(e).strip():
                failures[adapter_type] = "silent: ctor raised ValueError with no message"
            continue
        except Exception as e:  # noqa: BLE001 - TypeError/AttributeError at construction = the class cannot exist
            failures[adapter_type] = f"ctor: {type(e).__name__}: {str(e)[:120]}"
            continue
        try:
            await adapter.initialize()
        except SmokeEgressBlocked:
            if adapter_type not in IN_NETWORK_ON_INIT:
                failures[adapter_type] = "egress-on-init: initialize() reached outside the network with empty credentials"
        except Exception as e:  # noqa: BLE001 - a clear refusal is a real result
            if not str(e).strip():
                failures[adapter_type] = f"silent: initialize() raised {type(e).__name__} with no message"
    print(f"\\ntier-a adapters: {len(failures)} failing of {len(MAPPINGS)}: {failures}")
    import re as _re
    matched = {fid: [k for k, v in failures.items() if _re.search(rx, v)] for fid, rx in KNOWN_FAILING_PATTERNS.items()}
    accounted = {k for ks in matched.values() for k in ks}
    new = {k: v for k, v in failures.items() if k not in accounted}
    fixed = [fid for fid, ks in matched.items() if not ks]
    print(f"tier-a adapters by finding: { {fid: len(ks) for fid, ks in matched.items()} }")
    assert not new, f"NEW adapter failures (file them): {new}"
    assert not fixed, f"FIXED - remove from KNOWN_FAILING_PATTERNS and close: {fixed}"


@pytest.mark.asyncio
async def test_platform_adapters_reject_empty_credentials():
    """Second hierarchy (services/platform_adapters): n8n, Zapier, Make, Power Automate, IFTTT.
    validate_credentials({}) must be (False, <reason>) - (True, None) would be a placeholder."""
    import services.platform_adapters as pa  # importing the package registers all five
    reg = pa.PlatformAdapterRegistry
    platforms = list(reg.list_platforms())
    assert len(platforms) >= 5, platforms
    bad = {}
    for pt in platforms:
        adapter = reg.get_adapter(pt)
        ok, reason = await adapter.validate_credentials({})
        if ok or not reason:
            bad[str(pt)] = (ok, reason)
    assert not bad, f"platform adapters accepting empty credentials: {bad}"
