"""PHI egress guard for model providers (HIPAA risk analysis R-04).

Under AICTRLNET_PHI_MODE, a request carrying protected health information may
only reach a model provider the deployment has allowlisted, and a local
provider (Ollama, vLLM) only at a loopback, private or link-local address.
Cloud providers additionally need a BAA recorded. Refusal is an error, never a
silent skip, and never a fallback to another provider.

This module is imported by adapters/base_adapter.py at boot, so it must stay
free of imports from llm.* and adapters.* (a core -> llm -> adapters cycle).
It carries its own copy of the provider alias table for that reason; the R-04
spec pins the copy to llm.tier_resolver.PROVIDER_ALIASES.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import time
from typing import Optional
from urllib.parse import urlsplit


logger = logging.getLogger(__name__)

# Every message starts with this. The exception TYPE does not survive the trip
# from an adapter to a workflow run (base_node stringifies it, ai_process_node
# re-wraps it, the loop node re-raises result.error), so nodes that swallow
# child failures by design recognise a refusal by this string instead.
PHI_REFUSAL_MARKER = "[PHI-EGRESS-REFUSED]"

ALLOWLIST_SETTING = "AICTRLNET_PHI_LLM_PROVIDERS"
BAA_SETTING = "AICTRLNET_PHI_BAA_PROVIDERS"
# Hostnames of the EHR FHIR / identity servers the care-gap sync may call.
FHIR_HOSTS_SETTING = "AICTRLNET_PHI_FHIR_HOSTS"

# Providers served from the deployment's own machine. They need no BAA but
# their endpoint must resolve locally.
LOCAL_PROVIDERS = frozenset({"ollama", "vllm"})

# Not a model provider: the agent-framework-service container, which forwards
# prompts to whatever providers IT is configured with. Allowlisting it is the
# operator's attestation that that container is configured with allowed
# providers only.
PSEUDO_PROVIDERS = frozenset({"agent-framework"})

# llm.models.ModelProvider minus "custom", plus perplexity (registered as an
# adapter, not in the enum) and the pseudo-provider. Literal so this module
# stays import-light; the spec pins it to the enum.
KNOWN_PROVIDERS = frozenset(
    {
        "ollama",
        "vllm",
        "anthropic",
        "openai",
        "gemini",
        "vertex_ai",
        "huggingface",
        "bedrock",
        "azure_openai",
        "cohere",
        "deepseek",
        "dashscope",
        "perplexity",
    }
) | PSEUDO_PROVIDERS

# Copy of llm.tier_resolver.PROVIDER_ALIASES (see module docstring).
_ALIASES = {
    "claude": "anthropic",
    "google": "gemini",
    "vertex-ai": "vertex_ai",
    "vertexai": "vertex_ai",
    "google-gemini": "gemini",
}

# Names that are local by construction; checked before any DNS so the common
# case never blocks the event loop (adapter construction is synchronous and
# runs inside async node execution).
_STATIC_LOCAL_HOSTS = frozenset({"localhost", "host.docker.internal"})

_CACHE_TTL_SECONDS = 60.0
_endpoint_cache: dict[str, tuple[bool, float]] = {}


class PHIEgressRefused(RuntimeError):
    """A model call would send PHI to a provider not allowed to receive it."""


def normalize_provider(name: Optional[str]) -> Optional[str]:
    if not name:
        return name
    key = name.strip().lower()
    return _ALIASES.get(key, key)


def parse_provider_list(raw: Optional[str]) -> frozenset[str]:
    """Comma-separated provider names -> canonical, de-duplicated set."""
    if not raw:
        return frozenset()
    return frozenset(
        normalize_provider(part) for part in raw.split(",") if part and part.strip()
    )


def parse_host_list(raw: Optional[str]) -> frozenset[str]:
    """Comma-separated hostnames -> lowercased, de-duplicated set."""
    if not raw:
        return frozenset()
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


def reset_endpoint_cache() -> None:
    _endpoint_cache.clear()


def _is_local_ip(ip: ipaddress._BaseAddress) -> bool:
    return ip.is_loopback or ip.is_private or ip.is_link_local


def resolve_endpoint_addresses(url: str) -> tuple[str, list[str]]:
    """(host, resolved addresses). Empty list when the name does not resolve."""
    host = urlsplit(url if "//" in url else f"//{url}").hostname or ""
    try:
        ipaddress.ip_address(host)
        return host, [host]
    except ValueError:
        pass
    if host in _STATIC_LOCAL_HOSTS:
        return host, []
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError, OSError):
        return host, []
    return host, sorted({info[4][0] for info in infos if info and info[4]})


def is_local_endpoint(url: str) -> bool:
    """True when every address behind the URL's host is loopback, private or
    link-local, or when the host cannot be resolved at all.

    Unresolvable is allowed on purpose: nothing can egress to a name that does
    not resolve (the connection fails on its own), and the compose defaults
    use host.docker.internal, which does not resolve on the host where the
    security specs run. This deliberately differs from the test egress guard
    in tests/smoke_common/egress_guard.py, whose job is to block, and which
    also inspects only the first resolved address; this one inspects all of
    them, so a name with one private and one public record is not local.
    """
    host = urlsplit(url if "//" in url else f"//{url}").hostname or ""
    if not host:
        return False
    if host in _STATIC_LOCAL_HOSTS:
        return True
    try:
        return _is_local_ip(ipaddress.ip_address(host))
    except ValueError:
        pass

    now = time.monotonic()
    cached = _endpoint_cache.get(host)
    if cached and cached[1] > now:
        return cached[0]

    _, addresses = resolve_endpoint_addresses(url)
    if not addresses:
        verdict = True
    else:
        verdict = all(_is_local_ip(ipaddress.ip_address(a)) for a in addresses)
    _endpoint_cache[host] = (verdict, now + _CACHE_TTL_SECONDS)
    return verdict


def _settings(settings=None):
    if settings is not None:
        return settings
    from core.config import get_settings

    return get_settings()


def phi_mode_on(settings=None) -> bool:
    return bool(getattr(_settings(settings), "AICTRLNET_PHI_MODE", False))


def phi_allowed_providers(settings=None) -> frozenset[str]:
    return parse_provider_list(getattr(_settings(settings), ALLOWLIST_SETTING, ""))


def phi_baa_providers(settings=None) -> frozenset[str]:
    return parse_provider_list(getattr(_settings(settings), BAA_SETTING, ""))


def phi_fhir_hosts(settings=None) -> frozenset[str]:
    return parse_host_list(getattr(_settings(settings), FHIR_HOSTS_SETTING, ""))


def describe_endpoint(url: str) -> str:
    host, addresses = resolve_endpoint_addresses(url)
    if addresses and addresses != [host]:
        return f"'{url}' (host '{host}' resolves to {', '.join(addresses)})"
    return f"'{url}'"


def assert_phi_provider_allowed(
    provider: Optional[str], base_url: Optional[str] = None, *, settings=None
) -> None:
    """Refuse to send PHI to a provider the deployment has not allowlisted.

    No-op unless AICTRLNET_PHI_MODE is on. Raises PHIEgressRefused, whose
    message starts with PHI_REFUSAL_MARKER and names the provider, the setting
    that governs it, what the endpoint resolved to, and what is required.
    """
    settings = _settings(settings)
    if not phi_mode_on(settings):
        return

    canonical = normalize_provider(provider) or "<unknown>"
    allowed = phi_allowed_providers(settings)

    if canonical not in allowed:
        listed = ", ".join(sorted(allowed)) if allowed else "<empty>"
        raise PHIEgressRefused(
            f"{PHI_REFUSAL_MARKER} Refusing to send protected health information "
            f"to model provider '{canonical}': it is not in {ALLOWLIST_SETTING} "
            f"(currently: {listed}). Required: add '{canonical}' to "
            f"{ALLOWLIST_SETTING} (and, for a cloud provider, record its BAA in "
            f"{BAA_SETTING}), or route this work to an allowlisted provider."
        )

    if canonical in LOCAL_PROVIDERS and base_url and not is_local_endpoint(base_url):
        raise PHIEgressRefused(
            f"{PHI_REFUSAL_MARKER} Refusing to send protected health information "
            f"to '{canonical}' at {describe_endpoint(base_url)}: a local provider "
            f"must resolve to a loopback, private or link-local address. Required: "
            f"point '{canonical}' at the on-machine endpoint, or allowlist the "
            f"hosted provider by its own name with a BAA recorded in {BAA_SETTING}."
        )

    if canonical in PSEUDO_PROVIDERS and base_url and not is_local_endpoint(base_url):
        raise PHIEgressRefused(
            f"{PHI_REFUSAL_MARKER} Refusing to send protected health information "
            f"to '{canonical}' at {describe_endpoint(base_url)}: the agent "
            f"framework service must be a container on this deployment. Required: "
            f"a compose-local FRAMEWORK_SERVICE_URL."
        )


def assert_phi_fhir_host_allowed(url: str, *, settings=None) -> None:
    """Refuse to call an EHR FHIR or identity host that is not allowlisted.

    No-op unless AICTRLNET_PHI_MODE is on. The URL is never quoted back: a FHIR
    search carries patient identifiers in its query string, and refusals are
    shown to staff and written to logs. The host alone says what was refused.
    """
    settings = _settings(settings)
    if not phi_mode_on(settings):
        return

    parts = urlsplit(url if "//" in url else f"//{url}")
    host = (parts.hostname or "").lower()
    allowed = phi_fhir_hosts(settings)
    listed = ", ".join(sorted(allowed)) if allowed else "<empty>"

    if parts.scheme != "https":
        raise PHIEgressRefused(
            f"{PHI_REFUSAL_MARKER} Refusing to send protected health information "
            f"to FHIR host '{host or '<none>'}' over '{parts.scheme or '<none>'}': "
            f"PHI leaves this deployment over https only. Required: an https URL "
            f"for that host."
        )

    if host not in allowed:
        raise PHIEgressRefused(
            f"{PHI_REFUSAL_MARKER} Refusing to send protected health information "
            f"to FHIR host '{host or '<none>'}': it is not in {FHIR_HOSTS_SETTING} "
            f"(currently: {listed}). Required: add '{host}' to "
            f"{FHIR_HOSTS_SETTING} on a deployment covered by a BAA with that "
            f"vendor, or point the connection at an allowlisted host."
        )


def assert_local_service_endpoint(name: str, url: Optional[str], *, settings=None) -> None:
    """For platform-internal hops that are not model providers (the llm-service
    bridge, the MCP server): under PHI mode the target must be local."""
    settings = _settings(settings)
    if not phi_mode_on(settings) or not url:
        return
    if not is_local_endpoint(url):
        raise PHIEgressRefused(
            f"{PHI_REFUSAL_MARKER} Refusing to send protected health information "
            f"to internal service '{name}' at {describe_endpoint(url)}: under PHI "
            f"mode platform services must be local to this deployment. Required: "
            f"a loopback, private or compose-local address."
        )


def phi_boot_problems(settings) -> list[str]:
    """Configuration problems for validate_phi_mode() to report at startup.

    Only meaningful under PHI mode (the caller checks). An empty allowlist
    yields no problems: a PHI deployment that uses no LLM is legitimate, and
    every LLM call then refuses at runtime naming the setting.
    """
    problems: list[str] = []
    allowed = phi_allowed_providers(settings)
    baa = phi_baa_providers(settings)

    for setting_name, names in ((ALLOWLIST_SETTING, allowed), (BAA_SETTING, baa)):
        unknown = sorted(names - KNOWN_PROVIDERS)
        if unknown:
            problems.append(
                f"{setting_name} names unknown provider(s) {', '.join(unknown)}. "
                f"Required: one of {', '.join(sorted(KNOWN_PROVIDERS))}."
            )

    fhir_hosts = phi_fhir_hosts(settings)
    malformed = sorted(
        h
        for h in fhir_hosts
        if any(c in h for c in "/:*") or h.split() != [h]
    )
    if malformed:
        problems.append(
            f"{FHIR_HOSTS_SETTING} entries must be bare hostnames, not URLs, "
            f"ports or wildcards: {', '.join(malformed)}. Required: e.g. "
            f"'api.practicefusion.com'."
        )
    elif not fhir_hosts and getattr(settings, "CARE_GAPS_ENABLED", False):
        # Not a problem: a practice that keeps its roster by hand runs the
        # care-gap engine with no EHR egress at all.
        logger.warning(
            "CARE_GAPS_ENABLED is on under PHI mode but %s is empty: the FHIR "
            "sync will refuse every call. Set it to the EHR's FHIR host to "
            "enable syncing, or ignore this if the roster is maintained manually.",
            FHIR_HOSTS_SETTING,
        )

    if not allowed:
        return problems

    for name in sorted(allowed - LOCAL_PROVIDERS - PSEUDO_PROVIDERS):
        if name in KNOWN_PROVIDERS and name not in baa:
            problems.append(
                f"{ALLOWLIST_SETTING} allows '{name}', a provider outside this "
                f"deployment, but no BAA is recorded for it. Required: record the "
                f"BAA by adding '{name}' to {BAA_SETTING}, or remove it from "
                f"{ALLOWLIST_SETTING}."
            )

    # Every env name a local provider's endpoint can come from. Four names,
    # three precedences (vllm_adapter, llm/generation.py, llm/service.py);
    # checking a URL nothing reads would be a worthless check, so every set
    # one is checked and none is unified here.
    endpoint_settings = {
        "ollama": ("OLLAMA_URL", "OLLAMA_BASE_URL"),
        "vllm": ("VLLM_URL", "VLLM_BASE_URL"),
    }
    for provider in sorted(allowed & LOCAL_PROVIDERS):
        for setting_name in endpoint_settings[provider]:
            url = getattr(settings, setting_name, None)
            if url and not is_local_endpoint(url):
                problems.append(
                    f"{setting_name}={url!r} resolves to a non-local address "
                    f"({describe_endpoint(url)}), but '{provider}' is allowlisted "
                    f"as a LOCAL provider. Required: an endpoint on this machine "
                    f"or the clinic network."
                )

    default_provider = _default_model_provider(settings)
    if default_provider and default_provider not in allowed:
        problems.append(
            f"DEFAULT_LLM_MODEL={getattr(settings, 'DEFAULT_LLM_MODEL', '')!r} "
            f"routes to provider '{default_provider}', which is not in "
            f"{ALLOWLIST_SETTING}; every auto-selected model call would refuse. "
            f"Required: a default model served by an allowlisted provider."
        )
    return problems


def _default_model_provider(settings) -> Optional[str]:
    """Provider DEFAULT_LLM_MODEL routes to, via the canonical resolver.

    Imported lazily: llm.tier_resolver reads Settings at import time and this
    module must not import llm.* at module level.
    """
    model = getattr(settings, "DEFAULT_LLM_MODEL", None)
    if not model:
        return None
    from llm.tier_resolver import provider_for_model_name

    return normalize_provider(provider_for_model_name(model))
