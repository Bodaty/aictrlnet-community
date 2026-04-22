"""SSRF / injection defense for the ``browser_execute`` MCP tool.

The browser-automation microservice accepts a list of ``BrowserAction``
records (navigate / click / fill / screenshot / extract_text / wait_for
/ run_script / download). Exposing it directly via MCP would let any
``write:browser`` caller pivot into internal networks (cloud metadata,
localhost, link-local) or execute arbitrary JavaScript.

This module enforces, BEFORE the request leaves the MCP server:

1. **Action allow-list** — only safe action types by default.
   ``run_script`` and ``download`` disabled unless a tenant-scoped
   feature flag is set.
2. **URL deny-list** — RFC1918, link-local, loopback, cloud-metadata
   endpoints, and non-HTTP(S) schemes are rejected outright. Hostnames
   are resolved locally and re-checked against the same rules to block
   DNS-rebinding and ``0.0.0.0`` tricks.
3. **Hard cap on actions per call** (default 20).
4. **Read-side length caps** on ``selector`` / ``value`` / ``script``
   fields to keep payloads bounded.

The browser-service should apply the SAME rules as defense in depth,
but enforcing here removes the easiest attack path and gives us a
single place to audit.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from typing import Iterable, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

SAFE_ACTIONS: set[str] = {
    "navigate",
    "click",
    "fill",
    "screenshot",
    "extract_text",
    "wait_for",
}

# Actions that must be gated by a tenant feature flag.
DANGEROUS_ACTIONS: set[str] = {"run_script", "download"}

ALLOWED_SCHEMES: set[str] = {"http", "https"}

# Cloud / orchestrator metadata endpoints (literal IPs + common DNS).
METADATA_HOSTS: set[str] = {
    "169.254.169.254",          # AWS / GCP / Azure IMDS
    "metadata.google.internal",
    "metadata",
    "100.100.100.200",          # Alibaba ECS
}

# Hostnames that don't parse as IPs but always resolve to local
# addresses. Checked before DNS resolution so tests and offline
# environments still catch them.
BLOCKED_HOSTNAMES: set[str] = {
    "localhost",
    "localhost.localdomain",
    "ip6-localhost",
    "ip6-loopback",
    "broadcasthost",
}

MAX_ACTIONS_PER_CALL = 20
MAX_SELECTOR_LEN = 2_048
MAX_VALUE_LEN = 16_384
MAX_SCRIPT_LEN = 8_192
MAX_URL_LEN = 4_096


class BrowserSafetyError(ValueError):
    """Raised when an action or URL is denied by policy."""


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

def _is_blocked_ip(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _resolve_host(host: str) -> Iterable[str]:
    try:
        return {
            info[4][0]
            for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        }
    except Exception:
        return ()


def validate_url(url: str) -> None:
    """Raise ``BrowserSafetyError`` if the URL is not safe to fetch."""
    if not url or not isinstance(url, str):
        raise BrowserSafetyError("url is required and must be a string")
    if len(url) > MAX_URL_LEN:
        raise BrowserSafetyError(f"url exceeds max length {MAX_URL_LEN}")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise BrowserSafetyError(f"scheme '{scheme}' not allowed")

    host = (parsed.hostname or "").lower()
    if not host:
        raise BrowserSafetyError("url must include a hostname")
    if host in METADATA_HOSTS:
        raise BrowserSafetyError("metadata endpoint is blocked")
    if host in BLOCKED_HOSTNAMES:
        raise BrowserSafetyError(f"hostname '{host}' is blocked")

    # Literal IP in the URL
    if _is_blocked_ip(host):
        raise BrowserSafetyError(f"host {host} resolves to a blocked address")

    # Resolve to catch DNS rebinding / alias tricks
    for resolved in _resolve_host(host):
        if _is_blocked_ip(resolved):
            raise BrowserSafetyError(
                f"host {host} resolves to blocked address {resolved}"
            )


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

def _allowed_action_types(
    tenant_id: Optional[str], feature_flags: Optional[dict] = None
) -> set[str]:
    allowed = set(SAFE_ACTIONS)
    flags = feature_flags or {}
    if flags.get("browser_run_script"):
        allowed.add("run_script")
    if flags.get("browser_download"):
        allowed.add("download")
    return allowed


def validate_actions(
    actions: list[dict],
    tenant_id: Optional[str] = None,
    feature_flags: Optional[dict] = None,
) -> list[dict]:
    """Validate a list of ``BrowserAction`` dicts.

    Returns the (possibly normalized) list on success. Raises
    ``BrowserSafetyError`` on any violation.
    """
    if not isinstance(actions, list):
        raise BrowserSafetyError("'actions' must be a list")
    if len(actions) == 0:
        raise BrowserSafetyError("'actions' must contain at least one action")
    if len(actions) > MAX_ACTIONS_PER_CALL:
        raise BrowserSafetyError(
            f"too many actions (max {MAX_ACTIONS_PER_CALL})"
        )

    allowed_types = _allowed_action_types(tenant_id, feature_flags)
    out: list[dict] = []
    for i, action in enumerate(actions):
        if not isinstance(action, dict):
            raise BrowserSafetyError(f"action[{i}] must be an object")
        a_type = action.get("type") or action.get("action_type") or action.get("action")
        if not a_type:
            raise BrowserSafetyError(f"action[{i}] missing 'type'")
        if a_type not in allowed_types:
            if a_type in DANGEROUS_ACTIONS:
                raise BrowserSafetyError(
                    f"action[{i}] type '{a_type}' disabled for this tenant"
                )
            raise BrowserSafetyError(
                f"action[{i}] type '{a_type}' is not in the allow-list"
            )

        if "url" in action and action["url"]:
            validate_url(action["url"])

        for field, cap in (
            ("selector", MAX_SELECTOR_LEN),
            ("value", MAX_VALUE_LEN),
            ("script", MAX_SCRIPT_LEN),
        ):
            v = action.get(field)
            if isinstance(v, str) and len(v) > cap:
                raise BrowserSafetyError(
                    f"action[{i}].{field} exceeds max length {cap}"
                )

        out.append(action)

    return out


__all__ = [
    "ALLOWED_SCHEMES",
    "BLOCKED_HOSTNAMES",
    "BrowserSafetyError",
    "DANGEROUS_ACTIONS",
    "MAX_ACTIONS_PER_CALL",
    "METADATA_HOSTS",
    "SAFE_ACTIONS",
    "validate_actions",
    "validate_url",
]
