"""Shared outbound-URL guard (SSRF defense) — stdlib-only, no heavy deps.

Every server-side fetch of a user/LLM-supplied URL (webhook delivery, template
import, workflow api_call node, MCP server URLs, OAuth provider endpoints, …)
must run through ``validate_outbound_url`` at BOTH create time and send time.

Rules: https/http only; block cloud metadata endpoints, and any host that is
(or DNS-resolves to) a private / loopback / link-local / reserved / multicast
address — resolution defeats DNS-rebinding and hostname aliases.

This is the canonical implementation; mcp_server.browser_safety predates it and
enforces the same rules for the browser path.
"""

import ipaddress
import socket
from urllib.parse import urlparse

ALLOWED_SCHEMES = {"http", "https"}
MAX_URL_LEN = 2048
METADATA_HOSTS = {
    "169.254.169.254",              # AWS / GCP / Azure IMDS
    "metadata.google.internal",
    "metadata",
}
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


class SSRFError(ValueError):
    """Raised when an outbound URL targets a blocked/internal destination."""


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


def _resolve(host: str):
    try:
        return {info[4][0] for info in socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)}
    except Exception:
        return set()


def validate_outbound_url(url: str) -> str:
    """Validate an outbound URL or raise SSRFError. Returns the url on success."""
    if not url or not isinstance(url, str):
        raise SSRFError("url is required and must be a string")
    if len(url) > MAX_URL_LEN:
        raise SSRFError(f"url exceeds max length {MAX_URL_LEN}")

    parsed = urlparse(url)
    scheme = (parsed.scheme or "").lower()
    if scheme not in ALLOWED_SCHEMES:
        raise SSRFError(f"scheme '{scheme}' not allowed")

    host = (parsed.hostname or "").lower()
    if not host:
        raise SSRFError("url must include a hostname")
    if host in METADATA_HOSTS:
        raise SSRFError("metadata endpoint is blocked")
    if host in BLOCKED_HOSTNAMES:
        raise SSRFError(f"hostname '{host}' is blocked")
    if _is_blocked_ip(host):
        raise SSRFError(f"host {host} is a blocked address")

    # Resolve to catch DNS rebinding / alias tricks.
    for resolved in _resolve(host):
        if _is_blocked_ip(resolved):
            raise SSRFError(f"host {host} resolves to blocked address {resolved}")

    return url


# ---------------------------------------------------------------------------
# Connect-time IP pinning.
#
# validate_outbound_url resolves and checks, and then httpx resolves AGAIN when
# it connects — so a DNS record that flips between the two answers (rebinding)
# is not caught by the check above. Closing that needs the pin to happen at
# connect time, not at validation time.
#
# httpx-secure does exactly that: it captures the peer IP of a real connection
# via getpeername(), so there is no resolve->connect gap at all, then rewrites
# the request URL to that IP while preserving the Host header and the TLS SNI
# hostname (the mechanism documented in httpx's own "connecting to an IP
# address" guide). It installs as a request event hook, so redirects are
# covered too.
#
# It is additive, not a replacement: its own check is `is_global`, which says
# nothing about our scheme allowlist, URL length cap, or the metadata/blocked
# hostname lists. validate_outbound_url stays the pre-flight check, and
# _pin_validator re-applies the host rules at connect time against the IP that
# will actually be dialled.
# ---------------------------------------------------------------------------

def _pin_validator(hostname: str, ip_addr, port: int) -> bool:
    """Re-apply our host rules against the IP httpx is about to connect to."""
    host = (hostname or "").lower()
    if host in METADATA_HOSTS or host in BLOCKED_HOSTNAMES:
        return False
    return not _is_blocked_ip(str(ip_addr))


def pin_outbound_client(client):
    """Wrap an httpx.AsyncClient so every request connects to a validated IP.

    Returns the same client. Degrades to an unwrapped client if httpx-secure is
    unavailable — the caller's validate_outbound_url check still applies, so
    this weakens to the pre-existing behaviour rather than failing the request.
    """
    try:
        from httpx_secure import httpx_ssrf_protection
    except ImportError:  # pragma: no cover - dependency gate
        import logging

        logging.getLogger(__name__).warning(
            "httpx-secure not installed — outbound requests are validated but "
            "not IP-pinned, so a DNS rebind between check and connect is not "
            "caught. Install httpx-secure (see requirements.txt)."
        )
        return client

    return httpx_ssrf_protection(client, custom_validator=_pin_validator)
