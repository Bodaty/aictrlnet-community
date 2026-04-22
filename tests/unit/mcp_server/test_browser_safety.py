"""Unit tests for mcp_server.browser_safety."""

import pytest
from unittest.mock import patch

from mcp_server.browser_safety import (
    BrowserSafetyError,
    DANGEROUS_ACTIONS,
    MAX_ACTIONS_PER_CALL,
    METADATA_HOSTS,
    SAFE_ACTIONS,
    validate_actions,
    validate_url,
)


# ---------------------------------------------------------------------------
# URL validation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "http://localhost/foo",
        "http://127.0.0.1/foo",
        "http://127.0.0.1:8000/admin",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "http://169.254.169.254/latest/meta-data/",  # AWS IMDS
        "http://169.254.169.254",
        "http://metadata.google.internal/computeMetadata/v1/",
        "http://[::1]/",
        "http://0.0.0.0/",
    ],
)
def test_validate_url_rejects_private_and_metadata(url):
    with patch("mcp_server.browser_safety._resolve_host", return_value=set()):
        with pytest.raises(BrowserSafetyError):
            validate_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "chrome://settings",
        "data:text/html,<script>",
        "javascript:alert(1)",
        "ftp://example.com/",
    ],
)
def test_validate_url_rejects_non_http_schemes(url):
    with pytest.raises(BrowserSafetyError):
        validate_url(url)


def test_validate_url_rejects_empty():
    with pytest.raises(BrowserSafetyError):
        validate_url("")
    with pytest.raises(BrowserSafetyError):
        validate_url(None)  # type: ignore[arg-type]


def test_validate_url_rejects_overlong():
    with pytest.raises(BrowserSafetyError):
        validate_url("https://example.com/" + ("x" * 10_000))


def test_validate_url_accepts_public_https():
    # Stub DNS so the test doesn't depend on network
    with patch(
        "mcp_server.browser_safety._resolve_host", return_value={"93.184.216.34"}
    ):
        validate_url("https://example.com/")


def test_validate_url_rejects_dns_rebinding():
    # Hostname looks public but resolves to private IP
    with patch(
        "mcp_server.browser_safety._resolve_host", return_value={"10.0.0.5"}
    ):
        with pytest.raises(BrowserSafetyError):
            validate_url("https://evil.example.com/")


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

def test_validate_actions_rejects_non_list():
    with pytest.raises(BrowserSafetyError):
        validate_actions({"type": "navigate"})  # type: ignore[arg-type]


def test_validate_actions_rejects_empty_list():
    with pytest.raises(BrowserSafetyError):
        validate_actions([])


def test_validate_actions_rejects_too_many():
    with pytest.raises(BrowserSafetyError):
        validate_actions(
            [{"type": "click", "selector": "#x"}] * (MAX_ACTIONS_PER_CALL + 1)
        )


def test_validate_actions_accepts_safe_actions():
    with patch(
        "mcp_server.browser_safety._resolve_host", return_value={"93.184.216.34"}
    ):
        validate_actions(
            [
                {"type": "navigate", "url": "https://example.com/"},
                {"type": "click", "selector": "#submit"},
                {"type": "fill", "selector": "#email", "value": "a@b.c"},
                {"type": "screenshot"},
                {"type": "extract_text", "selector": ".results"},
                {"type": "wait_for", "selector": ".done"},
            ]
        )


def test_validate_actions_rejects_unknown_type():
    with pytest.raises(BrowserSafetyError):
        validate_actions([{"type": "kaboom", "selector": "#x"}])


def test_validate_actions_rejects_dangerous_without_flag():
    with pytest.raises(BrowserSafetyError):
        validate_actions([{"type": "run_script", "script": "alert(1)"}])
    with pytest.raises(BrowserSafetyError):
        validate_actions([{"type": "download", "url": "https://example.com/x"}])


def test_validate_actions_accepts_dangerous_with_feature_flag():
    with patch(
        "mcp_server.browser_safety._resolve_host", return_value={"93.184.216.34"}
    ):
        validate_actions(
            [{"type": "run_script", "script": "return 1"}],
            feature_flags={"browser_run_script": True},
        )


def test_validate_actions_rejects_missing_type():
    with pytest.raises(BrowserSafetyError):
        validate_actions([{"selector": "#x"}])


def test_validate_actions_caps_selector_length():
    with pytest.raises(BrowserSafetyError):
        validate_actions(
            [{"type": "click", "selector": "#" + ("x" * 100_000)}]
        )


def test_validate_actions_caps_value_length():
    with pytest.raises(BrowserSafetyError):
        validate_actions(
            [{"type": "fill", "selector": "#e", "value": "a" * 100_000}]
        )


def test_safe_and_dangerous_are_disjoint():
    assert SAFE_ACTIONS.isdisjoint(DANGEROUS_ACTIONS)


def test_metadata_hosts_include_aws_imds():
    assert "169.254.169.254" in METADATA_HOSTS
