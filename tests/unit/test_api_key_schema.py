"""Unit tests for APIKeyCreate / APIKeyUpdate scope validation.

WS1: legacy scopes (`read:all`, `write:all`) were dropped at request-time
in Wave 2 Phase B. Keys submitted with them would silently fail every
scope check, so fail fast at create/update instead.
"""

import pytest
from pydantic import ValidationError

from schemas.api_key import APIKeyCreate, APIKeyUpdate
from mcp_server.scopes import ALL_SCOPES


class TestAPIKeyCreateScopes:
    def test_accepts_all_71_new_scopes(self):
        k = APIKeyCreate(name="k", scopes=sorted(ALL_SCOPES))
        assert set(k.scopes) == ALL_SCOPES

    def test_accepts_empty_scopes(self):
        k = APIKeyCreate(name="k", scopes=[])
        assert k.scopes == []

    def test_accepts_single_known_scope(self):
        k = APIKeyCreate(name="k", scopes=["read:workflows"])
        assert k.scopes == ["read:workflows"]

    @pytest.mark.parametrize("legacy", ["read:all", "write:all", "admin:all"])
    def test_rejects_legacy_scope(self, legacy):
        with pytest.raises(ValidationError) as exc:
            APIKeyCreate(name="k", scopes=[legacy])
        assert legacy in str(exc.value)

    def test_rejects_unknown_scope(self):
        with pytest.raises(ValidationError) as exc:
            APIKeyCreate(name="k", scopes=["read:nonexistent"])
        assert "read:nonexistent" in str(exc.value)

    def test_reports_all_unknown_scopes(self):
        with pytest.raises(ValidationError) as exc:
            APIKeyCreate(name="k", scopes=["read:workflows", "read:all", "read:ghost"])
        msg = str(exc.value)
        assert "read:all" in msg
        assert "read:ghost" in msg


class TestAPIKeyCreateIPs:
    def test_accepts_valid_ip_and_cidr(self):
        k = APIKeyCreate(name="k", allowed_ips=["10.0.0.1", "192.168.0.0/16", "::1"])
        assert len(k.allowed_ips) == 3

    def test_rejects_garbage(self):
        with pytest.raises(ValidationError):
            APIKeyCreate(name="k", allowed_ips=["999.999.999.999"])


class TestAPIKeyCreateRateLimitPerTool:
    def test_accepts_valid_shape(self):
        k = APIKeyCreate(
            name="k",
            rate_limit_per_tool={"create_workflow": {"per_minute": 10, "per_day": 100}},
        )
        assert k.rate_limit_per_tool == {"create_workflow": {"per_minute": 10, "per_day": 100}}

    def test_accepts_none(self):
        k = APIKeyCreate(name="k", rate_limit_per_tool=None)
        assert k.rate_limit_per_tool is None

    def test_rejects_unknown_cap_key(self):
        with pytest.raises(ValidationError):
            APIKeyCreate(
                name="k",
                rate_limit_per_tool={"create_workflow": {"per_hour": 10}},
            )

    def test_rejects_non_positive(self):
        with pytest.raises(ValidationError):
            APIKeyCreate(
                name="k",
                rate_limit_per_tool={"create_workflow": {"per_minute": 0}},
            )


class TestAPIKeyUpdateScopes:
    def test_none_passes(self):
        u = APIKeyUpdate(scopes=None)
        assert u.scopes is None

    def test_accepts_valid(self):
        u = APIKeyUpdate(scopes=["read:workflows", "write:workflows"])
        assert u.scopes == ["read:workflows", "write:workflows"]

    def test_rejects_legacy(self):
        with pytest.raises(ValidationError):
            APIKeyUpdate(scopes=["read:all"])
