"""The seeds never create dev@/admin@/test@ (published passwords) in production.

25 Nov 2026: dev@aictrlnet.com signed in to production with testpass123 as
admin, because the hosted seed job ran with ENVIRONMENT=development.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from seed_profile import known_password_accounts_allowed  # noqa: E402


@pytest.mark.parametrize("value", ["development", "dev", "test", " Development "])
def test_allowed_outside_production(monkeypatch, value):
    monkeypatch.setenv("ENVIRONMENT", value)
    assert known_password_accounts_allowed() is True


@pytest.mark.parametrize("value", ["production", "prod", "PRODUCTION", " Production "])
def test_refused_in_production(monkeypatch, value):
    monkeypatch.setenv("ENVIRONMENT", value)
    assert known_password_accounts_allowed() is False


@pytest.mark.parametrize("value", [None, ""])
def test_unset_environment_counts_as_production(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("ENVIRONMENT", raising=False)
    else:
        monkeypatch.setenv("ENVIRONMENT", value)
    assert known_password_accounts_allowed() is False


@pytest.mark.parametrize("value", ["staging", "stage", " Staging "])
def test_staging_refused_by_default(monkeypatch, value):
    monkeypatch.setenv("ENVIRONMENT", value)
    monkeypatch.delenv("SEED_ALLOW_KNOWN_PASSWORDS", raising=False)
    assert known_password_accounts_allowed() is False


@pytest.mark.parametrize("flag", ["0", "true", ""])
def test_staging_needs_the_exact_flag(monkeypatch, flag):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("SEED_ALLOW_KNOWN_PASSWORDS", flag)
    assert known_password_accounts_allowed() is False


def test_staging_allowed_with_flag(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("SEED_ALLOW_KNOWN_PASSWORDS", "1")
    assert known_password_accounts_allowed() is True


def test_flag_does_not_open_production(monkeypatch):
    monkeypatch.setenv("ENVIRONMENT", "production")
    monkeypatch.setenv("SEED_ALLOW_KNOWN_PASSWORDS", "1")
    assert known_password_accounts_allowed() is False
