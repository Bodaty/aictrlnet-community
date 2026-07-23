"""Unit tests for the canonical resolve_model() (Phase 2)."""
import pytest

from llm.models import ModelTier
from llm.tier_resolver import ModelResolution, resolve_model


class FakeUser:
    def __init__(self, **kw):
        self.selected_model = kw.get("selected_model")
        self.preferredFastModel = kw.get("preferredFastModel")
        self.preferredBalancedModel = kw.get("preferredBalancedModel")
        self.preferredQualityModel = kw.get("preferredQualityModel")


class FakeOrg:
    def __init__(self, preferred_model=None, trial_mode=True, allowed_providers=None, own_key=False):
        self.preferred_model = preferred_model
        self.trial_mode = trial_mode
        self.allowed_providers = allowed_providers or []
        self._own_key = own_key

    def has_own_key(self, provider=None):
        return self._own_key


def test_explicit_wins_over_everything():
    r = resolve_model(
        explicit_model="claude-3-opus",
        user_settings=FakeUser(preferredQualityModel="mistral:7b"),
        org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=False),
    )
    assert r.model == "claude-3-opus"
    assert r.source == "explicit"


def test_user_tier_pref_beats_org(monkeypatch):
    r = resolve_model(
        user_settings=FakeUser(preferredBalancedModel="mistral:7b"),
        org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=False),
        tier=ModelTier.BALANCED,
        is_available=lambda m: m == "mistral:7b",
    )
    assert (r.model, r.source, r.tier) == ("mistral:7b", "user_tier_preference", ModelTier.BALANCED)


def test_legacy_selected_model_beats_org():
    r = resolve_model(
        user_settings=FakeUser(selected_model="mistral:7b"),
        org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=False),
        is_available=lambda m: m == "mistral:7b",
    )
    assert (r.model, r.source) == ("mistral:7b", "user_selected_model")


def test_tier_pref_unavailable_falls_back_to_legacy_selected_model():
    r = resolve_model(
        user_settings=FakeUser(preferredBalancedModel="unavailable:model", selected_model="mistral:7b"),
        org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=False),
        tier=ModelTier.BALANCED,
        is_available=lambda m: m == "mistral:7b",
    )
    assert (r.model, r.source) == ("mistral:7b", "user_selected_model")


def test_org_preferred_when_no_user_pref():
    r = resolve_model(org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=False))
    assert (r.model, r.source) == ("gpt-4o", "org_preferred")


def test_org_skipped_in_trial_without_own_key(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    r = resolve_model(org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=True))
    assert r.source == "system_default"


def test_org_honored_in_trial_with_own_key():
    r = resolve_model(org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=True, own_key=True))
    assert (r.model, r.source) == ("gpt-4o", "org_preferred")


def test_org_blocked_by_allowed_providers(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    r = resolve_model(
        org_settings=FakeOrg(preferred_model="gpt-4o", trial_mode=False, allowed_providers=["anthropic"])
    )
    assert r.source == "system_default"


def test_unavailable_candidates_fall_through(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
    r = resolve_model(
        user_settings=FakeUser(selected_model="gone:1b"),
        is_available=lambda m: False,
    )
    assert (r.model, r.source) == ("vllm:Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit", "system_default")


def test_system_default_when_nothing_configured(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    r = resolve_model()
    assert isinstance(r, ModelResolution)
    assert (r.model, r.source, r.provider) == ("llama3.1:8b-instruct-q4_K_M", "system_default", "ollama")
