"""Characterization matrix for model resolution (model-selection remediation).

Documents LLMGenerationEngine._select_model's precedence, which delegates to the
canonical llm.tier_resolver.resolve_model():
  explicit request → user prefs (tier → quality backfill → legacy selected_model)
  → org preferred_model → system default
These tests guard that precedence contract — a regression here means resolve_model
or its delegation from _select_model has drifted from the documented order.
Plan: .claude/plans/llm-generate-fallback-and-model-selection.md
"""
from unittest.mock import AsyncMock

import pytest

from llm.generation import LLMGenerationEngine
from llm.model_selection import classify_model_tier
from llm.models import LLMRequest, ModelTier, UserLLMSettings


@pytest.fixture
def engine():
    return LLMGenerationEngine()


def _settings(**kwargs):
    kwargs.setdefault("user_id", "u1")
    kwargs.setdefault("selected_model", "llama3.2:1b")
    return UserLLMSettings(**kwargs)


# --- today's precedence (characterization — keep green) ---------------------


async def test_explicit_override_beats_user_preferences(engine):
    request = LLMRequest(
        prompt="hi",
        model_override="claude-3-opus",
        user_settings=_settings(preferredQualityModel="llama3.1:8b-instruct-q4_K_M"),
    )

    model, tier = await engine._select_model(request)

    assert model == "claude-3-opus"
    assert tier == classify_model_tier("claude-3-opus")


async def test_user_tier_preference_wins_when_locally_available(engine, monkeypatch):
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=["mistral:7b"]))
    request = LLMRequest(
        prompt="hi",
        task_type="unknown_task_type",  # documents the silent BALANCED default too
        user_settings=_settings(preferredBalancedModel="mistral:7b"),
    )

    model, tier = await engine._select_model(request)

    assert model == "mistral:7b"
    assert tier == ModelTier.BALANCED


async def test_quality_preference_backfills_other_tiers(engine, monkeypatch):
    monkeypatch.setattr(
        engine, "_get_ollama_models", AsyncMock(return_value=["llama3.1:8b-instruct-q4_K_M"])
    )
    request = LLMRequest(
        prompt="hi",
        task_type="unknown_task_type",
        user_settings=_settings(preferredQualityModel="llama3.1:8b-instruct-q4_K_M"),
    )

    model, _ = await engine._select_model(request)

    assert model == "llama3.1:8b-instruct-q4_K_M"


async def test_legacy_selected_model_used_without_tier_prefs(engine, monkeypatch):
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=["mistral:7b"]))
    request = LLMRequest(prompt="hi", user_settings=_settings(selected_model="mistral:7b"))

    model, _ = await engine._select_model(request)

    assert model == "mistral:7b"


async def test_no_user_settings_uses_non_ollama_env_default_for_all_tiers(engine, monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
    request = LLMRequest(prompt="hi", task_type="general")

    model, _ = await engine._select_model(request)

    assert model == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"


# --- canonical resolver contract (resolve_model exists and is delegated to) -


def test_canonical_resolver_exists():
    from llm.tier_resolver import resolve_model  # noqa: F401


class _Org:
    def __init__(self, preferred_model, trial_mode=False, allowed_providers=None):
        self.preferred_model = preferred_model
        self.trial_mode = trial_mode
        self.allowed_providers = allowed_providers or []

    def has_own_key(self, provider=None):
        return False


async def test_org_preferred_model_used_when_no_user_prefs(engine, monkeypatch):
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=[]))
    request = LLMRequest(prompt="hi", org_settings=_Org("gpt-4o"))

    model, _ = await engine._select_model(request)

    assert model == "gpt-4o"
    assert request.resolution_source == "org_preferred"


async def test_trial_org_falls_to_system_default(engine, monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit")
    monkeypatch.setattr(engine, "_get_ollama_models", AsyncMock(return_value=[]))
    request = LLMRequest(prompt="hi", org_settings=_Org("gpt-4o", trial_mode=True))

    model, _ = await engine._select_model(request)

    assert model == "vllm:Qwen/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit"
    assert request.resolution_source == "system_default"
