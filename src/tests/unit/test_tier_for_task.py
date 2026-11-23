import logging

import pytest

from llm.generation import LLMGenerationEngine
from llm.models import LLMRequest, ModelTier


@pytest.fixture
def engine():
    return LLMGenerationEngine()


@pytest.mark.parametrize("task_type,expected", [
    ("structured_generation", ModelTier.BALANCED),
    ("planning", ModelTier.BALANCED),
    ("basic_agent", ModelTier.BALANCED),
    ("general", ModelTier.BALANCED),
    ("conversation", ModelTier.BALANCED),
    ("extraction", ModelTier.BALANCED),
    ("data_extraction", ModelTier.BALANCED),
    ("ai_task", ModelTier.BALANCED),
])
def test_explicit_mappings(engine, task_type, expected, caplog):
    with caplog.at_level(logging.WARNING, logger="llm.generation"):
        tier = engine._determine_tier_for_task(LLMRequest(prompt="x", task_type=task_type))
    assert tier == expected
    assert not caplog.records  # mapped tasks must not warn


def test_unknown_task_type_warns(engine, caplog):
    with caplog.at_level(logging.WARNING, logger="llm.generation"):
        tier = engine._determine_tier_for_task(LLMRequest(prompt="x", task_type="totally_new_thing"))
    assert tier == ModelTier.BALANCED
    assert any("totally_new_thing" in r.message for r in caplog.records)


def test_an_ai_calls_outcome_is_read_by_the_quality_tier(engine, caplog):
    """Reading a call into its outcome and the patient's own words (Business
    call_outcome_reader, task type "voice_outcome_read"): the 3B BALANCED model dropped
    preferred times the caller gave, the 8B QUALITY model reads them (23 Nov 2026)."""
    with caplog.at_level(logging.WARNING, logger="llm.generation"):
        tier = engine._determine_tier_for_task(LLMRequest(prompt="x", task_type="voice_outcome_read"))
    assert tier == ModelTier.QUALITY
    assert not caplog.records


def test_the_quality_tier_is_a_local_model_phi_mode_allows(monkeypatch):
    """On a local Ollama deployment QUALITY resolves to the 8B, an Ollama model, which
    PHI mode allows when Ollama is allowlisted and reached on a local endpoint."""
    from types import SimpleNamespace

    from core.phi_egress import LOCAL_PROVIDERS, assert_phi_provider_allowed
    from llm import tier_resolver

    monkeypatch.setattr(tier_resolver, "get_environment_default_model", lambda: "llama3.1:8b-instruct-q4_K_M")
    pulled = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b-instruct-q4_K_M"]
    quality = tier_resolver.effective_tier_map(pulled)["tiers"][ModelTier.QUALITY.value]["model"]
    assert quality == "llama3.1:8b-instruct-q4_K_M"
    provider = tier_resolver.provider_for_model_name(quality)
    assert provider == "ollama" and provider in LOCAL_PROVIDERS
    phi = SimpleNamespace(AICTRLNET_PHI_MODE=True, AICTRLNET_PHI_LLM_PROVIDERS="ollama")
    assert_phi_provider_allowed(provider, "http://localhost:11434", settings=phi)  # no refusal


# A deployment whose DEFAULT_LLM_MODEL is an Ollama tag and whose Ollama has none of the
# SYSTEM_TIER_DEFAULTS models: the practice's Studio on 23 Nov 2026.
STUDIO_DEFAULT = "qwen3:14b"
STUDIO_PULLED = ["qwen3:14b", "qwen3.5:35b-a3b-q4_K_M", "qwen3.5:35b-a3b-nvfp4"]


@pytest.fixture
def studio(engine, monkeypatch):
    from llm import tier_resolver

    monkeypatch.delenv("LLM_SCRIPTED_MODE", raising=False)
    monkeypatch.setattr(tier_resolver, "get_environment_default_model", lambda: STUDIO_DEFAULT)

    async def pulled():
        return list(STUDIO_PULLED)

    monkeypatch.setattr(engine, "_get_ollama_models", pulled)
    return engine


@pytest.mark.parametrize("task_type", ["voice_outcome_read", "intent_classification", "conversation", "totally_new_thing"])
async def test_without_the_tier_table_models_every_task_gets_a_pulled_model(studio, task_type):
    """No tier resolves to a model the deployment does not have (llama3.1:8b, llama3.2:3b):
    with the table's model absent, the largest pulled local model of the tier answers, then
    the largest of any tier, QUALITY first."""
    model, _ = await studio._select_model(LLMRequest(prompt="x", task_type=task_type))
    assert model in STUDIO_PULLED


async def test_the_call_outcome_read_gets_the_largest_pulled_quality_model(studio):
    model, tier = await studio._select_model(LLMRequest(prompt="x", task_type="voice_outcome_read"))
    assert tier == ModelTier.QUALITY and model.startswith("qwen3.5:35b-a3b")


def test_the_tier_map_names_only_pulled_models(monkeypatch):
    from llm import tier_resolver

    monkeypatch.setattr(tier_resolver, "get_environment_default_model", lambda: STUDIO_DEFAULT)
    tiers = tier_resolver.effective_tier_map(STUDIO_PULLED)["tiers"]
    assert all(entry["model"] in STUDIO_PULLED for entry in tiers.values())
