"""T7: the per-environment tier map (spec §7.1) and its startup log."""

import logging

import pytest

from llm.tier_resolver import effective_tier_map, log_tier_map

LOCAL_ALL = ["llama3.2:1b", "llama3.2:3b", "llama3.1:8b-instruct-q4_K_M"]


def test_local_uses_the_table_when_every_default_is_pulled(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    m = effective_tier_map(LOCAL_ALL)
    assert m["provider_class"] == "local"
    assert {t: m["tiers"][t]["model"] for t in ("fast", "balanced", "quality")} == {
        "fast": "llama3.2:1b",
        "balanced": "llama3.2:3b",
        "quality": "llama3.1:8b-instruct-q4_K_M",
    }
    assert all(m["tiers"][t]["pulled"] for t in m["tiers"])


def test_local_reports_tiers_whose_default_is_not_pulled(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    m = effective_tier_map(["llama3.1:8b-instruct-q4_K_M"])
    assert m["tiers"]["fast"]["pulled"] is False
    assert m["tiers"]["balanced"]["pulled"] is False
    assert m["tiers"]["quality"]["pulled"] is True
    # Nothing in the fast tier is pulled: the env default answers, as before.
    assert m["tiers"]["fast"]["model"] == "llama3.1:8b-instruct-q4_K_M"


@pytest.mark.parametrize("default, cls", [
    ("vllm:cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit", "self-hosted"),
    ("gemini-2.5-flash-vertex", "cloud"),
])
def test_self_hosted_and_cloud_use_the_environment_default_for_every_tier(monkeypatch, default, cls):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", default)
    m = effective_tier_map([])
    assert m["provider_class"] == cls
    for tier in m["tiers"].values():
        assert tier["model"] == default
        assert tier["source"] == "environment_default"
        assert tier["pulled"] is None


def test_log_warns_once_per_missing_local_tier(monkeypatch, caplog):
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    with caplog.at_level(logging.INFO, logger="llm.tier_resolver"):
        log_tier_map(effective_tier_map(["llama3.1:8b-instruct-q4_K_M"]))
    info = [r for r in caplog.records if r.levelno == logging.INFO and r.getMessage().startswith("[tier-map]")]
    warns = [r for r in caplog.records if r.levelno == logging.WARNING and "[tier-map]" in r.getMessage()]
    assert len(info) == 1 and "class=local" in info[0].getMessage()
    assert len(warns) == 2
    assert "make preflight-models" in warns[0].getMessage()


def test_discovery_never_picks_ocr_embedding_cloud_or_unsized_models(monkeypatch):
    # Seen 1 Oct: the balanced tier resolved to glm-ocr:latest (no size in the
    # name → default BALANCED → "largest" candidate). An Ollama `:cloud` model
    # would send the prompt off the machine.
    monkeypatch.setenv("DEFAULT_LLM_MODEL", "llama3.1:8b-instruct-q4_K_M")
    available = [
        "llama3.1:8b-instruct-q4_K_M", "glm-ocr:latest", "nemotron-3-super:cloud",
        "nomic-embed-text:latest",
    ]
    m = effective_tier_map(available)
    for tier in ("fast", "balanced"):
        assert m["tiers"][tier]["model"] == "llama3.1:8b-instruct-q4_K_M"
        assert m["tiers"][tier]["source"] == "environment_default"
    m = effective_tier_map(available + ["qwen3:4b"])
    assert m["tiers"]["balanced"]["model"] == "qwen3:4b"
