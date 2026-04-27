"""Unit tests for LLMGenerationEngine._is_api_model — prefix and substring matching."""
import pytest
from llm.generation import LLMGenerationEngine


@pytest.fixture
def engine():
    return LLMGenerationEngine()


@pytest.mark.parametrize("model", [
    "vllm:cpatonn/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit",
    "vllm:meta-llama/Llama-3.1-8B-Instruct",
    "VLLM:foo",  # case insensitive
])
def test_is_api_model_recognizes_vllm_prefix(engine, model):
    assert engine._is_api_model(model) is True


@pytest.mark.parametrize("model", [
    "claude-3-opus",
    "gpt-4",
    "gemini-2.5-flash",
    "deepseek-chat",
])
def test_is_api_model_recognizes_cloud_apis(engine, model):
    assert engine._is_api_model(model) is True


@pytest.mark.parametrize("model", [
    "llama3.2:3b",
    "llama3.1:8b-instruct-q4_K_M",
    "phi3:mini",
    "mistral:7b",
])
def test_is_api_model_rejects_ollama_names(engine, model):
    assert engine._is_api_model(model) is False
