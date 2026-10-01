"""A model that returns no workflow steps is an upstream failure, not a success.

Seen in the 30 Sep pre-push gate: a slow local model timed out inside the
Ollama workflow adapter, which logged an ERROR and returned None; the engine
answered with the text "Failed to generate workflow steps" as if the model had
said it, and POST /llm/workflow/generate returned 200. It now raises
UpstreamResponseError, which the endpoint maps to 502.
"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.exceptions import UpstreamResponseError
from llm.generation import LLMGenerationEngine
from llm.models import LLMRequest, ModelProvider


@pytest.mark.parametrize("path", ["_generate_with_ollama", "_generate_workflow_with_any_provider"])
async def test_no_steps_raises_upstream_error(path, monkeypatch):
    engine = LLMGenerationEngine()
    adapter = MagicMock()
    adapter.generate_workflow_steps = AsyncMock(return_value=None)
    monkeypatch.setattr("services.model_adapters.get_model_adapter", lambda *a, **k: adapter)
    monkeypatch.setattr("llm.generation.assert_phi_provider_allowed", lambda *a, **k: None, raising=False)
    request = LLMRequest(prompt="invoice approvals", task_type="workflow_generation")

    with pytest.raises(UpstreamResponseError):
        if path == "_generate_with_ollama":
            await engine._generate_with_ollama(request, "llama3.1:8b")
        else:
            await engine._generate_workflow_with_any_provider(request, "llama3.1:8b", ModelProvider.OLLAMA)


async def test_the_endpoint_answers_502():
    from fastapi import HTTPException
    import llm.api.endpoints as endpoints

    original = endpoints.llm_service.generate_workflow_steps
    endpoints.llm_service.generate_workflow_steps = AsyncMock(
        side_effect=UpstreamResponseError("llama3.1:8b returned no usable workflow steps")
    )
    original_ctx = endpoints._load_llm_context
    endpoints._load_llm_context = AsyncMock(return_value=(None, None))
    try:
        with pytest.raises(HTTPException) as caught:
            await endpoints.generate_workflow_steps(
                request=endpoints.WorkflowGenerationRequest(description="invoice approvals"),
                current_user=MagicMock(id="u1"), db=MagicMock(),
            )
        assert caught.value.status_code == 502
    finally:
        endpoints.llm_service.generate_workflow_steps = original
        endpoints._load_llm_context = original_ctx
