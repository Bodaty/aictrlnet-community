"""An adapter that reports an error must not become an empty, successful answer.

Seen on Beast 1 Oct: vLLM answered 404 (model not served); the adapter returned
status="error" with no data, the engine returned text="" and the conversation
turn completed `done` with an empty message (spec §7.3 R9: never silent).
"""
from types import SimpleNamespace

import pytest

from llm.generation import LLMGenerationEngine
from llm.models import LLMRequest, ModelProvider


class _FailingAdapter:
    capabilities = [SimpleNamespace(name="chat_completion")]

    async def execute(self, request):
        from adapters.models import AdapterResponse
        return AdapterResponse(
            request_id="r1", capability="chat_completion", status="error",
            error="vLLM API error 404: The model `x` does not exist.", duration_ms=3,
        )


async def test_adapter_error_raises_instead_of_returning_empty_text():
    engine = LLMGenerationEngine()
    with pytest.raises(RuntimeError, match="404"):
        await engine._execute_adapter_chat(
            _FailingAdapter(), LLMRequest(prompt="hi"), "vllm:x", ModelProvider.VLLM,
        )
