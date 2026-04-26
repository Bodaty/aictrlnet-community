"""Unit tests for the vLLM adapter."""

import json
import pytest
import httpx

from adapters.implementations.ai.vllm_adapter import VLLMAdapter, _strip_vllm_prefix
from adapters.models import (
    AdapterConfig, AdapterRequest, AdapterCategory,
)
from adapters.tool_calling import ToolCallingRequest


def _config(**kwargs):
    """Build an AdapterConfig with sensible defaults for unit tests."""
    defaults = dict(
        name="vllm-test",
        version="1.0.0",
        category=AdapterCategory.AI,
        custom_config={"discovery_only": True},  # avoid network calls in unit tests
    )
    defaults.update(kwargs)
    return AdapterConfig(**defaults)


class TestPrefixStripping:
    def test_strip_with_prefix(self):
        assert _strip_vllm_prefix("vllm:meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"

    def test_strip_without_prefix(self):
        assert _strip_vllm_prefix("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"

    def test_strip_none(self):
        assert _strip_vllm_prefix(None) is None

    def test_strip_empty(self):
        assert _strip_vllm_prefix("") == ""


class TestBaseURLNormalization:
    def test_localhost_translated_when_not_discovery(self):
        adapter = VLLMAdapter(_config(custom_config={}, base_url="http://localhost:8000"))
        assert adapter.base_url == "http://host.docker.internal:8000/v1"

    def test_localhost_not_translated_in_discovery_mode(self):
        adapter = VLLMAdapter(_config(base_url="http://localhost:8000"))
        assert adapter.base_url == "http://localhost:8000/v1"

    def test_v1_suffix_not_doubled(self):
        adapter = VLLMAdapter(_config(base_url="http://example.com:8000/v1"))
        assert adapter.base_url == "http://example.com:8000/v1"

    def test_trailing_slash_stripped(self):
        adapter = VLLMAdapter(_config(base_url="http://example.com:8000/"))
        assert adapter.base_url == "http://example.com:8000/v1"

    def test_default_when_no_base_url(self):
        adapter = VLLMAdapter(_config())
        # discovery_only by default in helper, so localhost not translated
        assert adapter.base_url == "http://localhost:8000/v1"

    def test_127_not_translated(self):
        # We deliberately match Ollama's behavior: only literal "localhost" is rewritten.
        adapter = VLLMAdapter(_config(custom_config={}, base_url="http://127.0.0.1:8000"))
        assert adapter.base_url == "http://127.0.0.1:8000/v1"


class TestAuthOptional:
    def test_no_api_key_does_not_raise(self):
        # Unlike OpenAI, vLLM commonly runs without auth.
        VLLMAdapter(_config())  # must not raise

    def test_api_key_attached_to_headers(self):
        adapter = VLLMAdapter(_config(api_key="sk-test"))
        headers = adapter._build_headers()
        assert headers.get("Authorization") == "Bearer sk-test"

    def test_no_api_key_omits_auth_header(self):
        adapter = VLLMAdapter(_config())
        headers = adapter._build_headers()
        assert "Authorization" not in headers


class TestCapabilities:
    def test_capability_names(self):
        adapter = VLLMAdapter(_config())
        names = {c.name for c in adapter.get_capabilities()}
        assert names == {"chat_completion", "list_models"}


class TestDiscoveryOnly:
    @pytest.mark.asyncio
    async def test_initialize_skips_http_in_discovery_mode(self):
        adapter = VLLMAdapter(_config())  # discovery_only=True via helper default
        await adapter.initialize()
        assert adapter.client is None
        await adapter.shutdown()

    @pytest.mark.asyncio
    async def test_list_models_returns_empty_in_discovery_mode(self):
        adapter = VLLMAdapter(_config())
        await adapter.initialize()
        assert await adapter.list_models() == []
        await adapter.shutdown()


class _StubResponse:
    """Minimal stand-in for httpx.Response used in adapter tests."""

    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload
        self.text = payload if isinstance(payload, str) else json.dumps(payload)
        self.request = httpx.Request("POST", "http://test/")

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}", request=self.request, response=self
            )


class _StubClient:
    """Async stub for httpx.AsyncClient with a configurable post() handler."""

    def __init__(self, post_handler):
        self._post_handler = post_handler
        self.last_post = None
        self.closed = False

    async def post(self, url, json=None):
        self.last_post = {"url": url, "json": json}
        return self._post_handler(url, json)

    async def aclose(self):
        self.closed = True


class TestExecuteDispatch:
    @pytest.mark.asyncio
    async def test_unknown_capability_raises(self):
        adapter = VLLMAdapter(_config())
        request = AdapterRequest(capability="nope", parameters={})
        with pytest.raises(ValueError, match="Unknown capability"):
            await adapter.execute(request)

    @pytest.mark.asyncio
    async def test_chat_completion_strips_vllm_prefix(self):
        adapter = VLLMAdapter(_config())

        def handler(url, payload):
            assert payload["model"] == "meta-llama/Llama-3.1-8B-Instruct"
            return _StubResponse(200, {
                "id": "x", "model": payload["model"],
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 5},
            })

        adapter.client = _StubClient(handler)
        response = await adapter.execute(AdapterRequest(
            capability="chat_completion",
            parameters={
                "model": "vllm:meta-llama/Llama-3.1-8B-Instruct",
                "messages": [{"role": "user", "content": "hi"}],
            },
        ))
        assert response.status == "success"
        assert response.tokens_used == 5
        assert response.cost == 0.0

    @pytest.mark.asyncio
    async def test_chat_capability_alias_for_workflow_path(self):
        # generation.py's _generate_workflow_with_any_provider sends
        # capability="chat" — the adapter must accept that.
        adapter = VLLMAdapter(_config())
        adapter.client = _StubClient(lambda url, payload: _StubResponse(200, {
            "id": "x", "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 1},
        }))
        response = await adapter.execute(AdapterRequest(
            capability="chat",
            parameters={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
        ))
        assert response.status == "success"


class TestChatWithTools:
    @pytest.mark.asyncio
    async def test_400_raises_clear_remediation(self):
        adapter = VLLMAdapter(_config())
        adapter.client = _StubClient(lambda url, payload: _StubResponse(
            400, "tool choice is not supported"
        ))

        request = ToolCallingRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d", "parameters": {}}],
            model="vllm:m",
        )
        with pytest.raises(Exception, match="--enable-auto-tool-choice"):
            await adapter.chat_with_tools(request)

    @pytest.mark.asyncio
    async def test_empty_arguments_string_does_not_crash(self):
        adapter = VLLMAdapter(_config())
        adapter.client = _StubClient(lambda url, payload: _StubResponse(200, {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},  # the quirk
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
        }))

        request = ToolCallingRequest(
            messages=[{"role": "user", "content": "weather"}],
            tools=[{"name": "get_weather", "description": "x", "parameters": {}}],
            model="vllm:m",
        )
        response = await adapter.chat_with_tools(request)
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0]["arguments"] == {}
        assert response.tool_calls[0]["name"] == "get_weather"
        assert response.tokens_used == 12

    @pytest.mark.asyncio
    async def test_strips_vllm_prefix_from_request_model(self):
        adapter = VLLMAdapter(_config())

        captured = {}

        def handler(url, payload):
            captured["model"] = payload["model"]
            return _StubResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        adapter.client = _StubClient(handler)
        request = ToolCallingRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d", "parameters": {}}],
            model="vllm:meta-llama/Llama-3.1-8B-Instruct",
        )
        await adapter.chat_with_tools(request)
        assert captured["model"] == "meta-llama/Llama-3.1-8B-Instruct"

    @pytest.mark.asyncio
    async def test_specific_tool_choice_passed_through_as_function_object(self):
        adapter = VLLMAdapter(_config())
        captured = {}

        def handler(url, payload):
            captured["tool_choice"] = payload.get("tool_choice")
            return _StubResponse(200, {
                "choices": [{"message": {"role": "assistant", "content": ""}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            })

        adapter.client = _StubClient(handler)
        request = ToolCallingRequest(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "get_weather", "description": "d", "parameters": {}}],
            model="m",
            tool_choice="get_weather",
        )
        await adapter.chat_with_tools(request)
        assert captured["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}
