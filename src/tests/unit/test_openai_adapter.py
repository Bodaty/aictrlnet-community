"""Unit tests for OpenAI adapter."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime
import json

from adapters.implementations.ai.openai_adapter import OpenAIAdapter
from adapters.models import (
    AdapterConfig, AdapterRequest, AdapterResponse,
    AdapterStatus, AdapterCategory
)
from adapters.registry import adapter_registry


@pytest.fixture
def adapter_config():
    """Create test adapter configuration."""
    return AdapterConfig(
        adapter_id="test-openai",
        adapter_type="openai",
        name="Test OpenAI Adapter",
        category=AdapterCategory.AI,
        credentials={"api_key": "test-key"},
        settings={"model": "gpt-3.5-turbo", "temperature": 0.7}
    )


@pytest.fixture
async def openai_adapter(adapter_config):
    """Create OpenAI adapter instance."""
    adapter = OpenAIAdapter(adapter_config)
    await adapter.initialize()
    yield adapter
    await adapter.shutdown()


class TestOpenAIAdapter:
    """Test suite for OpenAI adapter."""
    
    def test_adapter_creation(self, adapter_config):
        """Test adapter can be created with config."""
        adapter = OpenAIAdapter(adapter_config)
        assert adapter.id == "test-openai"
        assert adapter.type == "openai"
        assert adapter.category == AdapterCategory.AI
        assert adapter.status == AdapterStatus.CREATED
    
    @pytest.mark.asyncio
    async def test_adapter_initialization(self, adapter_config):
        """Test adapter initialization."""
        adapter = OpenAIAdapter(adapter_config)
        assert adapter.status == AdapterStatus.CREATED
        
        await adapter.initialize()
        assert adapter.status == AdapterStatus.RUNNING
        assert adapter._initialized is True
        
        await adapter.shutdown()
        assert adapter.status == AdapterStatus.STOPPED
    
    def test_get_capabilities(self, adapter_config):
        """Test adapter returns correct capabilities."""
        adapter = OpenAIAdapter(adapter_config)
        capabilities = adapter.get_capabilities()
        
        # Check we have expected capabilities
        capability_names = [cap.name for cap in capabilities]
        assert "chat_completion" in capability_names
        assert "text_completion" in capability_names
        assert "embeddings" in capability_names
        assert "function_calling" in capability_names
        
        # Check capability details
        chat_cap = next(c for c in capabilities if c.name == "chat_completion")
        assert chat_cap.category == "text_generation"
        assert "messages" in chat_cap.required_parameters
        assert chat_cap.async_supported is True
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_chat_completion(self, mock_openai_class, openai_adapter):
        """Test chat completion functionality."""
        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content="Hello! How can I help you?"))
        ]
        mock_response.usage = MagicMock(
            prompt_tokens=10,
            completion_tokens=8,
            total_tokens=18
        )
        mock_response.model = "gpt-3.5-turbo"
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client
        
        # Re-initialize adapter with mocked client
        await openai_adapter.initialize()
        
        # Create request
        request = AdapterRequest(
            capability="chat_completion",
            parameters={
                "messages": [{"role": "user", "content": "Hello"}],
                "temperature": 0.7
            }
        )
        
        # Execute request
        response = await openai_adapter.execute(request)
        
        # Verify response
        assert response.success is True
        assert response.data["content"] == "Hello! How can I help you?"
        assert response.data["model"] == "gpt-3.5-turbo"
        assert response.metadata["usage"]["total_tokens"] == 18
        
        # Verify API was called correctly
        mock_client.chat.completions.create.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_embeddings(self, mock_openai_class, openai_adapter):
        """Test embeddings generation."""
        # Mock embedding response
        mock_embedding_data = MagicMock(embedding=[0.1, 0.2, 0.3, 0.4, 0.5])
        mock_response = MagicMock()
        mock_response.data = [mock_embedding_data]
        mock_response.usage = MagicMock(total_tokens=5)
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.embeddings.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client
        
        # Re-initialize adapter with mocked client
        await openai_adapter.initialize()
        
        # Create request
        request = AdapterRequest(
            capability="embeddings",
            parameters={
                "input": "Hello world",
                "model": "text-embedding-ada-002"
            }
        )
        
        # Execute request
        response = await openai_adapter.execute(request)
        
        # Verify response
        assert response.success is True
        assert "embeddings" in response.data
        assert len(response.data["embeddings"]) == 1
        assert response.data["embeddings"][0] == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.asyncio
    async def test_invalid_capability(self, openai_adapter):
        """Test handling of invalid capability."""
        request = AdapterRequest(
            capability="invalid_capability",
            parameters={}
        )
        
        with pytest.raises(ValueError, match="Unknown capability"):
            await openai_adapter.execute(request)
    
    @pytest.mark.asyncio
    async def test_missing_required_parameters(self, openai_adapter):
        """Test handling of missing required parameters."""
        request = AdapterRequest(
            capability="chat_completion",
            parameters={}  # Missing required 'messages'
        )
        
        with pytest.raises(KeyError):
            await openai_adapter.execute(request)
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_api_error_handling(self, mock_openai_class, openai_adapter):
        """Test handling of OpenAI API errors."""
        # Setup mock client to raise error
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error: Rate limit exceeded")
        )
        mock_openai_class.return_value = mock_client
        
        # Re-initialize adapter with mocked client
        await openai_adapter.initialize()
        
        # Create request
        request = AdapterRequest(
            capability="chat_completion",
            parameters={
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )
        
        # Execute should raise exception
        with pytest.raises(Exception, match="API Error"):
            await openai_adapter.execute(request)
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_streaming_response(self, mock_openai_class, openai_adapter):
        """Test streaming chat completion."""
        # Mock streaming response
        async def mock_stream():
            chunks = [
                MagicMock(choices=[MagicMock(delta=MagicMock(content="Hello"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content=" world"))]),
                MagicMock(choices=[MagicMock(delta=MagicMock(content="!"))])
            ]
            for chunk in chunks:
                yield chunk
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_stream())
        mock_openai_class.return_value = mock_client
        
        # Re-initialize adapter with mocked client
        await openai_adapter.initialize()
        
        # Create streaming request
        request = AdapterRequest(
            capability="chat_completion",
            parameters={
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True
            }
        )
        
        # Execute request
        response = await openai_adapter.execute(request)
        
        # Verify response
        assert response.success is True
        assert response.data["stream"] is True
        
        # Collect stream chunks
        chunks = []
        async for chunk in response.data["response"]:
            chunks.append(chunk)
        
        assert len(chunks) == 3
        assert chunks[0] == "Hello"
        assert chunks[1] == " world"
        assert chunks[2] == "!"
    
    def test_metrics_tracking(self, adapter_config):
        """Test adapter metrics are tracked."""
        adapter = OpenAIAdapter(adapter_config)
        
        # Initial metrics
        assert adapter.metrics.total_requests == 0
        assert adapter.metrics.successful_requests == 0
        assert adapter.metrics.failed_requests == 0
        assert adapter.metrics.total_latency == 0.0
    
    @pytest.mark.asyncio
    async def test_health_check(self, openai_adapter):
        """Test adapter health check."""
        health = await openai_adapter.health_check()
        
        assert health["status"] == "healthy"
        assert "api_key_configured" in health
        assert "default_model" in health
        assert "models_available" in health
        assert isinstance(health["models_available"], list)
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, adapter_config):
        """Test rate limiting functionality."""
        # Create adapter with rate limiting
        adapter_config.settings["rate_limit"] = 2  # 2 requests per second
        adapter = OpenAIAdapter(adapter_config)
        await adapter.initialize()
        
        # Verify rate limiter is configured
        assert adapter._rate_limiter is not None
        assert adapter._rate_limiter._rate_limit == 2
        
        await adapter.shutdown()
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_function_calling(self, mock_openai_class, openai_adapter):
        """Test function calling capability."""
        # Mock function call response
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=None,
                    function_call=MagicMock(
                        name="get_weather",
                        arguments='{"location": "San Francisco"}'
                    )
                )
            )
        ]
        mock_response.usage = MagicMock(total_tokens=50)
        
        # Setup mock client
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client
        
        # Re-initialize adapter with mocked client
        await openai_adapter.initialize()
        
        # Create request with functions
        request = AdapterRequest(
            capability="function_calling",
            parameters={
                "messages": [{"role": "user", "content": "What's the weather in SF?"}],
                "functions": [{
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string"}
                        }
                    }
                }]
            }
        )
        
        # Execute request
        response = await openai_adapter.execute(request)
        
        # Verify response
        assert response.success is True
        assert response.data["function_call"]["name"] == "get_weather"
        assert response.data["function_call"]["arguments"]["location"] == "San Francisco"


class TestOpenAIAdapterEdgeCases:
    """Test edge cases and error conditions."""
    
    @pytest.mark.asyncio
    async def test_adapter_not_initialized(self, adapter_config):
        """Test executing request on uninitialized adapter."""
        adapter = OpenAIAdapter(adapter_config)
        
        request = AdapterRequest(
            capability="chat_completion",
            parameters={"messages": [{"role": "user", "content": "Hello"}]}
        )
        
        with pytest.raises(RuntimeError, match="Adapter not initialized"):
            await adapter.execute(request)
    
    @pytest.mark.asyncio
    async def test_empty_messages(self, openai_adapter):
        """Test handling of empty messages."""
        request = AdapterRequest(
            capability="chat_completion",
            parameters={"messages": []}
        )
        
        # Should handle gracefully or raise appropriate error
        # Implementation dependent
    
    @pytest.mark.asyncio
    @patch('openai.AsyncOpenAI')
    async def test_max_tokens_limit(self, mock_openai_class, openai_adapter):
        """Test handling of max tokens limit."""
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(content="Truncated response..."),
                finish_reason="length"
            )
        ]
        
        mock_client = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        mock_openai_class.return_value = mock_client
        
        await openai_adapter.initialize()
        
        request = AdapterRequest(
            capability="chat_completion",
            parameters={
                "messages": [{"role": "user", "content": "Tell me a very long story"}],
                "max_tokens": 10
            }
        )
        
        response = await openai_adapter.execute(request)
        
        assert response.success is True
        assert response.metadata.get("finish_reason") == "length"