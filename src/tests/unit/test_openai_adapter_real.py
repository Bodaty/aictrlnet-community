"""Unit tests for the actual OpenAI adapter implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import httpx

from adapters.implementations.ai.openai_adapter import OpenAIAdapter
from adapters.models import (
    AdapterConfig, AdapterRequest, AdapterResponse,
    AdapterStatus, AdapterCategory
)


@pytest.fixture
def adapter_config():
    """Create test adapter configuration."""
    return AdapterConfig(
        adapter_id="test-openai",
        adapter_type="openai",
        name="Test OpenAI Adapter",
        category=AdapterCategory.AI,
        credentials={"api_key": "test-key"},
        settings={"model": "gpt-3.5-turbo", "temperature": 0.7},
        timeout_seconds=30
    )


class TestOpenAIAdapter:
    """Test the actual OpenAI adapter implementation."""
    
    def test_adapter_creation(self, adapter_config):
        """Test creating an OpenAI adapter instance."""
        adapter = OpenAIAdapter(adapter_config)
        
        assert adapter.config.name == "Test OpenAI Adapter"
        assert adapter.config.category == AdapterCategory.AI
        assert adapter.api_key == "test-key"
        assert adapter.base_url == "https://api.openai.com/v1"
    
    def test_adapter_requires_api_key(self):
        """Test that adapter raises error without API key."""
        config = AdapterConfig(
            adapter_id="test-no-key",
            adapter_type="openai",
            name="No Key Adapter",
            category=AdapterCategory.AI,
            credentials={},  # No API key
            settings={}
        )
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIAdapter(config)
    
    @pytest.mark.asyncio
    async def test_adapter_initialize(self, adapter_config):
        """Test adapter initialization."""
        adapter = OpenAIAdapter(adapter_config)
        
        # Mock the HTTP client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock the test connection
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "gpt-3.5-turbo"}]}
            mock_client.get.return_value = mock_response
            
            await adapter.initialize()
            
            # Verify client was created with correct parameters
            mock_client_class.assert_called_once_with(
                base_url="https://api.openai.com/v1",
                headers={
                    "Authorization": "Bearer test-key",
                    "Content-Type": "application/json"
                },
                timeout=30
            )
            
            # Verify test connection was made
            mock_client.get.assert_called_once_with("/models")
    
    @pytest.mark.asyncio
    async def test_adapter_shutdown(self, adapter_config):
        """Test adapter shutdown."""
        adapter = OpenAIAdapter(adapter_config)
        
        # Initialize with mocked client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            
            await adapter.initialize()
            await adapter.shutdown()
            
            # Verify client was closed
            mock_client.aclose.assert_called_once()
    
    def test_get_capabilities(self, adapter_config):
        """Test getting adapter capabilities."""
        adapter = OpenAIAdapter(adapter_config)
        capabilities = adapter.get_capabilities()
        
        # Check that all expected capabilities are present
        capability_names = [cap.name for cap in capabilities]
        assert "chat_completion" in capability_names
        assert "embeddings" in capability_names
        assert "function_calling" in capability_names
        
        # Check capability details
        chat_cap = next(cap for cap in capabilities if cap.name == "chat_completion")
        assert "model" in chat_cap.parameters
        assert "messages" in chat_cap.parameters
        assert "messages" in chat_cap.required_parameters
    
    @pytest.mark.asyncio
    async def test_execute_chat_completion(self, adapter_config):
        """Test executing a chat completion request."""
        adapter = OpenAIAdapter(adapter_config)
        
        # Initialize with mocked client
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock initialization
            mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            await adapter.initialize()
            
            # Mock chat completion response
            mock_chat_response = Mock()
            mock_chat_response.status_code = 200
            mock_chat_response.json.return_value = {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Hello! How can I help you?"
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21
                }
            }
            mock_client.post.return_value = mock_chat_response
            
            # Create request
            request = AdapterRequest(
                capability="chat_completion",
                parameters={
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "model": "gpt-3.5-turbo"
                }
            )
            
            # Execute
            response = await adapter.execute(request)
            
            # Verify response
            assert response.success is True
            assert response.data["content"] == "Hello! How can I help you?"
            assert response.data["model"] == "gpt-3.5-turbo"
            assert response.data["usage"]["total_tokens"] == 21
            
            # Verify API was called correctly
            mock_client.post.assert_called_once_with(
                "/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello!"}],
                    "model": "gpt-3.5-turbo"
                }
            )
    
    @pytest.mark.asyncio
    async def test_execute_embeddings(self, adapter_config):
        """Test executing an embeddings request."""
        adapter = OpenAIAdapter(adapter_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock initialization
            mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            await adapter.initialize()
            
            # Mock embeddings response
            mock_embeddings_response = Mock()
            mock_embeddings_response.status_code = 200
            mock_embeddings_response.json.return_value = {
                "object": "list",
                "data": [{
                    "object": "embedding",
                    "index": 0,
                    "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
                }],
                "model": "text-embedding-ada-002",
                "usage": {
                    "prompt_tokens": 8,
                    "total_tokens": 8
                }
            }
            mock_client.post.return_value = mock_embeddings_response
            
            # Create request
            request = AdapterRequest(
                capability="embeddings",
                parameters={
                    "input": "Hello world",
                    "model": "text-embedding-ada-002"
                }
            )
            
            # Execute
            response = await adapter.execute(request)
            
            # Verify response
            assert response.success is True
            assert response.data["embeddings"][0] == [0.1, 0.2, 0.3, 0.4, 0.5]
            assert response.data["model"] == "text-embedding-ada-002"
            assert response.data["usage"]["total_tokens"] == 8
    
    @pytest.mark.asyncio
    async def test_execute_with_stream(self, adapter_config):
        """Test executing a streaming chat completion."""
        adapter = OpenAIAdapter(adapter_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock initialization
            mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            await adapter.initialize()
            
            # Mock streaming response
            async def mock_stream():
                chunks = [
                    b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n',
                    b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n',
                    b'data: [DONE]\n\n'
                ]
                for chunk in chunks:
                    yield chunk
            
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.aiter_lines = mock_stream
            
            mock_client.post.return_value = mock_response
            
            # Create streaming request
            request = AdapterRequest(
                capability="chat_completion",
                parameters={
                    "messages": [{"role": "user", "content": "Say hello"}],
                    "model": "gpt-3.5-turbo",
                    "stream": True
                }
            )
            
            # Execute and collect stream
            response = await adapter.execute(request)
            
            # For streaming, the adapter returns a generator
            assert response.success is True
            # In the real adapter, response.data would contain a stream handler
    
    @pytest.mark.asyncio
    async def test_execute_with_error(self, adapter_config):
        """Test handling of API errors."""
        adapter = OpenAIAdapter(adapter_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock initialization
            mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            await adapter.initialize()
            
            # Mock error response
            mock_client.post.side_effect = httpx.HTTPError("API Error")
            
            request = AdapterRequest(
                capability="chat_completion",
                parameters={
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )
            
            response = await adapter.execute(request)
            
            # Should return error response
            assert response.success is False
            assert "error" in response.data
            assert "API Error" in str(response.data["error"])
    
    @pytest.mark.asyncio
    async def test_execute_with_rate_limit(self, adapter_config):
        """Test handling of rate limit errors."""
        adapter = OpenAIAdapter(adapter_config)
        
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock initialization
            mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
            await adapter.initialize()
            
            # Mock rate limit response
            mock_response = Mock()
            mock_response.status_code = 429
            mock_response.json.return_value = {
                "error": {
                    "message": "Rate limit exceeded",
                    "type": "rate_limit_error"
                }
            }
            mock_client.post.return_value = mock_response
            
            request = AdapterRequest(
                capability="chat_completion",
                parameters={"messages": [{"role": "user", "content": "Test"}]}
            )
            
            response = await adapter.execute(request)
            
            assert response.success is False
            assert "rate limit" in response.data["error"].lower()