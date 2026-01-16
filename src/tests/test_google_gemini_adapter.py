"""Tests for Google Gemini adapter."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx
from datetime import datetime

from adapters.implementations.ai.google_gemini_adapter import GoogleGeminiAdapter
from adapters.models import (
    AdapterConfig, AdapterRequest, AdapterCategory
)


@pytest.fixture
def gemini_config():
    """Create a test Gemini adapter configuration."""
    return AdapterConfig(
        name="test-gemini",
        version="1.0.0",
        category=AdapterCategory.AI,
        credentials={"api_key": "test-api-key"},
        timeout_seconds=30.0
    )


@pytest.fixture
async def gemini_adapter(gemini_config):
    """Create a test Gemini adapter instance."""
    adapter = GoogleGeminiAdapter(gemini_config)
    # Mock the HTTP client
    adapter.client = AsyncMock(spec=httpx.AsyncClient)
    return adapter


class TestGoogleGeminiAdapter:
    """Test Google Gemini adapter functionality."""
    
    async def test_initialization(self, gemini_config):
        """Test adapter initialization."""
        with patch('httpx.AsyncClient') as mock_client_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            
            # Mock successful models endpoint
            mock_response = AsyncMock()
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            
            adapter = GoogleGeminiAdapter(gemini_config)
            await adapter.initialize()
            
            # Verify API key is set
            assert adapter.api_key == "test-api-key"
            
            # Verify client was created with correct params
            mock_client_class.assert_called_once()
            call_kwargs = mock_client_class.call_args[1]
            assert call_kwargs["params"] == {"key": "test-api-key"}
            assert call_kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta"
    
    async def test_get_capabilities(self, gemini_adapter):
        """Test getting adapter capabilities."""
        capabilities = gemini_adapter.get_capabilities()
        
        assert len(capabilities) == 4
        capability_names = [cap.name for cap in capabilities]
        assert "generate_content" in capability_names
        assert "chat" in capability_names
        assert "embed_content" in capability_names
        assert "count_tokens" in capability_names
        
        # Check generate_content capability details
        gen_cap = next(cap for cap in capabilities if cap.name == "generate_content")
        assert gen_cap.category == "text_generation"
        assert "model" in gen_cap.required_parameters
        assert "contents" in gen_cap.required_parameters
    
    async def test_generate_content(self, gemini_adapter):
        """Test content generation."""
        request = AdapterRequest(
            id="test-req-1",
            capability="generate_content",
            parameters={
                "model": "gemini-pro",
                "contents": [{
                    "parts": [{"text": "Hello, Gemini!"}]
                }],
                "temperature": 0.7
            }
        )
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "Hello! How can I help you today?"}],
                    "role": "model"
                },
                "finishReason": "STOP"
            }],
            "usageMetadata": {
                "promptTokenCount": 5,
                "candidatesTokenCount": 10,
                "totalTokenCount": 15
            }
        }
        gemini_adapter.client.post.return_value = mock_response
        
        response = await gemini_adapter.execute(request)
        
        assert response.status == "success"
        assert response.capability == "generate_content"
        assert len(response.data["candidates"]) == 1
        assert response.tokens_used == 15
        assert response.cost > 0
    
    async def test_chat_conversation(self, gemini_adapter):
        """Test chat capability."""
        request = AdapterRequest(
            id="test-req-2",
            capability="chat",
            parameters={
                "model": "gemini-pro",
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                    {"role": "user", "content": "How are you?"}
                ],
                "temperature": 0.8
            }
        )
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "candidates": [{
                "content": {
                    "parts": [{"text": "I'm doing well, thank you!"}],
                    "role": "model"
                }
            }],
            "usageMetadata": {"totalTokenCount": 20}
        }
        gemini_adapter.client.post.return_value = mock_response
        
        response = await gemini_adapter.execute(request)
        
        assert response.status == "success"
        assert response.capability == "chat"
        assert "message" in response.data
        assert response.data["message"]["role"] == "assistant"
        assert response.data["message"]["content"] == "I'm doing well, thank you!"
    
    async def test_embeddings(self, gemini_adapter):
        """Test embeddings generation."""
        request = AdapterRequest(
            id="test-req-3",
            capability="embed_content",
            parameters={
                "model": "embedding-001",
                "content": {
                    "parts": [{"text": "Generate embeddings for this text"}]
                },
                "task_type": "RETRIEVAL_QUERY"
            }
        )
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "embedding": {
                "values": [0.1, 0.2, 0.3, 0.4, 0.5]  # Simplified
            }
        }
        gemini_adapter.client.post.return_value = mock_response
        
        response = await gemini_adapter.execute(request)
        
        assert response.status == "success"
        assert response.capability == "embed_content"
        assert "embedding" in response.data
        assert "values" in response.data["embedding"]
    
    async def test_token_counting(self, gemini_adapter):
        """Test token counting."""
        request = AdapterRequest(
            id="test-req-4",
            capability="count_tokens",
            parameters={
                "model": "gemini-pro",
                "contents": [{
                    "parts": [{"text": "Count the tokens in this text"}]
                }]
            }
        )
        
        # Mock successful response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "totalTokens": 8
        }
        gemini_adapter.client.post.return_value = mock_response
        
        response = await gemini_adapter.execute(request)
        
        assert response.status == "success"
        assert response.capability == "count_tokens"
        assert response.data["totalTokens"] == 8
        assert response.cost == 0.0  # Token counting is free
    
    async def test_streaming_content(self, gemini_adapter):
        """Test streaming content generation."""
        request = AdapterRequest(
            id="test-req-5",
            capability="generate_content",
            parameters={
                "model": "gemini-pro",
                "contents": [{
                    "parts": [{"text": "Stream this response"}]
                }],
                "stream": True
            }
        )
        
        # Mock streaming response
        async def mock_aiter_bytes():
            chunks = [
                b'{"candidates":[{"content":{"parts":[{"text":"This "}]}}],"usageMetadata":{"totalTokenCount":5}}',
                b'{"candidates":[{"content":{"parts":[{"text":"is "}]}}],"usageMetadata":{"totalTokenCount":7}}',
                b'{"candidates":[{"content":{"parts":[{"text":"streaming!"}]}}],"usageMetadata":{"totalTokenCount":10}}'
            ]
            for chunk in chunks:
                yield chunk
        
        mock_stream = AsyncMock()
        mock_stream.raise_for_status = MagicMock()
        mock_stream.aiter_bytes = mock_aiter_bytes
        mock_stream.__aenter__.return_value = mock_stream
        mock_stream.__aexit__.return_value = None
        
        gemini_adapter.client.stream.return_value = mock_stream
        
        response = await gemini_adapter.execute(request)
        
        assert response.status == "success"
        assert response.capability == "generate_content"
        assert response.metadata.get("streaming") is True
        assert response.data["stream_chunks"] == 3
        
        # Check combined text
        candidate = response.data["candidates"][0]
        text = candidate["content"]["parts"][0]["text"]
        assert text == "This is streaming!"
    
    async def test_error_handling(self, gemini_adapter):
        """Test error handling."""
        request = AdapterRequest(
            id="test-req-6",
            capability="generate_content",
            parameters={
                "model": "gemini-pro",
                "contents": [{"parts": [{"text": "Test error"}]}]
            }
        )
        
        # Mock error response
        mock_response = AsyncMock()
        mock_response.status_code = 400
        mock_response.content = b'{"error": {"message": "Invalid API key"}}'
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        error = httpx.HTTPStatusError("Bad Request", request=None, response=mock_response)
        mock_response.raise_for_status.side_effect = error
        
        gemini_adapter.client.post.return_value = mock_response
        
        response = await gemini_adapter.execute(request)
        
        assert response.status == "error"
        assert response.error == "Invalid API key"
        assert response.error_code == "HTTP_400"
    
    async def test_health_check(self, gemini_adapter):
        """Test health check functionality."""
        # Mock successful models response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {
            "models": [
                {"name": "models/gemini-pro"},
                {"name": "models/gemini-pro-vision"},
                {"name": "models/gemini-1.5-pro"}
            ]
        }
        gemini_adapter.client.get.return_value = mock_response
        
        health = await gemini_adapter._perform_health_check()
        
        assert health["status"] == "healthy"
        assert health["available_models"] == 3
        assert "models/gemini-pro" in health["models"]