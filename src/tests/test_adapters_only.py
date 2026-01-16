"""Test only the adapter implementations without database dependencies."""

import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import adapters directly without going through the API
from adapters.implementations.ai.openai_adapter import OpenAIAdapter
from adapters.implementations.human.upwork_adapter import UpworkAdapter
from adapters.implementations.human.fiverr_adapter import FiverrAdapter
from adapters.models import AdapterConfig, AdapterRequest, AdapterCategory


def test_openai_adapter():
    """Test OpenAI adapter functionality."""
    print("\n=== Testing OpenAI Adapter ===")
    
    config = AdapterConfig(
        adapter_id="test-openai",
        adapter_type="openai",
        name="Test OpenAI",
        category=AdapterCategory.AI,
        credentials={"api_key": "test-key"}
    )
    
    adapter = OpenAIAdapter(config)
    
    # Test basic properties
    assert adapter.api_key == "test-key"
    assert adapter.base_url == "https://api.openai.com/v1"
    print("✓ Basic properties correct")
    
    # Test capabilities
    capabilities = adapter.get_capabilities()
    cap_names = [cap.name for cap in capabilities]
    assert "chat_completion" in cap_names
    assert "embeddings" in cap_names
    print(f"✓ Found {len(capabilities)} capabilities: {cap_names}")
    
    # Test missing API key
    try:
        bad_config = AdapterConfig(
            adapter_id="test-bad",
            adapter_type="openai",
            name="Bad OpenAI",
            category=AdapterCategory.AI,
            credentials={}
        )
        OpenAIAdapter(bad_config)
        assert False, "Should have raised error for missing API key"
    except ValueError as e:
        assert "API key is required" in str(e)
        print("✓ Correctly validates API key requirement")


def test_upwork_adapter():
    """Test Upwork adapter functionality."""
    print("\n=== Testing Upwork Adapter ===")
    
    config = AdapterConfig(
        adapter_id="test-upwork",
        adapter_type="upwork",
        name="Test Upwork",
        category=AdapterCategory.HUMAN,
        credentials={
            "client_id": "test-client",
            "client_secret": "test-secret",
            "access_token": "test-token",
            "access_secret": "test-secret-token"
        }
    )
    
    adapter = UpworkAdapter(config)
    
    # Test basic properties
    assert adapter.client_id == "test-client"
    assert adapter.base_url == "https://www.upwork.com/api"
    print("✓ Basic properties correct")
    
    # Test capabilities
    capabilities = adapter.get_capabilities()
    cap_names = [cap.name for cap in capabilities]
    expected_caps = ["search_freelancers", "post_job", "create_milestone", "send_offer"]
    for cap in expected_caps:
        assert cap in cap_names
    print(f"✓ Found {len(capabilities)} capabilities including: {expected_caps}")
    
    # Test OAuth validation
    try:
        bad_config = AdapterConfig(
            adapter_id="test-bad",
            adapter_type="upwork",
            name="Bad Upwork",
            category=AdapterCategory.HUMAN,
            credentials={"client_id": "only-one"}
        )
        UpworkAdapter(bad_config)
        assert False, "Should have raised error for missing OAuth credentials"
    except ValueError as e:
        assert "OAuth credentials required" in str(e)
        print("✓ Correctly validates OAuth credentials")


def test_fiverr_adapter():
    """Test Fiverr adapter functionality."""
    print("\n=== Testing Fiverr Adapter ===")
    
    config = AdapterConfig(
        adapter_id="test-fiverr",
        adapter_type="fiverr",
        name="Test Fiverr",
        category=AdapterCategory.HUMAN,
        credentials={"api_key": "test-api-key"}
    )
    
    adapter = FiverrAdapter(config)
    
    # Test basic properties
    assert adapter.api_key == "test-api-key"
    assert adapter.base_url == "https://api.fiverr.com/v1"
    print("✓ Basic properties correct")
    
    # Test capabilities
    capabilities = adapter.get_capabilities()
    cap_names = [cap.name for cap in capabilities]
    expected_caps = ["search_gigs", "create_order", "get_order_status"]
    for cap in expected_caps:
        assert cap in cap_names
    print(f"✓ Found {len(capabilities)} capabilities including: {expected_caps}")


async def test_adapter_execution():
    """Test adapter execution with mocking."""
    print("\n=== Testing Adapter Execution ===")
    
    config = AdapterConfig(
        adapter_id="test-exec",
        adapter_type="openai",
        name="Test Execution",
        category=AdapterCategory.AI,
        credentials={"api_key": "test-key"}
    )
    
    adapter = OpenAIAdapter(config)
    
    # Mock the HTTP client
    with patch('httpx.AsyncClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client
        
        # Mock successful init
        mock_client.get.return_value = Mock(status_code=200, json=Mock(return_value={"data": []}))
        
        await adapter.initialize()
        assert adapter._initialized is True
        print("✓ Adapter initialized successfully")
        
        # Mock chat completion
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"total_tokens": 10}
        }
        mock_client.post.return_value = mock_response
        
        request = AdapterRequest(
            capability="chat_completion",
            parameters={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
        
        response = await adapter.execute(request)
        
        assert response.success is True
        assert response.data["content"] == "Hello!"
        print("✓ Chat completion executed successfully")
        
        await adapter.shutdown()
        assert adapter.client is None
        print("✓ Adapter shutdown successfully")


def main():
    """Run all adapter tests."""
    print("=" * 50)
    print("Adapter-Only Test Suite")
    print("=" * 50)
    
    try:
        # Synchronous tests
        test_openai_adapter()
        test_upwork_adapter()
        test_fiverr_adapter()
        
        # Async tests
        asyncio.run(test_adapter_execution())
        
        print("\n" + "=" * 50)
        print("All adapter tests passed! ✓")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()