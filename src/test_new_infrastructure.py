"""Test script for the new FastAPI infrastructure."""

import asyncio
import logging
from datetime import datetime

from control_plane.models import ComponentRegistrationRequest, ComponentType
from control_plane.services import control_plane_service
from events.event_bus import event_bus
from adapters.factory import AdapterFactory
from adapters.models import AdapterRequest, AdapterCategory


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_control_plane():
    """Test control plane functionality."""
    logger.info("Testing Control Plane...")
    
    # Register a test component
    registration_request = ComponentRegistrationRequest(
        name="test-service",
        type=ComponentType.SERVICE,
        version="1.0.0",
        description="Test service for demonstration",
        capabilities=[
            {
                "name": "process_data",
                "description": "Process incoming data",
                "parameters": {"data": "object"}
            }
        ],
        edition="community"
    )
    
    response = await control_plane_service.register_component(
        registration_request,
        user_id="test-user"
    )
    
    logger.info(f"Component registered: {response.component.id}")
    logger.info(f"JWT Token: {response.token[:20]}...")
    
    # Get component
    component = await control_plane_service.get_component(response.component.id)
    logger.info(f"Retrieved component: {component.name} - Status: {component.status}")
    
    return response.component.id


async def test_event_bus():
    """Test event bus functionality."""
    logger.info("\nTesting Event Bus...")
    
    # Start event bus
    await event_bus.start()
    
    # Subscribe to events
    received_events = []
    
    async def event_handler(event):
        logger.info(f"Received event: {event.type} - {event.data}")
        received_events.append(event)
    
    # Register handler
    event_bus.register_handler(
        ["test.*", "component.*"],
        event_handler,
        name="test_handler"
    )
    
    # Publish test events
    await event_bus.publish(
        "test.hello",
        {"message": "Hello from event bus!"},
        source_id="test"
    )
    
    await event_bus.publish(
        "component.started",
        {"component": "test-component"},
        source_id="test"
    )
    
    # Wait for events to be processed
    await asyncio.sleep(1)
    
    logger.info(f"Received {len(received_events)} events")
    
    # Get stats
    stats = await event_bus.get_stats()
    logger.info(f"Event bus stats: {stats}")


async def test_openai_adapter():
    """Test OpenAI adapter (requires API key)."""
    logger.info("\nTesting OpenAI Adapter...")
    
    # Check if API key is available
    import os
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set, skipping OpenAI adapter test")
        return
    
    # Create adapter through factory
    adapter = await AdapterFactory.create_and_register_adapter(
        "openai",
        {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "category": AdapterCategory.AI,
            "version": "1.0.0"
        }
    )
    
    logger.info(f"OpenAI adapter created: {adapter.id}")
    
    # Test chat completion
    request = AdapterRequest(
        capability="chat_completion",
        parameters={
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Say hello in 5 words or less"}
            ],
            "max_tokens": 20
        }
    )
    
    response = await adapter.handle_request(request)
    
    if response.status == "success":
        content = response.data["choices"][0]["message"]["content"]
        logger.info(f"OpenAI response: {content}")
        logger.info(f"Duration: {response.duration_ms}ms, Cost: ${response.cost}")
    else:
        logger.error(f"OpenAI error: {response.error}")
    
    # Check adapter metrics
    info = adapter.get_info()
    logger.info(f"Adapter metrics: {info.metrics.dict()}")


async def main():
    """Run all tests."""
    logger.info("Starting AICtrlNet Infrastructure Tests\n")
    
    try:
        # Test control plane
        component_id = await test_control_plane()
        
        # Test event bus
        await test_event_bus()
        
        # Test OpenAI adapter
        await test_openai_adapter()
        
        logger.info("\nAll tests completed!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise
    
    finally:
        # Cleanup
        await event_bus.stop()


if __name__ == "__main__":
    asyncio.run(main())