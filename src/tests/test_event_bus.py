"""Tests for event bus functionality."""

import pytest
import asyncio
from datetime import datetime

from events.event_bus import event_bus, EventBus
from events.models import Event, EventSubscription, EventHandler, EventPriority


@pytest.fixture
async def clean_event_bus():
    """Create a clean event bus for testing."""
    test_bus = EventBus(enable_persistence=False)
    await test_bus.start()
    return test_bus


@pytest.mark.asyncio
async def test_event_publishing(clean_event_bus):
    """Test basic event publishing."""
    # Publish an event
    event = await clean_event_bus.publish(
        "test.event",
        {"message": "Hello, World!"},
        source_id="test",
        priority=EventPriority.NORMAL
    )
    
    # Verify event
    assert event.type == "test.event"
    assert event.data["message"] == "Hello, World!"
    assert event.source_id == "test"
    assert event.priority == EventPriority.NORMAL


@pytest.mark.asyncio
async def test_event_subscription(clean_event_bus):
    """Test event subscription and delivery."""
    received_events = []
    
    # Create handler
    async def test_handler(event: Event):
        received_events.append(event)
    
    # Register handler
    clean_event_bus.register_handler(
        ["test.*"],
        test_handler,
        name="test_handler"
    )
    
    # Publish events
    await clean_event_bus.publish("test.one", {"value": 1})
    await clean_event_bus.publish("test.two", {"value": 2})
    await clean_event_bus.publish("other.event", {"value": 3})
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify only test.* events were received
    assert len(received_events) == 2
    assert received_events[0].type == "test.one"
    assert received_events[1].type == "test.two"


@pytest.mark.asyncio
async def test_event_priority(clean_event_bus):
    """Test event priority handling."""
    received_order = []
    
    async def priority_handler(event: Event):
        received_order.append(event.priority)
    
    # Register handler for all events
    clean_event_bus.register_handler(
        ["*"],
        priority_handler,
        priority=EventPriority.HIGH
    )
    
    # Publish events with different priorities
    await clean_event_bus.publish("test", {}, priority=EventPriority.LOW)
    await clean_event_bus.publish("test", {}, priority=EventPriority.CRITICAL)
    await clean_event_bus.publish("test", {}, priority=EventPriority.NORMAL)
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Events should be processed in priority order
    assert len(received_order) == 3


@pytest.mark.asyncio
async def test_event_filtering(clean_event_bus):
    """Test event filtering with custom filter function."""
    received_events = []
    
    async def filtered_handler(event: Event):
        received_events.append(event)
    
    async def filter_func(event: Event) -> bool:
        return event.data.get("include", False)
    
    # Register handler with filter
    clean_event_bus.register_handler(
        ["test.*"],
        filtered_handler,
        filter_func=filter_func
    )
    
    # Publish events
    await clean_event_bus.publish("test.event", {"include": True, "value": 1})
    await clean_event_bus.publish("test.event", {"include": False, "value": 2})
    await clean_event_bus.publish("test.event", {"value": 3})  # No include field
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Only events with include=True should be received
    assert len(received_events) == 1
    assert received_events[0].data["value"] == 1


@pytest.mark.asyncio
async def test_wildcard_subscriptions(clean_event_bus):
    """Test wildcard pattern matching."""
    received_events = []
    
    async def wildcard_handler(event: Event):
        received_events.append(event.type)
    
    # Register handlers with different patterns
    clean_event_bus.register_handler(
        ["component.*", "workflow.started", "*.error"],
        wildcard_handler
    )
    
    # Publish various events
    await clean_event_bus.publish("component.registered", {})
    await clean_event_bus.publish("component.updated", {})
    await clean_event_bus.publish("workflow.started", {})
    await clean_event_bus.publish("workflow.completed", {})
    await clean_event_bus.publish("adapter.error", {})
    await clean_event_bus.publish("node.error", {})
    await clean_event_bus.publish("other.event", {})
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Verify pattern matching
    expected = [
        "component.registered",
        "component.updated",
        "workflow.started",
        "adapter.error",
        "node.error"
    ]
    assert sorted(received_events) == sorted(expected)


@pytest.mark.asyncio
async def test_event_stats(clean_event_bus):
    """Test event bus statistics."""
    # Register a handler
    async def dummy_handler(event: Event):
        pass
    
    clean_event_bus.register_handler(["test.*"], dummy_handler)
    
    # Publish some events
    for i in range(5):
        await clean_event_bus.publish("test.event", {"index": i})
    
    # Get stats
    stats = await clean_event_bus.get_stats()
    
    # Verify stats
    assert stats["events_published"] == 5
    assert stats["registered_handlers"] > 0
    assert stats["active_subscriptions"] == 0  # No subscriptions in this test


@pytest.mark.asyncio
async def test_handler_error_handling(clean_event_bus):
    """Test error handling in event handlers."""
    successful_events = []
    
    async def failing_handler(event: Event):
        if event.data.get("fail", False):
            raise ValueError("Handler failed")
        successful_events.append(event)
    
    # Register handler
    clean_event_bus.register_handler(["test.*"], failing_handler)
    
    # Publish events
    await clean_event_bus.publish("test.event", {"fail": False, "id": 1})
    await clean_event_bus.publish("test.event", {"fail": True, "id": 2})
    await clean_event_bus.publish("test.event", {"fail": False, "id": 3})
    
    # Wait for processing
    await asyncio.sleep(0.1)
    
    # Handler errors should not affect other events
    assert len(successful_events) == 2
    assert successful_events[0].data["id"] == 1
    assert successful_events[1].data["id"] == 3