"""Tests for control plane functionality."""

import pytest
import asyncio
from datetime import datetime, timedelta

from control_plane.models import (
    ComponentRegistrationRequest, ComponentType, ComponentStatus,
    ComponentCapability, ComponentHeartbeat
)
from control_plane.services import control_plane_service
from control_plane.auth import component_auth
from control_plane.registry import component_registry


@pytest.fixture
async def clean_registry():
    """Clean component registry before and after tests."""
    # Clear registry
    component_registry.components.clear()
    component_registry.components_by_type.clear()
    component_registry.components_by_capability.clear()
    yield
    # Clean up after test
    component_registry.components.clear()
    component_registry.components_by_type.clear()
    component_registry.components_by_capability.clear()


@pytest.mark.asyncio
async def test_component_registration(clean_registry):
    """Test component registration."""
    # Create registration request
    request = ComponentRegistrationRequest(
        name="test-adapter",
        type=ComponentType.ADAPTER,
        version="1.0.0",
        description="Test adapter",
        capabilities=[
            ComponentCapability(
                name="test_capability",
                description="Test capability"
            )
        ],
        edition="community"
    )
    
    # Register component
    response = await control_plane_service.register_component(
        request,
        user_id="test-user"
    )
    
    # Verify response
    assert response.component.name == "test-adapter"
    assert response.component.status == ComponentStatus.ACTIVE
    assert response.token is not None
    assert response.expires_at > datetime.utcnow()
    
    # Verify component in registry
    component = await control_plane_service.get_component(response.component.id)
    assert component is not None
    assert component.name == "test-adapter"


@pytest.mark.asyncio
async def test_component_heartbeat(clean_registry):
    """Test component heartbeat."""
    # Register a component first
    request = ComponentRegistrationRequest(
        name="heartbeat-test",
        type=ComponentType.SERVICE,
        version="1.0.0"
    )
    
    response = await control_plane_service.register_component(
        request,
        user_id="test-user"
    )
    
    component_id = response.component.id
    
    # Send heartbeat
    heartbeat = ComponentHeartbeat(
        component_id=component_id,
        health_score=95.0,
        metrics={"requests": 100, "errors": 5}
    )
    
    updated_component = await control_plane_service.process_heartbeat(heartbeat)
    
    # Verify update
    assert updated_component.health_score == 95.0
    assert updated_component.last_heartbeat is not None
    assert updated_component.status == ComponentStatus.ACTIVE


@pytest.mark.asyncio
async def test_component_jwt_auth(clean_registry):
    """Test component JWT authentication."""
    # Register component
    request = ComponentRegistrationRequest(
        name="auth-test",
        type=ComponentType.ADAPTER,
        version="1.0.0"
    )
    
    response = await control_plane_service.register_component(
        request,
        user_id="test-user"
    )
    
    # Verify token
    token = response.token
    payload = component_auth.verify_component_token(token)
    
    assert payload["sub"] == response.component.id
    assert payload["component_name"] == "auth-test"
    assert payload["type"] == "component"
    
    # Test token refresh
    new_token, new_expires = await control_plane_service.refresh_component_token(
        response.component.id,
        token
    )
    
    assert new_token != token
    assert new_expires > datetime.utcnow()


@pytest.mark.asyncio
async def test_component_discovery(clean_registry):
    """Test component discovery by type and capability."""
    # Register multiple components
    components = [
        ("adapter1", ComponentType.ADAPTER, ["chat", "completion"]),
        ("adapter2", ComponentType.ADAPTER, ["email", "notification"]),
        ("service1", ComponentType.SERVICE, ["monitoring", "health"])
    ]
    
    for name, comp_type, capabilities in components:
        caps = [
            ComponentCapability(name=cap, description=f"{cap} capability")
            for cap in capabilities
        ]
        
        request = ComponentRegistrationRequest(
            name=name,
            type=comp_type,
            version="1.0.0",
            capabilities=caps
        )
        
        await control_plane_service.register_component(
            request,
            user_id="test-user"
        )
    
    # Test discovery by type
    adapters = await control_plane_service.get_components(
        component_type=ComponentType.ADAPTER
    )
    assert len(adapters) == 2
    
    # Test discovery by capability
    chat_components = await control_plane_service.get_components(
        capability="chat"
    )
    assert len(chat_components) == 1
    assert chat_components[0].name == "adapter1"


@pytest.mark.asyncio
async def test_component_health_tracking(clean_registry):
    """Test component health and reputation tracking."""
    # Register component
    request = ComponentRegistrationRequest(
        name="health-test",
        type=ComponentType.SERVICE,
        version="1.0.0"
    )
    
    response = await control_plane_service.register_component(
        request,
        user_id="test-user"
    )
    
    component_id = response.component.id
    
    # Record successful operations
    for _ in range(10):
        await control_plane_service.record_component_result(
            component_id,
            success=True,
            metrics={"duration_ms": 100}
        )
    
    # Record some failures
    for _ in range(2):
        await control_plane_service.record_component_result(
            component_id,
            success=False,
            error_message="Test error"
        )
    
    # Check reputation
    component = await control_plane_service.get_component(component_id)
    assert component.success_count == 10
    assert component.error_count == 2
    assert component.reputation_score == pytest.approx(83.33, rel=0.1)


@pytest.mark.asyncio
async def test_component_cleanup(clean_registry):
    """Test inactive component cleanup."""
    # Register component
    request = ComponentRegistrationRequest(
        name="cleanup-test",
        type=ComponentType.SERVICE,
        version="1.0.0"
    )
    
    response = await control_plane_service.register_component(
        request,
        user_id="test-user"
    )
    
    component_id = response.component.id
    
    # Manually set old heartbeat
    component = component_registry.components[component_id]
    component.last_heartbeat = datetime.utcnow() - timedelta(hours=1)
    
    # Run cleanup
    inactive_ids = await control_plane_service.cleanup_inactive_components()
    
    # Verify cleanup
    assert component_id in inactive_ids
    component = await control_plane_service.get_component(component_id)
    assert component.status == ComponentStatus.INACTIVE