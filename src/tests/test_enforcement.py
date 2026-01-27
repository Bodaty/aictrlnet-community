"""Tests for the value ladder enforcement system."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from core.enforcement import (
    LicenseEnforcer, 
    EnforcementMode, 
    LimitType, 
    Edition,
    LimitExceededException
)
from core.usage_tracker import UsageTracker


@pytest.fixture
async def mock_db():
    """Mock database session."""
    db = AsyncMock()
    return db


@pytest.fixture
async def mock_cache():
    """Mock cache instance."""
    cache = AsyncMock()
    cache.get = AsyncMock(return_value=None)
    cache.set = AsyncMock()
    cache.delete = AsyncMock()
    return cache


class TestLicenseEnforcer:
    """Test license enforcement functionality."""
    
    @pytest.mark.asyncio
    async def test_enforcement_modes(self, mock_db):
        """Test different enforcement modes."""
        
        # NONE mode - should allow everything
        enforcer = LicenseEnforcer(mock_db, mode=EnforcementMode.NONE)
        result = await enforcer.check_limit("tenant1", LimitType.WORKFLOWS, 100)
        assert result["allowed"] is True
        assert result["warning"] is None
        
        # SOFT mode - should warn but allow
        enforcer = LicenseEnforcer(mock_db, mode=EnforcementMode.SOFT)
        with patch.object(enforcer, '_get_tenant_info', return_value={"edition": "community"}):
            with patch.object(enforcer, '_get_current_usage', return_value=15):
                result = await enforcer.check_limit("tenant1", LimitType.WORKFLOWS, 1)
                assert result["allowed"] is True
                assert result["warning"] is not None
                assert "exceeded" in result["warning"]
        
        # STRICT mode - should raise exception
        enforcer = LicenseEnforcer(mock_db, mode=EnforcementMode.STRICT)
        with patch.object(enforcer, '_get_tenant_info', return_value={"edition": "community"}):
            with patch.object(enforcer, '_get_current_usage', return_value=10):
                with pytest.raises(LimitExceededException) as exc:
                    await enforcer.check_limit("tenant1", LimitType.WORKFLOWS, 1)
                assert exc.value.limit_type == LimitType.WORKFLOWS
                assert exc.value.limit == 10
    
    @pytest.mark.asyncio
    async def test_edition_limits(self, mock_db):
        """Test limits by edition."""
        enforcer = LicenseEnforcer(mock_db, mode=EnforcementMode.STRICT)
        
        # Community limits
        assert enforcer.EDITION_LIMITS[Edition.COMMUNITY][LimitType.WORKFLOWS] == 10
        assert enforcer.EDITION_LIMITS[Edition.COMMUNITY][LimitType.ADAPTERS] == 5
        assert enforcer.EDITION_LIMITS[Edition.COMMUNITY][LimitType.USERS] == 1
        
        # Business limits are higher
        assert enforcer.EDITION_LIMITS[Edition.BUSINESS_STARTER][LimitType.WORKFLOWS] == 100
        assert enforcer.EDITION_LIMITS[Edition.BUSINESS_GROWTH][LimitType.WORKFLOWS] == 500
        assert enforcer.EDITION_LIMITS[Edition.BUSINESS_SCALE][LimitType.WORKFLOWS] == 1000
        
        # Enterprise is unlimited
        assert enforcer.EDITION_LIMITS[Edition.ENTERPRISE][LimitType.WORKFLOWS] == 999999
    
    @pytest.mark.asyncio
    async def test_feature_checking(self, mock_db):
        """Test feature availability checking."""
        enforcer = LicenseEnforcer(mock_db)
        
        # Community features
        with patch.object(enforcer, '_get_tenant_info', return_value={"edition": "community"}):
            assert await enforcer.check_feature("tenant1", "basic_workflows") is True
            assert await enforcer.check_feature("tenant1", "approval_workflows") is False
            assert await enforcer.check_feature("tenant1", "multi_tenant") is False
        
        # Business features
        with patch.object(enforcer, '_get_tenant_info', return_value={"edition": "business_starter"}):
            assert await enforcer.check_feature("tenant1", "basic_workflows") is True
            assert await enforcer.check_feature("tenant1", "approval_workflows") is True
            assert await enforcer.check_feature("tenant1", "multi_tenant") is False
        
        # Enterprise features
        with patch.object(enforcer, '_get_tenant_info', return_value={"edition": "enterprise"}):
            assert await enforcer.check_feature("tenant1", "basic_workflows") is True
            assert await enforcer.check_feature("tenant1", "approval_workflows") is True
            assert await enforcer.check_feature("tenant1", "multi_tenant") is True
    
    @pytest.mark.asyncio
    async def test_limit_warnings(self, mock_db):
        """Test warning thresholds."""
        enforcer = LicenseEnforcer(mock_db, mode=EnforcementMode.SOFT)
        
        with patch.object(enforcer, '_get_tenant_info', return_value={"edition": "community"}):
            # At 80% - should warn
            with patch.object(enforcer, '_get_current_usage', return_value=8):
                result = await enforcer.check_limit("tenant1", LimitType.WORKFLOWS, 0)
                assert result["warning"] is not None
                assert "Approaching limit" in result["warning"]
                assert result["percentage"] == 80.0
            
            # Below 80% - no warning
            with patch.object(enforcer, '_get_current_usage', return_value=5):
                result = await enforcer.check_limit("tenant1", LimitType.WORKFLOWS, 0)
                assert result["warning"] is None
                assert result["percentage"] == 50.0
    
    @pytest.mark.asyncio
    async def test_custom_overrides(self, mock_db):
        """Test custom limit overrides."""
        enforcer = LicenseEnforcer(mock_db)
        
        # Mock custom override
        mock_override = Mock()
        mock_override.limit_value = 50
        
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.scalar_one_or_none = Mock(return_value=mock_override)
        
        with patch.object(enforcer, '_get_tenant_info', return_value={"edition": "community"}):
            limit = await enforcer._get_limit_for_tenant("tenant1", "community", LimitType.WORKFLOWS)
            assert limit == 50  # Override value, not default 10


class TestUsageTracker:
    """Test usage tracking functionality."""
    
    @pytest.mark.asyncio
    async def test_usage_tracking(self, mock_db):
        """Test basic usage tracking."""
        tracker = UsageTracker(mock_db)
        
        # Track usage
        await tracker.track_usage(
            tenant_id="tenant1",
            metric_type="api_calls",
            value=5.0,
            metadata={"endpoint": "/api/test"}
        )
        
        # Check buffer
        assert "tenant1:api_calls" in tracker._buffer
        assert tracker._buffer["tenant1:api_calls"]["value"] == 5.0
        assert tracker._buffer["tenant1:api_calls"]["count"] == 1
    
    @pytest.mark.asyncio 
    async def test_buffer_flushing(self, mock_db):
        """Test buffer flushing to database."""
        tracker = UsageTracker(mock_db)
        
        # Add to buffer
        await tracker.track_usage("tenant1", "api_calls", 10)
        await tracker.track_usage("tenant1", "api_calls", 5)
        
        # Buffer should combine
        assert tracker._buffer["tenant1:api_calls"]["value"] == 15.0
        assert tracker._buffer["tenant1:api_calls"]["count"] == 2
        
        # Mock db operations
        mock_db.add = Mock()
        mock_db.commit = AsyncMock()
        
        # Flush buffer
        await tracker.flush_buffer()
        
        # Buffer should be empty
        assert len(tracker._buffer) == 0
        
        # Should have added to db
        assert mock_db.add.called
    
    @pytest.mark.asyncio
    async def test_api_call_tracking(self, mock_db):
        """Test API call tracking helper."""
        tracker = UsageTracker(mock_db)
        
        await tracker.track_api_call(
            tenant_id="tenant1",
            endpoint="/api/workflows",
            method="POST",
            response_time_ms=150.5,
            status_code=201
        )
        
        # Check metadata
        buffer_key = "tenant1:api_calls"
        assert buffer_key in tracker._buffer
        metadata = tracker._buffer[buffer_key].get("metadata", [])
        assert len(metadata) == 1
        assert metadata[0]["endpoint"] == "/api/workflows"
        assert metadata[0]["response_time_ms"] == 150.5
    
    @pytest.mark.asyncio
    async def test_usage_summary(self, mock_db):
        """Test usage summary generation."""
        tracker = UsageTracker(mock_db)
        
        # Mock query results
        mock_result = Mock()
        mock_result.metric_type = "api_calls"
        mock_result.total_value = 1000.0
        mock_result.total_count = 1000
        mock_result.record_count = 50
        
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.all = Mock(return_value=[mock_result])
        
        # Get summary
        summary = await tracker.get_usage_summary("tenant1")
        
        assert summary["tenant_id"] == "tenant1"
        assert "api_calls" in summary["metrics"]
        assert summary["metrics"]["api_calls"]["total_value"] == 1000.0


class TestEnforcementIntegration:
    """Test enforcement system integration."""
    
    @pytest.mark.asyncio
    async def test_upgrade_path_generation(self, mock_db):
        """Test upgrade path suggestions."""
        enforcer = LicenseEnforcer(mock_db)
        
        # Community upgrade path
        path = enforcer._get_upgrade_path("community")
        assert len(path["upgrade_options"]) == 2
        assert path["upgrade_options"][0]["edition"] == "business_starter"
        assert "$599/month" in path["upgrade_options"][0]["price"]
        
        # Business upgrade path
        path = enforcer._get_upgrade_path("business_starter")
        assert len(path["upgrade_options"]) == 2
        assert path["upgrade_options"][0]["edition"] == "business_growth"
        
        # Scale/Enterprise path
        path = enforcer._get_upgrade_path("business_scale")
        assert len(path["upgrade_options"]) == 1
        assert path["upgrade_options"][0]["edition"] == "enterprise"
        assert path["contact_sales"] is True
    
    @pytest.mark.asyncio
    async def test_trial_features(self, mock_db):
        """Test trial feature access."""
        enforcer = LicenseEnforcer(mock_db)
        
        # Mock trial
        mock_trial = Mock()
        mock_trial.edition_required = "business_starter"
        mock_trial.expires_at = datetime.utcnow() + timedelta(days=7)
        mock_trial.converted = False
        
        mock_db.execute = AsyncMock()
        mock_db.execute.return_value.scalars = Mock()
        mock_db.execute.return_value.scalars.return_value.all = Mock(return_value=[mock_trial])
        
        # Get trial features
        features = await enforcer._get_trial_features("tenant1")
        
        # Should include business features
        assert "approval_workflows" in features
        assert "rbac" in features