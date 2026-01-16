"""Unit tests for the actual SLA monitoring service implementation."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta
import time

from services.sla_monitoring import (
    SLAMonitoringService, SLAType, ViolationSeverity,
    SLAThreshold, SLAStatus
)


@pytest.fixture
async def sla_service():
    """Create SLA monitoring service instance."""
    service = SLAMonitoringService(check_interval=5)  # 5 seconds for testing
    await service.initialize()
    return service


class TestSLAMonitoringService:
    """Test the actual SLA monitoring service implementation."""
    
    @pytest.mark.asyncio
    async def test_service_initialization(self):
        """Test SLA service initialization."""
        service = SLAMonitoringService(check_interval=10)
        
        assert service._running is False
        assert len(service._sla_configs) == 0
        
        await service.initialize()
        
        assert service._running is True
        
        # Cleanup
        await service.shutdown()
    
    @pytest.mark.asyncio
    async def test_add_sla_config(self, sla_service):
        """Test adding SLA configuration."""
        threshold = SLAThreshold(
            metric_type=SLAType.RESPONSE_TIME,
            warning_threshold=1000,  # 1 second
            violation_threshold=2000,  # 2 seconds
            critical_threshold=5000,  # 5 seconds
            measurement_window=300,
            aggregation_method="average"
        )
        
        await sla_service.add_sla_config("workflow-123", [threshold])
        
        assert "workflow-123" in sla_service._sla_configs
        assert len(sla_service._sla_configs["workflow-123"]) == 1
        assert sla_service._sla_configs["workflow-123"][0].metric_type == SLAType.RESPONSE_TIME
    
    @pytest.mark.asyncio
    async def test_record_metric(self, sla_service):
        """Test recording SLA metrics."""
        # Add SLA config
        threshold = SLAThreshold(
            metric_type=SLAType.RESPONSE_TIME,
            warning_threshold=100,
            violation_threshold=200,
            critical_threshold=500
        )
        await sla_service.add_sla_config("workflow-456", [threshold])
        
        # Record some metrics
        await sla_service.record_metric(
            workflow_id="workflow-456",
            metric_type=SLAType.RESPONSE_TIME,
            value=50  # Good - under warning
        )
        
        status = sla_service.get_sla_status("workflow-456", SLAType.RESPONSE_TIME)
        assert status is not None
        assert status.current_value == 50
        assert status.is_violated is False
        assert status.severity is None
        
        # Record metric that triggers warning
        await sla_service.record_metric(
            workflow_id="workflow-456",
            metric_type=SLAType.RESPONSE_TIME,
            value=150  # Warning level
        )
        
        status = sla_service.get_sla_status("workflow-456", SLAType.RESPONSE_TIME)
        assert status.is_violated is True
        assert status.severity == ViolationSeverity.WARNING
    
    @pytest.mark.asyncio
    async def test_violation_detection(self, sla_service):
        """Test SLA violation detection."""
        # Add SLA config
        threshold = SLAThreshold(
            metric_type=SLAType.ERROR_RATE,
            warning_threshold=0.05,  # 5% error rate
            violation_threshold=0.10,  # 10% error rate
            critical_threshold=0.20,  # 20% error rate
            aggregation_method="average"
        )
        await sla_service.add_sla_config("service-789", [threshold])
        
        # Record metrics that will trigger different severity levels
        test_cases = [
            (0.03, None),  # Good
            (0.07, ViolationSeverity.WARNING),
            (0.15, ViolationSeverity.MINOR),
            (0.25, ViolationSeverity.CRITICAL)
        ]
        
        for error_rate, expected_severity in test_cases:
            await sla_service.record_metric(
                workflow_id="service-789",
                metric_type=SLAType.ERROR_RATE,
                value=error_rate
            )
            
            status = sla_service.get_sla_status("service-789", SLAType.ERROR_RATE)
            
            if expected_severity:
                assert status.is_violated is True
                assert status.severity == expected_severity
            else:
                assert status.is_violated is False
    
    @pytest.mark.asyncio
    async def test_check_workflow_instance_sla(self, sla_service):
        """Test checking SLA for a workflow instance."""
        # Add SLA config
        threshold = SLAThreshold(
            metric_type=SLAType.COMPLETION_TIME,
            warning_threshold=60000,  # 1 minute
            violation_threshold=120000,  # 2 minutes
            critical_threshold=300000  # 5 minutes
        )
        await sla_service.add_sla_config("workflow-template-1", [threshold])
        
        # Mock workflow instance
        mock_instance = Mock()
        mock_instance.workflow_id = "workflow-template-1"
        mock_instance.instance_id = "instance-123"
        mock_instance.start_time = datetime.utcnow() - timedelta(seconds=90)  # 1.5 minutes ago
        mock_instance.end_time = datetime.utcnow()
        mock_instance.status = "completed"
        
        violations = await sla_service.check_workflow_instance(mock_instance)
        
        # Should have one warning violation (90 seconds > 60 seconds warning threshold)
        assert len(violations) == 1
        assert violations[0]["metric_type"] == SLAType.COMPLETION_TIME
        assert violations[0]["severity"] == ViolationSeverity.WARNING
        assert violations[0]["actual_value"] == 90000  # 90 seconds in ms
    
    @pytest.mark.asyncio
    async def test_aggregation_methods(self, sla_service):
        """Test different aggregation methods for SLA metrics."""
        # Test average aggregation
        threshold_avg = SLAThreshold(
            metric_type=SLAType.RESPONSE_TIME,
            warning_threshold=100,
            violation_threshold=200,
            critical_threshold=500,
            aggregation_method="average"
        )
        await sla_service.add_sla_config("avg-test", [threshold_avg])
        
        # Record multiple values
        values = [50, 150, 100, 200, 100]
        for value in values:
            await sla_service.record_metric(
                workflow_id="avg-test",
                metric_type=SLAType.RESPONSE_TIME,
                value=value
            )
        
        status = sla_service.get_sla_status("avg-test", SLAType.RESPONSE_TIME)
        assert status.current_value == 120  # Average of values
        
        # Test percentile aggregation
        threshold_p95 = SLAThreshold(
            metric_type=SLAType.RESPONSE_TIME,
            warning_threshold=150,
            violation_threshold=250,
            critical_threshold=500,
            aggregation_method="p95"
        )
        await sla_service.add_sla_config("p95-test", [threshold_p95])
        
        # Record values
        for value in values:
            await sla_service.record_metric(
                workflow_id="p95-test",
                metric_type=SLAType.RESPONSE_TIME,
                value=value
            )
        
        status = sla_service.get_sla_status("p95-test", SLAType.RESPONSE_TIME)
        # P95 of [50, 100, 100, 150, 200] should be close to 200
        assert 190 <= status.current_value <= 200
    
    @pytest.mark.asyncio
    async def test_availability_monitoring(self, sla_service):
        """Test availability SLA monitoring."""
        threshold = SLAThreshold(
            metric_type=SLAType.AVAILABILITY,
            warning_threshold=0.99,  # 99% availability
            violation_threshold=0.95,  # 95% availability
            critical_threshold=0.90,  # 90% availability
            measurement_window=3600  # 1 hour
        )
        await sla_service.add_sla_config("api-service", [threshold])
        
        # Simulate uptime/downtime events
        await sla_service.record_availability_event("api-service", is_up=True)
        await asyncio.sleep(0.1)
        await sla_service.record_availability_event("api-service", is_up=False)
        await asyncio.sleep(0.1)
        await sla_service.record_availability_event("api-service", is_up=True)
        
        # Check availability calculation
        availability = await sla_service.calculate_availability("api-service")
        assert 0 < availability <= 1.0
    
    @pytest.mark.asyncio
    async def test_violation_notifications(self, sla_service):
        """Test SLA violation notifications."""
        # Track published events
        published_events = []
        
        async def event_handler(event):
            published_events.append(event)
        
        # Subscribe to SLA events
        from events.event_bus import event_bus
        await event_bus.subscribe("sla.violation.*", event_handler)
        
        # Add SLA config
        threshold = SLAThreshold(
            metric_type=SLAType.ERROR_RATE,
            warning_threshold=0.01,
            violation_threshold=0.05,
            critical_threshold=0.10
        )
        await sla_service.add_sla_config("critical-service", [threshold])
        
        # Trigger a critical violation
        await sla_service.record_metric(
            workflow_id="critical-service",
            metric_type=SLAType.ERROR_RATE,
            value=0.15  # 15% error rate - critical
        )
        
        # Give time for event to be published
        await asyncio.sleep(0.1)
        
        # Should have received violation event
        violation_events = [e for e in published_events if e.get("type", "").startswith("sla.violation")]
        assert len(violation_events) > 0
        
        if violation_events:
            event = violation_events[0]
            assert event["workflow_id"] == "critical-service"
            assert event["metric_type"] == SLAType.ERROR_RATE.value
            assert event["severity"] == ViolationSeverity.CRITICAL.value
    
    @pytest.mark.asyncio
    async def test_sla_report_generation(self, sla_service):
        """Test SLA compliance report generation."""
        # Add multiple SLA configs
        thresholds = [
            SLAThreshold(
                metric_type=SLAType.RESPONSE_TIME,
                warning_threshold=100,
                violation_threshold=200,
                critical_threshold=500
            ),
            SLAThreshold(
                metric_type=SLAType.ERROR_RATE,
                warning_threshold=0.01,
                violation_threshold=0.05,
                critical_threshold=0.10
            )
        ]
        await sla_service.add_sla_config("reporting-test", thresholds)
        
        # Record some metrics
        await sla_service.record_metric("reporting-test", SLAType.RESPONSE_TIME, 50)
        await sla_service.record_metric("reporting-test", SLAType.RESPONSE_TIME, 150)
        await sla_service.record_metric("reporting-test", SLAType.ERROR_RATE, 0.02)
        
        # Generate report
        report = await sla_service.generate_sla_report("reporting-test")
        
        assert "workflow_id" in report
        assert report["workflow_id"] == "reporting-test"
        assert "metrics" in report
        assert len(report["metrics"]) == 2
        assert "compliance_percentage" in report
        assert "violations" in report
    
    @pytest.mark.asyncio
    async def test_cleanup_old_metrics(self, sla_service):
        """Test cleanup of old metric data."""
        # Add SLA config
        threshold = SLAThreshold(
            metric_type=SLAType.RESPONSE_TIME,
            warning_threshold=100,
            violation_threshold=200,
            critical_threshold=500,
            measurement_window=60  # 1 minute window
        )
        await sla_service.add_sla_config("cleanup-test", [threshold])
        
        # Record metrics
        for i in range(10):
            await sla_service.record_metric(
                workflow_id="cleanup-test",
                metric_type=SLAType.RESPONSE_TIME,
                value=50 + i * 10
            )
        
        status = sla_service.get_sla_status("cleanup-test", SLAType.RESPONSE_TIME)
        initial_count = len(status.samples)
        
        # Simulate time passing and cleanup
        with patch('time.time', return_value=time.time() + 120):  # 2 minutes later
            await sla_service._cleanup_old_metrics()
        
        # Old metrics should be removed
        status = sla_service.get_sla_status("cleanup-test", SLAType.RESPONSE_TIME)
        assert len(status.samples) < initial_count