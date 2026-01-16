"""Cloud-aware configuration for platform integrations"""
import logging
from typing import Dict, Any, Optional
from datetime import timedelta

from services.cloud_detection import cloud_detector, CloudProvider, DeploymentEnvironment
from schemas.platform_integration import PlatformType

logger = logging.getLogger(__name__)


class PlatformCloudConfig:
    """Cloud-aware configuration for platform integrations"""
    
    def __init__(self):
        self.cloud_provider = cloud_detector.provider
        self.environment = cloud_detector.environment
        self.is_cloud = cloud_detector.is_cloud
        self.is_serverless = cloud_detector.is_serverless
        self.resource_limits = cloud_detector.get_resource_limits()
    
    def get_platform_limits(self, platform: PlatformType) -> Dict[str, Any]:
        """Get platform-specific limits adjusted for cloud environment"""
        
        # Base limits (local/development)
        base_limits = {
            PlatformType.N8N: {
                "max_executions_per_minute": 100,
                "max_execution_time": 3600,
                "max_payload_size_mb": 50,
                "max_concurrent_executions": 20
            },
            PlatformType.ZAPIER: {
                "max_executions_per_minute": 75,
                "max_execution_time": 30,
                "max_payload_size_mb": 10,
                "max_concurrent_executions": 10
            },
            PlatformType.MAKE: {
                "max_executions_per_minute": 60,
                "max_execution_time": 2400,
                "max_payload_size_mb": 25,
                "max_concurrent_executions": 15
            },
            PlatformType.POWER_AUTOMATE: {
                "max_executions_per_minute": 1000,
                "max_execution_time": 86400 * 30,  # 30 days
                "max_payload_size_mb": 100,
                "max_concurrent_executions": 50
            },
            PlatformType.IFTTT: {
                "max_executions_per_minute": 100,
                "max_execution_time": 15,
                "max_payload_size_mb": 1,
                "max_concurrent_executions": 5
            }
        }
        
        limits = base_limits.get(platform, {
            "max_executions_per_minute": 50,
            "max_execution_time": 300,
            "max_payload_size_mb": 10,
            "max_concurrent_executions": 10
        })
        
        # Adjust for cloud environment
        if self.is_serverless:
            # Serverless has stricter limits
            limits["max_execution_time"] = min(
                limits["max_execution_time"],
                self.resource_limits["max_execution_time_seconds"]
            )
            limits["max_payload_size_mb"] = min(
                limits["max_payload_size_mb"],
                self.resource_limits["max_payload_size_mb"]
            )
            limits["max_concurrent_executions"] = min(
                limits["max_concurrent_executions"],
                5  # Conservative for serverless
            )
        
        elif self.is_cloud:
            # Cloud but not serverless - adjust based on environment
            if self.environment == DeploymentEnvironment.PRODUCTION:
                # Production gets full limits
                pass
            elif self.environment == DeploymentEnvironment.STAGING:
                # Staging gets 50% of production limits
                limits["max_executions_per_minute"] = int(limits["max_executions_per_minute"] * 0.5)
                limits["max_concurrent_executions"] = int(limits["max_concurrent_executions"] * 0.5)
            else:
                # Dev/Test gets 25% of production limits
                limits["max_executions_per_minute"] = int(limits["max_executions_per_minute"] * 0.25)
                limits["max_concurrent_executions"] = int(limits["max_concurrent_executions"] * 0.25)
        
        return limits
    
    def get_retry_config(self, platform: PlatformType) -> Dict[str, Any]:
        """Get retry configuration adjusted for cloud environment"""
        
        # Base retry config
        config = {
            "max_attempts": 3,
            "initial_delay": 1.0,
            "max_delay": 60.0,
            "exponential_base": 2.0,
            "jitter": True
        }
        
        # Adjust for serverless
        if self.is_serverless:
            # More aggressive retries in serverless
            config.update({
                "max_attempts": 2,  # Fewer attempts
                "initial_delay": 0.5,  # Shorter initial delay
                "max_delay": 30.0  # Shorter max delay
            })
        
        # Platform-specific adjustments
        if platform == PlatformType.IFTTT:
            # IFTTT is fast, reduce delays
            config["initial_delay"] = 0.5
            config["max_delay"] = 10.0
        elif platform == PlatformType.POWER_AUTOMATE:
            # Power Automate can have long-running flows
            config["max_delay"] = 120.0
        
        return config
    
    def get_caching_config(self) -> Dict[str, Any]:
        """Get caching configuration for cloud environment"""
        
        config = {
            "enabled": True,
            "ttl_seconds": 300,  # 5 minutes default
            "max_cache_size_mb": 100,
            "cache_workflow_info": True,
            "cache_credentials": False  # Never cache credentials
        }
        
        if self.is_serverless:
            # Serverless has limited memory, reduce cache
            config.update({
                "ttl_seconds": 60,  # 1 minute
                "max_cache_size_mb": 10,
                "cache_workflow_info": False  # Don't cache in serverless
            })
        elif self.cloud_provider == CloudProvider.REDIS:
            # If we have Redis, use it more aggressively
            config.update({
                "ttl_seconds": 3600,  # 1 hour
                "max_cache_size_mb": 500
            })
        
        return config
    
    def get_webhook_config(self, platform: PlatformType) -> Dict[str, Any]:
        """Get webhook configuration for cloud environment"""
        
        config = {
            "timeout_seconds": 30,
            "max_retries": 3,
            "verify_ssl": True,
            "follow_redirects": True,
            "max_redirects": 5
        }
        
        # Adjust timeout for serverless
        if self.is_serverless:
            max_timeout = self.resource_limits["max_execution_time_seconds"]
            config["timeout_seconds"] = min(30, max_timeout - 5)  # Leave 5s buffer
        
        # Platform-specific adjustments
        if platform == PlatformType.IFTTT:
            config["timeout_seconds"] = 15  # IFTTT is fast
        elif platform == PlatformType.POWER_AUTOMATE:
            config["timeout_seconds"] = 120  # Power Automate can be slow
        
        # Development environments might use self-signed certs
        if self.environment == DeploymentEnvironment.DEVELOPMENT:
            config["verify_ssl"] = False
        
        return config
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration for cloud environment"""
        
        config = {
            "enabled": True,
            "metrics_enabled": True,
            "tracing_enabled": True,
            "logging_level": "INFO",
            "health_check_interval": 60,  # seconds
            "detailed_errors": False
        }
        
        # Adjust for environment
        if self.environment == DeploymentEnvironment.PRODUCTION:
            config.update({
                "health_check_interval": 30,
                "detailed_errors": False
            })
        elif self.environment in [DeploymentEnvironment.DEVELOPMENT, DeploymentEnvironment.TEST]:
            config.update({
                "logging_level": "DEBUG",
                "detailed_errors": True
            })
        
        # Cloud-specific monitoring
        if self.cloud_provider == CloudProvider.AWS:
            config["cloudwatch_enabled"] = True
        elif self.cloud_provider == CloudProvider.AZURE:
            config["app_insights_enabled"] = True
        elif self.cloud_provider == CloudProvider.GCP:
            config["stackdriver_enabled"] = True
        
        return config
    
    def get_security_config(self) -> Dict[str, Any]:
        """Get security configuration for cloud environment"""
        
        config = {
            "encryption_enabled": True,
            "credential_rotation_days": 90,
            "api_rate_limiting": True,
            "ip_whitelist_enabled": False,
            "audit_logging": True,
            "secure_webhooks": True
        }
        
        # Production security
        if self.environment == DeploymentEnvironment.PRODUCTION:
            config.update({
                "credential_rotation_days": 30,
                "ip_whitelist_enabled": True,
                "audit_logging": True
            })
        
        # Cloud-specific security
        if self.is_cloud:
            config["use_managed_identity"] = self.cloud_provider in [
                CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP
            ]
        
        return config
    
    def should_use_async(self, platform: PlatformType) -> bool:
        """Determine if platform operations should be async"""
        
        # Always async in serverless
        if self.is_serverless:
            return True
        
        # Platform-specific
        if platform == PlatformType.POWER_AUTOMATE:
            return True  # Long-running flows
        elif platform == PlatformType.IFTTT:
            return False  # Quick webhooks
        
        # Default to async in cloud
        return self.is_cloud
    
    def get_cost_config(self) -> Dict[str, Any]:
        """Get cost tracking configuration"""
        
        config = {
            "track_costs": False,
            "cost_alerts_enabled": False,
            "monthly_budget": None,
            "cost_per_execution": 0.0
        }
        
        if self.environment == DeploymentEnvironment.PRODUCTION and self.is_cloud:
            config.update({
                "track_costs": True,
                "cost_alerts_enabled": True,
                "monthly_budget": 1000.0  # Default $1000
            })
            
            # Cloud-specific costs
            if self.cloud_provider == CloudProvider.AWS:
                config["cost_per_execution"] = 0.0002  # Lambda pricing estimate
            elif self.cloud_provider == CloudProvider.AZURE:
                config["cost_per_execution"] = 0.00016  # Azure Functions estimate
            elif self.cloud_provider == CloudProvider.GCP:
                config["cost_per_execution"] = 0.00025  # Cloud Functions estimate
        
        return config
    
    def validate_platform_compatibility(self, platform: PlatformType) -> Dict[str, Any]:
        """Validate if platform is compatible with current cloud environment"""
        
        compatible = True
        warnings = []
        recommendations = []
        
        # Check execution time limits
        platform_limits = self.get_platform_limits(platform)
        if platform_limits["max_execution_time"] > self.resource_limits["max_execution_time_seconds"]:
            if platform == PlatformType.POWER_AUTOMATE:
                warnings.append("Power Automate flows may timeout in serverless environment")
                recommendations.append("Use webhooks for long-running flows")
            elif platform == PlatformType.MAKE:
                warnings.append("Make.com scenarios may exceed serverless timeout")
                recommendations.append("Break long scenarios into smaller parts")
        
        # Check payload size
        if platform_limits["max_payload_size_mb"] > self.resource_limits["max_payload_size_mb"]:
            warnings.append(f"Payload size limited to {self.resource_limits['max_payload_size_mb']}MB")
            recommendations.append("Use external storage for large payloads")
        
        # Serverless-specific checks
        if self.is_serverless:
            if platform == PlatformType.N8N:
                warnings.append("n8n webhooks may timeout in serverless")
                recommendations.append("Use queue-based processing for complex workflows")
        
        # Memory constraints
        if self.resource_limits["max_memory_mb"] < 256:
            warnings.append("Low memory environment detected")
            recommendations.append("Monitor memory usage closely")
        
        return {
            "compatible": compatible,
            "warnings": warnings,
            "recommendations": recommendations,
            "environment": {
                "provider": self.cloud_provider.value,
                "environment": self.environment.value,
                "is_serverless": self.is_serverless,
                "limits": self.resource_limits
            }
        }


# Global instance
platform_cloud_config = PlatformCloudConfig()