"""Cloud deployment detection and configuration service"""
import os
import logging
from typing import Dict, Any, Optional
from enum import Enum
import httpx
import socket

logger = logging.getLogger(__name__)


class CloudProvider(str, Enum):
    """Supported cloud providers"""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    HEROKU = "heroku"
    DIGITALOCEAN = "digitalocean"
    RAILWAY = "railway"
    RENDER = "render"
    FLY_IO = "fly_io"
    KUBERNETES = "kubernetes"
    DOCKER = "docker"
    LOCAL = "local"
    UNKNOWN = "unknown"


class DeploymentEnvironment(str, Enum):
    """Deployment environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"


class CloudDetectionService:
    """Service for detecting cloud deployment environment"""
    
    def __init__(self):
        self._provider: Optional[CloudProvider] = None
        self._environment: Optional[DeploymentEnvironment] = None
        self._metadata: Dict[str, Any] = {}
        self._detect_provider()
    
    def _detect_provider(self):
        """Detect the cloud provider based on environment variables and metadata endpoints"""
        
        # Check environment variables first
        if os.environ.get("AWS_EXECUTION_ENV") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
            self._provider = CloudProvider.AWS
            self._detect_aws_metadata()
        elif os.environ.get("WEBSITE_INSTANCE_ID") or os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT"):
            self._provider = CloudProvider.AZURE
            self._detect_azure_metadata()
        elif os.environ.get("GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT"):
            self._provider = CloudProvider.GCP
            self._detect_gcp_metadata()
        elif os.environ.get("DYNO") or os.environ.get("HEROKU_APP_NAME"):
            self._provider = CloudProvider.HEROKU
            self._detect_heroku_metadata()
        elif os.environ.get("DO_APP_ID") or os.environ.get("DIGITALOCEAN"):
            self._provider = CloudProvider.DIGITALOCEAN
            self._detect_do_metadata()
        elif os.environ.get("RAILWAY_PROJECT_ID"):
            self._provider = CloudProvider.RAILWAY
            self._detect_railway_metadata()
        elif os.environ.get("RENDER_SERVICE_ID"):
            self._provider = CloudProvider.RENDER
            self._detect_render_metadata()
        elif os.environ.get("FLY_APP_NAME") or os.environ.get("FLY_ALLOC_ID"):
            self._provider = CloudProvider.FLY_IO
            self._detect_fly_metadata()
        elif os.environ.get("KUBERNETES_SERVICE_HOST"):
            self._provider = CloudProvider.KUBERNETES
            self._detect_kubernetes_metadata()
        elif os.path.exists("/.dockerenv") or os.environ.get("DOCKER_CONTAINER"):
            self._provider = CloudProvider.DOCKER
            self._detect_docker_metadata()
        else:
            # Try metadata endpoints
            if self._check_aws_metadata():
                self._provider = CloudProvider.AWS
            elif self._check_azure_metadata():
                self._provider = CloudProvider.AZURE
            elif self._check_gcp_metadata():
                self._provider = CloudProvider.GCP
            else:
                self._provider = CloudProvider.LOCAL
        
        # Detect environment
        self._detect_environment()
    
    def _detect_environment(self):
        """Detect deployment environment"""
        env = os.environ.get("ENVIRONMENT", "").lower()
        if not env:
            env = os.environ.get("ENV", "").lower()
        if not env:
            env = os.environ.get("NODE_ENV", "").lower()
        
        if env in ["prod", "production"]:
            self._environment = DeploymentEnvironment.PRODUCTION
        elif env in ["stage", "staging"]:
            self._environment = DeploymentEnvironment.STAGING
        elif env in ["test", "testing"]:
            self._environment = DeploymentEnvironment.TEST
        elif env in ["dev", "development"]:
            self._environment = DeploymentEnvironment.DEVELOPMENT
        else:
            # Default based on provider
            if self._provider == CloudProvider.LOCAL:
                self._environment = DeploymentEnvironment.DEVELOPMENT
            else:
                self._environment = DeploymentEnvironment.PRODUCTION
    
    def _check_aws_metadata(self) -> bool:
        """Check AWS metadata endpoint"""
        try:
            # AWS IMDSv2 requires token
            token_response = httpx.put(
                "http://169.254.169.254/latest/api/token",
                headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
                timeout=1.0
            )
            if token_response.status_code == 200:
                self._metadata["aws_token"] = token_response.text
                return True
        except:
            pass
        return False
    
    def _check_azure_metadata(self) -> bool:
        """Check Azure metadata endpoint"""
        try:
            response = httpx.get(
                "http://169.254.169.254/metadata/instance?api-version=2021-02-01",
                headers={"Metadata": "true"},
                timeout=1.0
            )
            return response.status_code == 200
        except:
            pass
        return False
    
    def _check_gcp_metadata(self) -> bool:
        """Check GCP metadata endpoint"""
        try:
            response = httpx.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/",
                headers={"Metadata-Flavor": "Google"},
                timeout=1.0
            )
            return response.status_code == 200
        except:
            pass
        return False
    
    def _detect_aws_metadata(self):
        """Collect AWS-specific metadata"""
        self._metadata.update({
            "aws_region": os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION")),
            "aws_execution_env": os.environ.get("AWS_EXECUTION_ENV"),
            "aws_function_name": os.environ.get("AWS_LAMBDA_FUNCTION_NAME"),
            "aws_instance_id": self._get_aws_instance_id()
        })
    
    def _detect_azure_metadata(self):
        """Collect Azure-specific metadata"""
        self._metadata.update({
            "azure_website_name": os.environ.get("WEBSITE_SITE_NAME"),
            "azure_instance_id": os.environ.get("WEBSITE_INSTANCE_ID"),
            "azure_region": os.environ.get("REGION_NAME"),
            "azure_resource_group": os.environ.get("WEBSITE_RESOURCE_GROUP")
        })
    
    def _detect_gcp_metadata(self):
        """Collect GCP-specific metadata"""
        self._metadata.update({
            "gcp_project": os.environ.get("GCP_PROJECT", os.environ.get("GOOGLE_CLOUD_PROJECT")),
            "gcp_region": os.environ.get("FUNCTION_REGION"),
            "gcp_service": os.environ.get("K_SERVICE"),
            "gcp_revision": os.environ.get("K_REVISION")
        })
    
    def _detect_heroku_metadata(self):
        """Collect Heroku-specific metadata"""
        self._metadata.update({
            "heroku_app_name": os.environ.get("HEROKU_APP_NAME"),
            "heroku_dyno_id": os.environ.get("HEROKU_DYNO_ID"),
            "heroku_release_version": os.environ.get("HEROKU_RELEASE_VERSION"),
            "heroku_slug_commit": os.environ.get("HEROKU_SLUG_COMMIT")
        })
    
    def _detect_do_metadata(self):
        """Collect DigitalOcean-specific metadata"""
        self._metadata.update({
            "do_app_id": os.environ.get("DO_APP_ID"),
            "do_app_name": os.environ.get("DO_APP_NAME"),
            "do_region": os.environ.get("DO_REGION"),
            "do_component": os.environ.get("DO_COMPONENT_NAME")
        })
    
    def _detect_railway_metadata(self):
        """Collect Railway-specific metadata"""
        self._metadata.update({
            "railway_project_id": os.environ.get("RAILWAY_PROJECT_ID"),
            "railway_environment": os.environ.get("RAILWAY_ENVIRONMENT"),
            "railway_deployment_id": os.environ.get("RAILWAY_DEPLOYMENT_ID"),
            "railway_service_name": os.environ.get("RAILWAY_SERVICE_NAME")
        })
    
    def _detect_render_metadata(self):
        """Collect Render-specific metadata"""
        self._metadata.update({
            "render_service_id": os.environ.get("RENDER_SERVICE_ID"),
            "render_service_name": os.environ.get("RENDER_SERVICE_NAME"),
            "render_service_type": os.environ.get("RENDER_SERVICE_TYPE"),
            "render_instance_id": os.environ.get("RENDER_INSTANCE_ID")
        })
    
    def _detect_fly_metadata(self):
        """Collect Fly.io-specific metadata"""
        self._metadata.update({
            "fly_app_name": os.environ.get("FLY_APP_NAME"),
            "fly_alloc_id": os.environ.get("FLY_ALLOC_ID"),
            "fly_region": os.environ.get("FLY_REGION"),
            "fly_public_ip": os.environ.get("FLY_PUBLIC_IP")
        })
    
    def _detect_kubernetes_metadata(self):
        """Collect Kubernetes-specific metadata"""
        self._metadata.update({
            "k8s_namespace": os.environ.get("POD_NAMESPACE"),
            "k8s_pod_name": os.environ.get("POD_NAME"),
            "k8s_node_name": os.environ.get("NODE_NAME"),
            "k8s_service_account": os.environ.get("SERVICE_ACCOUNT")
        })
    
    def _detect_docker_metadata(self):
        """Collect Docker-specific metadata"""
        hostname = socket.gethostname()
        self._metadata.update({
            "docker_hostname": hostname,
            "docker_container": os.environ.get("HOSTNAME", hostname)
        })
    
    def _get_aws_instance_id(self) -> Optional[str]:
        """Get AWS instance ID from metadata"""
        if "aws_token" in self._metadata:
            try:
                response = httpx.get(
                    "http://169.254.169.254/latest/meta-data/instance-id",
                    headers={"X-aws-ec2-metadata-token": self._metadata["aws_token"]},
                    timeout=1.0
                )
                if response.status_code == 200:
                    return response.text
            except:
                pass
        return None
    
    @property
    def provider(self) -> CloudProvider:
        """Get detected cloud provider"""
        return self._provider or CloudProvider.UNKNOWN
    
    @property
    def environment(self) -> DeploymentEnvironment:
        """Get deployment environment"""
        return self._environment or DeploymentEnvironment.DEVELOPMENT
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get cloud metadata"""
        return self._metadata
    
    @property
    def is_cloud(self) -> bool:
        """Check if running in cloud"""
        return self._provider not in [CloudProvider.LOCAL, CloudProvider.UNKNOWN, CloudProvider.DOCKER]
    
    @property
    def is_serverless(self) -> bool:
        """Check if running in serverless environment"""
        serverless_indicators = {
            CloudProvider.AWS: lambda: bool(os.environ.get("AWS_LAMBDA_FUNCTION_NAME")),
            CloudProvider.AZURE: lambda: bool(os.environ.get("AZURE_FUNCTIONS_ENVIRONMENT")),
            CloudProvider.GCP: lambda: bool(os.environ.get("K_SERVICE")),
            CloudProvider.RENDER: lambda: os.environ.get("RENDER_SERVICE_TYPE") == "background"
        }
        
        checker = serverless_indicators.get(self._provider)
        return checker() if checker else False
    
    def get_resource_limits(self) -> Dict[str, Any]:
        """Get resource limits based on cloud provider"""
        # Default limits
        limits = {
            "max_memory_mb": 512,
            "max_cpu_cores": 1,
            "max_execution_time_seconds": 300,
            "max_payload_size_mb": 10,
            "max_concurrent_executions": 10
        }
        
        # Provider-specific limits
        if self._provider == CloudProvider.AWS:
            if self.is_serverless:
                limits.update({
                    "max_memory_mb": 3008,
                    "max_execution_time_seconds": 900,
                    "max_payload_size_mb": 6
                })
            else:
                # EC2 defaults
                limits.update({
                    "max_memory_mb": 8192,
                    "max_cpu_cores": 4,
                    "max_execution_time_seconds": 3600
                })
        
        elif self._provider == CloudProvider.AZURE:
            if self.is_serverless:
                limits.update({
                    "max_memory_mb": 1536,
                    "max_execution_time_seconds": 600,
                    "max_payload_size_mb": 100
                })
        
        elif self._provider == CloudProvider.GCP:
            if self.is_serverless:
                limits.update({
                    "max_memory_mb": 8192,
                    "max_execution_time_seconds": 3600,
                    "max_payload_size_mb": 32
                })
        
        elif self._provider == CloudProvider.HEROKU:
            dyno_type = os.environ.get("DYNO_RAM", "512")
            limits.update({
                "max_memory_mb": int(dyno_type) if dyno_type.isdigit() else 512,
                "max_execution_time_seconds": 1800  # 30 min request timeout
            })
        
        return limits
    
    def get_scaling_config(self) -> Dict[str, Any]:
        """Get auto-scaling configuration based on provider"""
        config = {
            "auto_scale": False,
            "min_instances": 1,
            "max_instances": 1,
            "scale_up_threshold": 80,
            "scale_down_threshold": 20
        }
        
        if self.is_serverless:
            config.update({
                "auto_scale": True,
                "min_instances": 0,
                "max_instances": 1000
            })
        elif self._provider in [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP]:
            config.update({
                "auto_scale": True,
                "min_instances": 1,
                "max_instances": 10
            })
        
        return config
    
    def get_networking_config(self) -> Dict[str, Any]:
        """Get networking configuration"""
        return {
            "provider": self.provider.value,
            "environment": self.environment.value,
            "is_vpc": self._provider in [CloudProvider.AWS, CloudProvider.AZURE, CloudProvider.GCP],
            "has_load_balancer": self.is_cloud and not self.is_serverless,
            "supports_websockets": not self.is_serverless,
            "supports_long_polling": not self.is_serverless
        }


# Global instance
cloud_detector = CloudDetectionService()