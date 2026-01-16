"""Base platform adapter for external automation platforms"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
from enum import Enum

from schemas.platform_integration import (
    PlatformType,
    AuthMethod,
    PlatformCapabilities,
    PlatformWorkflowInfo,
    WorkflowStatus
)

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Platform execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ExecutionRequest:
    """Platform execution request"""
    workflow_id: str
    input_data: Dict[str, Any]
    timeout: int = 300  # seconds
    retry_count: int = 0
    callback_url: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ExecutionResponse:
    """Platform execution response"""
    execution_id: str
    status: ExecutionStatus
    output_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    cost_estimate: Optional[int] = None  # In cents
    metadata: Dict[str, Any] = None


@dataclass
class HealthCheckResult:
    """Platform health check result"""
    is_healthy: bool
    response_time_ms: int
    error: Optional[str] = None
    details: Dict[str, Any] = None


class BasePlatformAdapter(ABC):
    """Abstract base class for platform adapters"""
    
    def __init__(self, platform_type: PlatformType):
        self.platform_type = platform_type
        self.logger = logging.getLogger(f"{__name__}.{platform_type.value}")
    
    @abstractmethod
    def get_supported_auth_methods(self) -> List[AuthMethod]:
        """Get supported authentication methods"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> PlatformCapabilities:
        """Get platform capabilities"""
        pass
    
    @abstractmethod
    async def validate_credentials(self, credentials: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate platform credentials
        Returns: (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    async def list_workflows(
        self,
        credentials: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0,
        search_term: Optional[str] = None,
        status_filter: Optional[WorkflowStatus] = None
    ) -> List[PlatformWorkflowInfo]:
        """List available workflows on the platform with optional filtering"""
        pass
    
    @abstractmethod
    async def get_workflow_info(
        self,
        credentials: Dict[str, Any],
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get detailed information about a specific workflow"""
        pass
    
    @abstractmethod
    async def execute_workflow(
        self,
        credentials: Dict[str, Any],
        request: ExecutionRequest
    ) -> ExecutionResponse:
        """Execute a workflow on the platform"""
        pass
    
    @abstractmethod
    async def get_execution_status(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> ExecutionResponse:
        """Get the status of a workflow execution"""
        pass
    
    @abstractmethod
    async def cancel_execution(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> bool:
        """Cancel a running workflow execution"""
        pass
    
    @abstractmethod
    async def health_check(
        self,
        credentials: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResult:
        """Check platform health and availability"""
        pass
    
    @abstractmethod
    def configure(self, credentials: Dict[str, Any]) -> None:
        """Configure adapter with credentials"""
        pass
    
    # New abstract methods for UI support
    
    @abstractmethod
    async def get_workflow(
        self,
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get details of a specific workflow"""
        pass
    
    @abstractmethod
    async def get_workflow_count(
        self,
        search_term: Optional[str] = None,
        status_filter: Optional[WorkflowStatus] = None
    ) -> int:
        """Get total count of workflows matching criteria"""
        pass
    
    @abstractmethod
    async def test_workflow(
        self,
        workflow_id: str,
        test_inputs: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Test execute a workflow with sample data"""
        pass
    
    # Optional methods with default implementations
    
    async def get_execution_logs(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> List[Dict[str, Any]]:
        """Get execution logs (if supported)"""
        return []
    
    async def estimate_cost(
        self,
        credentials: Dict[str, Any],
        workflow_id: str,
        input_data: Dict[str, Any]
    ) -> Optional[int]:
        """Estimate execution cost in cents (if supported)"""
        return None
    
    async def get_usage_metrics(
        self,
        credentials: Dict[str, Any],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Get usage metrics for the period (if supported)"""
        return {}
    
    async def create_webhook(
        self,
        credentials: Dict[str, Any],
        workflow_id: str,
        callback_url: str,
        events: List[str]
    ) -> Optional[str]:
        """Create a webhook for workflow events (if supported)"""
        return None
    
    async def delete_webhook(
        self,
        credentials: Dict[str, Any],
        webhook_id: str
    ) -> bool:
        """Delete a webhook (if supported)"""
        return False
    
    # Helper methods
    
    def _validate_auth_method(self, credentials: Dict[str, Any]) -> bool:
        """Validate that credentials contain required auth fields"""
        auth_method = credentials.get("auth_method")
        if not auth_method:
            return False
        
        try:
            method = AuthMethod(auth_method)
        except ValueError:
            return False
        
        if method not in self.get_supported_auth_methods():
            return False
        
        # Check required fields based on auth method
        if method == AuthMethod.API_KEY:
            return "api_key" in credentials
        elif method == AuthMethod.OAUTH2:
            return all(k in credentials for k in ["client_id", "client_secret", "access_token"])
        elif method == AuthMethod.BASIC:
            return all(k in credentials for k in ["username", "password"])
        elif method == AuthMethod.TOKEN:
            return "token" in credentials
        
        return True
    
    def _format_error(self, error: Exception, context: str = "") -> str:
        """Format error message consistently"""
        error_msg = f"{type(error).__name__}: {str(error)}"
        if context:
            error_msg = f"{context}: {error_msg}"
        self.logger.error(error_msg, exc_info=True)
        return error_msg
    
    async def _measure_response_time(self, func, *args, **kwargs) -> Tuple[Any, int]:
        """Measure response time of an async function"""
        import time
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            duration_ms = int((time.time() - start) * 1000)
            return result, duration_ms
        except Exception as e:
            duration_ms = int((time.time() - start) * 1000)
            raise e