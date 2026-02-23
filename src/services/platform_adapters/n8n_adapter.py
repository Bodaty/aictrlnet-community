"""n8n platform adapter implementation"""
import httpx
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
import logging

from schemas.platform_integration import (
    PlatformType,
    AuthMethod,
    PlatformCapabilities,
    PlatformWorkflowInfo,
    WorkflowStatus
)
from .base import (
    BasePlatformAdapter,
    ExecutionRequest,
    ExecutionResponse,
    ExecutionStatus,
    HealthCheckResult
)
from .registry import register_adapter

logger = logging.getLogger(__name__)


@register_adapter(PlatformType.N8N)
class N8nAdapter(BasePlatformAdapter):
    """Adapter for n8n automation platform"""
    
    def get_supported_auth_methods(self) -> List[AuthMethod]:
        """n8n supports API key and basic auth"""
        return [AuthMethod.API_KEY, AuthMethod.BASIC]
    
    def get_capabilities(self) -> PlatformCapabilities:
        """Get n8n platform capabilities"""
        return PlatformCapabilities(
            platform=PlatformType.N8N,
            supports_webhooks=True,
            supports_scheduling=True,
            supports_versioning=True,
            supports_rollback=False,
            supports_monitoring=True,
            max_execution_time=3600,  # 1 hour
            rate_limits={
                "executions_per_minute": 100,
                "webhooks_per_workflow": 10
            },
            available_triggers=[
                "webhook",
                "schedule",
                "manual",
                "email",
                "form"
            ],
            available_actions=[
                "http_request",
                "database",
                "transform",
                "conditional",
                "loop"
            ]
        )
    
    def _get_base_url(self, credentials: Dict[str, Any]) -> str:
        """Get n8n instance base URL"""
        return credentials.get("base_url", "").rstrip("/")
    
    def _get_headers(self, credentials: Dict[str, Any]) -> Dict[str, str]:
        """Get authentication headers"""
        headers = {"Content-Type": "application/json"}
        
        auth_method = AuthMethod(credentials.get("auth_method"))
        if auth_method == AuthMethod.API_KEY:
            headers["X-N8N-API-KEY"] = credentials.get("api_key", "")
        elif auth_method == AuthMethod.BASIC:
            # httpx will handle basic auth separately
            pass
        
        return headers
    
    def _get_auth(self, credentials: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Get basic auth tuple if needed"""
        auth_method = AuthMethod(credentials.get("auth_method"))
        if auth_method == AuthMethod.BASIC:
            username = credentials.get("username", "")
            password = credentials.get("password", "")
            return (username, password)
        return None
    
    async def validate_credentials(
        self,
        credentials: Dict[str, Any]
    ) -> Tuple[bool, Optional[str]]:
        """Validate n8n credentials"""
        if not self._validate_auth_method(credentials):
            return False, "Invalid authentication method or missing credentials"
        
        base_url = self._get_base_url(credentials)
        if not base_url:
            return False, "Missing base_url in credentials"
        
        try:
            async with httpx.AsyncClient() as client:
                # Test API connectivity
                response = await client.get(
                    f"{base_url}/api/v1/workflows",
                    headers=self._get_headers(credentials),
                    auth=self._get_auth(credentials),
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return True, None
                elif response.status_code == 401:
                    return False, "Invalid credentials"
                else:
                    return False, f"API error: {response.status_code}"
                
        except Exception as e:
            return False, self._format_error(e, "Credential validation failed")
    
    async def list_workflows(
        self,
        credentials: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[PlatformWorkflowInfo]:
        """List n8n workflows"""
        base_url = self._get_base_url(credentials)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/api/v1/workflows",
                    params={"limit": limit, "offset": offset},
                    headers=self._get_headers(credentials),
                    auth=self._get_auth(credentials),
                    timeout=30.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                workflows = []
                for workflow in data.get("data", []):
                    workflows.append(PlatformWorkflowInfo(
                        platform=PlatformType.N8N,
                        workflow_id=str(workflow["id"]),
                        name=workflow["name"],
                        description=workflow.get("description"),
                        version=str(workflow.get("versionId", "1")),
                        is_active=workflow.get("active", False),
                        last_modified=datetime.fromisoformat(
                            workflow["updatedAt"].replace("Z", "+00:00")
                        ) if "updatedAt" in workflow else None,
                        tags=workflow.get("tags", []),
                        input_schema=None,  # n8n doesn't provide schema
                        output_schema=None
                    ))
                
                return workflows
                
        except Exception as e:
            self.logger.error(f"Failed to list workflows: {e}")
            return []
    
    async def get_workflow_info(
        self,
        credentials: Dict[str, Any],
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get n8n workflow details"""
        base_url = self._get_base_url(credentials)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/api/v1/workflows/{workflow_id}",
                    headers=self._get_headers(credentials),
                    auth=self._get_auth(credentials),
                    timeout=30.0
                )
                
                if response.status_code == 404:
                    return None
                
                response.raise_for_status()
                workflow = response.json()
                
                return PlatformWorkflowInfo(
                    platform=PlatformType.N8N,
                    workflow_id=str(workflow["id"]),
                    name=workflow["name"],
                    description=workflow.get("description"),
                    version=str(workflow.get("versionId", "1")),
                    is_active=workflow.get("active", False),
                    last_modified=datetime.fromisoformat(
                        workflow["updatedAt"].replace("Z", "+00:00")
                    ) if "updatedAt" in workflow else None,
                    tags=workflow.get("tags", []),
                    input_schema=None,
                    output_schema=None
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow info: {e}")
            return None
    
    async def execute_workflow(
        self,
        credentials: Dict[str, Any],
        request: ExecutionRequest
    ) -> ExecutionResponse:
        """Execute n8n workflow"""
        base_url = self._get_base_url(credentials)
        
        try:
            async with httpx.AsyncClient() as client:
                # Execute workflow
                response = await client.post(
                    f"{base_url}/api/v1/workflows/{request.workflow_id}/execute",
                    json={"data": request.input_data},
                    headers=self._get_headers(credentials),
                    auth=self._get_auth(credentials),
                    timeout=float(request.timeout)
                )
                
                response.raise_for_status()
                result = response.json()
                
                # n8n executes synchronously by default
                execution_id = result.get("executionId", "sync")
                
                return ExecutionResponse(
                    execution_id=execution_id,
                    status=ExecutionStatus.COMPLETED,
                    output_data=result.get("data", {}),
                    error=None,
                    started_at=datetime.utcnow(),
                    completed_at=datetime.utcnow(),
                    duration_ms=None,  # n8n doesn't provide this
                    cost_estimate=0,  # n8n is self-hosted
                    metadata={"mode": "sync"}
                )
                
        except httpx.TimeoutException:
            return ExecutionResponse(
                execution_id="timeout",
                status=ExecutionStatus.TIMEOUT,
                error=f"Execution timed out after {request.timeout} seconds",
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
        except Exception as e:
            return ExecutionResponse(
                execution_id="error",
                status=ExecutionStatus.FAILED,
                error=self._format_error(e, "Execution failed"),
                started_at=datetime.utcnow(),
                completed_at=datetime.utcnow()
            )
    
    async def get_execution_status(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> ExecutionResponse:
        """Get n8n execution status"""
        base_url = self._get_base_url(credentials)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/api/v1/executions/{execution_id}",
                    headers=self._get_headers(credentials),
                    auth=self._get_auth(credentials),
                    timeout=30.0
                )
                
                if response.status_code == 404:
                    return ExecutionResponse(
                        execution_id=execution_id,
                        status=ExecutionStatus.FAILED,
                        error="Execution not found"
                    )
                
                response.raise_for_status()
                execution = response.json()
                
                # Map n8n status to our status
                n8n_status = execution.get("status", "unknown")
                status_map = {
                    "success": ExecutionStatus.COMPLETED,
                    "error": ExecutionStatus.FAILED,
                    "running": ExecutionStatus.RUNNING,
                    "waiting": ExecutionStatus.PENDING
                }
                status = status_map.get(n8n_status, ExecutionStatus.FAILED)
                
                return ExecutionResponse(
                    execution_id=execution_id,
                    status=status,
                    output_data=execution.get("data"),
                    error=execution.get("error"),
                    started_at=datetime.fromisoformat(
                        execution["startedAt"].replace("Z", "+00:00")
                    ) if "startedAt" in execution else None,
                    completed_at=datetime.fromisoformat(
                        execution["stoppedAt"].replace("Z", "+00:00")
                    ) if "stoppedAt" in execution else None,
                    duration_ms=None,
                    cost_estimate=0,
                    metadata={"n8n_status": n8n_status}
                )
                
        except Exception as e:
            return ExecutionResponse(
                execution_id=execution_id,
                status=ExecutionStatus.FAILED,
                error=self._format_error(e, "Failed to get execution status")
            )
    
    async def cancel_execution(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> bool:
        """Cancel n8n execution"""
        base_url = self._get_base_url(credentials)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{base_url}/api/v1/executions/{execution_id}/stop",
                    headers=self._get_headers(credentials),
                    auth=self._get_auth(credentials),
                    timeout=10.0
                )
                
                return response.status_code in [200, 204]
                
        except Exception as e:
            self.logger.error(f"Failed to cancel execution: {e}")
            return False
    
    async def health_check(
        self,
        credentials: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResult:
        """Check n8n instance health"""
        if not credentials:
            # Basic health check without auth
            return HealthCheckResult(
                is_healthy=True,
                response_time_ms=0,
                details={"note": "No credentials provided for health check"}
            )
        
        base_url = self._get_base_url(credentials)
        
        try:
            async with httpx.AsyncClient() as client:
                # Try to access workflows endpoint
                result, duration_ms = await self._measure_response_time(
                    client.get,
                    f"{base_url}/api/v1/workflows",
                    headers=self._get_headers(credentials),
                    auth=self._get_auth(credentials),
                    params={"limit": 1},
                    timeout=10.0
                )
                
                is_healthy = result.status_code == 200
                
                return HealthCheckResult(
                    is_healthy=is_healthy,
                    response_time_ms=duration_ms,
                    error=None if is_healthy else f"Status code: {result.status_code}",
                    details={
                        "status_code": result.status_code,
                        "version": result.headers.get("n8n-version")
                    }
                )
                
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                response_time_ms=0,
                error=str(e),
                details={"exception": type(e).__name__}
            )
    
    def configure(self, credentials: Dict[str, Any]) -> None:
        """Configure adapter with credentials"""
        self._credentials = credentials
    
    async def list_workflows(
        self,
        credentials: Dict[str, Any] = None,
        limit: int = 100,
        offset: int = 0,
        search_term: Optional[str] = None,
        status_filter: Optional[WorkflowStatus] = None
    ) -> List[PlatformWorkflowInfo]:
        """List n8n workflows with optional filtering"""
        # Use stored credentials if not provided
        creds = credentials or getattr(self, '_credentials', {})
        base_url = self._get_base_url(creds)
        
        try:
            async with httpx.AsyncClient() as client:
                # n8n API caps limit at 250, use cursor pagination for larger sets
                response = await client.get(
                    f"{base_url}/api/v1/workflows",
                    params={"limit": min(limit, 250)},
                    headers=self._get_headers(creds),
                    auth=self._get_auth(creds),
                    timeout=30.0
                )
                
                response.raise_for_status()
                data = response.json()
                
                workflows = []
                for workflow in data.get("data", []):
                    # Create workflow info
                    wf_info = PlatformWorkflowInfo(
                        workflow_id=str(workflow["id"]),
                        name=workflow["name"],
                        description=workflow.get("description"),
                        status=WorkflowStatus.ACTIVE if workflow.get("active", False) else WorkflowStatus.INACTIVE,
                        trigger_type="unknown",
                        created_at=datetime.fromisoformat(
                            workflow["createdAt"].replace("Z", "+00:00")
                        ) if "createdAt" in workflow else None,
                        updated_at=datetime.fromisoformat(
                            workflow["updatedAt"].replace("Z", "+00:00")
                        ) if "updatedAt" in workflow else None,
                        version=str(workflow.get("versionId", "1")),
                        tags=workflow.get("tags", []),
                        input_schema=None,
                        output_schema=None
                    )
                    
                    # Apply filters
                    if search_term and search_term.lower() not in wf_info.name.lower():
                        if not wf_info.description or search_term.lower() not in wf_info.description.lower():
                            continue
                    
                    if status_filter and wf_info.status != status_filter:
                        continue
                    
                    workflows.append(wf_info)
                
                # Apply pagination
                start = offset
                end = offset + limit
                return workflows[start:end]
                
        except Exception as e:
            self.logger.error(f"Failed to list workflows: {e}")
            return []
    
    async def get_workflow(
        self,
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get details of a specific workflow"""
        creds = getattr(self, '_credentials', {})
        base_url = self._get_base_url(creds)
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{base_url}/api/v1/workflows/{workflow_id}",
                    headers=self._get_headers(creds),
                    auth=self._get_auth(creds),
                    timeout=30.0
                )
                
                if response.status_code == 404:
                    return None
                
                response.raise_for_status()
                workflow = response.json()
                
                return PlatformWorkflowInfo(
                    workflow_id=str(workflow["id"]),
                    name=workflow["name"],
                    description=workflow.get("description"),
                    status=WorkflowStatus.ACTIVE if workflow.get("active", False) else WorkflowStatus.INACTIVE,
                    trigger_type="unknown",
                    created_at=datetime.fromisoformat(
                        workflow["createdAt"].replace("Z", "+00:00")
                    ) if "createdAt" in workflow else None,
                    updated_at=datetime.fromisoformat(
                        workflow["updatedAt"].replace("Z", "+00:00")
                    ) if "updatedAt" in workflow else None,
                    version=str(workflow.get("versionId", "1")),
                    tags=workflow.get("tags", []),
                    input_schema=None,
                    output_schema=None
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get workflow: {e}")
            return None
    
    async def get_workflow_count(
        self,
        search_term: Optional[str] = None,
        status_filter: Optional[WorkflowStatus] = None
    ) -> int:
        """Get total count of workflows matching criteria"""
        workflows = await self.list_workflows(
            limit=250,  # n8n API max
            offset=0,
            search_term=search_term,
            status_filter=status_filter
        )
        return len(workflows)
    
    async def test_workflow(
        self,
        workflow_id: str,
        test_inputs: Dict[str, Any],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Test execute a workflow with sample data"""
        creds = getattr(self, '_credentials', {})
        base_url = self._get_base_url(creds)
        
        try:
            async with httpx.AsyncClient() as client:
                # Execute workflow with test data
                response = await client.post(
                    f"{base_url}/api/v1/workflows/{workflow_id}/execute",
                    json={"data": test_inputs},
                    headers=self._get_headers(creds),
                    auth=self._get_auth(creds),
                    timeout=float(timeout)
                )
                
                response.raise_for_status()
                result = response.json()
                
                return {
                    "output": result.get("data", {}),
                    "logs": []  # n8n doesn't return logs in test mode
                }
                
        except httpx.TimeoutError:
            raise Exception(f"Test execution timed out after {timeout} seconds")
        except Exception as e:
            raise Exception(f"Test execution failed: {str(e)}")