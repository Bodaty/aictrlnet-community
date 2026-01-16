"""Microsoft Power Automate platform adapter implementation"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
import json

from schemas.platform_integration import (
    PlatformType,
    AuthMethod,
    PlatformCapabilities,
    PlatformWorkflowInfo,
    WorkflowStatus,
    ExecutionStatus
)
from .base import (
    BasePlatformAdapter,
    ExecutionRequest,
    ExecutionResponse,
    HealthCheckResult
)
from .registry import register_adapter

logger = logging.getLogger(__name__)


class PowerAutomateEndpoints:
    """Power Automate API endpoints"""
    BASE_URL = "https://api.flow.microsoft.com"
    GRAPH_URL = "https://graph.microsoft.com/v1.0"
    ENVIRONMENTS = "/providers/Microsoft.ProcessSimple/environments"
    FLOWS = "/flows"
    RUNS = "/runs"
    TRIGGERS = "/triggers"


@register_adapter(PlatformType.POWER_AUTOMATE)
class PowerAutomateAdapter(BasePlatformAdapter):
    """Adapter for Microsoft Power Automate platform integration"""
    
    def get_platform_type(self) -> PlatformType:
        return PlatformType.POWER_AUTOMATE
    
    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return [AuthMethod.OAUTH2, AuthMethod.API_KEY]
    
    def get_capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=PlatformType.POWER_AUTOMATE,
            supports_webhooks=True,
            supports_scheduling=True,
            supports_versioning=True,
            supports_rollback=False,
            supports_monitoring=True,
            max_execution_time=30 * 24 * 60 * 60,  # 30 days max
            rate_limits={
                "executions_per_5min": 100000,  # Per user
                "api_calls_per_day": 25000,     # Per user per day
                "flows_per_user": 600            # Max flows per user
            },
            available_triggers=[
                "http_request",
                "recurrence",
                "email_received",
                "file_created",
                "form_submitted",
                "sharepoint_item",
                "teams_message",
                "calendar_event",
                "button"
            ],
            available_actions=[
                "http",
                "email",
                "approval",
                "create_file",
                "update_excel",
                "post_teams",
                "create_task",
                "condition",
                "apply_to_each",
                "scope",
                "terminate"
            ]
        )
    
    async def validate_credentials(
        self,
        credentials: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate Power Automate credentials"""
        try:
            auth_method = credentials.get("auth_method", "oauth2")
            
            if auth_method == "oauth2":
                access_token = credentials.get("access_token")
                if not access_token:
                    return False, "Access token is required for OAuth2"
                
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                
                # Test with environments endpoint
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}",
                        headers=headers,
                        params={"api-version": "2016-11-01"},
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        return True, None
                    elif response.status_code == 401:
                        return False, "Invalid or expired access token"
                    else:
                        return False, f"Authentication failed: {response.status_code}"
            
            elif auth_method == "api_key":
                # For service accounts or app-only auth
                api_key = credentials.get("api_key")
                tenant_id = credentials.get("tenant_id")
                
                if not api_key or not tenant_id:
                    return False, "API key and tenant ID are required"
                
                # This would typically use client credentials flow
                return True, None
            
            else:
                return False, f"Unsupported auth method: {auth_method}"
                
        except Exception as e:
            logger.error(f"Credential validation error: {e}")
            return False, str(e)
    
    async def list_workflows(
        self,
        credentials: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[PlatformWorkflowInfo]:
        """List available flows"""
        try:
            access_token = credentials.get("access_token")
            environment_id = credentials.get("environment_id")
            
            if not environment_id:
                # Get default environment
                environment_id = await self._get_default_environment(access_token)
                if not environment_id:
                    logger.error("No environment found")
                    return []
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}/{environment_id}{PowerAutomateEndpoints.FLOWS}",
                    headers=headers,
                    params={
                        "api-version": "2016-11-01",
                        "$top": limit,
                        "$skip": offset
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to list flows: {response.status_code}")
                    return []
                
                data = response.json()
                workflows = []
                
                for flow in data.get("value", []):
                    properties = flow.get("properties", {})
                    
                    workflows.append(PlatformWorkflowInfo(
                        workflow_id=flow["name"],
                        name=properties.get("displayName", ""),
                        description=properties.get("description", ""),
                        status=WorkflowStatus.ACTIVE if properties.get("state") == "Started" else WorkflowStatus.INACTIVE,
                        trigger_type=self._extract_trigger_type(properties),
                        created_at=datetime.fromisoformat(properties.get("createdTime", "").replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(properties.get("lastModifiedTime", "").replace("Z", "+00:00")),
                        metadata={
                            "environment_id": environment_id,
                            "flow_id": flow["id"],
                            "api_id": properties.get("apiId"),
                            "trigger_url": self._extract_trigger_url(properties)
                        }
                    ))
                
                return workflows
                
        except Exception as e:
            logger.error(f"Error listing flows: {e}")
            return []
    
    async def get_workflow_info(
        self,
        credentials: Dict[str, Any],
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get information about a specific flow"""
        try:
            access_token = credentials.get("access_token")
            environment_id = credentials.get("environment_id")
            
            if not environment_id:
                environment_id = await self._get_default_environment(access_token)
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}/{environment_id}{PowerAutomateEndpoints.FLOWS}/{workflow_id}",
                    headers=headers,
                    params={"api-version": "2016-11-01"},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get flow info: {response.status_code}")
                    return None
                
                flow = response.json()
                properties = flow.get("properties", {})
                
                return PlatformWorkflowInfo(
                    workflow_id=flow["name"],
                    name=properties.get("displayName", ""),
                    description=properties.get("description", ""),
                    status=WorkflowStatus.ACTIVE if properties.get("state") == "Started" else WorkflowStatus.INACTIVE,
                    trigger_type=self._extract_trigger_type(properties),
                    created_at=datetime.fromisoformat(properties.get("createdTime", "").replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(properties.get("lastModifiedTime", "").replace("Z", "+00:00")),
                    metadata={
                        "environment_id": environment_id,
                        "flow_id": flow["id"],
                        "api_id": properties.get("apiId"),
                        "definition": properties.get("definition"),
                        "connection_references": properties.get("connectionReferences"),
                        "trigger_url": self._extract_trigger_url(properties)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error getting flow info: {e}")
            return None
    
    async def execute_workflow(
        self,
        credentials: Dict[str, Any],
        request: ExecutionRequest
    ) -> ExecutionResponse:
        """Execute a Power Automate flow"""
        start_time = datetime.utcnow()
        
        try:
            # Power Automate flows can be triggered via HTTP trigger URL
            trigger_url = request.metadata.get("trigger_url")
            
            if not trigger_url:
                # Try to get trigger URL from flow info
                workflow_info = await self.get_workflow_info(credentials, request.workflow_id)
                if workflow_info and workflow_info.metadata:
                    trigger_url = workflow_info.metadata.get("trigger_url")
            
            if trigger_url:
                # Execute via HTTP trigger
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        trigger_url,
                        json=request.input_data,
                        timeout=request.timeout or 300.0
                    )
                    
                    completed_at = datetime.utcnow()
                    duration_ms = int((completed_at - start_time).total_seconds() * 1000)
                    
                    if response.status_code in [200, 202]:
                        # Power Automate typically returns 202 Accepted
                        run_id = response.headers.get("x-ms-workflow-run-id", f"run-{datetime.utcnow().timestamp()}")
                        
                        return ExecutionResponse(
                            execution_id=run_id,
                            status=ExecutionStatus.RUNNING if response.status_code == 202 else ExecutionStatus.COMPLETED,
                            output_data={"status": "accepted", "run_id": run_id},
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                            metadata={
                                "status_code": response.status_code,
                                "workflow_id": response.headers.get("x-ms-workflow-id"),
                                "trigger_type": "http"
                            }
                        )
                    else:
                        return ExecutionResponse(
                            execution_id=f"pa-error-{datetime.utcnow().timestamp()}",
                            status=ExecutionStatus.FAILED,
                            error=f"Trigger failed: {response.status_code}",
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms
                        )
            else:
                # Manual trigger via API
                access_token = credentials.get("access_token")
                environment_id = credentials.get("environment_id")
                
                if not environment_id:
                    environment_id = await self._get_default_environment(access_token)
                
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}/{environment_id}{PowerAutomateEndpoints.FLOWS}/{request.workflow_id}/triggers/manual/run",
                        headers=headers,
                        params={"api-version": "2016-11-01"},
                        json={"inputs": request.input_data},
                        timeout=request.timeout or 300.0
                    )
                    
                    completed_at = datetime.utcnow()
                    duration_ms = int((completed_at - start_time).total_seconds() * 1000)
                    
                    if response.status_code in [200, 202]:
                        run_data = response.json()
                        run_id = run_data.get("name", f"run-{datetime.utcnow().timestamp()}")
                        
                        return ExecutionResponse(
                            execution_id=run_id,
                            status=ExecutionStatus.RUNNING,
                            output_data=run_data,
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                            metadata={
                                "status_code": response.status_code,
                                "trigger_type": "manual"
                            }
                        )
                    else:
                        error_msg = f"Execution failed: {response.status_code}"
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("error", {}).get("message", error_msg)
                        except:
                            pass
                        
                        return ExecutionResponse(
                            execution_id=f"pa-error-{datetime.utcnow().timestamp()}",
                            status=ExecutionStatus.FAILED,
                            error=error_msg,
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms
                        )
                    
        except httpx.TimeoutException:
            return ExecutionResponse(
                execution_id=f"pa-timeout-{datetime.utcnow().timestamp()}",
                status=ExecutionStatus.FAILED,
                error="Execution timeout",
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Power Automate execution error: {e}")
            return ExecutionResponse(
                execution_id=f"pa-error-{datetime.utcnow().timestamp()}",
                status=ExecutionStatus.FAILED,
                error=str(e),
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
    
    async def get_execution_status(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> Optional[ExecutionResponse]:
        """Get status of a flow run"""
        try:
            access_token = credentials.get("access_token")
            environment_id = credentials.get("environment_id")
            workflow_id = credentials.get("workflow_id")  # Need flow ID to get run
            
            if not environment_id:
                environment_id = await self._get_default_environment(access_token)
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}/{environment_id}{PowerAutomateEndpoints.FLOWS}/{workflow_id}/runs/{execution_id}",
                    headers=headers,
                    params={"api-version": "2016-11-01"},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    return None
                
                run = response.json()
                properties = run.get("properties", {})
                
                # Map Power Automate status to our ExecutionStatus
                status_map = {
                    "Succeeded": ExecutionStatus.COMPLETED,
                    "Failed": ExecutionStatus.FAILED,
                    "Cancelled": ExecutionStatus.FAILED,
                    "Running": ExecutionStatus.RUNNING,
                    "Waiting": ExecutionStatus.RUNNING,
                    "Skipped": ExecutionStatus.COMPLETED
                }
                
                return ExecutionResponse(
                    execution_id=execution_id,
                    status=status_map.get(properties.get("status"), ExecutionStatus.UNKNOWN),
                    output_data=properties.get("outputs", {}),
                    error=properties.get("error", {}).get("message") if properties.get("error") else None,
                    started_at=datetime.fromisoformat(properties.get("startTime", "").replace("Z", "+00:00")),
                    completed_at=datetime.fromisoformat(properties.get("endTime", "").replace("Z", "+00:00")) if properties.get("endTime") else None,
                    metadata={
                        "status_detail": properties.get("status"),
                        "correlation_id": properties.get("correlation", {}).get("clientTrackingId"),
                        "trigger": properties.get("trigger", {}).get("name")
                    }
                )
                
        except Exception as e:
            logger.error(f"Error getting execution status: {e}")
            return None
    
    async def cancel_execution(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> bool:
        """Cancel a running flow execution"""
        try:
            access_token = credentials.get("access_token")
            environment_id = credentials.get("environment_id")
            workflow_id = credentials.get("workflow_id")
            
            if not environment_id:
                environment_id = await self._get_default_environment(access_token)
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}/{environment_id}{PowerAutomateEndpoints.FLOWS}/{workflow_id}/runs/{execution_id}/cancel",
                    headers=headers,
                    params={"api-version": "2016-11-01"},
                    timeout=10.0
                )
                
                return response.status_code in [200, 202]
                
        except Exception as e:
            logger.error(f"Error cancelling execution: {e}")
            return False
    
    async def health_check(
        self,
        credentials: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResult:
        """Check Power Automate platform health"""
        start_time = datetime.utcnow()
        
        try:
            # Check Microsoft 365 service health
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://status.office365.com/api/feed",
                    timeout=10.0
                )
                
                response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                if response.status_code == 200:
                    # Parse service health (would need proper XML parsing)
                    return HealthCheckResult(
                        is_healthy=True,
                        response_time_ms=response_time,
                        details={
                            "source": "Office 365 Status",
                            "checked_at": datetime.utcnow().isoformat()
                        }
                    )
                else:
                    return HealthCheckResult(
                        is_healthy=False,
                        response_time_ms=response_time,
                        error=f"Status check failed: {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                response_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                error=str(e)
            )
    
    async def _get_default_environment(self, access_token: str) -> Optional[str]:
        """Get the default environment ID"""
        try:
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}",
                    headers=headers,
                    params={"api-version": "2016-11-01"},
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    environments = response.json().get("value", [])
                    if environments:
                        # Return the default or first environment
                        for env in environments:
                            if env.get("properties", {}).get("isDefault"):
                                return env["name"]
                        return environments[0]["name"]
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting default environment: {e}")
            return None
    
    def _extract_trigger_type(self, properties: Dict[str, Any]) -> str:
        """Extract trigger type from flow properties"""
        definition = properties.get("definition", {})
        triggers = definition.get("triggers", {})
        
        if triggers:
            # Get the first trigger
            trigger_name = list(triggers.keys())[0]
            trigger = triggers[trigger_name]
            return trigger.get("type", "unknown")
        
        return "unknown"
    
    def _extract_trigger_url(self, properties: Dict[str, Any]) -> Optional[str]:
        """Extract HTTP trigger URL if available"""
        definition = properties.get("definition", {})
        triggers = definition.get("triggers", {})
        
        for trigger in triggers.values():
            if trigger.get("type") == "Request" and trigger.get("kind") == "Http":
                # The actual URL would be in the trigger configuration
                # This is a placeholder - actual implementation would need to parse properly
                return trigger.get("inputs", {}).get("schema", {}).get("url")
        
        return None
    
    def configure(self, credentials: Dict[str, Any]) -> None:
        """Configure adapter with credentials"""
        # Store credentials for later use
        self._credentials = credentials
        logger.info("Power Automate adapter configured")
    
    async def get_workflow(
        self,
        credentials: Dict[str, Any],
        workflow_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get workflow details"""
        # This is essentially the same as get_workflow_info but returns raw dict
        workflow_info = await self.get_workflow_info(credentials, workflow_id)
        if workflow_info:
            return workflow_info.dict()
        return None
    
    async def get_workflow_count(
        self,
        credentials: Dict[str, Any]
    ) -> int:
        """Get total count of workflows"""
        try:
            access_token = credentials.get("access_token")
            environment_id = credentials.get("environment_id")
            
            if not environment_id:
                environment_id = await self._get_default_environment(access_token)
                if not environment_id:
                    return 0
            
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{PowerAutomateEndpoints.BASE_URL}{PowerAutomateEndpoints.ENVIRONMENTS}/{environment_id}{PowerAutomateEndpoints.FLOWS}",
                    headers=headers,
                    params={
                        "api-version": "2016-11-01",
                        "$top": 1  # Just get count
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Power Automate returns total count in @odata.count
                    return data.get("@odata.count", len(data.get("value", [])))
                else:
                    logger.error(f"Failed to get workflow count: {response.status_code}")
                    return 0
                    
        except Exception as e:
            logger.error(f"Error getting workflow count: {e}")
            return 0
    
    async def test_workflow(
        self,
        credentials: Dict[str, Any],
        workflow_id: str,
        test_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Test a workflow with sample data"""
        # Execute the workflow with test data
        test_request = ExecutionRequest(
            workflow_id=workflow_id,
            input_data=test_data or {"test": True, "timestamp": datetime.utcnow().isoformat()},
            metadata={"is_test": True},
            timeout=30.0
        )
        
        result = await self.execute_workflow(credentials, test_request)
        
        return {
            "success": result.status == ExecutionStatus.COMPLETED,
            "execution_id": result.execution_id,
            "status": result.status.value,
            "output": result.output_data,
            "error": result.error,
            "duration_ms": result.duration_ms
        }