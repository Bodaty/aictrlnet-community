"""Make.com (Integromat) platform adapter implementation"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx

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


class MakeEndpoints:
    """Make.com API endpoints"""
    BASE_URL = "https://api.make.com/v2"
    SCENARIOS = "/scenarios"
    EXECUTIONS = "/executions"
    WEBHOOKS = "/webhooks"
    ORGANIZATIONS = "/organizations"


@register_adapter(PlatformType.MAKE)
class MakeAdapter(BasePlatformAdapter):
    """Adapter for Make.com platform integration"""
    
    def get_platform_type(self) -> PlatformType:
        return PlatformType.MAKE
    
    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return [AuthMethod.API_KEY]
    
    def get_capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=PlatformType.MAKE,
            supports_webhooks=True,
            supports_scheduling=True,
            supports_versioning=True,
            supports_rollback=True,
            supports_monitoring=True,
            max_execution_time=40 * 60,  # 40 minutes max
            rate_limits={
                "executions_per_minute": 60,
                "scenarios_per_organization": 1000,
                "operations_per_month": 1000  # Free tier
            },
            available_triggers=[
                "webhook",
                "schedule",
                "watch_records",
                "watch_emails",
                "watch_files",
                "instant_trigger",
                "poll_trigger"
            ],
            available_actions=[
                "http_request",
                "transform_data",
                "filter",
                "router",
                "aggregator",
                "iterator",
                "error_handler",
                "data_store",
                "custom_js"
            ]
        )
    
    async def validate_credentials(
        self,
        credentials: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate Make.com credentials"""
        try:
            api_key = credentials.get("api_key")
            if not api_key:
                return False, "API key is required"
            
            # Make.com requires organization ID in some endpoints
            org_id = credentials.get("organization_id")
            
            async with httpx.AsyncClient() as client:
                # Test with organizations endpoint
                response = await client.get(
                    f"{MakeEndpoints.BASE_URL}{MakeEndpoints.ORGANIZATIONS}",
                    headers={
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    return True, None
                elif response.status_code == 401:
                    return False, "Invalid API key"
                else:
                    return False, f"Authentication failed: {response.status_code}"
                    
        except Exception as e:
            logger.error(f"Credential validation error: {e}")
            return False, str(e)
    
    async def list_workflows(
        self,
        credentials: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[PlatformWorkflowInfo]:
        """List available scenarios"""
        try:
            api_key = credentials.get("api_key")
            org_id = credentials.get("organization_id")
            
            params = {
                "limit": limit,
                "offset": offset
            }
            
            if org_id:
                params["organizationId"] = org_id
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MakeEndpoints.BASE_URL}{MakeEndpoints.SCENARIOS}",
                    headers={
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "application/json"
                    },
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to list scenarios: {response.status_code}")
                    return []
                
                data = response.json()
                workflows = []
                
                for scenario in data.get("scenarios", []):
                    workflows.append(PlatformWorkflowInfo(
                        workflow_id=str(scenario["id"]),
                        name=scenario["name"],
                        description=scenario.get("description", ""),
                        status=WorkflowStatus.ACTIVE if scenario.get("isEnabled") else WorkflowStatus.INACTIVE,
                        trigger_type=self._extract_trigger_type(scenario),
                        created_at=datetime.fromisoformat(scenario["created"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(scenario["updated"].replace("Z", "+00:00")),
                        metadata={
                            "team_id": scenario.get("teamId"),
                            "is_paused": scenario.get("isPaused", False),
                            "last_run": scenario.get("lastRun"),
                            "webhook_url": self._extract_webhook_url(scenario)
                        }
                    ))
                
                return workflows
                
        except Exception as e:
            logger.error(f"Error listing scenarios: {e}")
            return []
    
    async def get_workflow_info(
        self,
        credentials: Dict[str, Any],
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get information about a specific scenario"""
        try:
            api_key = credentials.get("api_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MakeEndpoints.BASE_URL}{MakeEndpoints.SCENARIOS}/{workflow_id}",
                    headers={
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get scenario info: {response.status_code}")
                    return None
                
                scenario = response.json().get("scenario", {})
                
                return PlatformWorkflowInfo(
                    workflow_id=str(scenario["id"]),
                    name=scenario["name"],
                    description=scenario.get("description", ""),
                    status=WorkflowStatus.ACTIVE if scenario.get("isEnabled") else WorkflowStatus.INACTIVE,
                    trigger_type=self._extract_trigger_type(scenario),
                    created_at=datetime.fromisoformat(scenario["created"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(scenario["updated"].replace("Z", "+00:00")),
                    metadata={
                        "team_id": scenario.get("teamId"),
                        "is_paused": scenario.get("isPaused", False),
                        "scheduling": scenario.get("scheduling"),
                        "webhook_url": self._extract_webhook_url(scenario),
                        "modules": len(scenario.get("modules", []))
                    }
                )
                
        except Exception as e:
            logger.error(f"Error getting scenario info: {e}")
            return None
    
    async def execute_workflow(
        self,
        credentials: Dict[str, Any],
        request: ExecutionRequest
    ) -> ExecutionResponse:
        """Execute a Make.com scenario"""
        start_time = datetime.utcnow()
        
        try:
            api_key = credentials.get("api_key")
            
            # Make.com can execute scenarios via API or webhook
            if request.metadata.get("webhook_url"):
                # Execute via webhook
                webhook_url = request.metadata["webhook_url"]
                
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        webhook_url,
                        json=request.input_data,
                        timeout=request.timeout or 300.0
                    )
                    
                    completed_at = datetime.utcnow()
                    duration_ms = int((completed_at - start_time).total_seconds() * 1000)
                    
                    if response.status_code in [200, 201, 202]:
                        return ExecutionResponse(
                            execution_id=f"make-{datetime.utcnow().timestamp()}",
                            status=ExecutionStatus.COMPLETED,
                            output_data={"webhook_response": response.text},
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                            metadata={
                                "status_code": response.status_code,
                                "execution_type": "webhook"
                            }
                        )
                    else:
                        return ExecutionResponse(
                            execution_id=f"make-error-{datetime.utcnow().timestamp()}",
                            status=ExecutionStatus.FAILED,
                            error=f"Webhook execution failed: {response.status_code}",
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms
                        )
            else:
                # Execute via API (requires scenario to be enabled)
                async with httpx.AsyncClient() as client:
                    response = await client.post(
                        f"{MakeEndpoints.BASE_URL}{MakeEndpoints.SCENARIOS}/{request.workflow_id}/run",
                        headers={
                            "Authorization": f"Token {api_key}",
                            "Content-Type": "application/json"
                        },
                        json={"data": request.input_data},
                        timeout=request.timeout or 300.0
                    )
                    
                    completed_at = datetime.utcnow()
                    duration_ms = int((completed_at - start_time).total_seconds() * 1000)
                    
                    if response.status_code in [200, 201]:
                        execution_data = response.json()
                        return ExecutionResponse(
                            execution_id=str(execution_data.get("executionId", f"make-{datetime.utcnow().timestamp()}")),
                            status=ExecutionStatus.COMPLETED,
                            output_data=execution_data.get("data", {}),
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms,
                            metadata={
                                "status_code": response.status_code,
                                "execution_type": "api",
                                "operations_used": execution_data.get("operationsUsed")
                            }
                        )
                    else:
                        error_msg = f"Execution failed: {response.status_code}"
                        try:
                            error_data = response.json()
                            error_msg = error_data.get("message", error_msg)
                        except:
                            pass
                        
                        return ExecutionResponse(
                            execution_id=f"make-error-{datetime.utcnow().timestamp()}",
                            status=ExecutionStatus.FAILED,
                            error=error_msg,
                            started_at=start_time,
                            completed_at=completed_at,
                            duration_ms=duration_ms
                        )
                    
        except httpx.TimeoutException:
            return ExecutionResponse(
                execution_id=f"make-timeout-{datetime.utcnow().timestamp()}",
                status=ExecutionStatus.FAILED,
                error="Execution timeout",
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Make.com execution error: {e}")
            return ExecutionResponse(
                execution_id=f"make-error-{datetime.utcnow().timestamp()}",
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
        """Get status of a scenario execution"""
        try:
            api_key = credentials.get("api_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MakeEndpoints.BASE_URL}{MakeEndpoints.EXECUTIONS}/{execution_id}",
                    headers={
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    return None
                
                execution = response.json().get("execution", {})
                
                # Map Make.com status to our ExecutionStatus
                status_map = {
                    "success": ExecutionStatus.COMPLETED,
                    "error": ExecutionStatus.FAILED,
                    "running": ExecutionStatus.RUNNING,
                    "waiting": ExecutionStatus.RUNNING,
                    "warning": ExecutionStatus.COMPLETED
                }
                
                return ExecutionResponse(
                    execution_id=execution_id,
                    status=status_map.get(execution.get("status"), ExecutionStatus.UNKNOWN),
                    output_data=execution.get("data", {}),
                    error=execution.get("error"),
                    started_at=datetime.fromisoformat(execution["createdAt"].replace("Z", "+00:00")),
                    completed_at=datetime.fromisoformat(execution["finishedAt"].replace("Z", "+00:00")) if execution.get("finishedAt") else None,
                    metadata={
                        "operations_used": execution.get("operationsUsed"),
                        "scenario_id": execution.get("scenarioId"),
                        "status_detail": execution.get("status")
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
        """Cancel a running scenario execution"""
        try:
            api_key = credentials.get("api_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{MakeEndpoints.BASE_URL}{MakeEndpoints.EXECUTIONS}/{execution_id}/cancel",
                    headers={
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "application/json"
                    },
                    timeout=10.0
                )
                
                return response.status_code in [200, 204]
                
        except Exception as e:
            logger.error(f"Error cancelling execution: {e}")
            return False
    
    async def health_check(
        self,
        credentials: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResult:
        """Check Make.com platform health"""
        start_time = datetime.utcnow()
        
        try:
            # If credentials provided, do authenticated health check
            if credentials and credentials.get("api_key"):
                api_key = credentials["api_key"]
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{MakeEndpoints.BASE_URL}{MakeEndpoints.ORGANIZATIONS}",
                        headers={
                            "Authorization": f"Token {api_key}",
                            "Content-Type": "application/json"
                        },
                        timeout=10.0
                    )
                    
                    response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    return HealthCheckResult(
                        is_healthy=response.status_code == 200,
                        response_time_ms=response_time,
                        details={
                            "authenticated": True,
                            "status_code": response.status_code
                        },
                        error=None if response.status_code == 200 else f"Health check failed: {response.status_code}"
                    )
            else:
                # Basic connectivity check
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://status.make.com/api/v2/status.json",
                        timeout=10.0
                    )
                    
                    response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                    
                    if response.status_code == 200:
                        status_data = response.json()
                        is_healthy = status_data.get("status", {}).get("indicator") in ["none", "minor"]
                        
                        return HealthCheckResult(
                            is_healthy=is_healthy,
                            response_time_ms=response_time,
                            details={
                                "status": status_data.get("status", {}).get("description", "Unknown"),
                                "updated_at": status_data.get("page", {}).get("updated_at")
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
    
    def _extract_trigger_type(self, scenario: Dict[str, Any]) -> str:
        """Extract trigger type from scenario data"""
        modules = scenario.get("modules", [])
        if modules and len(modules) > 0:
            trigger = modules[0]
            return trigger.get("type", "unknown")
        return "unknown"
    
    def _extract_webhook_url(self, scenario: Dict[str, Any]) -> Optional[str]:
        """Extract webhook URL if scenario has webhook trigger"""
        modules = scenario.get("modules", [])
        if modules and len(modules) > 0:
            trigger = modules[0]
            if trigger.get("type") == "webhook":
                return trigger.get("webhook", {}).get("url")
        return None
    
    def configure(self, credentials: Dict[str, Any]) -> None:
        """Configure adapter with credentials"""
        # Store credentials for later use
        self._credentials = credentials
        logger.info("Make.com adapter configured")
    
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
            api_key = credentials.get("api_key")
            org_id = credentials.get("organization_id")
            
            params = {"limit": 1}  # Just get count, not all data
            if org_id:
                params["organizationId"] = org_id
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{MakeEndpoints.BASE_URL}{MakeEndpoints.SCENARIOS}",
                    headers={
                        "Authorization": f"Token {api_key}",
                        "Content-Type": "application/json"
                    },
                    params=params,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Make.com returns pagination info
                    return data.get("pagination", {}).get("total", 0)
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