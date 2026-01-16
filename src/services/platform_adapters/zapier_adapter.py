"""Zapier platform adapter implementation"""
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import httpx
from enum import Enum

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


class ZapierEndpoints:
    """Zapier API endpoints"""
    BASE_URL = "https://api.zapier.com"
    HOOKS = "/v1/hooks"
    ZAP_RUNS = "/v1/zap-runs"
    ZAPS = "/v1/zaps"
    AUTH_TEST = "/v1/check"


@register_adapter(PlatformType.ZAPIER)
class ZapierAdapter(BasePlatformAdapter):
    """Adapter for Zapier platform integration"""
    
    def get_platform_type(self) -> PlatformType:
        return PlatformType.ZAPIER
    
    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return [AuthMethod.API_KEY, AuthMethod.OAUTH2]
    
    def get_capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=PlatformType.ZAPIER,
            supports_webhooks=True,
            supports_scheduling=True,
            supports_versioning=True,
            supports_rollback=False,
            supports_monitoring=True,
            max_execution_time=30,  # Zapier has 30 second timeout
            rate_limits={
                "executions_per_minute": 75,  # Free tier limit
                "webhooks_per_zap": 1,
                "tasks_per_month": 100  # Free tier
            },
            available_triggers=[
                "webhook",
                "schedule",
                "rss",
                "email",
                "form",
                "new_file",
                "updated_row",
                "new_email"
            ],
            available_actions=[
                "webhook",
                "email",
                "sms",
                "create_document",
                "update_spreadsheet",
                "post_social",
                "create_task",
                "send_notification"
            ]
        )
    
    async def validate_credentials(
        self,
        credentials: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate Zapier credentials"""
        try:
            api_key = credentials.get("api_key")
            if not api_key:
                return False, "API key is required"
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ZapierEndpoints.BASE_URL}{ZapierEndpoints.AUTH_TEST}",
                    headers={"X-API-Key": api_key},
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
        """List available Zaps"""
        try:
            api_key = credentials.get("api_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ZapierEndpoints.BASE_URL}{ZapierEndpoints.ZAPS}",
                    headers={"X-API-Key": api_key},
                    params={"limit": limit, "offset": offset},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to list Zaps: {response.status_code}")
                    return []
                
                data = response.json()
                workflows = []
                
                for zap in data.get("zaps", []):
                    workflows.append(PlatformWorkflowInfo(
                        workflow_id=zap["id"],
                        name=zap["title"],
                        description=zap.get("description", ""),
                        status=WorkflowStatus.ACTIVE if zap["state"] == "on" else WorkflowStatus.INACTIVE,
                        trigger_type=self._extract_trigger_type(zap),
                        created_at=datetime.fromisoformat(zap["created_at"].replace("Z", "+00:00")),
                        updated_at=datetime.fromisoformat(zap["modified_at"].replace("Z", "+00:00")),
                        metadata={
                            "steps": len(zap.get("steps", [])),
                            "last_run": zap.get("last_successful_run_date"),
                            "url": zap.get("url")
                        }
                    ))
                
                return workflows
                
        except Exception as e:
            logger.error(f"Error listing Zaps: {e}")
            return []
    
    async def get_workflow_info(
        self,
        credentials: Dict[str, Any],
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get information about a specific Zap"""
        try:
            api_key = credentials.get("api_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ZapierEndpoints.BASE_URL}{ZapierEndpoints.ZAPS}/{workflow_id}",
                    headers={"X-API-Key": api_key},
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get Zap info: {response.status_code}")
                    return None
                
                zap = response.json()
                
                return PlatformWorkflowInfo(
                    workflow_id=zap["id"],
                    name=zap["title"],
                    description=zap.get("description", ""),
                    status=WorkflowStatus.ACTIVE if zap["state"] == "on" else WorkflowStatus.INACTIVE,
                    trigger_type=self._extract_trigger_type(zap),
                    created_at=datetime.fromisoformat(zap["created_at"].replace("Z", "+00:00")),
                    updated_at=datetime.fromisoformat(zap["modified_at"].replace("Z", "+00:00")),
                    metadata={
                        "steps": len(zap.get("steps", [])),
                        "last_run": zap.get("last_successful_run_date"),
                        "url": zap.get("url"),
                        "sample_data": zap.get("sample_result")
                    }
                )
                
        except Exception as e:
            logger.error(f"Error getting Zap info: {e}")
            return None
    
    async def execute_workflow(
        self,
        credentials: Dict[str, Any],
        request: ExecutionRequest
    ) -> ExecutionResponse:
        """Execute a Zap via webhook"""
        start_time = datetime.utcnow()
        
        try:
            # Zapier executions are typically done via webhooks
            webhook_url = request.metadata.get("webhook_url")
            if not webhook_url:
                # Try to get webhook URL from workflow info
                workflow_info = await self.get_workflow_info(credentials, request.workflow_id)
                if workflow_info and workflow_info.metadata:
                    webhook_url = workflow_info.metadata.get("webhook_url")
            
            if not webhook_url:
                return ExecutionResponse(
                    execution_id=f"zapier-error-{datetime.utcnow().timestamp()}",
                    status=ExecutionStatus.FAILED,
                    error="Webhook URL not provided. Zapier workflows execute via webhooks.",
                    started_at=start_time,
                    completed_at=datetime.utcnow()
                )
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=request.input_data,
                    timeout=request.timeout or 30.0
                )
                
                completed_at = datetime.utcnow()
                duration_ms = int((completed_at - start_time).total_seconds() * 1000)
                
                if response.status_code in [200, 201]:
                    # Zapier typically returns a simple success response
                    response_data = {}
                    try:
                        response_data = response.json()
                    except:
                        response_data = {"status": "success", "response": response.text}
                    
                    return ExecutionResponse(
                        execution_id=response_data.get("id", f"zapier-{datetime.utcnow().timestamp()}"),
                        status=ExecutionStatus.COMPLETED,
                        output_data=response_data,
                        started_at=start_time,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        metadata={
                            "status_code": response.status_code,
                            "request_id": response.headers.get("X-Request-Id"),
                            "hook_id": response_data.get("hook_id")
                        }
                    )
                else:
                    error_msg = f"Webhook execution failed with status {response.status_code}"
                    try:
                        error_data = response.json()
                        error_msg = error_data.get("message", error_msg)
                    except:
                        error_msg += f": {response.text}"
                    
                    return ExecutionResponse(
                        execution_id=f"zapier-error-{datetime.utcnow().timestamp()}",
                        status=ExecutionStatus.FAILED,
                        error=error_msg,
                        started_at=start_time,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        metadata={
                            "status_code": response.status_code
                        }
                    )
                    
        except httpx.TimeoutException:
            return ExecutionResponse(
                execution_id=f"zapier-timeout-{datetime.utcnow().timestamp()}",
                status=ExecutionStatus.FAILED,
                error="Execution timeout",
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"Zapier execution error: {e}")
            return ExecutionResponse(
                execution_id=f"zapier-error-{datetime.utcnow().timestamp()}",
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
        """Get status of a Zap execution"""
        # Zapier doesn't provide real-time execution status via API
        # Would need to use their Partner API or webhooks for status updates
        logger.warning("Zapier execution status tracking requires Partner API access")
        return None
    
    async def cancel_execution(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> bool:
        """Cancel a running Zap execution"""
        # Zapier doesn't support cancelling executions
        logger.warning("Zapier does not support cancelling executions")
        return False
    
    async def health_check(
        self,
        credentials: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResult:
        """Check Zapier platform health"""
        start_time = datetime.utcnow()
        
        try:
            async with httpx.AsyncClient() as client:
                # Check Zapier status page API
                response = await client.get(
                    "https://status.zapier.com/api/v2/status.json",
                    timeout=10.0
                )
                
                response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                if response.status_code == 200:
                    status_data = response.json()
                    is_healthy = status_data.get("status", {}).get("indicator") == "none"
                    
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
    
    def _extract_trigger_type(self, zap: Dict[str, Any]) -> str:
        """Extract trigger type from Zap data"""
        steps = zap.get("steps", [])
        if steps and len(steps) > 0:
            trigger = steps[0]
            app = trigger.get("app", {})
            return app.get("title", "unknown")
        return "unknown"
    
    def configure(self, credentials: Dict[str, Any]) -> None:
        """Configure adapter with credentials"""
        # Store credentials for later use
        self._credentials = credentials
        logger.info("Zapier adapter configured")
    
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
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{ZapierEndpoints.BASE_URL}{ZapierEndpoints.ZAPS}",
                    headers={"X-API-Key": api_key},
                    params={"limit": 1},  # Just get count, not all data
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return data.get("total_count", 0)
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