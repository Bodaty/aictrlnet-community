"""IFTTT (If This Then That) platform adapter implementation"""
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


class IFTTTEndpoints:
    """IFTTT API endpoints"""
    BASE_URL = "https://api.ifttt.com/v2"
    WEBHOOK_URL = "https://maker.ifttt.com/trigger"
    USER = "/user"
    APPLETS = "/applets"
    SERVICES = "/services"
    TRIGGERS = "/triggers"


@register_adapter(PlatformType.IFTTT)
class IFTTTAdapter(BasePlatformAdapter):
    """Adapter for IFTTT platform integration"""
    
    def get_platform_type(self) -> PlatformType:
        return PlatformType.IFTTT
    
    def get_supported_auth_methods(self) -> List[AuthMethod]:
        return [AuthMethod.API_KEY]
    
    def get_capabilities(self) -> PlatformCapabilities:
        return PlatformCapabilities(
            platform=PlatformType.IFTTT,
            supports_webhooks=True,
            supports_scheduling=False,  # IFTTT doesn't have built-in scheduling
            supports_versioning=False,
            supports_rollback=False,
            supports_monitoring=False,  # Limited monitoring capabilities
            max_execution_time=15,  # IFTTT has quick execution
            rate_limits={
                "webhooks_per_hour": 100,  # Maker webhooks limit
                "applets_per_account": 3,  # Free tier
                "triggers_per_applet": 1,
                "actions_per_applet": 1
            },
            available_triggers=[
                "webhook",
                "time_based",
                "weather",
                "location",
                "email",
                "social_media",
                "smart_home",
                "calendar",
                "rss"
            ],
            available_actions=[
                "webhook",
                "email",
                "sms",
                "notification",
                "social_post",
                "smart_home_control",
                "spreadsheet_row",
                "note",
                "calendar_event"
            ]
        )
    
    async def validate_credentials(
        self,
        credentials: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate IFTTT credentials"""
        try:
            # IFTTT uses service key for API access
            service_key = credentials.get("service_key")
            if service_key:
                # Service key validation (for IFTTT Platform)
                headers = {
                    "IFTTT-Service-Key": service_key,
                    "Content-Type": "application/json"
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{IFTTTEndpoints.BASE_URL}/status",
                        headers=headers,
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        return True, None
                    elif response.status_code == 401:
                        return False, "Invalid service key"
                    else:
                        return False, f"Authentication failed: {response.status_code}"
            
            # For webhooks, we just need the webhook key
            webhook_key = credentials.get("webhook_key")
            if webhook_key:
                # Test webhook key with a test event
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        f"{IFTTTEndpoints.WEBHOOK_URL}/test/with/key/{webhook_key}",
                        timeout=10.0
                    )
                    
                    if response.status_code == 200:
                        return True, None
                    elif response.status_code == 401:
                        return False, "Invalid webhook key"
                    else:
                        return False, f"Webhook key validation failed: {response.status_code}"
            
            return False, "Either service_key or webhook_key is required"
                    
        except Exception as e:
            logger.error(f"Credential validation error: {e}")
            return False, str(e)
    
    async def list_workflows(
        self,
        credentials: Dict[str, Any],
        limit: int = 100,
        offset: int = 0
    ) -> List[PlatformWorkflowInfo]:
        """List available applets"""
        try:
            service_key = credentials.get("service_key")
            
            if not service_key:
                # Without service key, we can't list applets
                # Return webhook-based "workflows"
                webhook_key = credentials.get("webhook_key")
                if webhook_key:
                    # Return generic webhook workflows
                    return [
                        PlatformWorkflowInfo(
                            workflow_id="webhook_trigger",
                            name="Webhook Trigger",
                            description="Trigger any IFTTT applet via webhook",
                            status=WorkflowStatus.ACTIVE,
                            trigger_type="webhook",
                            created_at=datetime.utcnow(),
                            updated_at=datetime.utcnow(),
                            metadata={
                                "webhook_key": webhook_key,
                                "webhook_url_template": f"{IFTTTEndpoints.WEBHOOK_URL}/{{event}}/with/key/{webhook_key}"
                            }
                        )
                    ]
                return []
            
            # With service key, list actual applets
            headers = {
                "IFTTT-Service-Key": service_key,
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{IFTTTEndpoints.BASE_URL}{IFTTTEndpoints.APPLETS}",
                    headers=headers,
                    params={
                        "limit": limit,
                        "offset": offset
                    },
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to list applets: {response.status_code}")
                    return []
                
                data = response.json()
                workflows = []
                
                for applet in data.get("data", []):
                    workflows.append(PlatformWorkflowInfo(
                        workflow_id=applet["id"],
                        name=applet.get("name", "Unnamed Applet"),
                        description=applet.get("description", ""),
                        status=WorkflowStatus.ACTIVE if applet.get("status") == "enabled" else WorkflowStatus.INACTIVE,
                        trigger_type=self._extract_trigger_type(applet),
                        created_at=datetime.fromisoformat(applet.get("created_at", datetime.utcnow().isoformat())),
                        updated_at=datetime.fromisoformat(applet.get("updated_at", datetime.utcnow().isoformat())),
                        metadata={
                            "user_id": applet.get("user_id"),
                            "trigger_service": applet.get("trigger", {}).get("service", {}).get("name"),
                            "action_service": applet.get("action", {}).get("service", {}).get("name")
                        }
                    ))
                
                return workflows
                
        except Exception as e:
            logger.error(f"Error listing applets: {e}")
            return []
    
    async def get_workflow_info(
        self,
        credentials: Dict[str, Any],
        workflow_id: str
    ) -> Optional[PlatformWorkflowInfo]:
        """Get information about a specific applet"""
        try:
            service_key = credentials.get("service_key")
            
            if not service_key:
                # For webhook-only access
                if workflow_id == "webhook_trigger":
                    webhook_key = credentials.get("webhook_key")
                    return PlatformWorkflowInfo(
                        workflow_id="webhook_trigger",
                        name="Webhook Trigger",
                        description="Trigger any IFTTT applet via webhook",
                        status=WorkflowStatus.ACTIVE,
                        trigger_type="webhook",
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        metadata={
                            "webhook_key": webhook_key,
                            "webhook_url_template": f"{IFTTTEndpoints.WEBHOOK_URL}/{{event}}/with/key/{webhook_key}"
                        }
                    )
                return None
            
            headers = {
                "IFTTT-Service-Key": service_key,
                "Content-Type": "application/json"
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{IFTTTEndpoints.BASE_URL}{IFTTTEndpoints.APPLETS}/{workflow_id}",
                    headers=headers,
                    timeout=30.0
                )
                
                if response.status_code != 200:
                    logger.error(f"Failed to get applet info: {response.status_code}")
                    return None
                
                applet = response.json().get("data", {})
                
                return PlatformWorkflowInfo(
                    workflow_id=applet["id"],
                    name=applet.get("name", "Unnamed Applet"),
                    description=applet.get("description", ""),
                    status=WorkflowStatus.ACTIVE if applet.get("status") == "enabled" else WorkflowStatus.INACTIVE,
                    trigger_type=self._extract_trigger_type(applet),
                    created_at=datetime.fromisoformat(applet.get("created_at", datetime.utcnow().isoformat())),
                    updated_at=datetime.fromisoformat(applet.get("updated_at", datetime.utcnow().isoformat())),
                    metadata={
                        "user_id": applet.get("user_id"),
                        "trigger": applet.get("trigger"),
                        "action": applet.get("action"),
                        "run_count": applet.get("run_count", 0)
                    }
                )
                
        except Exception as e:
            logger.error(f"Error getting applet info: {e}")
            return None
    
    async def execute_workflow(
        self,
        credentials: Dict[str, Any],
        request: ExecutionRequest
    ) -> ExecutionResponse:
        """Execute an IFTTT applet via webhook"""
        start_time = datetime.utcnow()
        
        try:
            webhook_key = credentials.get("webhook_key")
            if not webhook_key:
                return ExecutionResponse(
                    execution_id=f"ifttt-error-{datetime.utcnow().timestamp()}",
                    status=ExecutionStatus.FAILED,
                    error="Webhook key is required for execution",
                    started_at=start_time,
                    completed_at=datetime.utcnow()
                )
            
            # IFTTT webhook executions require an event name
            event_name = request.metadata.get("event_name", "trigger")
            
            # IFTTT webhooks accept up to 3 values
            webhook_data = {}
            if isinstance(request.input_data, dict):
                # Map first 3 keys to value1, value2, value3
                for i, (key, value) in enumerate(request.input_data.items()):
                    if i < 3:
                        webhook_data[f"value{i+1}"] = str(value)
            elif isinstance(request.input_data, list) and len(request.input_data) > 0:
                # Map list items to value1, value2, value3
                for i in range(min(3, len(request.input_data))):
                    webhook_data[f"value{i+1}"] = str(request.input_data[i])
            
            webhook_url = f"{IFTTTEndpoints.WEBHOOK_URL}/{event_name}/with/key/{webhook_key}"
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    webhook_url,
                    json=webhook_data,
                    timeout=request.timeout or 30.0
                )
                
                completed_at = datetime.utcnow()
                duration_ms = int((completed_at - start_time).total_seconds() * 1000)
                
                if response.status_code == 200:
                    # IFTTT returns a simple text response
                    response_text = response.text.strip()
                    
                    return ExecutionResponse(
                        execution_id=f"ifttt-{event_name}-{datetime.utcnow().timestamp()}",
                        status=ExecutionStatus.COMPLETED,
                        output_data={
                            "message": response_text,
                            "event": event_name,
                            "values_sent": webhook_data
                        },
                        started_at=start_time,
                        completed_at=completed_at,
                        duration_ms=duration_ms,
                        metadata={
                            "webhook_url": webhook_url,
                            "event_name": event_name
                        }
                    )
                else:
                    return ExecutionResponse(
                        execution_id=f"ifttt-error-{datetime.utcnow().timestamp()}",
                        status=ExecutionStatus.FAILED,
                        error=f"Webhook execution failed: {response.status_code} - {response.text}",
                        started_at=start_time,
                        completed_at=completed_at,
                        duration_ms=duration_ms
                    )
                    
        except httpx.TimeoutException:
            return ExecutionResponse(
                execution_id=f"ifttt-timeout-{datetime.utcnow().timestamp()}",
                status=ExecutionStatus.FAILED,
                error="Execution timeout",
                started_at=start_time,
                completed_at=datetime.utcnow()
            )
        except Exception as e:
            logger.error(f"IFTTT execution error: {e}")
            return ExecutionResponse(
                execution_id=f"ifttt-error-{datetime.utcnow().timestamp()}",
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
        """Get status of an IFTTT execution"""
        # IFTTT doesn't provide execution status tracking
        logger.warning("IFTTT does not support execution status tracking")
        return None
    
    async def cancel_execution(
        self,
        credentials: Dict[str, Any],
        execution_id: str
    ) -> bool:
        """Cancel a running IFTTT execution"""
        # IFTTT doesn't support cancelling executions
        logger.warning("IFTTT does not support cancelling executions")
        return False
    
    async def health_check(
        self,
        credentials: Optional[Dict[str, Any]] = None
    ) -> HealthCheckResult:
        """Check IFTTT platform health"""
        start_time = datetime.utcnow()
        
        try:
            # IFTTT doesn't have a public status API
            # We'll do a basic connectivity check
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://ifttt.com",
                    timeout=10.0,
                    follow_redirects=True
                )
                
                response_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                if response.status_code == 200:
                    return HealthCheckResult(
                        is_healthy=True,
                        response_time_ms=response_time,
                        details={
                            "status": "IFTTT website is accessible",
                            "checked_at": datetime.utcnow().isoformat()
                        }
                    )
                else:
                    return HealthCheckResult(
                        is_healthy=False,
                        response_time_ms=response_time,
                        error=f"IFTTT website returned status {response.status_code}"
                    )
                    
        except Exception as e:
            return HealthCheckResult(
                is_healthy=False,
                response_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                error=str(e)
            )
    
    def _extract_trigger_type(self, applet: Dict[str, Any]) -> str:
        """Extract trigger type from applet data"""
        trigger = applet.get("trigger", {})
        if trigger:
            service = trigger.get("service", {})
            return service.get("name", "unknown")
        return "unknown"
    
    def configure(self, credentials: Dict[str, Any]) -> None:
        """Configure adapter with credentials"""
        # Store credentials for later use
        self._credentials = credentials
        logger.info("IFTTT adapter configured")
    
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
            service_key = credentials.get("service_key")
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{IFTTTEndpoints.BASE_URL}{IFTTTEndpoints.APPLETS}",
                    headers={"IFTTT-Service-Key": service_key},
                    params={"limit": 1},  # Just get count
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # IFTTT returns data array, need to get total from headers or pagination
                    return len(data.get("data", []))  # This is simplified, real API may have pagination info
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