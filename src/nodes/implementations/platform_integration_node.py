"""Platform Integration Node for external automation platforms"""
import json
from typing import Dict, Any, Optional
from datetime import datetime
import logging

from core.database import get_db
from models.platform_integration import PlatformExecution, PlatformType
from schemas.platform_integration import PlatformNodeConfig
from services.platform_credential_service import PlatformCredentialService
from services.platform_adapters import (
    PlatformAdapterService,
    ExecutionRequest,
    ExecutionStatus
)
from ..base_node import BaseNode
from ..models import NodeConfig, NodeExecutionResult, NodeStatus

logger = logging.getLogger(__name__)


class PlatformIntegrationNode(BaseNode):
    """Node for integrating with external automation platforms"""
    
    node_type = "platform"
    
    def __init__(self, node_id: str, config: Dict[str, Any]):
        super().__init__(node_id, config)
        
        # Validate and parse configuration
        try:
            self.platform_config = PlatformNodeConfig(**config)
        except Exception as e:
            raise ValueError(f"Invalid platform node configuration: {e}")
        
        # Initialize services (will be set in execute)
        self.credential_service = None
        self.adapter_service = None
        self.db = None
    
    async def validate_config(self) -> bool:
        """Validate node configuration"""
        try:
            # Check required fields
            if not self.platform_config.platform:
                logger.error("Missing platform in configuration")
                return False
            
            if not self.platform_config.workflow_id:
                logger.error("Missing workflow_id in configuration")
                return False
            
            if not self.platform_config.credential_id:
                logger.error("Missing credential_id in configuration")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Dict[str, Any]
    ) -> NodeExecutionResult:
        """Execute platform workflow"""
        start_time = datetime.utcnow()
        
        try:
            # Initialize services
            self.db = next(get_db())
            self.credential_service = PlatformCredentialService(self.db)
            self.adapter_service = PlatformAdapterService(self.db)
            
            # Get user from context
            user_id = context.user_id
            if not user_id:
                return NodeExecutionResult(
                    status=NodeStatus.FAILED,
                    error="User context not provided"
                )
            
            # Get credentials
            credential_data = await self.credential_service.get_credential_data(
                self.platform_config.credential_id,
                user_id
            )
            
            if not credential_data:
                return NodeExecutionResult(
                    status=NodeStatus.FAILED,
                    error="Platform credentials not found or unauthorized"
                )
            
            # Get platform adapter
            adapter = self.adapter_service.get_adapter_instance(
                self.platform_config.platform
            )
            
            if not adapter:
                return NodeExecutionResult(
                    status=NodeStatus.FAILED,
                    error=f"Platform adapter not found for {self.platform_config.platform.value}"
                )
            
            # Apply input mapping
            platform_input = self._apply_mapping(
                input_data,
                self.platform_config.input_mapping
            )
            
            # Create execution record
            db_execution = PlatformExecution(
                workflow_id=context.workflow_instance_id,
                node_id=self.node_id,
                execution_id=context.execution_id,
                platform=self.platform_config.platform.value,
                external_workflow_id=self.platform_config.workflow_id,
                credential_id=self.platform_config.credential_id,
                input_data=platform_input,
                status="running",
                started_at=start_time
            )
            self.db.add(db_execution)
            self.db.commit()
            
            # Execute on platform
            try:
                execution_request = ExecutionRequest(
                    workflow_id=self.platform_config.workflow_id,
                    input_data=platform_input,
                    timeout=self.platform_config.timeout,
                    retry_count=0,
                    metadata={
                        "node_id": self.node_id,
                        "workflow_instance_id": context.workflow_instance_id
                    }
                )
                
                response = await adapter.execute_workflow(
                    credential_data,
                    execution_request
                )
                
                # Update execution record
                db_execution.external_execution_id = response.execution_id
                db_execution.status = response.status.value
                db_execution.output_data = response.output_data
                db_execution.error_data = {"error": response.error} if response.error else None
                db_execution.completed_at = response.completed_at or datetime.utcnow()
                db_execution.duration_ms = response.duration_ms
                db_execution.estimated_cost = response.cost_estimate or 0
                
                # Update credential usage
                await self.credential_service.db.query(PlatformCredential).filter(
                    PlatformCredential.id == self.platform_config.credential_id
                ).update({
                    "execution_count": PlatformCredential.execution_count + 1
                })
                
                self.db.commit()
                
                # Handle execution result
                if response.status == ExecutionStatus.COMPLETED:
                    # Apply output mapping
                    output_data = self._apply_mapping(
                        response.output_data or {},
                        self.platform_config.output_mapping
                    )
                    
                    return NodeExecutionResult(
                        status=NodeStatus.COMPLETED,
                        output=output_data,
                        metadata={
                            "platform": self.platform_config.platform.value,
                            "execution_id": response.execution_id,
                            "duration_ms": response.duration_ms,
                            "cost_estimate": response.cost_estimate
                        }
                    )
                    
                elif response.status == ExecutionStatus.TIMEOUT:
                    return NodeExecutionResult(
                        status=NodeStatus.FAILED,
                        error=f"Platform execution timed out after {self.platform_config.timeout}s",
                        metadata={"execution_id": response.execution_id}
                    )
                    
                else:
                    # Failed or other status
                    await self.credential_service.record_error(
                        self.platform_config.credential_id,
                        response.error or "Unknown error"
                    )
                    
                    return NodeExecutionResult(
                        status=NodeStatus.FAILED,
                        error=response.error or f"Platform execution failed with status: {response.status}",
                        metadata={"execution_id": response.execution_id}
                    )
                    
            except Exception as e:
                # Update execution record with error
                db_execution.status = "failed"
                db_execution.error_data = {"error": str(e)}
                db_execution.completed_at = datetime.utcnow()
                self.db.commit()
                
                # Record error on credential
                await self.credential_service.record_error(
                    self.platform_config.credential_id,
                    str(e)
                )
                
                raise
                
        except Exception as e:
            logger.error(f"Platform node execution failed: {e}", exc_info=True)
            return NodeExecutionResult(
                status=NodeStatus.FAILED,
                error=f"Platform integration error: {str(e)}"
            )
            
        finally:
            if self.db:
                self.db.close()
    
    def _apply_mapping(
        self,
        data: Dict[str, Any],
        mapping: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply field mapping transformation"""
        if not mapping:
            return data
        
        result = {}
        
        for target_field, source_spec in mapping.items():
            if isinstance(source_spec, str):
                # Simple field mapping
                if source_spec.startswith("$."):
                    # JSONPath expression
                    try:
                        import jsonpath_ng
                        expr = jsonpath_ng.parse(source_spec)
                        matches = expr.find(data)
                        if matches:
                            result[target_field] = matches[0].value
                    except:
                        # Fallback to simple field access
                        field = source_spec[2:]  # Remove $. prefix
                        if field in data:
                            result[target_field] = data[field]
                else:
                    # Direct field name
                    if source_spec in data:
                        result[target_field] = data[source_spec]
                        
            elif isinstance(source_spec, dict):
                # Complex mapping with transformation
                source_field = source_spec.get("field")
                transform = source_spec.get("transform")
                default = source_spec.get("default")
                
                value = data.get(source_field, default)
                
                if value is not None and transform:
                    # Apply transformation
                    if transform == "string":
                        value = str(value)
                    elif transform == "number":
                        try:
                            value = float(value)
                        except:
                            value = default
                    elif transform == "boolean":
                        value = bool(value)
                    elif transform == "json":
                        if isinstance(value, str):
                            try:
                                value = json.loads(value)
                            except:
                                pass
                
                if value is not None:
                    result[target_field] = value
                    
            else:
                # Direct value assignment
                result[target_field] = source_spec
        
        return result
    
    def get_required_inputs(self) -> Dict[str, Any]:
        """Get required input schema"""
        # Build schema based on input mapping
        schema = {}
        
        if self.platform_config.input_mapping:
            for target_field, source_spec in self.platform_config.input_mapping.items():
                if isinstance(source_spec, str) and not source_spec.startswith("$."):
                    # This is a required field from input
                    schema[source_spec] = {"type": "any", "required": True}
                elif isinstance(source_spec, dict) and "field" in source_spec:
                    field = source_spec["field"]
                    required = source_spec.get("required", True)
                    schema[field] = {
                        "type": source_spec.get("type", "any"),
                        "required": required
                    }
        
        return schema
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Get output schema"""
        # Build schema based on output mapping
        schema = {}
        
        if self.platform_config.output_mapping:
            for target_field in self.platform_config.output_mapping.keys():
                schema[target_field] = {"type": "any"}
        
        # Add standard metadata fields
        schema["_platform_metadata"] = {
            "type": "object",
            "properties": {
                "platform": {"type": "string"},
                "execution_id": {"type": "string"},
                "duration_ms": {"type": "number"},
                "cost_estimate": {"type": "number"}
            }
        }
        
        return schema