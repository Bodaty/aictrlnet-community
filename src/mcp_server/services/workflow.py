"""MCP Workflow Service for workflow management via MCP protocol."""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from datetime import datetime
import uuid

from services.workflow_service import WorkflowService
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class MCPWorkflowService:
    """MCP service for workflow management."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.workflow_service = WorkflowService(db)
    
    async def create_workflow_via_mcp(
        self,
        workflow_definition: Dict[str, Any],
        execute_immediately: bool = False
    ) -> Dict[str, Any]:
        """Create a workflow via MCP protocol."""
        try:
            # Extract workflow details
            name = workflow_definition.get("name", f"MCP Workflow {uuid.uuid4().hex[:8]}")
            description = workflow_definition.get("description", "Created via MCP")
            steps = workflow_definition.get("steps", [])
            
            # Convert MCP workflow format to AICtrlNet format
            workflow_data = self._convert_mcp_to_aictrlnet_format(workflow_definition)
            
            # Create workflow
            workflow = await self.workflow_service.create_workflow(
                name=name,
                description=description,
                definition=workflow_data,
                metadata={
                    "created_via": "mcp",
                    "mcp_version": "1.0",
                    "original_definition": workflow_definition
                }
            )
            
            result = {
                "workflow_id": str(workflow.id),
                "name": workflow.name,
                "status": "created",
                "created_at": workflow.created_at.isoformat(),
                "metadata": {
                    "created_by": "AICtrlNet MCP Server",
                    "mcp_version": "1.0"
                }
            }
            
            # Execute if requested
            if execute_immediately:
                execution = await self.workflow_service.execute_workflow(
                    workflow_id=str(workflow.id),
                    input_data=workflow_definition.get("input_data", {})
                )
                
                result.update({
                    "status": "executing",
                    "execution_id": str(execution.id),
                    "execution_status": execution.status
                })
            
            return result
            
        except Exception as e:
            logger.error(f"MCP workflow creation failed: {str(e)}")
            return {
                "error": str(e),
                "status": "failed",
                "metadata": {
                    "created_by": "AICtrlNet MCP Server",
                    "failed_at": datetime.utcnow().isoformat()
                }
            }
    
    def _convert_mcp_to_aictrlnet_format(
        self, 
        mcp_workflow: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert MCP workflow format to AICtrlNet format."""
        nodes = []
        edges = []
        
        # Extract nodes from steps
        steps = mcp_workflow.get("steps", [])
        for i, step in enumerate(steps):
            node_id = step.get("id", f"node_{i}")
            
            node = {
                "id": node_id,
                "type": step.get("type", "task"),
                "data": {
                    "label": step.get("name", f"Step {i+1}"),
                    "description": step.get("description", ""),
                    "parameters": step.get("parameters", {}),
                    "config": step.get("config", {})
                },
                "position": {
                    "x": 100 + (i * 200),
                    "y": 100
                }
            }
            nodes.append(node)
            
            # Create edges for sequential flow
            if i > 0:
                prev_node_id = steps[i-1].get("id", f"node_{i-1}")
                edge = {
                    "id": f"edge_{i}",
                    "source": prev_node_id,
                    "target": node_id,
                    "type": "default"
                }
                edges.append(edge)
        
        # Handle connections if explicitly defined
        connections = mcp_workflow.get("connections", [])
        for conn in connections:
            edge = {
                "id": f"edge_custom_{uuid.uuid4().hex[:8]}",
                "source": conn.get("from"),
                "target": conn.get("to"),
                "type": conn.get("type", "default"),
                "data": conn.get("data", {})
            }
            edges.append(edge)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "metadata": mcp_workflow.get("metadata", {})
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a workflow created via MCP."""
        workflow = await self.workflow_service.get_workflow(workflow_id)
        
        if not workflow:
            return {
                "error": "Workflow not found",
                "workflow_id": workflow_id
            }
        
        # Get latest execution if any
        executions = await self.workflow_service.get_workflow_executions(
            workflow_id=workflow_id,
            limit=1
        )
        
        latest_execution = executions[0] if executions else None
        
        return {
            "workflow_id": str(workflow.id),
            "name": workflow.name,
            "status": workflow.status,
            "created_at": workflow.created_at.isoformat(),
            "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None,
            "latest_execution": {
                "execution_id": str(latest_execution.id),
                "status": latest_execution.status,
                "started_at": latest_execution.started_at.isoformat(),
                "completed_at": latest_execution.completed_at.isoformat() if latest_execution.completed_at else None
            } if latest_execution else None,
            "metadata": workflow.metadata
        }
    
    async def list_mcp_workflows(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """List workflows created via MCP."""
        workflows = await self.workflow_service.list_workflows(
            filters={"metadata.created_via": "mcp"},
            limit=limit,
            offset=offset
        )
        
        return {
            "workflows": [
                {
                    "workflow_id": str(wf.id),
                    "name": wf.name,
                    "status": wf.status,
                    "created_at": wf.created_at.isoformat(),
                    "metadata": wf.metadata
                }
                for wf in workflows
            ],
            "total": len(workflows),
            "limit": limit,
            "offset": offset
        }
    
    async def expose_workflow_as_mcp_endpoint(
        self,
        workflow_id: str,
        endpoint_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Expose an existing workflow as an MCP endpoint."""
        workflow = await self.workflow_service.get_workflow(workflow_id)
        
        if not workflow:
            return {
                "error": "Workflow not found",
                "workflow_id": workflow_id
            }
        
        # Register the workflow as an MCP endpoint
        endpoint_name = endpoint_config.get("endpoint_name", workflow.name.lower().replace(" ", "-"))
        
        # Store endpoint configuration in workflow metadata
        metadata = workflow.metadata or {}
        metadata["mcp_endpoint"] = {
            "enabled": True,
            "endpoint_name": endpoint_name,
            "auth_required": endpoint_config.get("auth_required", True),
            "allowed_operations": endpoint_config.get("allowed_operations", ["execute"]),
            "registered_at": datetime.utcnow().isoformat()
        }
        
        await self.workflow_service.update_workflow(
            workflow_id=workflow_id,
            metadata=metadata
        )
        
        return {
            "workflow_id": str(workflow.id),
            "endpoint_name": endpoint_name,
            "endpoint_url": f"/mcp/v1/workflows/{endpoint_name}",
            "status": "active",
            "metadata": {
                "exposed_by": "AICtrlNet MCP Server",
                "exposed_at": datetime.utcnow().isoformat()
            }
        }