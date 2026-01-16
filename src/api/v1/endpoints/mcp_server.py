"""MCP Server endpoints for exposing AICtrlNet capabilities via MCP protocol."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from core.database import get_db
from core.security import get_current_active_user
from core.exceptions import ValidationError
from mcp_server.services import (
    MCPOrchestrationService,
    MCPQualityService,
    MCPWorkflowService
)
from schemas.mcp import (
    MCPMessageRequest,
    MCPMessageResponse,
    MCPQualityRequest,
    MCPQualityResponse,
    MCPWorkflowRequest,
    MCPWorkflowResponse,
    MCPDiscoveryResponse
)
from models.user import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["MCP Server"])


@router.post("/messages", response_model=MCPMessageResponse)
async def handle_mcp_messages(
    request: MCPMessageRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Process MCP messages for task orchestration.
    
    This endpoint receives messages in MCP format and orchestrates tasks
    based on the content. It supports:
    - Multi-destination routing
    - Async execution
    - Result aggregation
    - Error handling
    """
    try:
        orchestration_service = MCPOrchestrationService(db)
        
        # Process orchestration request
        result = await orchestration_service.process_orchestration_request(
            messages=request.messages,
            context={
                "user_id": str(current_user.id),
                "execute_immediately": request.execute_immediately,
                "routing_preferences": request.routing_preferences,
                "mcp_version": "1.0"
            }
        )
        
        return MCPMessageResponse(**result)
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"MCP message processing failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process MCP messages"
        )


@router.post("/quality", response_model=MCPQualityResponse)
async def assess_quality_via_mcp(
    request: MCPQualityRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Assess quality of content via MCP.
    
    Supports quality assessment for:
    - Text content
    - JSON data
    - Code snippets
    
    Dimensions assessed:
    - Accuracy
    - Completeness
    - Relevance
    - Clarity
    - Consistency
    """
    try:
        quality_service = MCPQualityService(db)
        
        if request.batch_mode and request.items:
            # Batch assessment
            result = await quality_service.batch_assess_quality(
                items=request.items
            )
        else:
            # Single assessment
            result = await quality_service.assess_quality(
                content=request.content,
                content_type=request.content_type,
                criteria=request.criteria
            )
        
        return MCPQualityResponse(**result)
        
    except Exception as e:
        logger.error(f"MCP quality assessment failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to assess quality"
        )


@router.post("/workflows", response_model=MCPWorkflowResponse)
async def create_workflow_via_mcp(
    request: MCPWorkflowRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create and optionally execute workflows via MCP.
    
    Features:
    - Visual editor compatible format
    - Conditional logic support
    - Parallel execution
    - Error recovery
    """
    try:
        workflow_service = MCPWorkflowService(db)
        
        # Create workflow
        result = await workflow_service.create_workflow_via_mcp(
            workflow_definition=request.workflow_definition,
            execute_immediately=request.execute_immediately
        )
        
        return MCPWorkflowResponse(**result)
        
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"MCP workflow creation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create workflow"
        )


@router.get("/workflows/{workflow_id}/status")
async def get_workflow_status(
    workflow_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get status of a workflow created via MCP."""
    try:
        workflow_service = MCPWorkflowService(db)
        result = await workflow_service.get_workflow_status(workflow_id)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get workflow status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get workflow status"
        )


@router.get("/discovery", response_model=MCPDiscoveryResponse)
async def discover_services(
    category: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """
    Discover available AI services and capabilities.
    
    Categories:
    - orchestration: Task orchestration services
    - quality: Quality assessment services
    - workflow: Workflow management services
    - ai_models: Available AI model integrations
    """
    services = {
        "orchestration": {
            "name": "Task Orchestration",
            "endpoint": "/mcp/v1/messages",
            "description": "Multi-destination task routing and execution",
            "capabilities": [
                "Route tasks to multiple AI providers",
                "Aggregate results from multiple sources",
                "Handle errors and retries",
                "Track execution progress"
            ]
        },
        "quality": {
            "name": "Quality Assessment",
            "endpoint": "/mcp/v1/quality",
            "description": "Assess quality of AI-generated content",
            "capabilities": [
                "Multi-dimensional quality scoring",
                "Support for text, JSON, and code",
                "Batch assessment mode",
                "Customizable assessment criteria"
            ]
        },
        "workflow": {
            "name": "Workflow Management",
            "endpoint": "/mcp/v1/workflows",
            "description": "Create and manage complex AI workflows",
            "capabilities": [
                "Visual workflow editor compatible",
                "Conditional logic and branching",
                "Parallel task execution",
                "Error recovery and retry logic"
            ]
        },
        "ai_models": {
            "name": "AI Model Integration",
            "endpoint": "/mcp/v1/messages",
            "description": "Access to multiple AI providers",
            "capabilities": [
                "OpenAI GPT models",
                "Anthropic Claude models",
                "Local model support",
                "Custom model integration"
            ]
        }
    }
    
    # Filter by category if specified
    if category:
        if category not in services:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid category. Choose from: {', '.join(services.keys())}"
            )
        services = {category: services[category]}
    
    return MCPDiscoveryResponse(
        services=services,
        version="1.0",
        server_name="AICtrlNet MCP Server"
    )


@router.get("/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get status of a task orchestrated via MCP."""
    try:
        orchestration_service = MCPOrchestrationService(db)
        result = await orchestration_service.get_task_status(task_id)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get task status"
        )


@router.get("/tasks")
async def list_mcp_tasks(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List tasks orchestrated via MCP."""
    try:
        orchestration_service = MCPOrchestrationService(db)
        return await orchestration_service.list_orchestrated_tasks(
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Failed to list MCP tasks: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list tasks"
        )


@router.get("/workflows")
async def list_mcp_workflows(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List workflows created via MCP."""
    try:
        workflow_service = MCPWorkflowService(db)
        return await workflow_service.list_mcp_workflows(
            limit=limit,
            offset=offset
        )
    except Exception as e:
        logger.error(f"Failed to list MCP workflows: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflows"
        )


@router.post("/workflows/{workflow_id}/expose")
async def expose_workflow_as_mcp_endpoint(
    workflow_id: str,
    endpoint_config: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Expose an existing workflow as an MCP endpoint."""
    try:
        workflow_service = MCPWorkflowService(db)
        result = await workflow_service.expose_workflow_as_mcp_endpoint(
            workflow_id=workflow_id,
            endpoint_config=endpoint_config
        )
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to expose workflow: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to expose workflow as endpoint"
        )


# Workflow MCP endpoint management

@router.get("/workflow/endpoints")
async def list_workflow_endpoints(
    workflow_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """List MCP endpoints exposed by workflows."""
    try:
        from mcp_server.services.workflow_exposure import WorkflowMCPService
        workflow_mcp_service = WorkflowMCPService(db)
        
        if workflow_id:
            # Get endpoints for specific workflow
            endpoints = await workflow_mcp_service.get_workflow_endpoints(workflow_id)
        else:
            # Get all workflow endpoints
            endpoints = await workflow_mcp_service.get_all_workflow_endpoints()
        
        return {
            "endpoints": endpoints,
            "count": len(endpoints)
        }
        
    except Exception as e:
        logger.error(f"Failed to list workflow endpoints: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list workflow endpoints"
        )


@router.post("/workflow/endpoints/{endpoint_id}/request")
async def send_request_to_workflow_endpoint(
    endpoint_id: str,
    request: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Send a request to a workflow MCP endpoint."""
    try:
        from mcp_server.services.workflow_exposure import WorkflowMCPService
        workflow_mcp_service = WorkflowMCPService(db)
        
        # Add request metadata
        request_data = {
            "request_id": f"req-{datetime.utcnow().timestamp()}",
            "client_id": str(current_user.id),
            "operation": request.get("operation", "execute"),
            "payload": request.get("payload", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        result = await workflow_mcp_service.handle_endpoint_request(
            endpoint_id=endpoint_id,
            request_data=request_data
        )
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result.get("error", "Request failed")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to send request to workflow endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process workflow endpoint request"
        )


@router.get("/workflow/endpoints/{endpoint_id}/stats")
async def get_workflow_endpoint_stats(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Get statistics for a workflow MCP endpoint."""
    try:
        from mcp_server.services.workflow_exposure import WorkflowMCPService
        workflow_mcp_service = WorkflowMCPService(db)
        
        stats = await workflow_mcp_service.get_endpoint_stats(endpoint_id)
        
        if not stats.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=stats.get("error", "Endpoint not found")
            )
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get endpoint stats: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get endpoint statistics"
        )


@router.delete("/workflow/endpoints/{endpoint_id}")
async def unregister_workflow_endpoint(
    endpoint_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """Unregister a workflow MCP endpoint."""
    try:
        from mcp_server.services.workflow_exposure import WorkflowMCPService
        workflow_mcp_service = WorkflowMCPService(db)
        
        result = await workflow_mcp_service.unregister_workflow_endpoint(endpoint_id)
        
        if not result.get("success"):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result.get("error", "Endpoint not found")
            )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to unregister endpoint: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to unregister endpoint"
        )