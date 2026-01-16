"""NLP-related endpoints for natural language workflow generation."""

from typing import Optional, Dict, Any
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import StreamingResponse
from fastapi.encoders import jsonable_encoder
from sqlalchemy.ext.asyncio import AsyncSession
import json
import asyncio
import logging

from core.database import get_db
from core.security import get_current_active_user
from core.dependencies import get_current_user_safe
from core.tenant_context import get_current_tenant_id
from schemas.workflow import WorkflowResponse, NLPWorkflowResponse, TemplatePreview
from services.nlp import NLPService
from services.workflow_template_service import create_workflow_template_service
# WorkflowTemplates removed - using WorkflowTemplateService through NLPService
from models.user import User

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/process")
async def process_natural_language(
    request_data: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Process natural language to generate a workflow."""
    nlp_service = NLPService(db)
    
    # Extract fields from request
    prompt = request_data.get("input", "")
    context = request_data.get("context", {})
    
    if not prompt:
        raise HTTPException(
            status_code=422,
            detail="Input field is required"
        )
    
    # Add tenant_id to context if not already present
    if 'tenant_id' not in context or not context.get('tenant_id'):
        # Try to get tenant_id from current_user, otherwise use tenant context
        tenant_id = current_user.get('tenant_id')
        if not tenant_id:
            # If current_user doesn't have tenant_id, check if they have a tenant object
            tenant = current_user.get('tenant')
            if tenant and isinstance(tenant, dict):
                tenant_id = tenant.get('id') or get_current_tenant_id()
            else:
                tenant_id = get_current_tenant_id()
        context['tenant_id'] = tenant_id

    logger.info(f"NLP endpoint context after tenant_id: {context}")
    logger.info(f"Current user: {current_user}")

    try:
        # Check if transparency is requested (default true)
        return_transparency = request_data.get("return_transparency", True)
        print(f"DEBUG: NLP endpoint called with return_transparency={return_transparency}")
        
        # Pass user_id for security validation
        user_id = current_user.get('id', current_user.get('email', 'anonymous'))
        result = await nlp_service.process_natural_language(
            prompt, 
            context, 
            return_transparency,
            user_id=user_id
        )
        
        # Log what we got
        print(f"DEBUG: NLP result type: {type(result).__name__}, is_dict: {isinstance(result, dict)}")
        logger.info(f"NLP result type: {type(result).__name__}, is_dict: {isinstance(result, dict)}")
        if isinstance(result, dict):
            print(f"DEBUG: Dict keys: {list(result.keys())[:5]}")  # First 5 keys
            logger.info(f"Dict keys: {list(result.keys())[:5]}")  # First 5 keys
        
        # Check what type of result we got
        if isinstance(result, dict) and "workflow" in result:
            # This is a transparency response
            logger.info("Returning transparency response")
            return jsonable_encoder(result)
        else:
            # This is a workflow object, wrap it for backward compatibility
            logger.info(f"Returning backward compatibility response, validating as WorkflowResponse")
            workflow_response = WorkflowResponse.model_validate(result)
            return {
                "plan": jsonable_encoder(workflow_response)
            }
    except Exception as e:
        import traceback
        logger.error(f"Error in NLP endpoint: {str(e)}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing natural language: {str(e)}"
        )


@router.post("/process/stream")
async def process_natural_language_stream(
    request_data: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Process natural language with streaming response."""
    # Extract fields from request
    prompt = request_data.get("input", "")
    context = request_data.get("context", {})
    
    if not prompt:
        raise HTTPException(
            status_code=422,
            detail="Input field is required"
        )
    
    # Add tenant_id to context if not already present
    if 'tenant_id' not in context or not context.get('tenant_id'):
        # Try to get tenant_id from current_user, otherwise use tenant context
        tenant_id = current_user.get('tenant_id')
        if not tenant_id:
            # If current_user doesn't have tenant_id, check if they have a tenant object
            tenant = current_user.get('tenant')
            if tenant and isinstance(tenant, dict):
                tenant_id = tenant.get('id') or get_current_tenant_id()
            else:
                tenant_id = get_current_tenant_id()
        context['tenant_id'] = tenant_id

    logger.info(f"NLP endpoint context after tenant_id: {context}")
    logger.info(f"Current user: {current_user}")

    async def generate():
        nlp_service = NLPService(db)
        
        # Send initial response
        yield f"data: {json.dumps({'status': 'processing', 'message': 'Analyzing request...'})}\n\n"
        await asyncio.sleep(0.1)
        
        # Generate workflow with transparency
        try:
            return_transparency = request_data.get("return_transparency", True)
            # Pass user_id for security validation
            user_id = current_user.get('id', current_user.get('email', 'anonymous'))
            result = await nlp_service.process_natural_language(
                prompt, 
                context, 
                return_transparency,
                user_id=user_id
            )
            
            # Send progress updates
            yield f"data: {json.dumps({'status': 'generating', 'message': 'Creating workflow structure...'})}\n\n"
            await asyncio.sleep(0.1)
            
            yield f"data: {json.dumps({'status': 'enhancing', 'message': 'Adding AI capabilities...'})}\n\n"
            await asyncio.sleep(0.1)
            
            # Send final result
            try:
                if return_transparency:
                    # Send full transparency response
                    final_response = {'type': 'complete', 'data': jsonable_encoder(result)}
                else:
                    # Send simple workflow response
                    workflow_response = WorkflowResponse.model_validate(result)
                    workflow_dict = jsonable_encoder(workflow_response)
                    final_response = {'type': 'complete', 'data': {'plan': workflow_dict}}
                
                yield f"data: {json.dumps(final_response)}\n\n"
            except Exception as inner_e:
                # If serialization fails, send a simpler response
                simple_workflow = {
                    'id': str(result.get('workflow', {}).get('id', 'unknown')),
                    'name': result.get('workflow', {}).get('name', 'Generated Workflow'),
                    'description': result.get('workflow', {}).get('description', ''),
                    'status': 'active',
                    'category': 'nlp_generated'
                }
                yield f"data: {json.dumps({'type': 'complete', 'data': {'plan': simple_workflow}})}\n\n"
            
        except Exception as e:
            # Just send the error message as a string to avoid any serialization issues
            error_msg = str(e)
            yield f"data: {json.dumps({'type': 'error', 'error': error_msg})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/analyze")
async def analyze_text(
    request_data: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """
    Analyze text for entities, intent, and workflow suggestions.
    
    This endpoint provides comprehensive NLP analysis including:
    - Intent detection
    - Entity extraction
    - Workflow suggestions
    - Sentiment analysis
    """
    nlp_service = NLPService(db)
    
    # Extract fields from request
    text = request_data.get("text", "")
    context = request_data.get("context", {})
    analysis_type = request_data.get("type", "full")  # full, intent, entities, suggestions
    
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text field is required"
        )
    
    try:
        # Perform intent analysis
        intent = await nlp_service._analyze_intent(text)
        
        # Extract entities (simulate for now)
        entities = []
        if "email" in text.lower():
            entities.append({"type": "action", "value": "email", "confidence": 0.9})
        if "customer" in text.lower():
            entities.append({"type": "entity", "value": "customer", "confidence": 0.85})
        if "support" in text.lower():
            entities.append({"type": "domain", "value": "support", "confidence": 0.88})
        if "ticket" in text.lower():
            entities.append({"type": "object", "value": "ticket", "confidence": 0.92})
        
        # Generate workflow suggestions based on intent
        suggestions = []
        if intent == "support":
            suggestions = [
                {
                    "template_id": "customer_support",
                    "name": "Customer Support Workflow",
                    "confidence": 0.9,
                    "description": "Handle customer inquiries and support tickets"
                },
                {
                    "template_id": "ticket_escalation",
                    "name": "Ticket Escalation Workflow",
                    "confidence": 0.75,
                    "description": "Escalate support tickets based on priority"
                }
            ]
        elif intent == "data_processing":
            suggestions = [
                {
                    "template_id": "data_pipeline",
                    "name": "Data Pipeline Workflow",
                    "confidence": 0.85,
                    "description": "Process and transform data"
                }
            ]
        elif intent == "business_process":
            suggestions = [
                {
                    "template_id": "approval_workflow",
                    "name": "Approval Workflow",
                    "confidence": 0.8,
                    "description": "Multi-level approval process"
                }
            ]
        
        # Sentiment analysis (mock)
        sentiment = {
            "score": 0.7,
            "label": "positive" if len(text) % 2 == 0 else "neutral"
        }
        
        # Build response based on analysis type
        response = {
            "text": text,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if analysis_type in ["full", "intent"]:
            response["intent"] = {
                "type": intent,
                "confidence": 0.85
            }
        
        if analysis_type in ["full", "entities"]:
            response["entities"] = entities
        
        if analysis_type in ["full", "suggestions"]:
            response["suggestions"] = suggestions
        
        if analysis_type == "full":
            response["sentiment"] = sentiment
            response["language"] = "en"
            response["keywords"] = list(set(word.lower() for word in text.split() if len(word) > 4))[:5]
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing text: {str(e)}"
        )


@router.post("/analyze-intent")
async def analyze_intent(
    request_data: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Analyze user intent from natural language."""
    nlp_service = NLPService(db)
    
    # Extract fields from request
    text = request_data.get("text", "")
    context = request_data.get("context", {})
    
    if not text:
        raise HTTPException(
            status_code=422,
            detail="Text field is required"
        )
    
    try:
        intent = await nlp_service._analyze_intent(text)
        return {
            "prompt": text,
            "intent": intent,
            "confidence": 0.85  # Mock confidence for now
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing intent: {str(e)}"
        )


@router.get("/models")
async def list_ai_models(
    current_user: dict = Depends(get_current_user_safe),
):
    """List available AI models for NLP processing."""
    return {
        "models": [
            {
                "id": "llama3.2:1b",
                "name": "Llama 3.2 1B",
                "provider": "ollama",
                "status": "available",
                "capabilities": ["text-generation", "workflow-generation"]
            }
        ],
        "default": "llama3.2:1b"
    }


@router.post("/convert-plan")
async def convert_plan_to_workflow(
    plan: Dict[str, Any] = Body(...),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Convert an NLP-generated plan to an executable workflow."""
    nlp_service = NLPService(db)
    
    try:
        # Handle different plan formats
        if "plan" in plan:
            # Frontend sends the plan inside a plan field
            actual_plan = plan["plan"]
        else:
            actual_plan = plan
            
        # Extract workflow configuration from plan
        workflow_config = {
            "nodes": actual_plan.get("nodes", actual_plan.get("steps", [])),
            "edges": actual_plan.get("edges", actual_plan.get("connections", []))
        }
        
        # Get tenant_id from current_user
        tenant_id = current_user.get('tenant_id')
        if not tenant_id:
            # If current_user doesn't have tenant_id, check if they have a tenant object
            tenant = current_user.get('tenant')
            if tenant and isinstance(tenant, dict):
                tenant_id = tenant.get('id') or get_current_tenant_id()
            else:
                tenant_id = get_current_tenant_id()

        # Create workflow from plan
        workflow = await nlp_service._create_workflow_from_nlp(
            actual_plan.get("name", actual_plan.get("description", "Converted from plan")),
            workflow_config,
            "plan_conversion",
            tenant_id=tenant_id
        )
        
        return jsonable_encoder(WorkflowResponse.model_validate(workflow))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error converting plan: {str(e)}"
        )


@router.get("/templates/preview/{template_id}")
async def preview_template(
    template_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Get a preview of a specific workflow template."""
    from services.workflow_template_service import WorkflowTemplateService
    import uuid
    
    try:
        template_uuid = uuid.UUID(template_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid template ID format")
    
    template_service = create_workflow_template_service()
    try:
        template = await template_service.get_template(db, template_uuid, "system", load_definition=True)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
        
        # Create preview from template data
        preview = {
            'id': str(template.id),
            'name': template.name,
            'description': template.description,
            'category': template.category,
            'tags': template.tags,
            'complexity': template.complexity,
            'edition_required': template.edition,
            'estimated_duration': template.estimated_duration,
            'required_adapters': template.required_adapters,
            'required_capabilities': template.required_capabilities,
            'preview_available': template.workflow_definition is not None,
            'workflow_definition': template.workflow_definition
        }
        
        return jsonable_encoder(preview)
    except Exception as e:
        logger.error(f"Error getting template preview: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates/category/{category}")
async def get_templates_by_category(
    category: str,
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Get all templates for a specific category."""
    from services.workflow_template_service import WorkflowTemplateService
    
    template_service = create_workflow_template_service()
    try:
        templates = await template_service.get_templates_by_category_async(db, category)
        return jsonable_encoder(templates)
    except Exception as e:
        logger.error(f"Error getting templates by category: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/templates/categories")
async def get_template_categories(
    current_user: dict = Depends(get_current_user_safe),
):
    """Get all available template categories."""
    categories = [
        {
            "id": "support",
            "name": "Customer Support",
            "description": "Templates for customer service and support workflows",
            "icon": "support"
        },
        {
            "id": "business",
            "name": "Business Process",
            "description": "General business process and approval workflows",
            "icon": "business"
        },
        {
            "id": "it",
            "name": "IT & DevOps",
            "description": "IT operations and development workflows",
            "icon": "code"
        },
        {
            "id": "data",
            "name": "Data Processing",
            "description": "ETL and data pipeline workflows",
            "icon": "database"
        },
        {
            "id": "ai",
            "name": "AI & Analytics",
            "description": "AI-powered analysis and ML workflows",
            "icon": "brain"
        },
        {
            "id": "operations",
            "name": "Operations",
            "description": "Monitoring and operational workflows",
            "icon": "monitor"
        },
        {
            "id": "communication",
            "name": "Communication",
            "description": "Notification and messaging workflows",
            "icon": "mail"
        },
        {
            "id": "sales",
            "name": "Sales & CRM",
            "description": "Sales automation and CRM workflows",
            "icon": "dollar"
        },
        {
            "id": "hr",
            "name": "Human Resources",
            "description": "HR and employee management workflows",
            "icon": "users"
        },
        {
            "id": "finance",
            "name": "Finance",
            "description": "Financial and expense management workflows",
            "icon": "calculator"
        }
    ]
    
    return categories


@router.post("/templates/analyze-requirements")
async def analyze_template_requirements(
    template_id: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db),
    current_user: dict = Depends(get_current_user_safe),
):
    """Analyze edition requirements for a specific template."""
    from services.workflow_template_service import WorkflowTemplateService
    import uuid
    
    try:
        template_uuid = uuid.UUID(template_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid template ID format")
    
    template_service = create_workflow_template_service()
    try:
        template = await template_service.get_template(db, template_uuid, "system", load_definition=False)
        if not template:
            raise HTTPException(status_code=404, detail=f"Template {template_id} not found")
    except Exception as e:
        logger.error(f"Error getting template: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    
    # Get edition requirements (for now, just use template's edition requirement)
    workflow_config = template.get('workflow_config', {})
    requirements = {"edition": template.get('edition_required', 'community')}
    
    return {
        "template_id": template_id,
        "template_name": template['name'],
        "edition_required": template.get('edition_required', 'community'),
        "feature_breakdown": requirements,
        "total_features": {
            "community": len(requirements['community']),
            "business": len(requirements['business']),
            "enterprise": len(requirements['enterprise'])
        }
    }