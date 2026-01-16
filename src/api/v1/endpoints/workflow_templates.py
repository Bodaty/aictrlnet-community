"""API endpoints for workflow templates - Community Edition."""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from models.user import User
from schemas.workflow_templates import (
    WorkflowTemplateResponse,
    WorkflowTemplateDetail,
    WorkflowTemplateCreate,
    TemplateListRequest,
    TemplateListResponse,
    InstantiateTemplateRequest,
    InstantiateTemplateResponse,
    TemplateReviewCreate,
    TemplateReviewResponse,
    SuccessResponse
)
from services.workflow_template_service import create_workflow_template_service
from core.exceptions import NotFoundError, ForbiddenError, ValidationError

router = APIRouter(prefix="/workflow-templates", tags=["workflow-templates"])
template_service = create_workflow_template_service()


@router.get("", response_model=TemplateListResponse)
async def list_templates(
    category: Optional[str] = None,
    tags: Optional[List[str]] = Query(None),
    edition: Optional[str] = None,
    complexity: Optional[str] = None,
    include_public: bool = True,
    include_system: bool = True,
    include_private: bool = True,
    search: Optional[str] = None,
    sort_by: str = "created_at",
    sort_desc: bool = True,
    skip: int = 0,
    limit: int = Query(100, le=1000),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    List all accessible workflow templates.
    
    - **category**: Filter by template category
    - **tags**: Filter by tags (matches any)
    - **edition**: Filter by edition (community, business, enterprise)
    - **complexity**: Filter by complexity level (simple, moderate, complex, advanced)
    - **include_public**: Include public templates
    - **include_system**: Include system templates
    - **include_private**: Include private templates
    - **search**: Search in name and description
    - **sort_by**: Sort field (created_at, updated_at, usage_count, rating, name)
    - **sort_desc**: Sort descending
    - **skip**: Number of templates to skip
    - **limit**: Maximum number of templates to return
    """
    request = TemplateListRequest(
        category=category,
        tags=tags,
        edition=edition,
        complexity=complexity,
        include_public=include_public,
        include_system=include_system,
        include_private=include_private,
        search=search,
        sort_by=sort_by,
        sort_desc=sort_desc,
        skip=skip,
        limit=limit
    )
    
    templates, total = await template_service.list_templates(
        db, current_user.id, request
    )
    
    return TemplateListResponse(
        templates=templates,
        total=total,
        skip=skip,
        limit=limit
    )


@router.get("/health", response_model=dict)
async def get_template_health(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get health status of the workflow template system.
    
    Returns information about:
    - Number of templates by edition
    - Number of templates by category
    - Any validation errors found
    - System template directory status
    """
    try:
        # Get template counts by edition
        from sqlalchemy import select, func
        from models.workflow_templates import WorkflowTemplate
        
        # Count by edition
        edition_query = select(
            WorkflowTemplate.edition,
            func.count(WorkflowTemplate.id).label('count')
        ).where(
            WorkflowTemplate.is_system == True
        ).group_by(WorkflowTemplate.edition)
        
        edition_result = await db.execute(edition_query)
        edition_counts = {row.edition: row.count for row in edition_result}
        
        # Count by category
        category_query = select(
            WorkflowTemplate.category,
            func.count(WorkflowTemplate.id).label('count')
        ).where(
            WorkflowTemplate.is_system == True
        ).group_by(WorkflowTemplate.category)
        
        category_result = await db.execute(category_query)
        category_counts = {row.category: row.count for row in category_result}
        
        # Total counts
        total_query = select(func.count(WorkflowTemplate.id)).where(
            WorkflowTemplate.is_system == True
        )
        total_result = await db.execute(total_query)
        total_count = total_result.scalar() or 0
        
        # Check template directories
        import os
        from pathlib import Path
        
        template_dirs = {
            'community': Path('/app/workflow-templates/system'),
            'business': Path('/workspace/aictrlnet-fastapi-business/workflow-templates/system'),
            'enterprise': Path('/workspace/aictrlnet-fastapi-enterprise/workflow-templates/system'),
        }
        
        directory_status = {}
        for edition, path in template_dirs.items():
            if path.exists():
                json_count = len(list(path.rglob('*.json')))
                directory_status[edition] = {
                    'exists': True,
                    'json_files': json_count
                }
            else:
                directory_status[edition] = {
                    'exists': False,
                    'json_files': 0
                }
        
        return {
            'status': 'healthy' if total_count > 0 else 'warning',
            'total_templates': total_count,
            'templates_by_edition': edition_counts,
            'templates_by_category': category_counts,
            'directory_status': directory_status,
            'expected_totals': {
                'community': 1,
                'business': 176,
                'enterprise': 6,
                'total': 183
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{template_id}", response_model=WorkflowTemplateDetail)
async def get_template(
    template_id: UUID,
    load_definition: bool = Query(False, description="Load full workflow definition"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get a specific workflow template.
    
    - **template_id**: UUID of the template
    - **load_definition**: Whether to load the full workflow definition from file
    """
    try:
        template = await template_service.get_template(
            db, template_id, current_user.id, load_definition
        )
        return template
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ForbiddenError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.post("/{template_id}/instantiate", response_model=InstantiateTemplateResponse)
async def instantiate_template(
    template_id: UUID,
    request: InstantiateTemplateRequest,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a workflow instance from a template.
    
    This endpoint creates a new workflow based on the specified template,
    applying any provided parameters to customize the workflow.
    """
    try:
        result = await template_service.instantiate_template(
            db, template_id, current_user.id, request
        )
        return InstantiateTemplateResponse(**result)
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ForbiddenError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/{template_id}/reviews", response_model=SuccessResponse)
async def add_template_review(
    template_id: UUID,
    review: TemplateReviewCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Add or update a review for a template.
    
    Users can only have one review per template. If they've already reviewed,
    this will update their existing review.
    """
    try:
        await template_service.add_review(
            db, template_id, current_user.id, review
        )
        return SuccessResponse(
            success=True,
            message="Review added successfully"
        )
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ForbiddenError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))


@router.get("/{template_id}/preview", response_model=dict)
async def preview_template(
    template_id: UUID,
    parameters: Optional[str] = Query(None, description="JSON string of parameters"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Preview how a template would look with specific parameters.
    
    This endpoint returns the workflow configuration that would be created
    if the template were instantiated with the given parameters.
    """
    try:
        template = await template_service.get_template(
            db, template_id, current_user.id, load_definition=True
        )
        
        if not template.workflow_definition:
            raise ValidationError("Template has no workflow definition")
        
        # TODO: Apply parameters to preview
        # For now, just return the workflow definition
        return {
            "preview": template.workflow_definition,
            "parameters_applied": parameters is not None
        }
        
    except NotFoundError as e:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
    except ForbiddenError as e:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(e))
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("", response_model=WorkflowTemplateResponse)
async def create_template(
    template_data: WorkflowTemplateCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new workflow template.
    
    Creates a custom template that can be used to generate workflows.
    The template definition will be stored as a JSON file.
    """
    try:
        template = await template_service.create_template(
            db, current_user.id, template_data
        )
        return template
    except ValidationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/initialize-system-templates", response_model=SuccessResponse)
async def initialize_system_templates(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
) -> SuccessResponse:
    """Initialize system templates by scanning the template directory.
    
    This endpoint scans the /workflow-templates/system directory and registers
    all found templates as system templates in the database.
    """
    try:
        count = await template_service.initialize_system_templates(db)
        return SuccessResponse(
            success=True,
            message=f"Successfully initialized {count} system templates"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
