"""
Data Quality API endpoints - Community Edition
ISO 25012 compliant quality assessment with edition-based features
"""

from typing import List, Optional, Dict, Any
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import get_current_active_user
from schemas.data_quality import (
    AssessmentRequest, AssessmentResponse,
    QualityMetrics, QualityDimensionInfo,
    RuleCreate, RuleResponse,
    UsageSummary
)
from services.data_quality_service import DataQualityService
from core.usage_tracker import UsageTracker
from models.data_quality import QualityDimension, QualityDimensionCategory

router = APIRouter()


def user_to_dict(user) -> dict:
    """Convert User object to dict, handling both dict and object inputs"""
    if isinstance(user, dict):
        return user
    
    # Convert User object to dict
    return {
        'id': str(getattr(user, 'id', 'unknown')),
        'user_id': str(getattr(user, 'id', 'unknown')),
        'email': getattr(user, 'email', 'unknown@example.com'),
        'tenant_id': str(getattr(user, 'tenant_id', '00000000-0000-0000-0000-000000000000')),
        'edition': getattr(user, 'edition', 'community'),
        'is_active': getattr(user, 'is_active', True),
        'is_superuser': getattr(user, 'is_superuser', False),
    }


@router.get("/dimensions", response_model=List[QualityDimensionInfo])
async def get_available_dimensions(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get available quality dimensions for current edition"""
    user_dict = user_to_dict(current_user)
    service = DataQualityService(db)
    edition = user_dict.get('edition', 'community')
    
    available = service.get_available_dimensions(edition)
    
    # Build dimension info
    dimensions = []
    for dim in QualityDimension:
        dimensions.append(QualityDimensionInfo(
            name=dim.value,
            category=QualityDimensionCategory.INHERENT if dim.value in [
                'accuracy', 'completeness', 'consistency', 'credibility', 'currentness'
            ] else QualityDimensionCategory.SYSTEM_DEPENDENT,
            description=f"Assess data {dim.value}",
            available_in_edition=dim in available,
            measurement_method=f"Automated {dim.value} assessment"
        ))
    
    return dimensions


@router.post("/assess", response_model=AssessmentResponse)
async def assess_data_quality(
    request: AssessmentRequest,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Assess data quality across available dimensions
    
    Community: accuracy, completeness
    Business: +8 dimensions (10 total)
    Enterprise: +5 dimensions (15 total)
    """
    user_dict = user_to_dict(current_user)
    service = DataQualityService(db)
    
    # Extract user info safely
    user_id_str = user_dict.get('id', user_dict.get('user_id', 'unknown'))
    try:
        user_id = UUID(user_id_str)
    except (ValueError, TypeError):
        # Use default UUID if user_id is invalid
        user_id = UUID('00000000-0000-0000-0000-000000000000')
    # Handle non-UUID tenant_id
    tenant_id_str = user_dict.get('tenant_id', '00000000-0000-0000-0000-000000000000')
    try:
        tenant_id = UUID(tenant_id_str)
    except ValueError:
        # Use default UUID if tenant_id is invalid
        tenant_id = UUID('00000000-0000-0000-0000-000000000000')
    edition = user_dict.get('edition', 'community')
    
    try:
        result = await service.assess_quality(
            request=request,
            user_id=user_id,
            tenant_id=tenant_id,
            edition=edition
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/assessment/{assessment_id}", response_model=AssessmentResponse)
async def get_assessment(
    assessment_id: UUID,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get a specific quality assessment by ID"""
    user_dict = user_to_dict(current_user)
    from models.data_quality import DataQualityAssessment
    from sqlalchemy import select
    
    result = await db.execute(
        select(DataQualityAssessment).where(DataQualityAssessment.id == assessment_id)
    )
    assessment = result.scalar_one_or_none()
    
    if not assessment:
        raise HTTPException(status_code=404, detail="Assessment not found")
    
    # Check access
    if str(assessment.tenant_id) != user_dict.get('tenant_id'):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return AssessmentResponse(
        id=assessment.id,
        overall_score=assessment.overall_score,
        dimension_scores=assessment.dimension_scores,
        issues_found=assessment.issues_found or [],
        suggestions=assessment.suggestions,
        data_profile=assessment.data_profile,
        assessment_time=assessment.assessment_time,
        dimensions_assessed=list(assessment.dimension_scores.keys()),
        edition=assessment.edition
    )


@router.get("/dashboard", response_model=QualityMetrics)
async def get_quality_dashboard(
    time_range: int = Query(30, description="Time range in days"),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get quality metrics for dashboard"""
    user_dict = user_to_dict(current_user)
    service = DataQualityService(db)
    
    # Handle non-UUID tenant_id
    tenant_id_str = user_dict.get('tenant_id', '00000000-0000-0000-0000-000000000000')
    try:
        tenant_id = UUID(tenant_id_str)
    except ValueError:
        # Use default UUID if tenant_id is invalid
        tenant_id = UUID('00000000-0000-0000-0000-000000000000')
    
    metrics = await service.get_quality_metrics(
        tenant_id=tenant_id,
        time_range=time_range
    )
    
    return QualityMetrics(**metrics)


@router.get("/rules", response_model=List[RuleResponse])
async def get_quality_rules(
    dimension: Optional[str] = None,
    is_active: bool = True,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get quality rules (system and custom)"""
    user_dict = user_to_dict(current_user)
    service = DataQualityService(db)
    
    # Handle non-UUID tenant_id
    tenant_id_str = user_dict.get('tenant_id', '00000000-0000-0000-0000-000000000000')
    try:
        tenant_id = UUID(tenant_id_str)
    except ValueError:
        # Use default UUID if tenant_id is invalid
        tenant_id = UUID('00000000-0000-0000-0000-000000000000')
    
    rules = await service.get_rules(
        tenant_id=tenant_id,
        dimension=dimension,
        is_active=is_active
    )
    
    return rules


@router.post("/rules", response_model=RuleResponse, status_code=201)
async def create_quality_rule(
    rule_data: RuleCreate,
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new quality rule"""
    user_dict = user_to_dict(current_user)
    service = DataQualityService(db)
    
    user_id_str = user_dict.get('id', user_dict.get('user_id', 'unknown'))
    try:
        user_id = UUID(user_id_str)
    except (ValueError, TypeError):
        # Use default UUID if user_id is invalid
        user_id = UUID('00000000-0000-0000-0000-000000000000')
    # Handle non-UUID tenant_id
    tenant_id_str = user_dict.get('tenant_id', '00000000-0000-0000-0000-000000000000')
    try:
        tenant_id = UUID(tenant_id_str)
    except ValueError:
        # Use default UUID if tenant_id is invalid
        tenant_id = UUID('00000000-0000-0000-0000-000000000000')
    edition = user_dict.get('edition', 'community')
    
    # Check if dimension is available in edition
    available_dimensions = service.get_available_dimensions(edition)
    if rule_data.dimension not in available_dimensions:
        raise HTTPException(
            status_code=403,
            detail=f"Dimension {rule_data.dimension.value} not available in {edition} edition"
        )
    
    try:
        rule = await service.create_rule(
            rule_data=rule_data,
            user_id=user_id,
            tenant_id=tenant_id,
            edition=edition
        )
        return rule
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/usage", response_model=UsageSummary)
async def get_quality_usage(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get quality feature usage statistics"""
    user_dict = user_to_dict(current_user)
    tracker = UsageTracker(db)
    
    # Handle non-UUID tenant_id
    tenant_id_str = user_dict.get('tenant_id', '00000000-0000-0000-0000-000000000000')
    try:
        tenant_id = UUID(tenant_id_str)
    except ValueError:
        # Use default UUID if tenant_id is invalid
        tenant_id = UUID('00000000-0000-0000-0000-000000000000')
    edition = user_dict.get('edition', 'community')
    
    # Get usage statistics
    from datetime import datetime, timedelta
    period_start = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    period_end = datetime.utcnow()
    
    # Get assessment count
    from models.data_quality import DataQualityAssessment
    from sqlalchemy import func, and_, select
    
    result = await db.execute(
        select(func.count(DataQualityAssessment.id)).where(
            and_(
                DataQualityAssessment.tenant_id == tenant_id,
                DataQualityAssessment.assessment_time >= period_start
            )
        )
    )
    assessment_count = result.scalar() or 0
    
    # Get limits by edition
    limits = {
        'community': 1000,
        'business': 100000,
        'enterprise': None
    }
    
    return UsageSummary(
        edition=edition,
        period_start=period_start,
        period_end=period_end,
        total_assessments=assessment_count,
        assessments_limit=limits.get(edition),
        total_rules_created=len(await service.get_rules(tenant_id)),
        total_profiles_used=0,  # Profiles are Business/Enterprise only
        features_used=[],
        upgrade_recommended=assessment_count > (limits.get(edition) or 0) * 0.8
    )


@router.get("/profiles", response_model=List[Dict[str, Any]])
async def get_quality_profiles(
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get quality profiles for the current user"""
    # Quality profiles are a Business/Enterprise feature
    # Community edition returns empty list
    return []


@router.get("/audit-trail", response_model=List[Dict[str, Any]])
async def get_audit_trail(
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    current_user = Depends(get_current_active_user),
    db: AsyncSession = Depends(get_db)
):
    """Get data quality audit trail"""
    # Audit trail is a Business/Enterprise feature
    # Community edition returns empty list
    return []