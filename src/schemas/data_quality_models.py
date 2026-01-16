"""
Data quality model schemas that properly handle metadata field aliasing.

These schemas correspond to the SQLAlchemy models and handle the quality_metadata field mapping.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
from uuid import UUID
from pydantic import BaseModel, Field, ConfigDict


# Data Quality Assessment Schemas
class DataQualityAssessmentBase(BaseModel):
    """Base schema for data quality assessment."""
    workflow_instance_id: Optional[UUID] = None
    node_id: Optional[UUID] = None
    task_id: Optional[str] = None
    data_reference: str
    overall_score: float = Field(..., ge=0.0, le=1.0)
    dimension_scores: Dict[str, float]
    issues_found: List[Dict[str, Any]]
    suggestions: Optional[List[Dict[str, Any]]] = None
    data_profile: Optional[Dict[str, Any]] = None
    edition: str
    tenant_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="quality_metadata")


class DataQualityAssessmentCreate(DataQualityAssessmentBase):
    """Schema for creating data quality assessment."""
    pass


class DataQualityAssessmentResponse(DataQualityAssessmentBase):
    """Schema for data quality assessment response."""
    id: UUID
    assessment_time: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )


# Quality Rule Schemas
class QualityRuleBase(BaseModel):
    """Base schema for quality rule."""
    name: str
    description: Optional[str] = None
    dimension: str
    rule_type: str
    rule_definition: Dict[str, Any]
    severity: str = "warning"
    edition_required: str = "community"
    is_active: bool = True
    is_system: bool = False
    metadata: Optional[Dict[str, Any]] = Field(None, alias="quality_metadata")


class QualityRuleCreate(QualityRuleBase):
    """Schema for creating quality rule."""
    created_by: Optional[UUID] = None


class QualityRuleResponse(QualityRuleBase):
    """Schema for quality rule response."""
    id: UUID
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )


# Quality Profile Schemas
class QualityProfileBase(BaseModel):
    """Base schema for quality profile."""
    name: str
    description: Optional[str] = None
    profile_type: str
    applicable_dimensions: List[str]
    dimension_weights: Dict[str, float]
    thresholds: Dict[str, Dict[str, float]]
    edition_required: str = "business"
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = Field(None, alias="quality_metadata")


class QualityProfileCreate(QualityProfileBase):
    """Schema for creating quality profile."""
    created_by: Optional[UUID] = None


class QualityProfileResponse(QualityProfileBase):
    """Schema for quality profile response."""
    id: UUID
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )


# Quality Improvement Schemas
class QualityImprovementBase(BaseModel):
    """Base schema for quality improvement."""
    assessment_id: UUID
    improvement_type: str
    dimension: str
    original_score: float = Field(..., ge=0.0, le=1.0)
    improved_score: float = Field(..., ge=0.0, le=1.0)
    improvement_method: str
    improvement_details: Dict[str, Any]
    applied: bool = False
    edition_required: str = "business"
    metadata: Optional[Dict[str, Any]] = Field(None, alias="quality_metadata")


class QualityImprovementCreate(QualityImprovementBase):
    """Schema for creating quality improvement."""
    created_by: Optional[UUID] = None


class QualityImprovementResponse(QualityImprovementBase):
    """Schema for quality improvement response."""
    id: UUID
    created_by: Optional[UUID] = None
    created_at: datetime
    applied_at: Optional[datetime] = None
    applied_by: Optional[UUID] = None

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )


# Data Lineage Schemas (Enterprise)
class DataLineageBase(BaseModel):
    """Base schema for data lineage."""
    source_node_id: UUID
    target_node_id: UUID
    transformation_type: str
    transformation_details: Dict[str, Any]
    quality_impact: Optional[Dict[str, float]] = None
    created_by: Optional[UUID] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="quality_metadata")


class DataLineageCreate(DataLineageBase):
    """Schema for creating data lineage."""
    pass


class DataLineageResponse(DataLineageBase):
    """Schema for data lineage response."""
    id: UUID
    created_at: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )


# Quality SLA Schemas (Enterprise)
class QualitySLABase(BaseModel):
    """Base schema for quality SLA."""
    name: str
    description: Optional[str] = None
    resource_type: str
    resource_id: str
    minimum_quality_score: float = Field(..., ge=0.0, le=1.0)
    dimension_requirements: Dict[str, float]
    check_frequency: str
    alert_channels: List[str]
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = Field(None, alias="quality_metadata")


class QualitySLACreate(QualitySLABase):
    """Schema for creating quality SLA."""
    created_by: Optional[UUID] = None


class QualitySLAResponse(QualitySLABase):
    """Schema for quality SLA response."""
    id: UUID
    created_by: Optional[UUID] = None
    created_at: datetime
    updated_at: datetime
    last_checked: Optional[datetime] = None
    current_status: Optional[str] = None

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )


# Quality Usage Tracking Schemas (Enterprise)
class QualityUsageTrackingBase(BaseModel):
    """Base schema for quality usage tracking."""
    tenant_id: UUID
    feature_used: str
    usage_count: int = 1
    dimension: Optional[str] = None
    resource_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(None, alias="quality_metadata")


class QualityUsageTrackingCreate(QualityUsageTrackingBase):
    """Schema for creating quality usage tracking."""
    pass


class QualityUsageTrackingResponse(QualityUsageTrackingBase):
    """Schema for quality usage tracking response."""
    id: UUID
    usage_date: datetime

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True
    ,
        protected_namespaces=()
    )