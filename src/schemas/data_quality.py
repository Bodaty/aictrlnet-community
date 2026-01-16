"""
Pydantic schemas for Data Quality API
ISO 25012 compliant with edition-based features
"""

from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from uuid import UUID
from pydantic import BaseModel, Field, validator, ConfigDict

from enum import Enum

class QualityDimension(str, Enum):
    """ISO 25012 Quality Dimensions"""
    # Inherent dimensions
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    CONSISTENCY = "consistency"
    CREDIBILITY = "credibility"
    CURRENTNESS = "currentness"
    
    # System-dependent dimensions
    ACCESSIBILITY = "accessibility"
    COMPLIANCE = "compliance"
    EFFICIENCY = "efficiency"
    PRECISION = "precision"
    UNDERSTANDABILITY = "understandability"
    
    # Enterprise dimensions
    CONFIDENTIALITY = "confidentiality"
    TRACEABILITY = "traceability"
    AVAILABILITY = "availability"
    PORTABILITY = "portability"
    RECOVERABILITY = "recoverability"

class QualityDimensionCategory(str, Enum):
    """Categories of quality dimensions"""
    INHERENT = "inherent"
    SYSTEM_DEPENDENT = "system_dependent"

class RuleType(str, Enum):
    """Types of quality rules"""
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    CUSTOM = "custom"
    REGEX = "regex"
    RANGE = "range"

class Severity(str, Enum):
    """Severity levels for quality issues"""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# Base schemas
class QualityDimensionInfo(BaseModel):
    """Information about a quality dimension"""
    name: str
    category: QualityDimensionCategory
    description: str
    available_in_edition: bool
    measurement_method: Optional[str] = None


class QualityScore(BaseModel):
    """Individual quality dimension score"""
    dimension: str
    score: float = Field(..., ge=0, le=1)
    issues: List[Dict[str, Any]] = []
    suggestions: List[str] = []


class DataProfile(BaseModel):
    """Statistical profile of data"""
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    null_percentage: Optional[float] = None
    unique_percentage: Optional[float] = None
    data_types: Optional[Dict[str, int]] = None
    patterns_detected: Optional[List[str]] = None
    anomalies: Optional[List[Dict[str, Any]]] = None


# Assessment schemas
class AssessmentRequest(BaseModel):
    """Request to assess data quality"""
    data: Union[Dict[str, Any], List[Dict[str, Any]], str]
    workflow_instance_id: Optional[UUID] = None
    node_id: Optional[UUID] = None
    task_id: Optional[str] = None
    dimensions: Optional[List[str]] = None  # None = use all available
    profile_id: Optional[UUID] = None  # Use specific quality profile
    include_suggestions: bool = True
    include_profile: bool = False


class AssessmentResponse(BaseModel):
    """Data quality assessment result"""
    id: UUID
    overall_score: float = Field(..., ge=0, le=1)
    dimension_scores: Dict[str, float]
    issues_found: List[Dict[str, Any]]
    suggestions: Optional[List[Dict[str, Any]]] = None
    data_profile: Optional[DataProfile] = None
    assessment_time: datetime
    dimensions_assessed: List[str]
    edition: str


class AssessmentHistory(BaseModel):
    """Historical assessment data"""
    assessments: List[AssessmentResponse]
    total_count: int
    average_score: float
    trend: str  # improving, declining, stable


# Rule schemas
class RuleCreate(BaseModel):
    """Create a new quality rule"""
    name: str
    description: Optional[str] = None
    dimension: QualityDimension
    rule_type: RuleType
    rule_definition: Dict[str, Any]
    severity: Severity = Severity.WARNING
    
    @validator('rule_definition')
    def validate_rule_definition(cls, v, values):
        rule_type = values.get('rule_type')
        if rule_type == 'regex':
            if 'pattern' not in v:
                raise ValueError("Regex rules must include 'pattern'")
        elif rule_type == 'range':
            if 'min' not in v and 'max' not in v:
                raise ValueError("Range rules must include 'min' or 'max'")
        return v


class RuleResponse(BaseModel):
    """Quality rule details"""
    id: UUID
    name: str
    description: Optional[str]
    dimension: str
    rule_type: str
    rule_definition: Dict[str, Any]
    severity: str
    is_active: bool
    is_system: bool
    created_at: datetime
    usage_count: Optional[int] = 0
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class RuleEvaluation(BaseModel):
    """Result of rule evaluation"""
    rule_id: UUID
    rule_name: str
    passed: bool
    score: float
    issues: List[Dict[str, Any]]
    severity: str


# Profile schemas (Business/Enterprise)
class ProfileCreate(BaseModel):
    """Create a quality profile"""
    name: str
    description: Optional[str] = None
    profile_type: str = "custom"  # industry, data_type, custom
    applicable_dimensions: List[QualityDimension]
    dimension_weights: Optional[Dict[str, float]] = None
    thresholds: Optional[Dict[str, Dict[str, float]]] = None
    
    @validator('dimension_weights')
    def validate_weights(cls, v):
        if v and sum(v.values()) != 1.0:
            raise ValueError("Dimension weights must sum to 1.0")
        return v


class ProfileResponse(BaseModel):
    """Quality profile details"""
    id: UUID
    name: str
    description: Optional[str]
    profile_type: str
    applicable_dimensions: List[str]
    dimension_weights: Dict[str, float]
    thresholds: Dict[str, Dict[str, float]]
    is_active: bool
    created_at: datetime
    usage_count: int
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class ProfileLearnRequest(BaseModel):
    """Request to learn quality profile from data"""
    training_data: List[Dict[str, Any]]
    quality_labels: Optional[List[float]] = None  # Known quality scores
    profile_type: str = "ml_learned"
    target_dimensions: Optional[List[QualityDimension]] = None


# Lineage schemas (Enterprise only)
class LineageNode(BaseModel):
    """Node in data lineage"""
    node_id: UUID
    node_type: str
    node_name: str
    timestamp: datetime
    quality_score: Optional[float] = None
    transformations_applied: List[str] = []


class LineageResponse(BaseModel):
    """Data lineage information"""
    data_reference: str
    lineage_path: List[LineageNode]
    total_transformations: int
    quality_impact: Dict[str, float]  # dimension -> impact
    start_time: datetime
    end_time: datetime


# SLA schemas (Business/Enterprise)
class SLACreate(BaseModel):
    """Create quality SLA"""
    name: str
    description: Optional[str] = None
    target_score: float = Field(..., ge=0, le=1)
    dimension_targets: Optional[Dict[str, float]] = None
    measurement_frequency: str = "daily"  # realtime, hourly, daily
    alert_threshold: Optional[float] = None
    escalation_rules: Optional[Dict[str, Any]] = None


class SLAResponse(BaseModel):
    """Quality SLA details"""
    id: UUID
    name: str
    description: Optional[str]
    target_score: float
    dimension_targets: Dict[str, float]
    measurement_frequency: str
    alert_threshold: Optional[float]
    is_active: bool
    current_score: Optional[float] = None
    is_violated: bool = False
    
    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class SLAStatus(BaseModel):
    """Current SLA status"""
    sla_id: UUID
    sla_name: str
    current_score: float
    target_score: float
    is_violated: bool
    violation_details: Optional[List[Dict[str, Any]]] = None
    last_measured: datetime


# Dashboard schemas
class QualityMetrics(BaseModel):
    """Quality metrics for dashboard"""
    total_assessments: int
    average_score: float
    assessments_by_dimension: Dict[str, int]
    scores_by_dimension: Dict[str, float]
    top_issues: List[Dict[str, Any]]
    quality_trend: List[Dict[str, Any]]  # time series data
    active_slas: Optional[List[SLAStatus]] = None
    recent_improvements: Optional[List[Dict[str, Any]]] = None


class QualityTrend(BaseModel):
    """Quality trend analysis"""
    dimension: str
    trend_direction: str  # improving, declining, stable
    current_score: float
    previous_score: float
    change_percentage: float
    forecast: Optional[List[Dict[str, float]]] = None  # ML forecast


# Compliance report (Enterprise)
class ComplianceReport(BaseModel):
    """Data quality compliance report"""
    report_id: UUID
    report_date: datetime
    compliance_standard: str  # ISO25012, GDPR, HIPAA, etc.
    overall_compliance_score: float
    dimension_compliance: Dict[str, Dict[str, Any]]
    violations: List[Dict[str, Any]]
    recommendations: List[str]
    certification_ready: bool


# Usage tracking
class UsageStats(BaseModel):
    """Quality feature usage statistics"""
    feature: str
    usage_count: int
    usage_limit: Optional[int] = None
    percentage_used: Optional[float] = None
    reset_date: Optional[datetime] = None


class UsageSummary(BaseModel):
    """Usage summary for quality features"""
    edition: str
    period_start: datetime
    period_end: datetime
    total_assessments: int
    assessments_limit: Optional[int] = None
    total_rules_created: int
    total_profiles_used: int
    features_used: List[UsageStats]
    upgrade_recommended: bool = False