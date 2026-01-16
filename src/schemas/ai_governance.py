"""AI Governance schemas for API requests and responses."""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    CRITICAL = "critical"


class QualityStandardType(str, Enum):
    """Quality verification standards."""
    ISO_25012 = "iso_25012"
    NIST_AI_RMF = "nist_ai_rmf"
    EU_AI_ACT = "eu_ai_act"
    CUSTOM = "custom"


class QualityStandard(BaseModel):
    """Quality standard definition for task types."""
    task_type: str = Field(..., description="Task type this standard applies to")
    minimum_score: float = Field(..., ge=0.0, le=1.0, description="Minimum quality score required")
    required_fields: List[str] = Field(..., description="Required fields in the result")
    format_rules: Dict[str, Any] = Field(default_factory=dict, description="Format validation rules")
    semantic_rules: Dict[str, Any] = Field(default_factory=dict, description="Semantic validation rules")


class RiskAssessmentRequest(BaseModel):
    """Request for risk assessment."""
    task_type: str = Field(..., description="Type of task (ai_task, human_task, etc.)")
    content: Dict[str, Any] = Field(..., description="Task content to assess")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Additional context")
    policies: Optional[List[str]] = Field(default=None, description="Specific policies to check")


class RiskAssessmentResponse(BaseModel):
    """Risk assessment result."""
    risk_score: float = Field(..., ge=0.0, le=1.0, description="Risk score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    risk_factors: List[Dict[str, Any]] = Field(default_factory=list, description="Identified risk factors")
    recommendation: str = Field(..., description="Action recommendation (approve/review/block)")
    policy_violations: Optional[List[Dict[str, Any]]] = Field(default=None, description="Policy violations found")
    require_approval: bool = Field(default=False, description="Whether manual approval is required")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Assessment confidence")
    assessment_id: str = Field(..., description="Unique assessment ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 0.3,
                "risk_level": "low",
                "risk_factors": [{"factor": "data_sensitivity", "score": 0.2}],
                "recommendation": "approve",
                "policy_violations": [],
                "require_approval": False,
                "confidence": 0.85,
                "assessment_id": "assess-123"
            }
        }


class BatchRiskAssessmentRequest(BaseModel):
    """Request for batch risk assessment."""
    tasks: List[RiskAssessmentRequest] = Field(..., max_items=100, description="Tasks to assess")
    parallel: bool = Field(default=True, description="Process in parallel")
    optimization_mode: str = Field(default="balanced", description="Optimization mode: fast, balanced, or thorough")


class BatchRiskAssessmentResponse(BaseModel):
    """Batch risk assessment results."""
    assessments: List[RiskAssessmentResponse] = Field(..., description="Individual assessments")
    summary: Dict[str, Any] = Field(..., description="Batch summary statistics")
    processing_time: float = Field(..., description="Processing time in seconds")


class QualityVerificationRequest(BaseModel):
    """Request for quality verification."""
    task_id: str = Field(..., description="Task ID to verify")
    result: Dict[str, Any] = Field(..., description="Result to verify")
    expected_format: Optional[Dict[str, Any]] = Field(default=None, description="Expected format schema")
    quality_standards: Optional[List[QualityStandardType]] = Field(default=None, description="Standards to check")


class QualityVerificationResponse(BaseModel):
    """Quality verification result."""
    quality_score: float = Field(..., ge=0.0, le=1.0, description="Overall quality score")
    passed: bool = Field(..., description="Whether quality check passed")
    format_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Format compliance score")
    semantic_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Semantic quality score")
    completeness_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Completeness score")
    issues: List[Dict[str, Any]] = Field(default_factory=list, description="Quality issues found")
    anomalies: Optional[List[Dict[str, Any]]] = Field(default=None, description="Detected anomalies")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")
    verification_id: str = Field(..., description="Unique verification ID")


class GovernanceOptimizationRequest(BaseModel):
    """Request for governance optimization analysis."""
    period: str = Field(default="last_30_days", description="Analysis period")
    focus_areas: Optional[List[str]] = Field(default=None, description="Specific areas to analyze")
    include_predictions: bool = Field(default=True, description="Include predictive analysis")


class GovernanceOptimizationResponse(BaseModel):
    """Governance optimization recommendations."""
    optimization_id: str = Field(..., description="Unique optimization ID")
    performance_metrics: Dict[str, Any] = Field(..., description="Current performance metrics")
    suggestions: List[Dict[str, Any]] = Field(..., description="Optimization suggestions")
    predicted_impact: Dict[str, Any] = Field(..., description="Predicted impact of suggestions")
    recommendations: List[str] = Field(..., description="Actionable recommendations")
    confidence_scores: Dict[str, float] = Field(default_factory=dict, description="Confidence in suggestions")


class ApplyOptimizationRequest(BaseModel):
    """Request to apply optimization suggestions."""
    optimization_id: Optional[str] = Field(None, description="Reference to previous optimization")
    suggestion_ids: List[str] = Field(..., description="Suggestions to apply")
    test_mode: bool = Field(default=True, description="Apply in test mode first")


class ApplyOptimizationResponse(BaseModel):
    """Result of applying optimizations."""
    applied: List[Dict[str, Any]] = Field(..., description="Successfully applied suggestions")
    failed: List[Dict[str, Any]] = Field(default_factory=list, description="Failed suggestions")
    test_results: Optional[Dict[str, Any]] = Field(None, description="Test mode results")
    rollback_available: bool = Field(default=True, description="Whether rollback is available")


class AIGovernanceMetrics(BaseModel):
    """AI Governance system metrics."""
    risk_assessment_metrics: Dict[str, Any] = Field(..., description="Risk assessment statistics")
    quality_verification_metrics: Dict[str, Any] = Field(..., description="Quality verification statistics")
    policy_compliance_metrics: Dict[str, Any] = Field(..., description="Policy compliance statistics")
    optimization_metrics: Dict[str, Any] = Field(..., description="Optimization effectiveness metrics")
    active_models: List[Dict[str, Any]] = Field(..., description="Active ML models information")
    ml_insights: Optional[Dict[str, Any]] = Field(default=None, description="ML-powered insights and recommendations")


class ModelTrainingRequest(BaseModel):
    """Request to train custom governance model."""
    ai_model_type: str = Field(..., description="Type of model (risk/quality/compliance)")
    tenant_id: str = Field(..., description="Tenant ID for the model")
    training_data_path: Optional[str] = Field(None, description="Path to training data")
    training_config: Dict[str, Any] = Field(default_factory=dict, description="Training configuration")
    created_by: str = Field(..., description="User initiating training")


class ModelTrainingStatus(BaseModel):
    """Status of model training job."""
    job_id: str = Field(..., description="Training job ID")
    status: str = Field(..., description="Job status")
    progress: Optional[float] = Field(None, ge=0.0, le=1.0, description="Training progress")
    started_at: Optional[datetime] = Field(None, description="Start time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Training metrics")
    error_message: Optional[str] = Field(None, description="Error if failed")


class CrossTenantMetricsRequest(BaseModel):
    """Request for cross-tenant governance metrics."""
    tenant_group_id: str = Field(..., description="Tenant group ID")
    period: str = Field(default="last_30_days", description="Analysis period")
    compare_with_baseline: bool = Field(default=True, description="Include baseline comparison")


class CrossTenantPolicyRequest(BaseModel):
    """Request to apply policy across tenants."""
    policy_id: str = Field(..., description="Policy to apply")
    tenant_ids: List[str] = Field(..., description="Target tenant IDs")
    test_mode: bool = Field(default=True, description="Test mode first")


class SelfImprovementStatus(BaseModel):
    """Status of self-improvement analysis."""
    last_analysis: datetime = Field(..., description="Last analysis time")
    improvements_applied: int = Field(..., description="Number of improvements applied")
    current_accuracy: Dict[str, float] = Field(..., description="Current model accuracies")
    next_scheduled: datetime = Field(..., description="Next scheduled analysis")


# Re-export commonly used types
__all__ = [
    "RiskLevel",
    "QualityStandard",
    "RiskAssessmentRequest",
    "RiskAssessmentResponse",
    "BatchRiskAssessmentRequest",
    "BatchRiskAssessmentResponse",
    "QualityVerificationRequest",
    "QualityVerificationResponse",
    "GovernanceOptimizationRequest",
    "GovernanceOptimizationResponse",
    "ApplyOptimizationRequest",
    "ApplyOptimizationResponse",
    "AIGovernanceMetrics",
    "ModelTrainingRequest",
    "ModelTrainingStatus",
    "CrossTenantMetricsRequest",
    "CrossTenantPolicyRequest",
    "SelfImprovementStatus",
]