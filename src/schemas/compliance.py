"""Compliance schemas."""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# Enums

class ComplianceStatus(str, Enum):
    """Compliance status."""
    COMPLIANT = "compliant"
    PARTIAL = "partial"
    NON_COMPLIANT = "non_compliant"
    PENDING = "pending"
    NOT_APPLICABLE = "not_applicable"


class ComplianceFramework(str, Enum):
    """Compliance framework."""
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    SOC2 = "SOC2"
    ISO27001 = "ISO27001"
    PCI_DSS = "PCI_DSS"
    CCPA = "CCPA"
    NIST = "NIST"


# Compliance Standards schemas

class ComplianceStandard(BaseModel):
    """Compliance standard."""
    id: str
    name: str
    version: str
    description: str
    requirements_count: int
    categories: List[str]
    is_active: bool


class ComplianceRequirement(BaseModel):
    """Compliance requirement."""
    id: str
    standard_id: str
    title: str
    description: str
    category: str
    severity: str
    validation_rules: List[str]
    evidence_required: bool
    automated_check: bool


# Compliance Check schemas

class ComplianceCheck(BaseModel):
    """Compliance check request."""
    standard_id: str = Field(..., description="Compliance standard ID")
    scope: Dict[str, Any] = Field(..., description="Scope of compliance check")
    include_evidence: bool = Field(True, description="Include evidence in report")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class ComplianceViolation(BaseModel):
    """Compliance violation."""
    requirement_id: str
    requirement_title: str
    severity: str
    description: str
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    evidence: Dict[str, Any]
    detected_at: datetime
    remediation_suggestion: Optional[str] = None


class ComplianceReport(BaseModel):
    """Compliance report."""
    id: str
    standard_id: str
    standard_name: str
    check_date: datetime
    status: ComplianceStatus
    compliance_score: float
    total_requirements: int
    passed_requirements: int
    violations: List[ComplianceViolation]
    recommendations: List[str]
    next_review_date: datetime
    metadata: Optional[Dict[str, Any]] = None


# Compliance Assessment schemas

class ComplianceAssessment(BaseModel):
    """Comprehensive compliance assessment."""
    id: str
    assessment_date: datetime
    overall_status: ComplianceStatus
    overall_score: float
    standards_assessed: List[str]
    detailed_reports: List[ComplianceReport]
    executive_summary: str
    risk_areas: List[Dict[str, Any]]
    remediation_priority: List[str]


class RemediationPlan(BaseModel):
    """Remediation plan."""
    id: str
    report_id: str
    created_at: datetime
    priority_items: List[Dict[str, Any]]
    estimated_completion: datetime
    resource_requirements: Dict[str, Any]
    milestones: List[Dict[str, Any]]


# Audit Log schemas

class AuditLogEntry(BaseModel):
    """Audit log entry."""
    id: str
    timestamp: datetime
    event_type: str
    action: str
    component_id: Optional[str] = None
    resource_type: Optional[str] = None
    resource_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    endpoint: Optional[str] = None
    method: Optional[str] = None
    status: Optional[str] = None
    details: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True, protected_namespaces=())

class AuditLogQuery(BaseModel):
    """Audit log query."""
    event_type: Optional[str] = None
    resource_type: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(100, ge=1, le=1000)


# Export schemas

class ComplianceExport(BaseModel):
    """Compliance export request."""
    report_ids: List[str] = Field(..., description="Report IDs to export")
    format: str = Field("pdf", description="Export format (pdf, csv, json)")
    include_evidence: bool = Field(True, description="Include evidence")
    include_recommendations: bool = Field(True, description="Include recommendations")