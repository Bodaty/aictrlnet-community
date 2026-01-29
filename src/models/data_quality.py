"""
Data Quality Models for ISO 25012 Implementation
Supports Community, Business, and Enterprise editions with progressive features
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID
from enum import Enum

from sqlalchemy import Column, String, Float, Boolean, JSON, DateTime, Integer, ForeignKey, Index, Text, ARRAY
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import relationship

from .base import Base


class QualityDimensionCategory(str, Enum):
    """ISO 25012 dimension categories"""
    INHERENT = "inherent"
    SYSTEM_DEPENDENT = "system_dependent"


class QualityDimension(str, Enum):
    """ISO 25012 Quality Dimensions"""
    # Community Edition (2)
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    
    # Business Edition (10 total)
    CONSISTENCY = "consistency"
    CREDIBILITY = "credibility"
    CURRENTNESS = "currentness"
    ACCESSIBILITY = "accessibility"
    COMPLIANCE = "compliance"
    EFFICIENCY = "efficiency"
    PRECISION = "precision"
    UNDERSTANDABILITY = "understandability"
    
    # Enterprise Edition (15 total)
    CONFIDENTIALITY = "confidentiality"
    TRACEABILITY = "traceability"
    AVAILABILITY = "availability"
    PORTABILITY = "portability"
    RECOVERABILITY = "recoverability"


class RuleType(str, Enum):
    """Types of quality rules"""
    REGEX = "regex"
    RANGE = "range"
    ML_MODEL = "ml_model"
    CUSTOM = "custom"
    STATISTICAL = "statistical"


class Severity(str, Enum):
    """Severity levels for quality issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityDimensionModel(Base):
    """Quality dimensions reference table"""
    __tablename__ = "quality_dimensions"
    
    id = Column(PGUUID, primary_key=True)
    name = Column(String(50), nullable=False, unique=True)
    category = Column(String(20), nullable=False)
    description = Column(Text)
    edition_required = Column(String(20), nullable=False, default="community")
    measurement_method = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class DataQualityAssessment(Base):
    """Data quality assessment results"""
    __tablename__ = "data_quality_assessments"
    
    id = Column(PGUUID, primary_key=True)
    workflow_instance_id = Column(PGUUID, index=True)
    node_id = Column(PGUUID)
    task_id = Column(String(255))
    data_reference = Column(String(500))
    assessment_time = Column(DateTime, default=datetime.utcnow, index=True)
    overall_score = Column(Float)  # 0.0 to 1.0
    dimension_scores = Column(JSON)  # {dimension_name: score}
    issues_found = Column(JSON)  # [{dimension, issue, severity}]
    suggestions = Column(JSON)  # ML-generated suggestions
    data_profile = Column(JSON)  # Statistical profile
    edition = Column(String(20), nullable=False)
    tenant_id = Column(PGUUID, index=True)
    user_id = Column(PGUUID)
    quality_metadata = Column('metadata', JSON)
    
    __table_args__ = (
        Index('idx_quality_workflow', 'workflow_instance_id'),
        Index('idx_quality_time', 'assessment_time'),
        Index('idx_quality_tenant', 'tenant_id'),
    )


class QualityRule(Base):
    """Quality rules for automated checking"""
    __tablename__ = "quality_rules"
    
    id = Column(PGUUID, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    dimension = Column(String(50), nullable=False)
    rule_type = Column(String(50), nullable=False)
    rule_definition = Column(JSON, nullable=False)
    severity = Column(String(20), default="warning")
    edition_required = Column(String(20), nullable=False, default="community")
    is_active = Column(Boolean, default=True)
    is_system = Column(Boolean, default=False)
    created_by = Column(PGUUID)
    tenant_id = Column(PGUUID, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    quality_metadata = Column('metadata', JSON)
    
    __table_args__ = (
        Index('idx_rules_dimension', 'dimension'),
        Index('idx_rules_tenant', 'tenant_id'),
        Index('idx_rules_active', 'is_active'),
    )


class DataLineage(Base):
    """Data lineage tracking (Enterprise only)"""
    __tablename__ = "data_lineage"
    
    id = Column(PGUUID, primary_key=True)
    workflow_instance_id = Column(PGUUID, index=True)
    source_node_id = Column(PGUUID)
    target_node_id = Column(PGUUID)
    transformation_type = Column(String(100))
    transformation_details = Column(JSON)
    data_snapshot_before = Column(JSON)
    data_snapshot_after = Column(JSON)
    quality_impact = Column(JSON)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    tenant_id = Column(PGUUID)
    quality_metadata = Column('metadata', JSON)
    
    __table_args__ = (
        Index('idx_lineage_workflow', 'workflow_instance_id'),
        Index('idx_lineage_time', 'timestamp'),
        Index('idx_lineage_nodes', 'source_node_id', 'target_node_id'),
    )


class QualityProfile(Base):
    """Quality profiles for different data types (Business/Enterprise)"""
    __tablename__ = "quality_profiles"
    
    id = Column(PGUUID, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    profile_type = Column(String(50))  # industry, data_type, custom
    applicable_dimensions = Column(ARRAY(Text))
    dimension_weights = Column(JSON)  # {dimension: weight}
    thresholds = Column(JSON)  # {dimension: {min, target}}
    ml_model_id = Column(String(255))
    is_active = Column(Boolean, default=True)
    edition_required = Column(String(20), nullable=False, default="business")
    created_by = Column(PGUUID)
    tenant_id = Column(PGUUID, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    usage_count = Column(Integer, default=0)
    quality_metadata = Column('metadata', JSON)
    
    __table_args__ = (
        Index('idx_profiles_tenant', 'tenant_id'),
        Index('idx_profiles_type', 'profile_type'),
    )


class QualitySLA(Base):
    """Quality SLAs with alerting (Business/Enterprise)"""
    __tablename__ = "quality_slas"
    
    id = Column(PGUUID, primary_key=True)
    name = Column(String(255), nullable=False)
    description = Column(Text)
    target_score = Column(Float, nullable=False)
    dimension_targets = Column(JSON)  # {dimension: min_score}
    measurement_frequency = Column(String(50))  # realtime, hourly, daily
    alert_threshold = Column(Float)
    escalation_rules = Column(JSON)
    is_active = Column(Boolean, default=True)
    tenant_id = Column(PGUUID, index=True)
    created_by = Column(PGUUID)
    created_at = Column(DateTime, default=datetime.utcnow)
    quality_metadata = Column('metadata', JSON)
    
    __table_args__ = (
        Index('idx_sla_tenant', 'tenant_id'),
    )


class QualityImprovement(Base):
    """History of quality improvements"""
    __tablename__ = "quality_improvements"
    
    id = Column(PGUUID, primary_key=True)
    assessment_id = Column(PGUUID, ForeignKey('data_quality_assessments.id'))
    improvement_type = Column(String(100))  # manual, automated, ml_suggested
    dimension = Column(String(50), nullable=False)
    before_score = Column(Float)
    after_score = Column(Float)
    improvement_details = Column(JSON)
    implemented_by = Column(PGUUID)
    implemented_at = Column(DateTime, default=datetime.utcnow, index=True)
    tenant_id = Column(PGUUID)
    quality_metadata = Column('metadata', JSON)
    
    # Relationships
    assessment = relationship("DataQualityAssessment", backref="improvements")
    
    __table_args__ = (
        Index('idx_improvements_assessment', 'assessment_id'),
        Index('idx_improvements_time', 'implemented_at'),
    )


class QualityUsageTracking(Base):
    """Track usage of quality features for billing"""
    __tablename__ = "quality_usage_tracking"
    
    id = Column(PGUUID, primary_key=True)
    tenant_id = Column(PGUUID, nullable=False)
    user_id = Column(PGUUID)
    feature = Column(String(100), nullable=False)
    edition = Column(String(20), nullable=False)
    usage_date = Column(DateTime, nullable=False)
    usage_count = Column(Integer, default=1)
    quality_metadata = Column('metadata', JSON)
    
    __table_args__ = (
        Index('idx_usage_date', 'usage_date'),
        Index('idx_usage_tenant_date', 'tenant_id', 'usage_date'),
    )