"""
Data Quality Service - Core implementation for ISO 25012 compliance
Provides edition-aware quality assessment, rule evaluation, and improvement suggestions
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
import numpy as np
from collections import Counter
import asyncio

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, func, select

from models.data_quality import (
    DataQualityAssessment, QualityRule, QualityDimension,
    RuleType, Severity, QualityProfile, DataLineage
)
from schemas.data_quality import (
    AssessmentRequest, AssessmentResponse, QualityScore,
    RuleCreate, RuleResponse, DataProfile
)
from core.config import settings
from core.exceptions import ValidationError, FeatureNotAvailableError
from core.usage_tracker import UsageTracker
from core.cache import cache_result


class DataQualityService:
    """Core data quality service for all editions"""
    
    # Edition-based dimension access
    COMMUNITY_DIMENSIONS = [QualityDimension.ACCURACY, QualityDimension.COMPLETENESS]
    BUSINESS_DIMENSIONS = COMMUNITY_DIMENSIONS + [
        QualityDimension.CONSISTENCY, QualityDimension.CREDIBILITY,
        QualityDimension.CURRENTNESS, QualityDimension.ACCESSIBILITY,
        QualityDimension.COMPLIANCE, QualityDimension.EFFICIENCY,
        QualityDimension.PRECISION, QualityDimension.UNDERSTANDABILITY
    ]
    ENTERPRISE_DIMENSIONS = BUSINESS_DIMENSIONS + [
        QualityDimension.CONFIDENTIALITY, QualityDimension.TRACEABILITY,
        QualityDimension.AVAILABILITY, QualityDimension.PORTABILITY,
        QualityDimension.RECOVERABILITY
    ]
    
    def __init__(self, db: AsyncSession, usage_tracker: Optional[UsageTracker] = None):
        self.db = db
        self.usage_tracker = usage_tracker or UsageTracker(db)
    
    def get_available_dimensions(self, edition: str) -> List[QualityDimension]:
        """Get quality dimensions available for the edition"""
        if edition == "enterprise":
            return self.ENTERPRISE_DIMENSIONS
        elif edition == "business":
            return self.BUSINESS_DIMENSIONS
        else:
            return self.COMMUNITY_DIMENSIONS
    
    async def assess_quality(
        self,
        request: AssessmentRequest,
        user_id: UUID,
        tenant_id: UUID,
        edition: str
    ) -> AssessmentResponse:
        """Assess data quality across available dimensions"""
        
        # Track usage
        await self.usage_tracker.track_usage(
            tenant_id=str(tenant_id),
            metric_type="data_quality_assessment",
            value=1.0
        )
        
        # Check usage limits
        usage_limit = self._get_usage_limit(edition)
        current_usage = await self._get_current_usage(tenant_id)
        if usage_limit and current_usage >= usage_limit:
            raise ValidationError(f"Monthly quality assessment limit ({usage_limit}) reached")
        
        # Get available dimensions
        available_dimensions = self.get_available_dimensions(edition)
        requested_dimensions = request.dimensions or [d.value for d in available_dimensions]
        
        # Filter to only available dimensions
        dimensions_to_assess = [
            d for d in requested_dimensions 
            if d in [dim.value for dim in available_dimensions]
        ]
        
        # Convert data to assessable format
        data = self._prepare_data(request.data)
        
        # Assess each dimension
        dimension_scores = {}
        all_issues = []
        all_suggestions = []
        
        for dimension in dimensions_to_assess:
            score, issues, suggestions = await self._assess_dimension(
                data, dimension, edition
            )
            # Ensure score is not NaN
            if np.isnan(score):
                score = 0.0
            dimension_scores[dimension] = float(score)
            all_issues.extend(issues)
            if request.include_suggestions:
                all_suggestions.extend(suggestions)
        
        # Calculate overall score
        if dimension_scores:
            scores = list(dimension_scores.values())
            # Filter out any NaN values
            valid_scores = [s for s in scores if not np.isnan(s)]
            if valid_scores:
                overall_score = float(np.mean(valid_scores))
            else:
                overall_score = 0.0
            # Ensure no NaN
            if np.isnan(overall_score):
                overall_score = 0.0
        else:
            overall_score = 0.0  # No dimensions assessed
        
        # Generate data profile if requested
        data_profile = None
        if request.include_profile and edition in ["business", "enterprise"]:
            data_profile = self._generate_data_profile(data)
        
        # Save assessment to database
        assessment = DataQualityAssessment(
            id=uuid4(),
            workflow_instance_id=request.workflow_instance_id,
            node_id=request.node_id,
            task_id=request.task_id,
            data_reference=self._generate_data_reference(data),
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues_found=all_issues,
            suggestions=all_suggestions if request.include_suggestions else None,
            data_profile=data_profile.dict() if data_profile else None,
            edition=edition,
            tenant_id=tenant_id,
            user_id=user_id,
            metadata={
                "dimensions_requested": request.dimensions,
                "profile_id": str(request.profile_id) if request.profile_id else None
            }
        )
        
        self.db.add(assessment)
        await self.db.commit()
        
        return AssessmentResponse(
            id=assessment.id,
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            issues_found=all_issues,
            suggestions=all_suggestions if request.include_suggestions else None,
            data_profile=data_profile,
            assessment_time=assessment.assessment_time,
            dimensions_assessed=dimensions_to_assess,
            edition=edition
        )
    
    async def _assess_dimension(
        self,
        data: Any,
        dimension: str,
        edition: str
    ) -> tuple[float, List[Dict], List[Dict]]:
        """Assess a specific quality dimension"""
        
        if dimension == QualityDimension.ACCURACY.value:
            return await self._assess_accuracy(data)
        elif dimension == QualityDimension.COMPLETENESS.value:
            return await self._assess_completeness(data)
        elif dimension == QualityDimension.CONSISTENCY.value:
            return await self._assess_consistency(data)
        elif dimension == QualityDimension.CREDIBILITY.value:
            return await self._assess_credibility(data)
        elif dimension == QualityDimension.CURRENTNESS.value:
            return await self._assess_currentness(data)
        elif dimension == QualityDimension.ACCESSIBILITY.value:
            return await self._assess_accessibility(data)
        elif dimension == QualityDimension.COMPLIANCE.value:
            return await self._assess_compliance(data)
        elif dimension == QualityDimension.EFFICIENCY.value:
            return await self._assess_efficiency(data)
        elif dimension == QualityDimension.PRECISION.value:
            return await self._assess_precision(data)
        elif dimension == QualityDimension.UNDERSTANDABILITY.value:
            return await self._assess_understandability(data)
        # Enterprise dimensions
        elif dimension == QualityDimension.CONFIDENTIALITY.value:
            return await self._assess_confidentiality(data)
        elif dimension == QualityDimension.TRACEABILITY.value:
            return await self._assess_traceability(data)
        elif dimension == QualityDimension.AVAILABILITY.value:
            return await self._assess_availability(data)
        elif dimension == QualityDimension.PORTABILITY.value:
            return await self._assess_portability(data)
        elif dimension == QualityDimension.RECOVERABILITY.value:
            return await self._assess_recoverability(data)
        else:
            return 0.5, [], []  # Default neutral score
    
    # Dimension assessment methods - Community Edition
    async def _assess_accuracy(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data accuracy"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            # Check for common accuracy issues
            for key, value in data.items():
                if value is None:
                    continue
                    
                # Check email format
                if 'email' in key.lower() and isinstance(value, str):
                    if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', value):
                        issues.append({
                            "dimension": "accuracy",
                            "field": key,
                            "issue": "Invalid email format",
                            "severity": "error"
                        })
                        score -= 0.2
                
                # Check phone format
                if 'phone' in key.lower() and isinstance(value, str):
                    if not re.match(r'^[\d\s\-\+\(\)]+$', value):
                        issues.append({
                            "dimension": "accuracy",
                            "field": key,
                            "issue": "Invalid phone format",
                            "severity": "warning"
                        })
                        score -= 0.1
                
                # Check date formats
                if 'date' in key.lower() and isinstance(value, str):
                    try:
                        datetime.fromisoformat(value.replace('Z', '+00:00'))
                    except:
                        issues.append({
                            "dimension": "accuracy",
                            "field": key,
                            "issue": "Invalid date format",
                            "severity": "error"
                        })
                        score -= 0.15
        
        # Generate suggestions
        if issues:
            suggestions.append({
                "dimension": "accuracy",
                "suggestion": "Implement input validation for email, phone, and date fields",
                "impact": "high"
            })
        
        return max(0, score), issues, suggestions
    
    async def _assess_completeness(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data completeness"""
        issues = []
        suggestions = []
        
        if isinstance(data, dict):
            total_fields = len(data)
            if total_fields == 0:
                return 0, [{"dimension": "completeness", "issue": "No data provided", "severity": "critical"}], []
            
            null_fields = sum(1 for v in data.values() if v is None or v == "")
            completeness_score = 1 - (null_fields / total_fields)
            
            if null_fields > 0:
                issues.append({
                    "dimension": "completeness",
                    "issue": f"{null_fields} out of {total_fields} fields are empty",
                    "severity": "warning" if completeness_score > 0.7 else "error",
                    "fields": [k for k, v in data.items() if v is None or v == ""]
                })
                
                suggestions.append({
                    "dimension": "completeness",
                    "suggestion": "Consider making critical fields required",
                    "impact": "medium"
                })
            
            return completeness_score, issues, suggestions
        
        elif isinstance(data, list):
            if not data:
                return 0, [{"dimension": "completeness", "issue": "Empty list provided", "severity": "error"}], []
            
            # For lists, check completeness of each item
            scores = []
            for item in data:
                if isinstance(item, dict):
                    item_score, _, _ = await self._assess_completeness(item)
                    scores.append(item_score)
            
            return np.mean(scores) if scores else 1.0, issues, suggestions
        
        return 1.0, issues, suggestions
    
    # Business Edition dimension assessments
    async def _assess_consistency(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data consistency"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            # Check for inconsistent date formats
            date_formats = set()
            for key, value in data.items():
                if 'date' in key.lower() and isinstance(value, str):
                    if 'T' in value:
                        date_formats.add('ISO')
                    elif '/' in value:
                        date_formats.add('slash')
                    elif '-' in value:
                        date_formats.add('dash')
            
            if len(date_formats) > 1:
                issues.append({
                    "dimension": "consistency",
                    "issue": "Inconsistent date formats detected",
                    "severity": "warning"
                })
                score -= 0.1
                suggestions.append({
                    "dimension": "consistency",
                    "suggestion": "Standardize all dates to ISO 8601 format",
                    "impact": "medium"
                })
        
        return max(0, score), issues, suggestions
    
    async def _assess_credibility(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data credibility"""
        # Simplified credibility check - in real implementation would check sources
        score = 0.8  # Default credibility score
        issues = []
        suggestions = []
        
        if isinstance(data, dict):
            # Check for source information
            if 'source' not in data and 'created_by' not in data:
                issues.append({
                    "dimension": "credibility",
                    "issue": "No data source information provided",
                    "severity": "info"
                })
                score -= 0.1
                suggestions.append({
                    "dimension": "credibility",
                    "suggestion": "Include data source or creator information",
                    "impact": "low"
                })
        
        return max(0, score), issues, suggestions
    
    async def _assess_currentness(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data currentness"""
        issues = []
        suggestions = []
        score = 1.0
        now = datetime.utcnow()
        
        if isinstance(data, dict):
            for key, value in data.items():
                if any(term in key.lower() for term in ['date', 'time', 'updated', 'created']):
                    if isinstance(value, str):
                        try:
                            date = datetime.fromisoformat(value.replace('Z', '+00:00'))
                            age_days = (now - date).days
                            
                            if age_days > 365:
                                issues.append({
                                    "dimension": "currentness",
                                    "field": key,
                                    "issue": f"Data is {age_days} days old",
                                    "severity": "warning"
                                })
                                score -= min(0.5, age_days / 1000)
                        except:
                            pass
        
        if issues:
            suggestions.append({
                "dimension": "currentness",
                "suggestion": "Implement data refresh policies for stale data",
                "impact": "medium"
            })
        
        return max(0, score), issues, suggestions
    
    async def _assess_accessibility(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data accessibility - how easily data can be retrieved"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            # Check for deeply nested structures
            max_depth = self._get_max_depth(data)
            if max_depth > 5:
                issues.append({
                    "dimension": "accessibility",
                    "issue": f"Data structure too deeply nested (depth: {max_depth})",
                    "severity": "warning"
                })
                score -= 0.1 * (max_depth - 5)
                suggestions.append({
                    "dimension": "accessibility",
                    "suggestion": "Flatten data structure for easier access",
                    "impact": "medium"
                })
            
            # Check for complex keys
            for key in data.keys():
                if len(key) > 50 or not key.replace('_', '').isalnum():
                    issues.append({
                        "dimension": "accessibility",
                        "field": key,
                        "issue": "Complex or non-standard field name",
                        "severity": "info"
                    })
                    score -= 0.05
        
        return max(0, score), issues, suggestions
    
    async def _assess_compliance(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data compliance with standards and regulations"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            # Check for PII indicators
            pii_fields = ['ssn', 'social_security', 'credit_card', 'password', 'pin']
            for key, value in data.items():
                if any(pii in key.lower() for pii in pii_fields):
                    if isinstance(value, str) and not value.startswith('***'):
                        issues.append({
                            "dimension": "compliance",
                            "field": key,
                            "issue": "Potential unencrypted PII detected",
                            "severity": "critical"
                        })
                        score -= 0.3
                        suggestions.append({
                            "dimension": "compliance",
                            "suggestion": f"Encrypt or mask {key} field",
                            "impact": "critical"
                        })
            
            # Check for required compliance fields
            if 'gdpr_consent' not in data and 'consent_timestamp' not in data:
                issues.append({
                    "dimension": "compliance",
                    "issue": "Missing GDPR consent tracking",
                    "severity": "warning"
                })
                score -= 0.1
        
        return max(0, score), issues, suggestions
    
    async def _assess_efficiency(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data processing efficiency"""
        issues = []
        suggestions = []
        score = 1.0
        
        # Estimate data size
        data_size = len(json.dumps(data)) if not isinstance(data, str) else len(data)
        
        if data_size > 1000000:  # 1MB
            issues.append({
                "dimension": "efficiency",
                "issue": f"Large data size ({data_size} bytes) may impact processing",
                "severity": "warning"
            })
            score -= 0.2
            suggestions.append({
                "dimension": "efficiency",
                "suggestion": "Consider data compression or pagination",
                "impact": "high"
            })
        
        if isinstance(data, list) and len(data) > 1000:
            issues.append({
                "dimension": "efficiency",
                "issue": f"Large dataset ({len(data)} records) without pagination",
                "severity": "warning"
            })
            score -= 0.1
        
        return max(0, score), issues, suggestions
    
    async def _assess_precision(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data precision - level of detail and granularity"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Check numeric precision
                if isinstance(value, float):
                    decimal_places = len(str(value).split('.')[-1]) if '.' in str(value) else 0
                    if decimal_places > 10:
                        issues.append({
                            "dimension": "precision",
                            "field": key,
                            "issue": "Excessive decimal precision",
                            "severity": "info"
                        })
                        score -= 0.05
                
                # Check timestamp precision
                if 'timestamp' in key.lower() and isinstance(value, str):
                    if not re.match(r'.*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}', value):
                        issues.append({
                            "dimension": "precision",
                            "field": key,
                            "issue": "Low timestamp precision (missing time component)",
                            "severity": "warning"
                        })
                        score -= 0.1
        
        return max(0, score), issues, suggestions
    
    async def _assess_understandability(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess how clear and understandable the data is"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            # Check for cryptic field names
            for key in data.keys():
                if len(key) < 3 or key.isupper() or '_' not in key and len(key) > 20:
                    issues.append({
                        "dimension": "understandability",
                        "field": key,
                        "issue": "Field name not self-documenting",
                        "severity": "info"
                    })
                    score -= 0.05
            
            # Check for missing units in numeric fields
            numeric_fields = [(k, v) for k, v in data.items() if isinstance(v, (int, float))]
            for key, value in numeric_fields:
                if not any(unit in key.lower() for unit in ['_ms', '_seconds', '_bytes', '_percent', '_count']):
                    if 'amount' in key.lower() or 'size' in key.lower() or 'duration' in key.lower():
                        issues.append({
                            "dimension": "understandability",
                            "field": key,
                            "issue": "Numeric field without unit indication",
                            "severity": "warning"
                        })
                        score -= 0.05
        
        return max(0, score), issues, suggestions
    
    # Enterprise dimension assessments
    async def _assess_confidentiality(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data confidentiality controls"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            # Check for sensitive data exposure
            sensitive_patterns = {
                'api_key': r'^[a-zA-Z0-9]{20,}$',
                'token': r'^[a-zA-Z0-9\-_]{20,}$',
                'secret': r'^[a-zA-Z0-9]{10,}$'
            }
            
            for key, value in data.items():
                for pattern_name, pattern in sensitive_patterns.items():
                    if pattern_name in key.lower() and isinstance(value, str):
                        if re.match(pattern, value):
                            issues.append({
                                "dimension": "confidentiality",
                                "field": key,
                                "issue": "Potential exposed sensitive data",
                                "severity": "critical"
                            })
                            score -= 0.4
                            suggestions.append({
                                "dimension": "confidentiality",
                                "suggestion": f"Remove or encrypt {key} before storage",
                                "impact": "critical"
                            })
        
        return max(0, score), issues, suggestions
    
    async def _assess_traceability(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data lineage and audit trail"""
        issues = []
        suggestions = []
        score = 1.0
        
        if isinstance(data, dict):
            # Check for audit fields
            audit_fields = ['created_at', 'updated_at', 'created_by', 'modified_by', 'version']
            missing_audit = [field for field in audit_fields if field not in data]
            
            if len(missing_audit) > 2:
                issues.append({
                    "dimension": "traceability",
                    "issue": f"Missing audit fields: {', '.join(missing_audit)}",
                    "severity": "warning"
                })
                score -= 0.1 * len(missing_audit)
                suggestions.append({
                    "dimension": "traceability",
                    "suggestion": "Add audit fields for better traceability",
                    "impact": "medium"
                })
        
        return max(0, score), issues, suggestions
    
    async def _assess_availability(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data availability and redundancy"""
        issues = []
        suggestions = []
        score = 0.95  # Default high availability
        
        # In a real implementation, this would check:
        # - Data replication status
        # - Backup timestamps
        # - Cache headers
        # - Failover configuration
        
        if isinstance(data, dict) and 'cache_control' not in data:
            issues.append({
                "dimension": "availability",
                "issue": "No cache control headers",
                "severity": "info"
            })
            score -= 0.05
        
        return max(0, score), issues, suggestions
    
    async def _assess_portability(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data portability between systems"""
        issues = []
        suggestions = []
        score = 1.0
        
        # Check for standard formats
        if isinstance(data, dict):
            # Check for non-portable data types
            for key, value in data.items():
                if isinstance(value, bytes):
                    issues.append({
                        "dimension": "portability",
                        "field": key,
                        "issue": "Binary data not portable across systems",
                        "severity": "warning"
                    })
                    score -= 0.1
                    suggestions.append({
                        "dimension": "portability",
                        "suggestion": "Encode binary data as base64",
                        "impact": "medium"
                    })
        
        return max(0, score), issues, suggestions
    
    async def _assess_recoverability(self, data: Any) -> tuple[float, List[Dict], List[Dict]]:
        """Assess data recoverability after failure"""
        issues = []
        suggestions = []
        score = 0.9  # Default good recoverability
        
        if isinstance(data, dict):
            # Check for transaction/checkpoint info
            if 'transaction_id' not in data and 'checkpoint' not in data:
                issues.append({
                    "dimension": "recoverability",
                    "issue": "No transaction or checkpoint information",
                    "severity": "info"
                })
                score -= 0.1
        
        return max(0, score), issues, suggestions
    
    # Helper methods
    def _get_max_depth(self, obj: Any, current_depth: int = 0) -> int:
        """Get maximum depth of nested structure"""
        if not isinstance(obj, (dict, list)):
            return current_depth
        
        if isinstance(obj, dict):
            if not obj:
                return current_depth
            return max(self._get_max_depth(v, current_depth + 1) for v in obj.values())
        else:  # list
            if not obj:
                return current_depth
            return max(self._get_max_depth(item, current_depth + 1) for item in obj)
    
    def _prepare_data(self, data: Any) -> Any:
        """Prepare data for assessment"""
        if isinstance(data, str):
            try:
                return json.loads(data)
            except:
                return {"content": data}
        return data
    
    def _generate_data_reference(self, data: Any) -> str:
        """Generate a reference identifier for the data"""
        if isinstance(data, dict) and 'id' in data:
            return f"data_{data['id']}"
        return f"data_{uuid4().hex[:8]}"
    
    def _generate_data_profile(self, data: Any) -> DataProfile:
        """Generate statistical profile of data"""
        profile = DataProfile()
        
        if isinstance(data, dict):
            profile.column_count = len(data)
            null_count = sum(1 for v in data.values() if v is None or v == "")
            profile.null_percentage = (null_count / len(data) * 100) if data else 0
            
            # Analyze data types
            type_counts = Counter()
            for v in data.values():
                type_counts[type(v).__name__] += 1
            profile.data_types = dict(type_counts)
            
        elif isinstance(data, list):
            profile.row_count = len(data)
            if data and isinstance(data[0], dict):
                profile.column_count = len(data[0])
        
        return profile
    
    def _get_usage_limit(self, edition: str) -> Optional[int]:
        """Get monthly usage limit for edition"""
        limits = {
            "community": 1000,
            "business": 100000,
            "enterprise": None  # Unlimited
        }
        return limits.get(edition)
    
    async def _get_current_usage(self, tenant_id: UUID) -> int:
        """Get current month's usage count"""
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        result = await self.db.execute(
            select(func.count(DataQualityAssessment.id)).where(
                and_(
                    DataQualityAssessment.tenant_id == tenant_id,
                    DataQualityAssessment.assessment_time >= start_of_month
                )
            )
        )
        count = result.scalar()
        
        return count or 0
    
    # Rule management methods
    async def create_rule(
        self,
        rule_data: RuleCreate,
        user_id: UUID,
        tenant_id: UUID,
        edition: str
    ) -> RuleResponse:
        """Create a new quality rule"""
        
        # Check if dimension is available in edition
        available_dimensions = self.get_available_dimensions(edition)
        if rule_data.dimension not in available_dimensions:
            raise FeatureNotAvailableError(
                f"Dimension {rule_data.dimension} not available in {edition} edition"
            )
        
        rule = QualityRule(
            id=uuid4(),
            name=rule_data.name,
            description=rule_data.description,
            dimension=rule_data.dimension.value,
            rule_type=rule_data.rule_type.value,
            rule_definition=rule_data.rule_definition,
            severity=rule_data.severity.value,
            edition_required=edition,
            created_by=user_id,
            tenant_id=tenant_id
        )
        
        self.db.add(rule)
        await self.db.commit()
        await self.db.refresh(rule)
        
        return RuleResponse.from_orm(rule)
    
    async def get_rules(
        self,
        tenant_id: UUID,
        dimension: Optional[str] = None,
        is_active: bool = True
    ) -> List[RuleResponse]:
        """Get quality rules"""
        query = select(QualityRule).where(
            and_(
                or_(
                    QualityRule.tenant_id == tenant_id,
                    QualityRule.is_system == True
                ),
                QualityRule.is_active == is_active
            )
        )
        
        if dimension:
            query = query.where(QualityRule.dimension == dimension)
        
        result = await self.db.execute(query)
        rules = result.scalars().all()
        return [RuleResponse.from_orm(rule) for rule in rules]
    
    @cache_result(prefix="quality_trends", expire=300)  # Cache for 5 minutes
    async def get_quality_metrics(
        self,
        tenant_id: UUID,
        time_range: int = 30  # days
    ) -> Dict[str, Any]:
        """Get quality metrics for dashboard"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=time_range)
        
        result = await self.db.execute(
            select(DataQualityAssessment).where(
                and_(
                    DataQualityAssessment.tenant_id == tenant_id,
                    DataQualityAssessment.assessment_time >= start_date
                )
            )
        )
        assessments = result.scalars().all()
        
        if not assessments:
            return {
                "total_assessments": 0,
                "average_score": 0,
                "assessments_by_dimension": {},
                "scores_by_dimension": {},
                "top_issues": [],
                "quality_trend": []
            }
        
        # Calculate metrics
        total = len(assessments)
        scores = [a.overall_score for a in assessments]
        avg_score = float(np.mean(scores)) if scores else 0.0
        
        # Aggregate by dimension
        dimension_counts = Counter()
        dimension_scores = {}
        
        for assessment in assessments:
            for dim, score in assessment.dimension_scores.items():
                dimension_counts[dim] += 1
                if dim not in dimension_scores:
                    dimension_scores[dim] = []
                dimension_scores[dim].append(score)
        
        # Calculate average scores by dimension
        avg_dimension_scores = {
            dim: float(np.mean(scores)) if scores else 0.0 
            for dim, scores in dimension_scores.items()
        }
        
        # Get top issues
        all_issues = []
        for assessment in assessments:
            all_issues.extend(assessment.issues_found or [])
        
        issue_counts = Counter(issue['issue'] for issue in all_issues if 'issue' in issue)
        top_issues = [
            {"issue": issue, "count": count}
            for issue, count in issue_counts.most_common(10)
        ]
        
        # Generate trend data
        daily_scores = {}
        for assessment in assessments:
            date_key = assessment.assessment_time.date().isoformat()
            if date_key not in daily_scores:
                daily_scores[date_key] = []
            daily_scores[date_key].append(assessment.overall_score)
        
        quality_trend = [
            {"date": date, "average_score": float(np.mean(scores)) if scores else 0.0}
            for date, scores in sorted(daily_scores.items())
        ]
        
        return {
            "total_assessments": total,
            "average_score": float(avg_score) if not np.isnan(avg_score) else 0.0,
            "assessments_by_dimension": dict(dimension_counts),
            "scores_by_dimension": {
                k: (float(v) if not np.isnan(v) else 0.0) 
                for k, v in avg_dimension_scores.items()
            },
            "top_issues": top_issues,
            "quality_trend": quality_trend
        }