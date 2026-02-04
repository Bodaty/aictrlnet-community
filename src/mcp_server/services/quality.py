"""MCP Quality Service for quality assessment via MCP protocol."""

from typing import Dict, Any, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from datetime import datetime

from services.data_quality_service import DataQualityService, DataQualityAssessment
from core.exceptions import ValidationError

logger = logging.getLogger(__name__)


class MCPQualityService:
    """MCP service for quality assessment."""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.quality_service = DataQualityService(db)
        self._default_criteria = {
            "accuracy": {"weight": 0.3, "threshold": 0.8},
            "completeness": {"weight": 0.25, "threshold": 0.9},
            "relevance": {"weight": 0.25, "threshold": 0.85},
            "clarity": {"weight": 0.2, "threshold": 0.8}
        }
    
    async def assess_quality(
        self,
        content: str,
        content_type: str = "text",
        criteria: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Assess quality of content via MCP."""
        try:
            # Use provided criteria or defaults
            assessment_criteria = criteria or self._default_criteria
            
            # Prepare data for quality assessment
            data_to_assess = {
                "content": content,
                "type": content_type,
                "metadata": {
                    "source": "mcp",
                    "assessed_via": "mcp_protocol"
                }
            }
            
            # Perform quality assessment using real DataQualityService
            prepared_data = self.quality_service._prepare_data(data_to_assess.get("content", ""))

            dimensions = {}
            overall_score = 0.0
            total_weight = 0.0

            for criterion, config in assessment_criteria.items():
                weight = config.get("weight", 0.25)
                try:
                    score, issues, _ = await self.quality_service._assess_dimension(
                        prepared_data, criterion, "community"
                    )
                except Exception:
                    score = 0.5
                    issues = []

                dimensions[criterion] = {
                    "score": score,
                    "weight": weight,
                    "issues": issues or []
                }
                overall_score += score * weight
                total_weight += weight

            if total_weight > 0 and total_weight != 1.0:
                overall_score = overall_score / total_weight
            
            # Generate recommendations
            recommendations = self._generate_recommendations(dimensions, content_type)
            
            return {
                "quality_score": overall_score,
                "dimensions": dimensions,
                "recommendations": recommendations,
                "content_type": content_type,
                "metadata": {
                    "assessed_by": "AICtrlNet Quality Service",
                    "assessment_model": "iso-25012-based",
                    "assessed_at": datetime.utcnow().isoformat(),
                    "mcp_version": "1.0"
                }
            }
            
        except Exception as e:
            logger.error(f"MCP quality assessment failed: {str(e)}")
            return {
                "error": str(e),
                "quality_score": 0.0,
                "dimensions": {},
                "metadata": {
                    "assessed_by": "AICtrlNet Quality Service",
                    "status": "failed",
                    "failed_at": datetime.utcnow().isoformat()
                }
            }
    
    def _generate_recommendations(
        self, 
        dimensions: Dict[str, Any], 
        content_type: str
    ) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        for dim_name, dim_data in dimensions.items():
            score = dim_data.get("score", 0)
            issues = dim_data.get("issues", [])
            
            if score < 0.7:
                if dim_name == "accuracy":
                    recommendations.append(
                        "Consider fact-checking and verifying information accuracy"
                    )
                elif dim_name == "completeness":
                    recommendations.append(
                        "Add more details to make the content comprehensive"
                    )
                elif dim_name == "relevance":
                    recommendations.append(
                        "Ensure content is focused on the main topic"
                    )
                elif dim_name == "clarity":
                    recommendations.append(
                        "Simplify language and improve structure for better clarity"
                    )
            
            # Add specific issues as recommendations
            for issue in issues[:2]:  # Limit to 2 issues per dimension
                recommendations.append(f"{dim_name.capitalize()}: {issue}")
        
        # Content-type specific recommendations
        if content_type == "code":
            if any(dim["score"] < 0.8 for dim in dimensions.values()):
                recommendations.append("Consider adding comments and documentation")
                recommendations.append("Run linting and formatting tools")
        elif content_type == "json":
            recommendations.append("Validate JSON structure and schema compliance")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    async def batch_assess_quality(
        self,
        items: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess quality of multiple items."""
        results = []
        
        for item in items:
            result = await self.assess_quality(
                content=item.get("content", ""),
                content_type=item.get("content_type", "text"),
                criteria=item.get("criteria")
            )
            results.append({
                "item_id": item.get("id", ""),
                "quality_assessment": result
            })
        
        # Calculate aggregate metrics
        total_score = sum(r["quality_assessment"]["quality_score"] for r in results)
        avg_score = total_score / len(results) if results else 0
        
        return {
            "batch_size": len(items),
            "average_quality_score": avg_score,
            "results": results,
            "metadata": {
                "assessed_by": "AICtrlNet Quality Service",
                "batch_assessment": True,
                "assessed_at": datetime.utcnow().isoformat()
            }
        }