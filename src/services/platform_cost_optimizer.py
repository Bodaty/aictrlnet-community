"""Platform cost optimization service with ML-powered predictions"""
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import statistics
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import selectinload

from models.platform_integration import (
    PlatformExecution, PlatformCredential, PlatformHealth,
    PlatformAdapter
)
from schemas.platform_integration import PlatformType, ExecutionStatus
from core.cache import RedisCache
from core.events import event_bus
from services.platform_monitoring_service import PlatformMonitoringService


class PlatformCostOptimizer:
    """ML-powered cost optimization for platform integrations"""
    
    def __init__(self, db: AsyncSession, cache: RedisCache = None):
        self.db = db
        self.cache = cache or RedisCache()
        self.monitoring_service = PlatformMonitoringService(db, cache)
        
        # Cost models per platform (cents per execution minute)
        self.platform_cost_models = {
            PlatformType.N8N: {
                "base_cost": 0.01,  # $0.01 per execution
                "per_minute": 0.005,  # $0.005 per minute
                "data_gb": 0.10,  # $0.10 per GB transferred
                "free_tier_minutes": 1000,  # 1000 minutes free
                "volume_discount": 0.1  # 10% discount after 10k executions
            },
            PlatformType.ZAPIER: {
                "base_cost": 0.05,
                "per_minute": 0.01,
                "data_gb": 0.15,
                "free_tier_minutes": 100,
                "volume_discount": 0.15
            },
            PlatformType.MAKE: {
                "base_cost": 0.03,
                "per_minute": 0.008,
                "data_gb": 0.12,
                "free_tier_minutes": 500,
                "volume_discount": 0.12
            },
            PlatformType.POWER_AUTOMATE: {
                "base_cost": 0.02,
                "per_minute": 0.006,
                "data_gb": 0.08,
                "free_tier_minutes": 2000,
                "volume_discount": 0.2
            },
            PlatformType.IFTTT: {
                "base_cost": 0.00,  # Free tier
                "per_minute": 0.00,
                "data_gb": 0.00,
                "free_tier_minutes": float('inf'),
                "volume_discount": 0.0
            }
        }
        
        # ML model parameters for cost prediction
        self.cost_prediction_weights = {
            "historical_cost": 0.4,
            "execution_time": 0.3,
            "data_volume": 0.2,
            "failure_rate": 0.1
        }
    
    async def predict_execution_cost(
        self,
        platform: PlatformType,
        workflow_id: str,
        estimated_duration_ms: Optional[int] = None,
        estimated_data_mb: Optional[float] = None,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Predict cost for a workflow execution using ML"""
        
        # Get historical data for this workflow
        historical_data = await self._get_workflow_history(
            platform, workflow_id, user_id
        )
        
        # Use historical averages if not provided
        if not estimated_duration_ms and historical_data["executions"]:
            durations = [e.duration_ms for e in historical_data["executions"] if e.duration_ms]
            estimated_duration_ms = int(statistics.mean(durations)) if durations else 5000
        
        if not estimated_data_mb and historical_data["executions"]:
            # Estimate based on payload sizes
            estimated_data_mb = historical_data.get("avg_data_mb", 1.0)
        
        # Get platform cost model
        cost_model = self.platform_cost_models.get(
            platform, 
            self.platform_cost_models[PlatformType.N8N]
        )
        
        # Calculate base cost
        duration_minutes = (estimated_duration_ms or 5000) / 60000.0
        data_gb = (estimated_data_mb or 1.0) / 1024.0
        
        base_cost = cost_model["base_cost"]
        time_cost = duration_minutes * cost_model["per_minute"]
        data_cost = data_gb * cost_model["data_gb"]
        
        # Apply free tier
        if user_id:
            used_minutes = await self._get_monthly_usage_minutes(user_id, platform)
            if used_minutes < cost_model["free_tier_minutes"]:
                time_cost = 0
        
        # Apply volume discount
        if historical_data["total_executions"] > 10000:
            discount = cost_model["volume_discount"]
            base_cost *= (1 - discount)
            time_cost *= (1 - discount)
            data_cost *= (1 - discount)
        
        # ML adjustment based on failure rate
        if historical_data["failure_rate"] > 0.1:
            # Higher costs due to retries
            retry_multiplier = 1 + historical_data["failure_rate"]
            base_cost *= retry_multiplier
            time_cost *= retry_multiplier
        
        total_cost = base_cost + time_cost + data_cost
        
        # Confidence calculation
        confidence = min(0.95, 0.5 + (historical_data["total_executions"] / 1000) * 0.45)
        
        return {
            "predicted_cost": round(total_cost, 4),
            "cost_breakdown": {
                "base_cost": round(base_cost, 4),
                "time_cost": round(time_cost, 4),
                "data_cost": round(data_cost, 4)
            },
            "confidence": confidence,
            "historical_avg_cost": historical_data.get("avg_cost", 0),
            "estimated_duration_ms": estimated_duration_ms,
            "estimated_data_mb": estimated_data_mb,
            "free_tier_remaining": max(0, cost_model["free_tier_minutes"] - used_minutes) if user_id else 0,
            "volume_discount_applied": historical_data["total_executions"] > 10000
        }
    
    async def _get_workflow_history(
        self,
        platform: PlatformType,
        workflow_id: str,
        user_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get historical execution data for a workflow"""
        query = select(PlatformExecution).where(
            and_(
                PlatformExecution.platform == platform.value,
                PlatformExecution.external_workflow_id == workflow_id
            )
        )
        
        if user_id:
            query = query.join(PlatformCredential).where(
                PlatformCredential.user_id == user_id
            )
        
        result = await self.db.execute(query)
        executions = result.scalars().all()
        
        if not executions:
            return {
                "executions": [],
                "total_executions": 0,
                "failure_rate": 0,
                "avg_cost": 0,
                "avg_data_mb": 1.0
            }
        
        total = len(executions)
        failures = sum(1 for e in executions if e.status == "failed")
        costs = [e.estimated_cost for e in executions if e.estimated_cost]
        
        # Estimate data volume from payloads
        data_sizes = []
        for exec in executions:
            size_mb = 0
            if exec.input_data:
                size_mb += len(str(exec.input_data)) / (1024 * 1024)
            if exec.output_data:
                size_mb += len(str(exec.output_data)) / (1024 * 1024)
            data_sizes.append(size_mb)
        
        return {
            "executions": executions,
            "total_executions": total,
            "failure_rate": failures / total if total > 0 else 0,
            "avg_cost": statistics.mean(costs) if costs else 0,
            "avg_data_mb": statistics.mean(data_sizes) if data_sizes else 1.0
        }
    
    async def _get_monthly_usage_minutes(
        self,
        user_id: str,
        platform: PlatformType
    ) -> float:
        """Get total execution minutes for current month"""
        start_of_month = datetime.utcnow().replace(day=1, hour=0, minute=0, second=0)
        
        result = await self.db.execute(
            select(func.sum(PlatformExecution.duration_ms))
            .join(PlatformCredential)
            .where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.platform == platform.value,
                    PlatformExecution.started_at >= start_of_month,
                    PlatformExecution.status == "completed"
                )
            )
        )
        
        total_ms = result.scalar() or 0
        return total_ms / 60000.0
    
    async def optimize_workflow_routing(
        self,
        workflow_type: str,
        requirements: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Determine optimal platform for workflow execution"""
        
        # Get available platforms for user
        available_platforms = await self._get_user_platforms(user_id)
        
        if not available_platforms:
            return {
                "recommended_platform": None,
                "reason": "No platforms configured",
                "alternatives": []
            }
        
        # Score each platform
        platform_scores = []
        
        for platform in available_platforms:
            score = await self._score_platform_for_workflow(
                platform, workflow_type, requirements, user_id
            )
            platform_scores.append((platform, score))
        
        # Sort by score
        platform_scores.sort(key=lambda x: x[1]["total_score"], reverse=True)
        
        # Get top recommendation
        best_platform, best_score = platform_scores[0]
        
        # Get alternatives
        alternatives = [
            {
                "platform": p.value,
                "score": s["total_score"],
                "estimated_cost": s["cost_score"],
                "estimated_duration": s["performance_score"]
            }
            for p, s in platform_scores[1:4]  # Top 3 alternatives
        ]
        
        return {
            "recommended_platform": best_platform.value,
            "reason": self._generate_recommendation_reason(best_score),
            "estimated_cost": best_score["estimated_cost"],
            "estimated_duration_ms": best_score["estimated_duration"],
            "confidence": best_score["confidence"],
            "alternatives": alternatives,
            "optimization_factors": {
                "cost_weight": requirements.get("cost_priority", 0.5),
                "performance_weight": requirements.get("performance_priority", 0.5),
                "reliability_weight": requirements.get("reliability_priority", 0.5)
            }
        }
    
    async def _get_user_platforms(self, user_id: str) -> List[PlatformType]:
        """Get platforms with active credentials for user"""
        result = await self.db.execute(
            select(PlatformCredential.platform)
            .where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformCredential.is_active == True
                )
            )
            .distinct()
        )
        
        platforms = result.scalars().all()
        return [PlatformType(p) for p in platforms]
    
    async def _score_platform_for_workflow(
        self,
        platform: PlatformType,
        workflow_type: str,
        requirements: Dict[str, Any],
        user_id: str
    ) -> Dict[str, Any]:
        """Score a platform for specific workflow requirements"""
        
        # Get platform health
        health = await self._get_platform_health(platform)
        
        # Get historical performance
        analytics = await self.monitoring_service.get_cross_platform_analytics(
            user_id=user_id,
            start_date=datetime.utcnow() - timedelta(days=30)
        )
        
        platform_metrics = analytics.get("platform_comparison", {}).get(platform.value, {})
        
        # Calculate scores
        cost_score = await self._calculate_cost_score(
            platform, workflow_type, requirements
        )
        
        performance_score = self._calculate_performance_score(
            platform_metrics, requirements
        )
        
        reliability_score = self._calculate_reliability_score(
            platform_metrics, health
        )
        
        # Weight scores based on requirements
        weights = {
            "cost": requirements.get("cost_priority", 0.33),
            "performance": requirements.get("performance_priority", 0.33),
            "reliability": requirements.get("reliability_priority", 0.34)
        }
        
        total_score = (
            cost_score["score"] * weights["cost"] +
            performance_score * weights["performance"] +
            reliability_score * weights["reliability"]
        )
        
        return {
            "total_score": total_score,
            "cost_score": cost_score["score"],
            "performance_score": performance_score,
            "reliability_score": reliability_score,
            "estimated_cost": cost_score["estimated_cost"],
            "estimated_duration": platform_metrics.get("avg_duration_ms", 5000),
            "confidence": min(0.9, 0.5 + len(platform_metrics) / 1000 * 0.4)
        }
    
    async def _calculate_cost_score(
        self,
        platform: PlatformType,
        workflow_type: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate cost score for platform (0-1, higher is better)"""
        
        # Predict cost
        prediction = await self.predict_execution_cost(
            platform=platform,
            workflow_id=workflow_type,
            estimated_duration_ms=requirements.get("estimated_duration_ms"),
            estimated_data_mb=requirements.get("estimated_data_mb")
        )
        
        estimated_cost = prediction["predicted_cost"]
        
        # Normalize score (inverse of cost, capped at $1)
        score = max(0, 1 - (estimated_cost / 1.0))
        
        return {
            "score": score,
            "estimated_cost": estimated_cost
        }
    
    def _calculate_performance_score(
        self,
        platform_metrics: Dict[str, Any],
        requirements: Dict[str, Any]
    ) -> float:
        """Calculate performance score (0-1, higher is better)"""
        
        avg_duration = platform_metrics.get("avg_duration_ms", 10000)
        p95_duration = platform_metrics.get("p95_duration_ms", 30000)
        
        # Target duration from requirements
        target_duration = requirements.get("target_duration_ms", 5000)
        
        # Score based on how close to target
        if avg_duration <= target_duration:
            score = 1.0
        else:
            # Decay score as duration increases
            score = max(0, 1 - ((avg_duration - target_duration) / target_duration) * 0.5)
        
        # Penalty for high variance (p95 much higher than average)
        if p95_duration > avg_duration * 2:
            score *= 0.8
        
        return score
    
    def _calculate_reliability_score(
        self,
        platform_metrics: Dict[str, Any],
        health: Dict[str, Any]
    ) -> float:
        """Calculate reliability score (0-1, higher is better)"""
        
        success_rate = platform_metrics.get("success_rate", 0.9)
        reliability_score = platform_metrics.get("reliability_score", 0.9)
        is_healthy = health.get("is_healthy", True)
        
        # Base score from success rate
        score = success_rate
        
        # Adjust for reliability metrics
        score = (score + reliability_score) / 2
        
        # Penalty for unhealthy platform
        if not is_healthy:
            score *= 0.5
        
        return score
    
    async def _get_platform_health(self, platform: PlatformType) -> Dict[str, Any]:
        """Get current platform health status"""
        result = await self.db.execute(
            select(PlatformHealth)
            .where(PlatformHealth.platform == platform.value)
            .order_by(PlatformHealth.last_check_at.desc())
            .limit(1)
        )
        
        health = result.scalar_one_or_none()
        
        if health:
            return {
                "is_healthy": health.is_healthy,
                "response_time_ms": health.response_time_ms,
                "consecutive_failures": health.consecutive_failures
            }
        
        return {"is_healthy": True, "response_time_ms": 100, "consecutive_failures": 0}
    
    def _generate_recommendation_reason(self, score: Dict[str, Any]) -> str:
        """Generate human-readable recommendation reason"""
        reasons = []
        
        if score["cost_score"] > 0.8:
            reasons.append("lowest cost")
        elif score["cost_score"] > 0.6:
            reasons.append("competitive pricing")
        
        if score["performance_score"] > 0.8:
            reasons.append("fastest execution")
        elif score["performance_score"] > 0.6:
            reasons.append("good performance")
        
        if score["reliability_score"] > 0.9:
            reasons.append("highest reliability")
        elif score["reliability_score"] > 0.7:
            reasons.append("reliable service")
        
        if not reasons:
            reasons.append("balanced option")
        
        return f"Recommended for {' and '.join(reasons)}"
    
    async def create_budget_alert(
        self,
        user_id: str,
        budget_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create budget alert configuration"""
        
        alert_id = f"budget_{user_id}_{datetime.utcnow().timestamp()}"
        
        # Store in cache (in production, use database)
        await self.cache.set(
            f"budget_alert:{alert_id}",
            {
                "user_id": user_id,
                "monthly_limit": budget_config.get("monthly_limit", 100),
                "daily_limit": budget_config.get("daily_limit"),
                "per_execution_limit": budget_config.get("per_execution_limit"),
                "alert_thresholds": budget_config.get("alert_thresholds", [0.5, 0.8, 0.95]),
                "notification_channels": budget_config.get("notification_channels", ["email"]),
                "created_at": datetime.utcnow().isoformat(),
                "is_active": True
            },
            ttl=86400 * 30  # 30 days
        )
        
        return {
            "alert_id": alert_id,
            "status": "created",
            "config": budget_config
        }
    
    async def check_budget_status(self, user_id: str) -> Dict[str, Any]:
        """Check current budget status and alerts"""
        
        # Get all budget alerts for user
        # In production, this would query from database
        budget_alerts = []
        
        # Calculate current spend
        current_spend = await self._calculate_current_spend(user_id)
        
        # Check against limits
        alerts_triggered = []
        
        for alert in budget_alerts:
            if alert["monthly_limit"] and current_spend["monthly"] > alert["monthly_limit"]:
                alerts_triggered.append({
                    "type": "monthly_limit_exceeded",
                    "limit": alert["monthly_limit"],
                    "current": current_spend["monthly"],
                    "severity": "critical"
                })
            
            # Check thresholds
            for threshold in alert.get("alert_thresholds", []):
                if current_spend["monthly"] > alert["monthly_limit"] * threshold:
                    alerts_triggered.append({
                        "type": "threshold_reached",
                        "threshold": threshold,
                        "limit": alert["monthly_limit"],
                        "current": current_spend["monthly"],
                        "severity": "warning" if threshold < 0.9 else "critical"
                    })
        
        return {
            "current_spend": current_spend,
            "budget_status": "ok" if not alerts_triggered else "alert",
            "alerts": alerts_triggered,
            "recommendations": self._generate_budget_recommendations(
                current_spend, alerts_triggered
            )
        }
    
    async def _calculate_current_spend(self, user_id: str) -> Dict[str, float]:
        """Calculate current spending for user"""
        now = datetime.utcnow()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0)
        start_of_day = now.replace(hour=0, minute=0, second=0)
        
        # Monthly spend
        monthly_result = await self.db.execute(
            select(func.sum(PlatformExecution.estimated_cost))
            .join(PlatformCredential)
            .where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.started_at >= start_of_month,
                    PlatformExecution.estimated_cost > 0
                )
            )
        )
        
        monthly_spend = (monthly_result.scalar() or 0) / 100  # Convert cents to dollars
        
        # Daily spend
        daily_result = await self.db.execute(
            select(func.sum(PlatformExecution.estimated_cost))
            .join(PlatformCredential)
            .where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.started_at >= start_of_day,
                    PlatformExecution.estimated_cost > 0
                )
            )
        )
        
        daily_spend = (daily_result.scalar() or 0) / 100
        
        return {
            "monthly": monthly_spend,
            "daily": daily_spend,
            "as_of": now.isoformat()
        }
    
    def _generate_budget_recommendations(
        self,
        current_spend: Dict[str, float],
        alerts: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate budget optimization recommendations"""
        recommendations = []
        
        if any(a["severity"] == "critical" for a in alerts):
            recommendations.append("Consider pausing non-critical workflows to avoid overage")
            recommendations.append("Review and optimize high-cost workflows")
        
        if current_spend["daily"] > current_spend["monthly"] / 30 * 1.5:
            recommendations.append("Daily spending is above average - investigate recent changes")
        
        recommendations.append("Use platform routing to select most cost-effective options")
        recommendations.append("Enable execution caching to reduce duplicate runs")
        
        return recommendations
    
    async def get_cost_optimization_opportunities(
        self,
        user_id: str,
        lookback_days: int = 30
    ) -> List[Dict[str, Any]]:
        """Identify cost optimization opportunities"""
        
        start_date = datetime.utcnow() - timedelta(days=lookback_days)
        
        # Get execution data
        result = await self.db.execute(
            select(PlatformExecution)
            .join(PlatformCredential)
            .where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.started_at >= start_date
                )
            )
            .options(selectinload(PlatformExecution.credential))
        )
        
        executions = result.scalars().all()
        
        opportunities = []
        
        # 1. Identify duplicate executions
        duplicate_opps = self._find_duplicate_execution_opportunities(executions)
        opportunities.extend(duplicate_opps)
        
        # 2. Platform switching opportunities
        switching_opps = await self._find_platform_switching_opportunities(
            executions, user_id
        )
        opportunities.extend(switching_opps)
        
        # 3. Timing optimization
        timing_opps = self._find_timing_optimization_opportunities(executions)
        opportunities.extend(timing_opps)
        
        # 4. Batch processing opportunities
        batch_opps = self._find_batch_processing_opportunities(executions)
        opportunities.extend(batch_opps)
        
        # Sort by potential savings
        opportunities.sort(key=lambda x: x["potential_savings"], reverse=True)
        
        return opportunities[:10]  # Top 10 opportunities
    
    def _find_duplicate_execution_opportunities(
        self,
        executions: List[PlatformExecution]
    ) -> List[Dict[str, Any]]:
        """Find opportunities to cache duplicate executions"""
        opportunities = []
        
        # Group by workflow and input hash
        execution_groups = defaultdict(list)
        
        for exec in executions:
            if exec.status == "completed" and exec.input_data:
                # Create hash of workflow + input
                input_hash = hash(str(exec.external_workflow_id) + str(exec.input_data))
                execution_groups[input_hash].append(exec)
        
        # Find groups with duplicates
        for input_hash, group in execution_groups.items():
            if len(group) > 1:
                total_cost = sum(e.estimated_cost or 0 for e in group) / 100
                potential_savings = total_cost * 0.9  # Could save 90% with caching
                
                opportunities.append({
                    "type": "duplicate_executions",
                    "description": f"Workflow {group[0].external_workflow_id} executed {len(group)} times with same input",
                    "potential_savings": potential_savings,
                    "recommendation": "Enable execution caching for this workflow",
                    "workflow_id": group[0].external_workflow_id,
                    "platform": group[0].platform,
                    "duplicate_count": len(group)
                })
        
        return opportunities
    
    async def _find_platform_switching_opportunities(
        self,
        executions: List[PlatformExecution],
        user_id: str
    ) -> List[Dict[str, Any]]:
        """Find opportunities to switch to cheaper platforms"""
        opportunities = []
        
        # Group by workflow
        workflow_groups = defaultdict(list)
        for exec in executions:
            workflow_groups[(exec.platform, exec.external_workflow_id)].append(exec)
        
        for (platform, workflow_id), group in workflow_groups.items():
            if len(group) < 5:  # Need enough data
                continue
            
            # Calculate current cost
            current_total_cost = sum(e.estimated_cost or 0 for e in group) / 100
            avg_duration = statistics.mean([e.duration_ms for e in group if e.duration_ms])
            
            # Check if cheaper platform available
            routing = await self.optimize_workflow_routing(
                workflow_type=workflow_id,
                requirements={
                    "estimated_duration_ms": avg_duration,
                    "cost_priority": 0.8
                },
                user_id=user_id
            )
            
            if routing["recommended_platform"] and routing["recommended_platform"] != platform:
                # Estimate savings
                new_cost_per_execution = routing["estimated_cost"]
                new_total_cost = new_cost_per_execution * len(group)
                potential_savings = current_total_cost - new_total_cost
                
                if potential_savings > 1:  # At least $1 savings
                    opportunities.append({
                        "type": "platform_switch",
                        "description": f"Switch workflow from {platform} to {routing['recommended_platform']}",
                        "potential_savings": potential_savings,
                        "recommendation": routing["reason"],
                        "current_platform": platform,
                        "recommended_platform": routing["recommended_platform"],
                        "workflow_id": workflow_id,
                        "execution_count": len(group)
                    })
        
        return opportunities
    
    def _find_timing_optimization_opportunities(
        self,
        executions: List[PlatformExecution]
    ) -> List[Dict[str, Any]]:
        """Find opportunities to run workflows at cheaper times"""
        opportunities = []
        
        # Group by hour of day
        hourly_costs = defaultdict(list)
        for exec in executions:
            if exec.started_at and exec.estimated_cost:
                hour = exec.started_at.hour
                hourly_costs[hour].append(exec.estimated_cost)
        
        # Find expensive hours
        avg_hourly_costs = {
            hour: statistics.mean(costs) / 100
            for hour, costs in hourly_costs.items()
        }
        
        if avg_hourly_costs:
            peak_hour = max(avg_hourly_costs.items(), key=lambda x: x[1])
            off_peak_hour = min(avg_hourly_costs.items(), key=lambda x: x[1])
            
            if peak_hour[1] > off_peak_hour[1] * 1.5:  # 50% more expensive
                potential_savings = (peak_hour[1] - off_peak_hour[1]) * len(hourly_costs[peak_hour[0]])
                
                opportunities.append({
                    "type": "timing_optimization",
                    "description": f"Shift workflows from peak hour {peak_hour[0]}:00 to off-peak {off_peak_hour[0]}:00",
                    "potential_savings": potential_savings,
                    "recommendation": "Schedule non-urgent workflows during off-peak hours",
                    "peak_hour": peak_hour[0],
                    "off_peak_hour": off_peak_hour[0],
                    "cost_difference": peak_hour[1] - off_peak_hour[1]
                })
        
        return opportunities
    
    def _find_batch_processing_opportunities(
        self,
        executions: List[PlatformExecution]
    ) -> List[Dict[str, Any]]:
        """Find opportunities to batch process workflows"""
        opportunities = []
        
        # Group by workflow and time window
        time_window = timedelta(minutes=5)
        workflow_time_groups = defaultdict(list)
        
        for exec in executions:
            if exec.started_at:
                window_start = exec.started_at.replace(
                    minute=(exec.started_at.minute // 5) * 5,
                    second=0,
                    microsecond=0
                )
                workflow_time_groups[(exec.external_workflow_id, window_start)].append(exec)
        
        # Find groups that could be batched
        for (workflow_id, window), group in workflow_time_groups.items():
            if len(group) > 3:  # Multiple executions in short time
                total_cost = sum(e.estimated_cost or 0 for e in group) / 100
                # Batching typically saves 30-50% on overhead
                potential_savings = total_cost * 0.3
                
                opportunities.append({
                    "type": "batch_processing",
                    "description": f"Batch {len(group)} executions of workflow {workflow_id}",
                    "potential_savings": potential_savings,
                    "recommendation": "Implement batch processing for rapid successive executions",
                    "workflow_id": workflow_id,
                    "execution_count": len(group),
                    "time_window": window.isoformat()
                })
        
        return opportunities