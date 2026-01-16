"""Advanced platform execution monitoring service"""
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
    PlatformWebhook, PlatformWebhookDelivery
)
from schemas.platform_integration import PlatformType, ExecutionStatus
from core.cache import RedisCache
from core.events import event_bus


class PlatformMonitoringService:
    """Advanced monitoring service for platform integrations"""
    
    def __init__(self, db: AsyncSession, cache: RedisCache = None):
        self.db = db
        self.cache = cache or RedisCache()
        self._monitoring_tasks = {}
        self._anomaly_thresholds = {
            "execution_time_stddev": 3,  # Standard deviations from mean
            "failure_rate_threshold": 0.3,  # 30% failure rate
            "response_time_spike": 2.0,  # 2x normal response time
            "consecutive_failures": 5
        }
    
    async def start_real_time_monitoring(self, user_id: str):
        """Start real-time monitoring for a user"""
        task_key = f"monitor_{user_id}"
        
        if task_key in self._monitoring_tasks:
            return {"status": "already_monitoring"}
        
        # Create monitoring task
        task = asyncio.create_task(self._monitor_executions(user_id))
        self._monitoring_tasks[task_key] = task
        
        return {"status": "monitoring_started"}
    
    async def stop_real_time_monitoring(self, user_id: str):
        """Stop real-time monitoring for a user"""
        task_key = f"monitor_{user_id}"
        
        if task_key in self._monitoring_tasks:
            self._monitoring_tasks[task_key].cancel()
            del self._monitoring_tasks[task_key]
            return {"status": "monitoring_stopped"}
        
        return {"status": "not_monitoring"}
    
    async def _monitor_executions(self, user_id: str):
        """Monitor executions in real-time"""
        while True:
            try:
                # Check for new executions
                recent_executions = await self._get_recent_executions(user_id)
                
                # Analyze for anomalies
                anomalies = await self._detect_anomalies(recent_executions)
                
                if anomalies:
                    # Publish anomaly events
                    for anomaly in anomalies:
                        await event_bus.publish(
                            "platform.anomaly_detected",
                            {
                                "user_id": user_id,
                                "anomaly": anomaly,
                                "timestamp": datetime.utcnow().isoformat()
                            }
                        )
                
                # Update real-time metrics
                metrics = await self._calculate_real_time_metrics(recent_executions)
                
                # Cache metrics for quick access
                await self.cache.set(
                    f"platform_metrics:{user_id}",
                    metrics,
                    ttl=60  # 1 minute TTL
                )
                
                # Publish metrics update
                await event_bus.publish(
                    "platform.metrics_updated",
                    {
                        "user_id": user_id,
                        "metrics": metrics
                    }
                )
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue monitoring
                print(f"Monitoring error for user {user_id}: {e}")
                await asyncio.sleep(10)
    
    async def _get_recent_executions(self, user_id: str, minutes: int = 5) -> List[PlatformExecution]:
        """Get executions from the last N minutes"""
        cutoff_time = datetime.utcnow() - timedelta(minutes=minutes)
        
        result = await self.db.execute(
            select(PlatformExecution)
            .join(PlatformCredential)
            .where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.started_at >= cutoff_time
                )
            )
            .order_by(PlatformExecution.started_at.desc())
            .limit(100)
        )
        
        return result.scalars().all()
    
    async def _detect_anomalies(self, executions: List[PlatformExecution]) -> List[Dict[str, Any]]:
        """Detect anomalies in execution patterns"""
        anomalies = []
        
        if not executions:
            return anomalies
        
        # Group by platform
        platform_executions = defaultdict(list)
        for exec in executions:
            platform_executions[exec.platform].append(exec)
        
        for platform, platform_execs in platform_executions.items():
            # Check execution time anomalies
            exec_times = [
                e.duration_ms for e in platform_execs 
                if e.duration_ms is not None
            ]
            
            if len(exec_times) > 5:
                mean_time = statistics.mean(exec_times)
                stddev = statistics.stdev(exec_times)
                
                for exec in platform_execs:
                    if exec.duration_ms and abs(exec.duration_ms - mean_time) > self._anomaly_thresholds["execution_time_stddev"] * stddev:
                        anomalies.append({
                            "type": "execution_time_anomaly",
                            "platform": platform,
                            "execution_id": exec.id,
                            "duration_ms": exec.duration_ms,
                            "expected_range": (
                                mean_time - 2 * stddev,
                                mean_time + 2 * stddev
                            )
                        })
            
            # Check failure rate
            total = len(platform_execs)
            failures = sum(1 for e in platform_execs if e.status == "failed")
            
            if total > 0:
                failure_rate = failures / total
                if failure_rate > self._anomaly_thresholds["failure_rate_threshold"]:
                    anomalies.append({
                        "type": "high_failure_rate",
                        "platform": platform,
                        "failure_rate": failure_rate,
                        "failures": failures,
                        "total": total
                    })
            
            # Check consecutive failures
            consecutive_failures = 0
            for exec in sorted(platform_execs, key=lambda x: x.started_at):
                if exec.status == "failed":
                    consecutive_failures += 1
                    if consecutive_failures >= self._anomaly_thresholds["consecutive_failures"]:
                        anomalies.append({
                            "type": "consecutive_failures",
                            "platform": platform,
                            "count": consecutive_failures,
                            "start_time": exec.started_at
                        })
                        break
                else:
                    consecutive_failures = 0
        
        return anomalies
    
    async def _calculate_real_time_metrics(self, executions: List[PlatformExecution]) -> Dict[str, Any]:
        """Calculate real-time metrics from recent executions"""
        metrics = {
            "total_executions": len(executions),
            "platforms": {},
            "overall": {
                "success_rate": 0,
                "avg_duration_ms": 0,
                "active_platforms": 0
            }
        }
        
        if not executions:
            return metrics
        
        # Group by platform
        platform_executions = defaultdict(list)
        for exec in executions:
            platform_executions[exec.platform].append(exec)
        
        # Calculate per-platform metrics
        total_success = 0
        total_duration = 0
        total_with_duration = 0
        
        for platform, platform_execs in platform_executions.items():
            success_count = sum(1 for e in platform_execs if e.status == "completed")
            total_count = len(platform_execs)
            
            durations = [e.duration_ms for e in platform_execs if e.duration_ms is not None]
            avg_duration = statistics.mean(durations) if durations else 0
            
            metrics["platforms"][platform] = {
                "total": total_count,
                "success": success_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "avg_duration_ms": avg_duration,
                "min_duration_ms": min(durations) if durations else 0,
                "max_duration_ms": max(durations) if durations else 0,
                "active": any(e.status == "running" for e in platform_execs)
            }
            
            total_success += success_count
            if durations:
                total_duration += sum(durations)
                total_with_duration += len(durations)
        
        # Calculate overall metrics
        metrics["overall"]["success_rate"] = total_success / len(executions) if executions else 0
        metrics["overall"]["avg_duration_ms"] = total_duration / total_with_duration if total_with_duration > 0 else 0
        metrics["overall"]["active_platforms"] = len(platform_executions)
        
        return metrics
    
    async def get_cross_platform_analytics(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Get comprehensive cross-platform analytics"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()
        
        # Get all executions in date range
        result = await self.db.execute(
            select(PlatformExecution)
            .join(PlatformCredential)
            .where(
                and_(
                    PlatformCredential.user_id == user_id,
                    PlatformExecution.started_at >= start_date,
                    PlatformExecution.started_at <= end_date
                )
            )
            .options(selectinload(PlatformExecution.credential))
        )
        
        executions = result.scalars().all()
        
        analytics = {
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": await self._calculate_summary_analytics(executions),
            "platform_comparison": await self._calculate_platform_comparison(executions),
            "time_series": await self._calculate_time_series(executions, start_date, end_date),
            "cost_analysis": await self._calculate_cost_analysis(executions),
            "reliability_metrics": await self._calculate_reliability_metrics(executions),
            "performance_trends": await self._calculate_performance_trends(executions)
        }
        
        return analytics
    
    async def _calculate_summary_analytics(self, executions: List[PlatformExecution]) -> Dict[str, Any]:
        """Calculate summary analytics"""
        if not executions:
            return {
                "total_executions": 0,
                "total_platforms": 0,
                "overall_success_rate": 0,
                "total_cost": 0,
                "avg_duration_ms": 0
            }
        
        platforms = set(e.platform for e in executions)
        success_count = sum(1 for e in executions if e.status == "completed")
        total_cost = sum(e.estimated_cost for e in executions if e.estimated_cost)
        durations = [e.duration_ms for e in executions if e.duration_ms is not None]
        
        return {
            "total_executions": len(executions),
            "total_platforms": len(platforms),
            "overall_success_rate": success_count / len(executions),
            "total_cost": total_cost,
            "avg_duration_ms": statistics.mean(durations) if durations else 0,
            "median_duration_ms": statistics.median(durations) if durations else 0,
            "p95_duration_ms": np.percentile(durations, 95) if durations else 0
        }
    
    async def _calculate_platform_comparison(self, executions: List[PlatformExecution]) -> Dict[str, Any]:
        """Compare metrics across platforms"""
        platform_metrics = {}
        
        # Group by platform
        platform_executions = defaultdict(list)
        for exec in executions:
            platform_executions[exec.platform].append(exec)
        
        for platform, platform_execs in platform_executions.items():
            success_count = sum(1 for e in platform_execs if e.status == "completed")
            failure_count = sum(1 for e in platform_execs if e.status == "failed")
            total_count = len(platform_execs)
            
            durations = [e.duration_ms for e in platform_execs if e.duration_ms is not None]
            costs = [e.estimated_cost for e in platform_execs if e.estimated_cost]
            
            platform_metrics[platform] = {
                "total_executions": total_count,
                "success_rate": success_count / total_count if total_count > 0 else 0,
                "failure_rate": failure_count / total_count if total_count > 0 else 0,
                "avg_duration_ms": statistics.mean(durations) if durations else 0,
                "median_duration_ms": statistics.median(durations) if durations else 0,
                "p95_duration_ms": np.percentile(durations, 95) if durations else 0,
                "total_cost": sum(costs),
                "avg_cost_per_execution": statistics.mean(costs) if costs else 0,
                "reliability_score": self._calculate_reliability_score(platform_execs)
            }
        
        return platform_metrics
    
    async def _calculate_time_series(
        self,
        executions: List[PlatformExecution],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, Any]:
        """Calculate time series data"""
        # Group by day
        daily_data = defaultdict(lambda: {
            "total": 0,
            "success": 0,
            "failed": 0,
            "platforms": set(),
            "total_duration_ms": 0,
            "execution_count_with_duration": 0
        })
        
        for exec in executions:
            day_key = exec.started_at.date().isoformat()
            daily_data[day_key]["total"] += 1
            daily_data[day_key]["platforms"].add(exec.platform)
            
            if exec.status == "completed":
                daily_data[day_key]["success"] += 1
            elif exec.status == "failed":
                daily_data[day_key]["failed"] += 1
            
            if exec.duration_ms:
                daily_data[day_key]["total_duration_ms"] += exec.duration_ms
                daily_data[day_key]["execution_count_with_duration"] += 1
        
        # Convert to list format
        time_series = []
        current_date = start_date.date()
        
        while current_date <= end_date.date():
            day_key = current_date.isoformat()
            data = daily_data.get(day_key, {
                "total": 0,
                "success": 0,
                "failed": 0,
                "platforms": set(),
                "total_duration_ms": 0,
                "execution_count_with_duration": 0
            })
            
            time_series.append({
                "date": day_key,
                "total_executions": data["total"],
                "success_count": data["success"],
                "failure_count": data["failed"],
                "success_rate": data["success"] / data["total"] if data["total"] > 0 else 0,
                "active_platforms": len(data["platforms"]),
                "avg_duration_ms": (
                    data["total_duration_ms"] / data["execution_count_with_duration"]
                    if data["execution_count_with_duration"] > 0 else 0
                )
            })
            
            current_date += timedelta(days=1)
        
        return {"daily": time_series}
    
    async def _calculate_cost_analysis(self, executions: List[PlatformExecution]) -> Dict[str, Any]:
        """Analyze costs across platforms and workflows"""
        cost_by_platform = defaultdict(float)
        cost_by_workflow = defaultdict(float)
        
        for exec in executions:
            if exec.estimated_cost:
                cost_by_platform[exec.platform] += exec.estimated_cost
                cost_by_workflow[exec.workflow_id] += exec.estimated_cost
        
        # Get top cost workflows
        top_workflows = sorted(
            cost_by_workflow.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "by_platform": dict(cost_by_platform),
            "total_cost": sum(cost_by_platform.values()),
            "top_cost_workflows": [
                {"workflow_id": wf_id, "total_cost": cost}
                for wf_id, cost in top_workflows
            ],
            "cost_distribution": {
                "min": min(cost_by_platform.values()) if cost_by_platform else 0,
                "max": max(cost_by_platform.values()) if cost_by_platform else 0,
                "avg": (
                    sum(cost_by_platform.values()) / len(cost_by_platform)
                    if cost_by_platform else 0
                )
            }
        }
    
    async def _calculate_reliability_metrics(self, executions: List[PlatformExecution]) -> Dict[str, Any]:
        """Calculate reliability metrics"""
        # Group by platform
        platform_reliability = {}
        
        platform_executions = defaultdict(list)
        for exec in executions:
            platform_executions[exec.platform].append(exec)
        
        for platform, platform_execs in platform_executions.items():
            # Calculate MTBF (Mean Time Between Failures)
            failures = [e for e in platform_execs if e.status == "failed"]
            
            if len(failures) > 1:
                failure_intervals = []
                sorted_failures = sorted(failures, key=lambda x: x.started_at)
                
                for i in range(1, len(sorted_failures)):
                    interval = (sorted_failures[i].started_at - sorted_failures[i-1].started_at).total_seconds()
                    failure_intervals.append(interval)
                
                mtbf = statistics.mean(failure_intervals) if failure_intervals else 0
            else:
                mtbf = 0
            
            # Calculate availability
            total_time = sum(e.duration_ms or 0 for e in platform_execs)
            failed_time = sum(e.duration_ms or 0 for e in platform_execs if e.status == "failed")
            availability = (total_time - failed_time) / total_time if total_time > 0 else 0
            
            platform_reliability[platform] = {
                "mtbf_seconds": mtbf,
                "availability": availability,
                "reliability_score": self._calculate_reliability_score(platform_execs),
                "total_failures": len(failures),
                "failure_patterns": self._analyze_failure_patterns(failures)
            }
        
        return platform_reliability
    
    async def _calculate_performance_trends(self, executions: List[PlatformExecution]) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        # Group by week
        weekly_performance = defaultdict(lambda: {
            "durations": [],
            "success_rates": [],
            "platforms": defaultdict(list)
        })
        
        for exec in executions:
            week_key = exec.started_at.isocalendar()[:2]  # (year, week_number)
            
            if exec.duration_ms:
                weekly_performance[week_key]["durations"].append(exec.duration_ms)
                weekly_performance[week_key]["platforms"][exec.platform].append(exec.duration_ms)
        
        # Calculate trends
        trends = []
        for week_key in sorted(weekly_performance.keys()):
            week_data = weekly_performance[week_key]
            
            trend_point = {
                "year": week_key[0],
                "week": week_key[1],
                "avg_duration_ms": statistics.mean(week_data["durations"]) if week_data["durations"] else 0,
                "platforms": {}
            }
            
            for platform, durations in week_data["platforms"].items():
                trend_point["platforms"][platform] = {
                    "avg_duration_ms": statistics.mean(durations) if durations else 0,
                    "execution_count": len(durations)
                }
            
            trends.append(trend_point)
        
        # Calculate trend direction (improving/degrading)
        if len(trends) >= 2:
            recent_avg = statistics.mean([t["avg_duration_ms"] for t in trends[-4:]])
            older_avg = statistics.mean([t["avg_duration_ms"] for t in trends[:-4]])
            
            if recent_avg < older_avg * 0.9:
                trend_direction = "improving"
            elif recent_avg > older_avg * 1.1:
                trend_direction = "degrading"
            else:
                trend_direction = "stable"
        else:
            trend_direction = "insufficient_data"
        
        return {
            "weekly_trends": trends,
            "trend_direction": trend_direction
        }
    
    def _calculate_reliability_score(self, executions: List[PlatformExecution]) -> float:
        """Calculate reliability score (0-1) for a set of executions"""
        if not executions:
            return 0.0
        
        success_count = sum(1 for e in executions if e.status == "completed")
        total_count = len(executions)
        
        # Basic success rate
        success_rate = success_count / total_count
        
        # Penalty for consecutive failures
        consecutive_failures = 0
        max_consecutive = 0
        
        for exec in sorted(executions, key=lambda x: x.started_at):
            if exec.status == "failed":
                consecutive_failures += 1
                max_consecutive = max(max_consecutive, consecutive_failures)
            else:
                consecutive_failures = 0
        
        # Apply penalty
        penalty = min(0.3, max_consecutive * 0.05)
        
        return max(0, success_rate - penalty)
    
    def _analyze_failure_patterns(self, failures: List[PlatformExecution]) -> Dict[str, Any]:
        """Analyze patterns in failures"""
        if not failures:
            return {"patterns": []}
        
        patterns = []
        
        # Time-based patterns
        failure_hours = [f.started_at.hour for f in failures]
        if failure_hours:
            most_common_hour = max(set(failure_hours), key=failure_hours.count)
            hour_frequency = failure_hours.count(most_common_hour) / len(failure_hours)
            
            if hour_frequency > 0.3:  # 30% of failures in same hour
                patterns.append({
                    "type": "time_based",
                    "description": f"High failure rate at hour {most_common_hour}",
                    "frequency": hour_frequency
                })
        
        # Error type patterns
        error_types = defaultdict(int)
        for failure in failures:
            if failure.error_data:
                error_type = failure.error_data.get("type", "unknown")
                error_types[error_type] += 1
        
        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            if most_common_error[1] / len(failures) > 0.3:
                patterns.append({
                    "type": "error_type",
                    "description": f"Common error: {most_common_error[0]}",
                    "frequency": most_common_error[1] / len(failures)
                })
        
        return {"patterns": patterns}
    
    async def get_performance_recommendations(
        self,
        user_id: str,
        analytics: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Generate performance recommendations based on analytics"""
        if not analytics:
            analytics = await self.get_cross_platform_analytics(user_id)
        
        recommendations = []
        
        # Platform-specific recommendations
        platform_comparison = analytics.get("platform_comparison", {})
        
        for platform, metrics in platform_comparison.items():
            # High failure rate
            if metrics["failure_rate"] > 0.2:
                recommendations.append({
                    "type": "reliability",
                    "severity": "high",
                    "platform": platform,
                    "title": f"High failure rate on {platform}",
                    "description": f"{platform} has a {metrics['failure_rate']:.1%} failure rate",
                    "action": "Review error logs and consider adjusting retry settings"
                })
            
            # Slow performance
            if metrics["p95_duration_ms"] > 30000:  # 30 seconds
                recommendations.append({
                    "type": "performance",
                    "severity": "medium",
                    "platform": platform,
                    "title": f"Slow execution times on {platform}",
                    "description": f"95th percentile execution time is {metrics['p95_duration_ms']/1000:.1f}s",
                    "action": "Consider optimizing workflows or increasing timeout limits"
                })
            
            # Cost optimization
            if metrics["avg_cost_per_execution"] > 0.1:  # $0.10 per execution
                recommendations.append({
                    "type": "cost",
                    "severity": "low",
                    "platform": platform,
                    "title": f"High cost per execution on {platform}",
                    "description": f"Average cost is ${metrics['avg_cost_per_execution']:.3f} per execution",
                    "action": "Review workflow complexity and consider batching operations"
                })
        
        # Time-based recommendations
        time_series = analytics.get("time_series", {}).get("daily", [])
        if time_series:
            recent_days = time_series[-7:]
            recent_success_rates = [d["success_rate"] for d in recent_days if d["total_executions"] > 0]
            
            if recent_success_rates and statistics.mean(recent_success_rates) < 0.8:
                recommendations.append({
                    "type": "reliability",
                    "severity": "high",
                    "platform": "all",
                    "title": "Declining overall success rate",
                    "description": "Success rate has dropped below 80% in the last week",
                    "action": "Investigate recent changes and platform health"
                })
        
        # Performance trend recommendations
        performance_trends = analytics.get("performance_trends", {})
        if performance_trends.get("trend_direction") == "degrading":
            recommendations.append({
                "type": "performance",
                "severity": "medium",
                "platform": "all",
                "title": "Performance degradation detected",
                "description": "Execution times have increased over recent weeks",
                "action": "Review system load and optimize workflow configurations"
            })
        
        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        recommendations.sort(key=lambda x: severity_order.get(x["severity"], 3))
        
        return recommendations
    
    async def detect_anomalies(
        self,
        user_id: str,
        time_window_minutes: int = 60
    ) -> List[Dict[str, Any]]:
        """Detect anomalies in recent executions"""
        # Get recent executions
        executions = await self._get_recent_executions(user_id, time_window_minutes)
        
        # Detect anomalies
        anomalies = await self._detect_anomalies(executions)
        
        # Add severity and recommendations
        for anomaly in anomalies:
            if anomaly["type"] == "high_failure_rate":
                anomaly["severity"] = "high"
                anomaly["recommendation"] = "Investigate platform connectivity and error logs"
            elif anomaly["type"] == "consecutive_failures":
                anomaly["severity"] = "critical"
                anomaly["recommendation"] = "Platform may be down - check credentials and service status"
            elif anomaly["type"] == "execution_time_anomaly":
                anomaly["severity"] = "medium"
                anomaly["recommendation"] = "Monitor for performance degradation"
        
        return anomalies