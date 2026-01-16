"""
Action Orchestrator for Multi-Turn Conversation System.

This service bridges conversation intents to existing AICtrlNet services,
leveraging the already-implemented NLP workflow creation and other features.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from uuid import UUID
from enum import Enum
from dataclasses import asdict

from sqlalchemy.ext.asyncio import AsyncSession

from services.nlp import NLPService
from services.task import TaskService
from services.iam import IAMService
from core.edition_discovery import get_edition_discovery, Edition
from services.action_planner import ActionPlanner, ActionPlan
from services.progressive_executor import ProgressiveExecutor, ExecutionProgress

logger = logging.getLogger(__name__)


def dataclass_to_dict(obj):
    """Convert dataclass to dict with datetime serialization."""
    result = asdict(obj)

    def convert_value(v):
        if isinstance(v, datetime):
            return v.isoformat()
        elif isinstance(v, dict):
            return {k: convert_value(val) for k, val in v.items()}
        elif isinstance(v, list):
            return [convert_value(item) for item in v]
        return v

    return convert_value(result)
edition_discovery = get_edition_discovery()


class ActionType(str, Enum):
    """Types of actions the orchestrator can execute."""
    COMPANY_AUTOMATION = "company_automation"
    CREATE_WORKFLOW = "create_workflow"
    CREATE_AGENT = "create_agent"
    CREATE_POD = "create_pod"
    EXECUTE_TASK = "execute_task"
    SEARCH_RESOURCES = "search_resources"
    GET_HELP = "get_help"
    UNKNOWN = "unknown"


class ParameterRequirements:
    """Define required and optional parameters for each action type."""
    
    REQUIREMENTS = {
        ActionType.CREATE_WORKFLOW: {
            "essential": ["name"],  # Absolute minimum
            "recommended": ["workflow_type", "description"],  # Improves quality
            "optional": ["trigger", "schedule", "notifications"],  # Nice to have
            "contextual": {  # Required based on type
                "data_processing": ["source", "destination"],
                "approval": ["approvers", "routing_rules"],
                "monitoring": ["target", "frequency", "alerts"]
            },
            "clarification_questions": {
                "name": "What would you like to name this workflow?",
                "workflow_type": "What type of workflow? (e.g., approval, data processing, monitoring)",
                "description": "How would you describe this workflow's purpose?",
                "trigger": "How should this workflow be triggered? (manual, scheduled, event-based)",
                "source": "What's the data source for this workflow?",
                "destination": "Where should the results be stored?",
                "approvers": "Who should approve these requests?",
                "target": "What should this workflow monitor?",
                "frequency": "How often should it check?"
            }
        },
        ActionType.CREATE_AGENT: {
            "essential": ["name", "agent_type"],
            "recommended": ["capabilities", "description"],
            "optional": ["config", "memory_size"],
            "clarification_questions": {
                "name": "What would you like to name this agent?",
                "agent_type": "What type of agent? (monitor, executor, analyzer)",
                "capabilities": "What capabilities should this agent have?",
                "description": "What is this agent's purpose?"
            }
        },
        ActionType.CREATE_POD: {
            "essential": ["objective"],
            "recommended": ["name", "agents"],
            "optional": ["strategy", "max_size"],
            "clarification_questions": {
                "objective": "What is the pod's objective?",
                "name": "What would you like to name this pod?",
                "agents": "Which agents should be part of this pod?",
                "strategy": "What coordination strategy? (consensus, leader-follower, voting)"
            }
        },
        ActionType.EXECUTE_TASK: {
            "essential": ["task_type"],
            "recommended": ["name"],
            "optional": ["config", "params", "priority"],
            "clarification_questions": {
                "task_type": "What type of task would you like to execute?",
                "name": "What should we call this task?",
                "params": "Any specific parameters for this task?"
            }
        }
    }
    
    @classmethod
    def get_missing_essential_params(cls, action_type: ActionType, provided_params: Dict[str, Any]) -> List[str]:
        """Identify which essential parameters are missing."""
        requirements = cls.REQUIREMENTS.get(action_type, {})
        essential = requirements.get("essential", [])
        
        missing = []
        for param in essential:
            if param not in provided_params or not provided_params[param]:
                missing.append(param)
        
        return missing
    
    @classmethod
    def get_contextual_params(cls, action_type: ActionType, workflow_type: str) -> List[str]:
        """Get contextual parameters based on workflow type."""
        requirements = cls.REQUIREMENTS.get(action_type, {})
        contextual = requirements.get("contextual", {})
        return contextual.get(workflow_type, [])
    
    @classmethod
    def get_clarification_question(cls, action_type: ActionType, param: str) -> str:
        """Get the clarification question for a specific parameter."""
        requirements = cls.REQUIREMENTS.get(action_type, {})
        questions = requirements.get("clarification_questions", {})
        return questions.get(param, f"Please provide the {param}")


class SmartDefaults:
    """Provides intelligent defaults based on context and explains them."""
    
    @staticmethod
    def generate_defaults(action_type: str, context: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Generate intelligent defaults with explanations.
        Returns dict with default values and explanations.
        """
        defaults = {}
        
        # Workflow-specific defaults
        if action_type == ActionType.CREATE_WORKFLOW:
            workflow_type = context.get("workflow_type", "")
            
            # Name defaults
            if "name" not in context:
                if workflow_type:
                    suggested_name = f"{workflow_type.replace('_', ' ').title()} Workflow"
                else:
                    suggested_name = f"Workflow_{datetime.now().strftime('%Y%m%d_%H%M')}"
                
                defaults["name"] = {
                    "value": suggested_name,
                    "explanation": f"I'll name it '{suggested_name}' based on the type - you can rename it anytime"
                }
            
            # Schedule defaults based on type
            if workflow_type == "daily_report":
                defaults["schedule"] = {
                    "value": "0 9 * * *",  # 9 AM
                    "explanation": "Scheduled for 9 AM when team typically starts work"
                }
            elif workflow_type == "backup":
                defaults["schedule"] = {
                    "value": "0 2 * * *",  # 2 AM
                    "explanation": "Scheduled for 2 AM during low system usage"
                }
            elif workflow_type == "monitoring":
                defaults["schedule"] = {
                    "value": "*/5 * * * *",  # Every 5 minutes
                    "explanation": "Checks every 5 minutes for timely monitoring"
                }
            else:
                defaults["trigger"] = {
                    "value": "manual",
                    "explanation": "Set to manual trigger so you have full control - add automatic triggers later"
                }
            
            # Data processing defaults
            if workflow_type == "data_processing":
                if "source" not in context:
                    defaults["source"] = {
                        "value": "input",
                        "explanation": "Using flexible 'input' source - connect to specific sources when ready"
                    }
                if "destination" not in context:
                    defaults["destination"] = {
                        "value": "storage",
                        "explanation": "Using standard data storage - accessible through dashboard"
                    }
            
            # Notification defaults
            if context.get("criticality") == "high":
                defaults["notifications"] = {
                    "value": ["email", "slack"],
                    "explanation": "Multiple channels for critical workflows"
                }
            else:
                defaults["notifications"] = {
                    "value": ["email"],
                    "explanation": "Email notification for standard workflows"
                }
        
        # Agent-specific defaults
        elif action_type == ActionType.CREATE_AGENT:
            if "agent_type" not in context:
                defaults["agent_type"] = {
                    "value": "executor",
                    "explanation": "Using 'executor' type - the most versatile for general tasks"
                }
            
            if "capabilities" not in context:
                agent_type = context.get("agent_type", "executor")
                if agent_type == "monitor":
                    defaults["capabilities"] = {
                        "value": ["observe", "alert", "log"],
                        "explanation": "Standard monitoring capabilities"
                    }
                elif agent_type == "analyzer":
                    defaults["capabilities"] = {
                        "value": ["analyze", "report", "visualize"],
                        "explanation": "Standard analysis capabilities"
                    }
                else:
                    defaults["capabilities"] = {
                        "value": ["execute", "report"],
                        "explanation": "Basic execution and reporting capabilities"
                    }
        
        return defaults
    
    @staticmethod
    def explain_default(param_name: str, default_value: Any, context: Dict[str, Any]) -> str:
        """Generate a user-friendly explanation for why this default was chosen."""
        
        explanations = {
            "manual": "Manual trigger gives you full control over when to run",
            "scheduled": "Scheduled execution for regular automated runs",
            "email": "Email notifications to keep you informed",
            "executor": "Executor agents can perform a wide variety of tasks",
            "monitor": "Monitor agents watch systems and alert on issues",
            "input": "Flexible input that can connect to various sources",
            "storage": "Standard storage location accessible from dashboard"
        }
        
        # Look for specific explanation
        if isinstance(default_value, str):
            return explanations.get(default_value, f"Standard default for {param_name}")
        
        return f"Recommended setting for {param_name}"


class ActionValidator:
    """Validates action plans before execution."""

    @staticmethod
    async def validate_plan(plan: ActionPlan, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate an action plan for safety and feasibility."""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "suggestions": []
        }

        # Check resource availability
        for resource in plan.required_resources:
            if resource.status == "required" and not context.get(f"has_{resource.name}"):
                validation_result["errors"].append(
                    f"Missing required resource: {resource.name} ({resource.type})"
                )
                validation_result["valid"] = False

        # Check step dependencies
        completed_steps = set()
        for step in plan.steps:
            for dep in step.dependencies:
                if dep not in completed_steps:
                    validation_result["errors"].append(
                        f"Step '{step.name}' depends on '{dep}' which comes later"
                    )
                    validation_result["valid"] = False
            completed_steps.add(step.id)

        # Check time constraints
        if plan.estimated_time_seconds > 600:  # More than 10 minutes
            validation_result["warnings"].append(
                f"This operation will take approximately {plan.estimated_time_seconds // 60} minutes"
            )

        # Check for potential issues
        if plan.potential_issues:
            for issue in plan.potential_issues:
                validation_result["warnings"].append(issue)

        # Provide suggestions
        if not plan.rollback_strategy:
            validation_result["suggestions"].append(
                "Consider defining a rollback strategy for safer execution"
            )

        return validation_result


class ActionMonitor:
    """Monitors action execution and provides real-time feedback."""

    def __init__(self):
        self._execution_metrics = {}
        self._performance_data = {}

    async def start_monitoring(self, plan_id: str) -> None:
        """Start monitoring an execution."""
        self._execution_metrics[plan_id] = {
            "started_at": datetime.utcnow(),
            "steps_completed": 0,
            "steps_failed": 0,
            "warnings": [],
            "performance": {}
        }

    async def record_step_completion(
        self,
        plan_id: str,
        step_name: str,
        duration_seconds: float,
        success: bool
    ) -> None:
        """Record completion of a step."""
        if plan_id not in self._execution_metrics:
            return

        metrics = self._execution_metrics[plan_id]

        if success:
            metrics["steps_completed"] += 1
        else:
            metrics["steps_failed"] += 1

        # Track performance
        metrics["performance"][step_name] = {
            "duration": duration_seconds,
            "success": success,
            "timestamp": datetime.utcnow().isoformat()
        }

        # Check for performance issues
        if duration_seconds > 30:
            metrics["warnings"].append(
                f"Step '{step_name}' took {duration_seconds:.1f}s (longer than expected)"
            )

    async def get_execution_metrics(self, plan_id: str) -> Dict[str, Any]:
        """Get current execution metrics."""
        return self._execution_metrics.get(plan_id, {})

    async def stop_monitoring(self, plan_id: str) -> Dict[str, Any]:
        """Stop monitoring and return final metrics."""
        if plan_id in self._execution_metrics:
            metrics = self._execution_metrics[plan_id]
            metrics["completed_at"] = datetime.utcnow()

            # Calculate total duration
            duration = (metrics["completed_at"] - metrics["started_at"]).total_seconds()
            metrics["total_duration_seconds"] = duration

            # Store for learning
            self._performance_data[plan_id] = metrics

            # Clean up active monitoring
            del self._execution_metrics[plan_id]

            return metrics
        return {}


class ActionOptimizer:
    """Optimizes action plans based on historical performance."""

    def __init__(self):
        self._optimization_history = {}
        self._pattern_cache = {}

    async def optimize_plan(self, plan: ActionPlan, context: Dict[str, Any]) -> ActionPlan:
        """Optimize an action plan based on patterns and performance data."""

        # Check for parallel execution opportunities
        plan = self._identify_parallel_steps(plan)

        # Reorder steps for efficiency
        plan = self._reorder_steps(plan)

        # Apply caching where possible
        plan = self._apply_caching_hints(plan, context)

        # Adjust time estimates based on history
        plan = self._adjust_time_estimates(plan)

        return plan

    def _identify_parallel_steps(self, plan: ActionPlan) -> ActionPlan:
        """Identify steps that can be executed in parallel."""
        # Group steps by dependencies
        dependency_levels = {}
        for step in plan.steps:
            level = 0
            if step.dependencies:
                # Find max level of dependencies
                for dep_id in step.dependencies:
                    dep_step = next((s for s in plan.steps if s.id == dep_id), None)
                    if dep_step and dep_step.id in dependency_levels:
                        level = max(level, dependency_levels[dep_step.id] + 1)
            dependency_levels[step.id] = level

        # Mark steps at same level as parallelizable
        for step in plan.steps:
            level = dependency_levels.get(step.id, 0)
            parallel_candidates = [
                s.id for s in plan.steps
                if dependency_levels.get(s.id, -1) == level and s.id != step.id
            ]
            if parallel_candidates:
                step.params["parallel_with"] = parallel_candidates

        return plan

    def _reorder_steps(self, plan: ActionPlan) -> ActionPlan:
        """Reorder steps for optimal execution."""
        # Simple optimization: Put validation steps early
        validation_steps = [s for s in plan.steps if "validate" in s.name.lower()]
        other_steps = [s for s in plan.steps if "validate" not in s.name.lower()]

        # Rebuild steps list with validations first (respecting dependencies)
        optimized_steps = []
        for step in validation_steps:
            if not step.dependencies:  # Can be moved to front
                optimized_steps.append(step)

        optimized_steps.extend([s for s in validation_steps if s not in optimized_steps])
        optimized_steps.extend(other_steps)

        plan.steps = optimized_steps
        return plan

    def _apply_caching_hints(self, plan: ActionPlan, context: Dict[str, Any]) -> ActionPlan:
        """Apply caching hints to steps that can benefit from it."""
        # Check for repeated patterns
        pattern_key = f"{plan.name}_{len(plan.steps)}"

        if pattern_key in self._pattern_cache:
            # Apply cached optimizations
            cached_hints = self._pattern_cache[pattern_key]
            for step in plan.steps:
                if step.name in cached_hints:
                    step.params["use_cache"] = True
                    step.params["cache_ttl"] = cached_hints[step.name]

        return plan

    def _adjust_time_estimates(self, plan: ActionPlan) -> ActionPlan:
        """Adjust time estimates based on historical data."""
        # Look for similar plans in history
        for historical_plan_id, historical_data in self._optimization_history.items():
            if historical_data.get("name") == plan.name:
                # Apply historical timing data
                historical_steps = historical_data.get("steps", {})
                for step in plan.steps:
                    if step.name in historical_steps:
                        # Use 90th percentile of historical times
                        historical_time = historical_steps[step.name].get("p90_duration", step.duration_seconds)
                        step.duration_seconds = int(historical_time * 1.1)  # Add 10% buffer

        # Recalculate total time
        plan.estimated_time_seconds = sum(step.duration_seconds for step in plan.steps)

        return plan

    async def record_execution_performance(
        self,
        plan: ActionPlan,
        metrics: Dict[str, Any]
    ) -> None:
        """Record execution performance for future optimization."""
        self._optimization_history[plan.id] = {
            "name": plan.name,
            "steps": {},
            "total_duration": metrics.get("total_duration_seconds", 0)
        }

        for step in plan.steps:
            step_perf = metrics.get("performance", {}).get(step.name, {})
            if step_perf:
                self._optimization_history[plan.id]["steps"][step.name] = {
                    "avg_duration": step_perf.get("duration", step.duration_seconds),
                    "p90_duration": step_perf.get("duration", step.duration_seconds) * 1.2,
                    "success_rate": 1.0 if step_perf.get("success") else 0.0
                }


class ExecutionLearner:
    """Learns from execution results to improve future performance."""

    def __init__(self):
        self._learning_data = {
            "success_patterns": {},
            "failure_patterns": {},
            "optimization_rules": []
        }

    async def learn_from_execution(
        self,
        plan: ActionPlan,
        progress: ExecutionProgress,
        context: Dict[str, Any]
    ) -> None:
        """Learn from execution results."""

        # Analyze success/failure patterns
        if progress.status == "completed":
            self._record_success_pattern(plan, progress, context)
        elif progress.status == "failed":
            self._record_failure_pattern(plan, progress, context)

        # Extract optimization rules
        self._extract_optimization_rules(plan, progress)

        # Update confidence scores
        self._update_confidence_scores(plan, progress)

    def _record_success_pattern(self, plan: ActionPlan, progress: ExecutionProgress, context: Dict[str, Any]) -> None:
        """Record successful execution patterns."""
        pattern_key = f"{plan.name}_{context.get('workflow_type', 'default')}"

        if pattern_key not in self._learning_data["success_patterns"]:
            self._learning_data["success_patterns"][pattern_key] = {
                "count": 0,
                "avg_duration": 0,
                "common_params": {},
                "optimal_sequence": []
            }

        pattern = self._learning_data["success_patterns"][pattern_key]
        pattern["count"] += 1

        # Update average duration
        total_duration = (progress.completed_at - progress.started_at).total_seconds()
        pattern["avg_duration"] = (
            (pattern["avg_duration"] * (pattern["count"] - 1) + total_duration) / pattern["count"]
        )

        # Track common parameters
        for step in plan.steps:
            if step.id in progress.completed_steps:
                for param, value in step.params.items():
                    if param not in pattern["common_params"]:
                        pattern["common_params"][param] = {}
                    pattern["common_params"][param][str(value)] = \
                        pattern["common_params"][param].get(str(value), 0) + 1

        # Record successful sequence
        pattern["optimal_sequence"] = [s.name for s in plan.steps if s.id in progress.completed_steps]

    def _record_failure_pattern(self, plan: ActionPlan, progress: ExecutionProgress, context: Dict[str, Any]) -> None:
        """Record failure patterns for analysis."""
        pattern_key = f"{plan.name}_{context.get('workflow_type', 'default')}"

        if pattern_key not in self._learning_data["failure_patterns"]:
            self._learning_data["failure_patterns"][pattern_key] = {
                "count": 0,
                "common_failure_points": {},
                "common_errors": []
            }

        pattern = self._learning_data["failure_patterns"][pattern_key]
        pattern["count"] += 1

        # Track failure points
        for failed_step_id in progress.failed_steps:
            failed_step = next((s for s in plan.steps if s.id == failed_step_id), None)
            if failed_step:
                step_name = failed_step.name
                pattern["common_failure_points"][step_name] = \
                    pattern["common_failure_points"].get(step_name, 0) + 1

        # Track error messages
        for result in progress.results:
            if result.error:
                if result.error not in pattern["common_errors"]:
                    pattern["common_errors"].append(result.error)

    def _extract_optimization_rules(self, plan: ActionPlan, progress: ExecutionProgress) -> None:
        """Extract optimization rules from execution."""
        # Example: If certain steps always succeed together, group them
        consecutive_successes = []
        current_group = []

        for step in plan.steps:
            if step.id in progress.completed_steps:
                current_group.append(step.name)
            else:
                if len(current_group) > 1:
                    consecutive_successes.append(current_group)
                current_group = []

        if len(current_group) > 1:
            consecutive_successes.append(current_group)

        # Create optimization rules
        for group in consecutive_successes:
            rule = {
                "type": "group_execution",
                "steps": group,
                "confidence": 0.8,
                "reason": "These steps consistently succeed together"
            }
            if rule not in self._learning_data["optimization_rules"]:
                self._learning_data["optimization_rules"].append(rule)

    def _update_confidence_scores(self, plan: ActionPlan, progress: ExecutionProgress) -> None:
        """Update confidence score based on execution results."""
        # Adjust plan confidence based on success rate
        success_rate = len(progress.completed_steps) / len(plan.steps) if plan.steps else 0

        # Weighted update: new_confidence = 0.7 * old + 0.3 * observed
        plan.confidence_score = 0.7 * plan.confidence_score + 0.3 * success_rate

    def get_recommendations(self, intent: str, context: Dict[str, Any]) -> List[str]:
        """Get recommendations based on learned patterns."""
        recommendations = []

        pattern_key = f"{intent}_{context.get('workflow_type', 'default')}"

        # Check for success patterns
        if pattern_key in self._learning_data["success_patterns"]:
            success_pattern = self._learning_data["success_patterns"][pattern_key]
            if success_pattern["count"] > 3:  # Enough data
                recommendations.append(
                    f"Based on {success_pattern['count']} successful executions, "
                    f"this typically takes {success_pattern['avg_duration']:.0f} seconds"
                )

                # Recommend common parameters
                for param, values in success_pattern["common_params"].items():
                    most_common = max(values.items(), key=lambda x: x[1])
                    if most_common[1] > success_pattern["count"] * 0.7:  # Used in 70% of cases
                        recommendations.append(
                            f"Consider using {param}='{most_common[0]}' (used in {most_common[1]} successful runs)"
                        )

        # Check for failure patterns
        if pattern_key in self._learning_data["failure_patterns"]:
            failure_pattern = self._learning_data["failure_patterns"][pattern_key]
            if failure_pattern["count"] > 2:
                for step_name, count in failure_pattern["common_failure_points"].items():
                    if count > failure_pattern["count"] * 0.5:  # Fails 50% of time
                        recommendations.append(
                            f"⚠️ Step '{step_name}' has failed {count} times - ensure prerequisites are met"
                        )

        return recommendations


class ActionOrchestrator:
    """
    Enhanced orchestrator with validation, monitoring, optimization, and learning.
    Routes conversation intents to existing AICtrlNet services.
    """

    def __init__(self, db: AsyncSession, user_edition: Edition = Edition.COMMUNITY):
        self.db = db
        self.user_edition = user_edition
        self.nlp_service = NLPService(db)
        self.smart_defaults = SmartDefaults()

        # Phase 2 enhancements
        self.action_planner = ActionPlanner(db)
        self.progressive_executor = ProgressiveExecutor(db)
        self.action_validator = ActionValidator()
        self.action_monitor = ActionMonitor()
        self.action_optimizer = ActionOptimizer()
        self.execution_learner = ExecutionLearner()
    
    async def should_ask_for_more(
        self,
        action_type: ActionType,
        current_params: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Intelligently determines if more parameters are needed.
        Returns: (should_ask, param_name, question)
        """
        
        # 1. Essential params - ALWAYS required
        missing_essential = ParameterRequirements.get_missing_essential_params(action_type, current_params)
        if missing_essential:
            question = ParameterRequirements.get_clarification_question(action_type, missing_essential[0])
            return True, missing_essential[0], question
        
        # 2. User signals urgency - Skip optional
        last_message = conversation_context.get("last_user_message", "").lower()
        if any(signal in last_message for signal in ["quickly", "just create", "basic", "simple"]):
            return False, None, None
        
        # 3. Conversation fatigue - Limit to 3 questions
        questions_asked = conversation_context.get("questions_asked", 0)
        if questions_asked >= 3:
            return False, None, None
        
        # 4. Rich initial context - Extract don't ask
        if self._has_rich_context(current_params):
            return False, None, None
        
        # 5. Type-specific contextual params
        if action_type == ActionType.CREATE_WORKFLOW:
            workflow_type = current_params.get("workflow_type", "")
            contextual_params = ParameterRequirements.get_contextual_params(action_type, workflow_type)
            
            for param in contextual_params:
                if param not in current_params:
                    question = ParameterRequirements.get_clarification_question(action_type, param)
                    return True, param, question
        
        # 6. Good defaults available - Proceed
        defaults = self.smart_defaults.generate_defaults(action_type, current_params)
        if defaults and len(defaults) >= 2:  # Have enough defaults to proceed
            return False, None, None
        
        return False, None, None
    
    def _has_rich_context(self, params: Dict[str, Any]) -> bool:
        """Check if we have enough context to proceed without more questions."""
        # Count meaningful parameters (not just name)
        meaningful_params = [k for k, v in params.items() 
                           if v and k not in ["user_id", "session_id", "timestamp"]]
        return len(meaningful_params) >= 3
    
    async def prepare_action_with_defaults(
        self,
        action_type: ActionType,
        current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prepare action by filling in smart defaults where needed.
        Returns enriched parameters with defaults and explanations.
        """
        # Generate defaults for missing params
        defaults = self.smart_defaults.generate_defaults(action_type, current_params)
        
        # Merge defaults with current params
        enriched_params = current_params.copy()
        explanations = {}
        
        for param_name, default_info in defaults.items():
            if param_name not in enriched_params:
                enriched_params[param_name] = default_info["value"]
                explanations[param_name] = default_info["explanation"]
        
        return {
            "params": enriched_params,
            "defaults_used": list(defaults.keys()),
            "explanations": explanations
        }
    
    async def execute_action(
        self,
        action_type: str,
        params: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an action using the appropriate existing service.
        This method assumes all required parameters have been collected.
        """
        
        try:
            if action_type == ActionType.COMPANY_AUTOMATION:
                # Route to action_planner for abstract/complex request decomposition
                user_query = context.get('user_query', params.get('query', 'automate my company'))
                plan_context = {
                    **context,
                    'user_query': user_query,
                    'is_abstract_request': True
                }
                return await self.action_planner.plan_and_execute_action(
                    intent="company_automation",
                    context=plan_context,
                    parameters=params,
                    user_id=user_id
                )
            elif action_type == ActionType.CREATE_WORKFLOW:
                return await self._create_workflow_via_nlp(params, user_id, context)
            elif action_type == ActionType.CREATE_AGENT:
                return await self._create_agent_via_iam(params, user_id, context)
            elif action_type == ActionType.CREATE_POD:
                return await self._form_pod_via_service(params, user_id, context)
            elif action_type == ActionType.EXECUTE_TASK:
                return await self._execute_task_via_service(params, user_id, context)
            else:
                return {
                    "success": False,
                    "error": f"Unknown action type: {action_type}"
                }
        except Exception as e:
            logger.error(f"Action execution failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def _create_workflow_via_nlp(
        self,
        params: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Create a workflow using the EXISTING NLP service.
        This preserves all the sophisticated workflow generation logic we've already built.
        """
        
        # Build a natural language prompt from the collected parameters
        prompt_parts = [f"Create a {params.get('workflow_type', 'workflow')} workflow"]
        
        if 'name' in params:
            prompt_parts.append(f"named '{params['name']}'")
        
        if 'description' in params:
            prompt_parts.append(f"that {params['description']}")
        
        if 'source' in params and 'destination' in params:
            prompt_parts.append(f"to process data from {params['source']} and store in {params['destination']}")
        elif 'source' in params:
            prompt_parts.append(f"using data from {params['source']}")
        elif 'destination' in params:
            prompt_parts.append(f"outputting to {params['destination']}")
        
        if 'trigger' in params:
            prompt_parts.append(f"triggered by {params['trigger']}")
        elif 'schedule' in params:
            prompt_parts.append(f"scheduled at {params['schedule']}")
        
        prompt = " ".join(prompt_parts)
        
        # Merge conversation context with provided context
        full_context = {
            **(context or {}),
            **params,  # Include all collected parameters
            "source": "conversation",
            "conversation_params": params
        }
        
        # Call the existing NLP service
        try:
            result = await self.nlp_service.process_nlp_request(
                prompt=prompt,
                context=full_context,
                user_id=user_id,
                return_transparency=True  # Get detailed info about what was created
            )
            
            if result and result.get("success"):
                return {
                    "success": True,
                    "workflow": result.get("workflow"),
                    "message": f"Successfully created workflow '{params.get('name', 'Untitled')}'",
                    "method": result.get('generation_method', 'nlp'),
                    "next_steps": [
                        "View the workflow in the Workflows section",
                        "Test the workflow execution",
                        "Configure additional settings"
                    ]
                }
            else:
                return {
                    "success": False,
                    "error": result.get("error", "Failed to create workflow through NLP service")
                }
                
        except Exception as e:
            logger.error(f"NLP workflow creation failed: {e}")
            return {
                "success": False,
                "error": f"Workflow creation failed: {str(e)}"
            }
    
    async def _create_agent_via_iam(
        self,
        params: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create an agent using the existing IAM service."""
        try:
            iam_service = IAMService(self.db)
            
            # Map conversation params to IAM service params
            agent_data = {
                "name": params.get("name", "Unnamed Agent"),
                "agent_type": params.get("agent_type", "executor"),
                "capabilities": params.get("capabilities", []),
                "description": params.get("description", ""),
                "config": params.get("config", {}),
                "created_by": user_id
            }
            
            agent = await iam_service.create_agent(agent_data)
            
            return {
                "success": True,
                "agent": {
                    "id": str(agent.id),
                    "name": agent.name,
                    "type": agent.agent_type
                },
                "message": f"Successfully created agent '{agent.name}'",
                "next_steps": [
                    "Configure agent capabilities",
                    "Assign to workflows",
                    "Form pods with other agents"
                ]
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Agent creation failed: {str(e)}"
            }
    
    async def _form_pod_via_service(
        self,
        params: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Form a pod using the existing pod orchestration service (Business edition)."""
        
        # Check edition
        if self.user_edition == Edition.COMMUNITY:
            return {
                "success": False,
                "error": "Pod formation requires Business edition or higher",
                "upgrade_required": True,
                "upgrade_to": "business"
            }
        
        # This would call the actual pod service when available
        # For now, return a placeholder
        return {
            "success": True,
            "message": "Pod formation would be executed here",
            "params_received": params
        }
    
    async def _execute_task_via_service(
        self,
        params: Dict[str, Any],
        user_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute a task using the existing task service."""
        try:
            task_service = TaskService(self.db)
            
            task_data = {
                "name": params.get("name", "Quick Task"),
                "task_type": params.get("task_type", "simple"),
                "config": params.get("config", {}),
                "params": params.get("params", {}),
                "created_by": user_id
            }
            
            task = await task_service.create_and_execute(task_data)
            
            return {
                "success": True,
                "task": {
                    "id": str(task.id),
                    "name": task.name,
                    "status": task.status
                },
                "message": f"Task '{task.name}' is now {task.status}"
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"Task execution failed: {str(e)}"
            }
    
    def generate_default_explanation_message(
        self,
        param_name: str,
        default_value: Any,
        explanation: str
    ) -> str:
        """Generate a user-friendly message explaining a default choice."""
        
        # Handle "I don't know" responses
        uncertainty_messages = {
            "workflow_type": f"""No problem! {explanation}
            
Other options include:
• 'approval' - for review and sign-off processes
• 'monitoring' - for checking system status
• 'integration' - for connecting systems

Should I proceed with '{default_value}' or would you prefer a different type?""",
            
            "trigger": f"""That's fine! {explanation}

Common triggers are:
• manual - start when you click run
• scheduled - run at specific times
• event - triggered by system events
• webhook - triggered by external systems

I'll go with '{default_value}' - sound good?""",
            
            "name": f"""No worries! {explanation}

A good name helps you find this workflow later.
I'll use '{default_value}' for now - works?"""
        }
        
        return uncertainty_messages.get(
            param_name,
            f"I'll use '{default_value}' as a default. {explanation}"
        )

    async def plan_and_execute_action(
        self,
        intent: str,
        parameters: Dict[str, Any],
        context: Dict[str, Any],
        user_id: str,
        preview_only: bool = False,
        confirm_callback: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Enhanced action execution with planning, validation, and monitoring.

        Args:
            intent: The action intent
            parameters: Action parameters
            context: Execution context
            user_id: User ID
            preview_only: If True, only plan and preview without executing
            confirm_callback: Callback for user confirmations

        Returns:
            Execution result with detailed information
        """

        # Step 1: Create action plan
        plan = await self.action_planner.plan_action(intent, context, parameters)

        # Step 2: Optimize the plan
        plan = await self.action_optimizer.optimize_plan(plan, context)

        # Step 3: Validate the plan
        validation = await self.action_validator.validate_plan(plan, context)

        if not validation["valid"]:
            return {
                "success": False,
                "errors": validation["errors"],
                "warnings": validation["warnings"],
                "plan": dataclass_to_dict(plan),
                "message": "Action plan validation failed"
            }

        # Step 4: Get learning recommendations
        recommendations = self.execution_learner.get_recommendations(intent, context)

        # Step 5: Generate preview
        preview = await self.action_planner.preview_action(plan)

        if preview_only:
            return {
                "success": True,
                "preview": dataclass_to_dict(preview),
                "plan": dataclass_to_dict(plan),
                "validation": validation,
                "recommendations": recommendations,
                "message": "Action plan ready for execution"
            }

        # Step 6: Start monitoring
        await self.action_monitor.start_monitoring(plan.id)

        try:
            # Step 7: Execute progressively
            progress = await self.progressive_executor.execute_plan(
                plan,
                context,
                confirm_callback
            )

            # Step 8: Stop monitoring and get metrics
            metrics = await self.action_monitor.stop_monitoring(plan.id)

            # Step 9: Learn from execution
            await self.execution_learner.learn_from_execution(plan, progress, context)

            # Step 10: Record performance for optimization
            await self.action_optimizer.record_execution_performance(plan, metrics)

            return {
                "success": progress.status == "completed",
                "progress": dataclass_to_dict(progress),
                "metrics": metrics,
                "plan": dataclass_to_dict(plan),
                "message": f"Execution {progress.status}: {len(progress.completed_steps)}/{len(plan.steps)} steps completed"
            }

        except Exception as e:
            logger.error(f"Execution failed: {str(e)}")

            # Stop monitoring on error
            metrics = await self.action_monitor.stop_monitoring(plan.id)

            return {
                "success": False,
                "error": str(e),
                "metrics": metrics,
                "plan": dataclass_to_dict(plan),
                "message": f"Execution failed: {str(e)}"
            }