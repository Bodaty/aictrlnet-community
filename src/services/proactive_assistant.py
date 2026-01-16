"""Proactive Assistant Service for AICtrlNet Intelligent Assistant.

This service generates proactive suggestions based on learned patterns,
user history, and context to help users discover automation opportunities.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, func, select
import logging

from models.knowledge import LearnedPattern, KnowledgeItem
from models.conversation import ConversationSession, ConversationMessage, ConversationPattern, ConversationAction
from models.workflow_execution import WorkflowExecution
from models.workflow_templates import WorkflowTemplate
from models.community_complete import WorkflowInstance
from services.pattern_learning_service import PatternLearningService
from services.knowledge.knowledge_retrieval_service import KnowledgeRetrievalService
from services.knowledge.system_manifest_service import SystemManifestService

logger = logging.getLogger(__name__)


class ProactiveAssistant:
    """Service for generating proactive suggestions to users."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.pattern_service = PatternLearningService(db)
        self.knowledge_service = KnowledgeRetrievalService(db)
        self.manifest_service = SystemManifestService(db)

    async def generate_suggestions(self, user_id: str, context: Dict = None) -> List[Dict]:
        """Generate proactive suggestions for a user.

        Args:
            user_id: The user ID
            context: Optional current context

        Returns:
            List of suggestions
        """
        suggestions = []

        # 1. Pattern-based suggestions
        pattern_suggestions = await self._generate_pattern_suggestions(user_id, context)
        suggestions.extend(pattern_suggestions)

        # 2. Capability discovery suggestions
        discovery_suggestions = await self._generate_discovery_suggestions(user_id, context)
        suggestions.extend(discovery_suggestions)

        # 3. Workflow optimization suggestions
        optimization_suggestions = await self._generate_optimization_suggestions(user_id)
        suggestions.extend(optimization_suggestions)

        # 4. Usage-based suggestions
        usage_suggestions = await self._generate_usage_suggestions(user_id)
        suggestions.extend(usage_suggestions)

        # 5. Time-based suggestions
        time_suggestions = await self._generate_time_based_suggestions(user_id)
        suggestions.extend(time_suggestions)

        # Rank and filter suggestions
        ranked_suggestions = self._rank_suggestions(suggestions, context)

        # Limit to top suggestions
        return ranked_suggestions[:5]

    async def _generate_pattern_suggestions(self, user_id: str,
                                          context: Optional[Dict]) -> List[Dict]:
        """Generate suggestions based on learned patterns."""
        suggestions = []

        # Get user's recent sessions
        stmt = select(ConversationSession).filter(
            and_(
                ConversationSession.user_id == user_id,
                ConversationSession.started_at >= datetime.utcnow() - timedelta(days=30)
            )
        ).order_by(ConversationSession.started_at.desc()).limit(5)
        result = await self.db.execute(stmt)
        recent_sessions = result.scalars().all()

        if recent_sessions:
            # Build context from recent activity
            recent_context = {
                "user_id": user_id,
                "recent_intents": [],
                "recent_actions": []
            }

            for session in recent_sessions:
                if session.primary_intent:
                    recent_context["recent_intents"].append(session.primary_intent)

                # Get successful actions from session
                stmt = select(ConversationAction).filter(
                    and_(
                        ConversationAction.session_id == session.id,
                        ConversationAction.status == "completed"
                    )
                )
                result = await self.db.execute(stmt)
                actions = result.scalars().all()

                for action in actions:
                    recent_context["recent_actions"].append(action.action_type)

            # Merge with provided context
            if context:
                recent_context.update(context)

            # Get relevant patterns
            patterns = await self.pattern_service.get_relevant_patterns(recent_context)

            for pattern in patterns[:3]:  # Top 3 pattern suggestions
                pattern_suggestion = await self.pattern_service.apply_pattern(pattern, recent_context)

                suggestion = {
                    "type": "pattern_based",
                    "title": self._generate_pattern_title(pattern),
                    "description": self._generate_pattern_description(pattern),
                    "confidence": pattern.confidence_score,
                    "action": pattern_suggestion.get("suggested_action"),
                    "params": pattern_suggestion.get("action_params_template"),
                    "pattern_id": str(pattern.id),
                    "priority": pattern.confidence_score * pattern.success_count
                }

                suggestions.append(suggestion)

        return suggestions

    async def _generate_discovery_suggestions(self, user_id: str,
                                            context: Optional[Dict]) -> List[Dict]:
        """Generate suggestions for discovering new capabilities."""
        suggestions = []

        # Get current system capabilities
        manifest = await self.manifest_service.get_manifest()

        if not manifest:
            return suggestions

        # Get user's used capabilities
        used_capabilities = await self._get_user_capabilities(user_id)

        # Find unused capabilities
        all_capabilities = []

        # Templates
        if "workflow_templates" in manifest:
            for template_id, template_info in manifest["workflow_templates"].items():
                if template_id not in used_capabilities.get("templates", []):
                    all_capabilities.append({
                        "type": "template",
                        "id": template_id,
                        "info": template_info
                    })

        # Agents
        if "agents" in manifest:
            for agent_id, agent_info in manifest["agents"].items():
                if agent_id not in used_capabilities.get("agents", []):
                    all_capabilities.append({
                        "type": "agent",
                        "id": agent_id,
                        "info": agent_info
                    })

        # Recommend top unused capabilities
        for cap in all_capabilities[:2]:  # Top 2 discovery suggestions
            suggestion = {
                "type": "capability_discovery",
                "title": f"Try {cap['info'].get('name', cap['id'])}",
                "description": cap['info'].get('description', f"Discover the {cap['type']} capability"),
                "confidence": 0.6,  # Medium confidence for discovery
                "action": f"explore_{cap['type']}",
                "params": {"capability_id": cap['id']},
                "priority": 0.6
            }

            suggestions.append(suggestion)

        return suggestions

    async def _generate_optimization_suggestions(self, user_id: str) -> List[Dict]:
        """Generate suggestions for optimizing existing workflows."""
        suggestions = []

        # Get user's recent workflow executions
        stmt = select(WorkflowExecution).filter(
            WorkflowExecution.user_id == user_id
        ).order_by(WorkflowExecution.started_at.desc()).limit(10)
        result = await self.db.execute(stmt)
        recent_executions = result.scalars().all()

        # Analyze for optimization opportunities
        workflow_stats = {}
        for execution in recent_executions:
            workflow_id = execution.workflow_id
            if workflow_id not in workflow_stats:
                workflow_stats[workflow_id] = {
                    "count": 0,
                    "failures": 0,
                    "avg_duration": 0,
                    "total_duration": 0
                }

            stats = workflow_stats[workflow_id]
            stats["count"] += 1

            if execution.status != "completed":
                stats["failures"] += 1

            if execution.ended_at and execution.started_at:
                duration = (execution.ended_at - execution.started_at).total_seconds()
                stats["total_duration"] += duration

        # Generate optimization suggestions
        for workflow_id, stats in workflow_stats.items():
            if stats["count"] > 0:
                stats["avg_duration"] = stats["total_duration"] / stats["count"]

                # Suggest optimization for frequently failing workflows
                if stats["failures"] / stats["count"] > 0.3:
                    suggestion = {
                        "type": "optimization",
                        "title": f"Improve reliability of workflow {workflow_id[:8]}",
                        "description": f"This workflow has a {stats['failures']/stats['count']*100:.0f}% failure rate",
                        "confidence": 0.8,
                        "action": "optimize_workflow",
                        "params": {"workflow_id": workflow_id},
                        "priority": 0.8 * stats["count"]
                    }
                    suggestions.append(suggestion)

                # Suggest optimization for slow workflows
                elif stats["avg_duration"] > 300:  # > 5 minutes
                    suggestion = {
                        "type": "optimization",
                        "title": f"Speed up workflow {workflow_id[:8]}",
                        "description": f"Average runtime is {stats['avg_duration']/60:.1f} minutes",
                        "confidence": 0.7,
                        "action": "optimize_workflow",
                        "params": {"workflow_id": workflow_id},
                        "priority": 0.7 * stats["count"]
                    }
                    suggestions.append(suggestion)

        return suggestions[:1]  # Return top optimization suggestion

    async def _generate_usage_suggestions(self, user_id: str) -> List[Dict]:
        """Generate suggestions based on usage patterns."""
        suggestions = []

        # Get user's usage patterns
        stmt = select(ConversationSession).filter(
            ConversationSession.user_id == user_id
        ).order_by(ConversationSession.started_at.desc()).limit(20)
        result = await self.db.execute(stmt)
        recent_sessions = result.scalars().all()

        # Analyze usage frequency
        if recent_sessions:
            # Check if user frequently creates similar workflows
            intents = [s.primary_intent for s in recent_sessions if s.primary_intent]
            intent_counts = {}
            for intent in intents:
                intent_counts[intent] = intent_counts.get(intent, 0) + 1

            # Suggest automation for frequent intents
            for intent, count in intent_counts.items():
                if count >= 3:  # If intent appears 3+ times
                    suggestion = {
                        "type": "usage_pattern",
                        "title": f"Automate '{intent}' workflow",
                        "description": f"You've requested this {count} times recently",
                        "confidence": min(0.9, 0.5 + count * 0.1),
                        "action": "create_template",
                        "params": {"base_intent": intent},
                        "priority": count * 0.2
                    }
                    suggestions.append(suggestion)
                    break  # Only one usage suggestion

        return suggestions

    async def _generate_time_based_suggestions(self, user_id: str) -> List[Dict]:
        """Generate suggestions based on time patterns."""
        suggestions = []

        current_hour = datetime.utcnow().hour
        current_day = datetime.utcnow().weekday()

        # Business hours suggestions (Monday-Friday, 9-5)
        if 0 <= current_day <= 4 and 9 <= current_hour <= 17:
            suggestion = {
                "type": "time_based",
                "title": "Set up business hours automation",
                "description": "Automate routine tasks during work hours",
                "confidence": 0.5,
                "action": "explore_business_automation",
                "params": {"time_context": "business_hours"},
                "priority": 0.5
            }
            suggestions.append(suggestion)

        # End of day suggestions
        elif 0 <= current_day <= 4 and 16 <= current_hour <= 18:
            suggestion = {
                "type": "time_based",
                "title": "Create end-of-day report workflow",
                "description": "Automate daily summary and reporting",
                "confidence": 0.4,
                "action": "create_report_workflow",
                "params": {"report_type": "daily_summary"},
                "priority": 0.4
            }
            suggestions.append(suggestion)

        return suggestions[:1]  # Max one time-based suggestion

    async def _get_user_capabilities(self, user_id: str) -> Dict:
        """Get capabilities the user has already used."""
        used = {
            "templates": set(),
            "agents": set(),
            "adapters": set()
        }

        # Get from workflow executions
        stmt = select(WorkflowExecution).filter(
            WorkflowExecution.user_id == user_id
        )
        result = await self.db.execute(stmt)
        executions = result.scalars().all()

        for execution in executions:
            if execution.workflow_id:
                # Check if it's from a template
                stmt = select(WorkflowInstance).filter_by(
                    id=execution.workflow_id
                )
                result = await self.db.execute(stmt)
                workflow = result.scalar_one_or_none()
                if workflow and workflow.template_id:
                    used["templates"].add(workflow.template_id)

        # Get from conversation actions
        stmt = select(ConversationAction).join(
            ConversationSession
        ).filter(
            ConversationSession.user_id == user_id
        )
        result = await self.db.execute(stmt)
        actions = result.scalars().all()

        for action in actions:
            if action.agent_id:
                used["agents"].add(str(action.agent_id))

        return used

    def _generate_pattern_title(self, pattern: LearnedPattern) -> str:
        """Generate a user-friendly title for a pattern."""
        pattern_data = pattern.pattern_data

        if pattern.pattern_type == "intent_action":
            intent = pattern_data.get("intent", "task")
            action = pattern_data.get("action_type", "automation")
            return f"Automate {intent} with {action}"

        elif pattern.pattern_type == "sequence":
            return "Complete multi-step workflow"

        elif pattern.pattern_type == "parameter":
            intent = pattern_data.get("intent", "task")
            return f"Quick setup for {intent}"

        return "Automation suggestion"

    def _generate_pattern_description(self, pattern: LearnedPattern) -> str:
        """Generate a user-friendly description for a pattern."""
        success_rate = (pattern.success_count / pattern.occurrence_count * 100
                       if pattern.occurrence_count > 0 else 0)

        return (f"Based on {pattern.occurrence_count} similar requests "
               f"with {success_rate:.0f}% success rate")

    def _rank_suggestions(self, suggestions: List[Dict],
                         context: Optional[Dict]) -> List[Dict]:
        """Rank suggestions by relevance and priority."""
        # Calculate final score for each suggestion
        for suggestion in suggestions:
            base_score = suggestion.get("priority", 0.5)
            confidence = suggestion.get("confidence", 0.5)

            # Adjust based on context
            if context:
                # Boost if relates to current intent
                if context.get("primary_intent"):
                    if context["primary_intent"] in str(suggestion):
                        base_score *= 1.2

                # Boost if user is actively engaged
                if context.get("is_active_session"):
                    base_score *= 1.1

            suggestion["final_score"] = base_score * confidence

        # Sort by final score
        suggestions.sort(key=lambda s: s["final_score"], reverse=True)

        return suggestions

    async def should_show_suggestions(self, user_id: str, session_id: str) -> bool:
        """Determine if proactive suggestions should be shown.

        Args:
            user_id: The user ID
            session_id: Current session ID

        Returns:
            True if suggestions should be shown
        """
        # Get current session
        stmt = select(ConversationSession).filter_by(id=session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            return False

        # Don't show during active task execution
        if session.state in ["executing", "confirming"]:
            return False

        # Check message count in session
        stmt = select(func.count(ConversationMessage.id)).filter(
            ConversationMessage.session_id == session_id
        )
        result = await self.db.execute(stmt)
        message_count = result.scalar()

        # Show suggestions after greeting or if conversation is idle
        if session.state == "greeting" and message_count <= 2:
            return True

        # Show if conversation has been idle
        if session.last_activity:
            idle_time = (datetime.utcnow() - session.last_activity).total_seconds()
            if idle_time > 30:  # 30 seconds idle
                return True

        # Show every N messages to avoid being intrusive
        if message_count > 0 and message_count % 5 == 0:
            return True

        return False

    async def format_suggestions_for_ui(self, suggestions: List[Dict]) -> Dict:
        """Format suggestions for display in the UI.

        Args:
            suggestions: List of suggestion dictionaries

        Returns:
            Formatted response for UI
        """
        if not suggestions:
            return {
                "has_suggestions": False,
                "suggestions": []
            }

        formatted = {
            "has_suggestions": True,
            "suggestions": [],
            "intro_text": "Here are some suggestions based on your activity:"
        }

        for i, suggestion in enumerate(suggestions, 1):
            formatted_suggestion = {
                "id": f"suggestion_{i}",
                "title": suggestion["title"],
                "description": suggestion["description"],
                "type": suggestion["type"],
                "confidence": suggestion["confidence"],
                "action_button": {
                    "label": self._get_action_label(suggestion["action"]),
                    "action": suggestion["action"],
                    "params": suggestion.get("params", {})
                }
            }

            # Add icon based on type
            formatted_suggestion["icon"] = self._get_suggestion_icon(suggestion["type"])

            formatted["suggestions"].append(formatted_suggestion)

        return formatted

    def _get_action_label(self, action: str) -> str:
        """Get user-friendly label for action button."""
        action_labels = {
            "create_workflow": "Create Workflow",
            "create_template": "Create Template",
            "optimize_workflow": "Optimize",
            "explore_template": "Explore",
            "explore_agent": "Try Agent",
            "explore_business_automation": "Learn More",
            "create_report_workflow": "Set Up Report"
        }
        return action_labels.get(action, "Try This")

    def _get_suggestion_icon(self, suggestion_type: str) -> str:
        """Get icon for suggestion type."""
        icons = {
            "pattern_based": "🔄",
            "capability_discovery": "🔍",
            "optimization": "⚡",
            "usage_pattern": "📊",
            "time_based": "⏰"
        }
        return icons.get(suggestion_type, "💡")