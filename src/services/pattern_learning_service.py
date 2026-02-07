"""Pattern Learning Service for AICtrlNet Intelligent Assistant.

This service extracts and stores patterns from successful user interactions
to improve future automation suggestions and responses.
"""

from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, or_, func, select
import hashlib
import json
import logging

from models.knowledge import LearnedPattern
from models.conversation import ConversationSession, ConversationMessage, ConversationAction, ConversationPattern
from models.workflow_execution import WorkflowExecution, WorkflowExecutionStatus
from schemas.knowledge import LearnedPatternResponse

logger = logging.getLogger(__name__)


class PatternLearningService:
    """Service for learning patterns from user interactions."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.min_confidence = 0.3  # Minimum confidence to store pattern
        self.activation_threshold = 0.7  # Minimum confidence to activate pattern

    async def analyze_session(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze a conversation session for patterns after completion.

        Args:
            session_id: The conversation session ID
            user_id: User ID to scope patterns to user level
            organization_id: Organization ID to scope patterns to org level

        Returns:
            Extracted pattern data or None if no pattern found
        """
        # Get the session with all messages and actions
        stmt = select(ConversationSession).filter_by(id=session_id)
        result = await self.db.execute(stmt)
        session = result.scalar_one_or_none()

        if not session:
            logger.warning(f"Session {session_id} not found")
            return None

        # If user_id not provided, get from session
        if not user_id and hasattr(session, 'user_id'):
            user_id = session.user_id

        # Check if session resulted in successful action
        stmt = select(ConversationAction).filter(
            and_(
                ConversationAction.session_id == session_id,
                ConversationAction.status == "completed"
            )
        )
        result = await self.db.execute(stmt)
        successful_actions = result.scalars().all()

        if not successful_actions:
            logger.info(f"No successful actions in session {session_id}")
            return None

        # Extract patterns from the session
        patterns = []

        # Pattern 1: Intent to action mapping
        for action in successful_actions:
            if session.primary_intent:
                intent_pattern = self._extract_intent_pattern(session, action)
                if intent_pattern:
                    patterns.append(intent_pattern)

        # Pattern 2: Message sequence patterns
        stmt = select(ConversationMessage).filter_by(
            session_id=session_id
        ).order_by(ConversationMessage.timestamp)
        result = await self.db.execute(stmt)
        messages = result.scalars().all()

        if len(messages) >= 2:
            sequence_pattern = self._extract_sequence_pattern(messages, successful_actions)
            if sequence_pattern:
                patterns.append(sequence_pattern)

        # Pattern 3: Parameter extraction patterns
        if session.extracted_params:
            param_pattern = self._extract_parameter_pattern(session, successful_actions)
            if param_pattern:
                patterns.append(param_pattern)

        # Store patterns with user/org scoping
        stored_patterns = []
        for pattern in patterns:
            stored = await self._store_pattern(
                pattern,
                user_id=user_id,
                organization_id=organization_id
            )
            if stored:
                stored_patterns.append(stored)

        return {
            "session_id": session_id,
            "user_id": user_id,
            "organization_id": organization_id,
            "patterns_found": len(patterns),
            "patterns_stored": len(stored_patterns),
            "pattern_types": [p["type"] for p in patterns]
        }

    def _extract_intent_pattern(self, session: ConversationSession,
                                action: ConversationAction) -> Optional[Dict]:
        """Extract intent to action mapping pattern."""
        pattern_data = {
            "intent": session.primary_intent,
            "intent_confidence": session.intent_confidence or 0.5,
            "action_type": action.action_type,
            "action_params_template": self._generalize_params(action.action_params),
            "context_requirements": self._extract_context_requirements(session.context)
        }

        # Generate pattern signature
        signature = self._generate_signature("intent_action", pattern_data)

        return {
            "type": "intent_action",
            "signature": signature,
            "data": pattern_data,
            "confidence": session.intent_confidence or 0.5
        }

    def _extract_sequence_pattern(self, messages: List[ConversationMessage],
                                 actions: List[ConversationAction]) -> Optional[Dict]:
        """Extract message sequence patterns."""
        # Get user messages only
        user_messages = [m for m in messages if m.role == "user"]
        if len(user_messages) < 2:
            return None

        # Extract the sequence of intents
        intent_sequence = []
        for msg in user_messages:
            if msg.detected_intent:
                intent_sequence.append({
                    "intent": msg.detected_intent,
                    "confidence": msg.intent_confidence or 0.5
                })

        if len(intent_sequence) < 2:
            return None

        pattern_data = {
            "intent_sequence": intent_sequence,
            "final_action": actions[0].action_type if actions else None,
            "turn_count": len(user_messages)
        }

        signature = self._generate_signature("sequence", pattern_data)

        return {
            "type": "sequence",
            "signature": signature,
            "data": pattern_data,
            "confidence": min([i["confidence"] for i in intent_sequence])
        }

    def _extract_parameter_pattern(self, session: ConversationSession,
                                  actions: List[ConversationAction]) -> Optional[Dict]:
        """Extract parameter extraction patterns."""
        if not session.extracted_params or not actions:
            return None

        pattern_data = {
            "intent": session.primary_intent,
            "extracted_params": list(session.extracted_params.keys()),
            "param_sources": self._analyze_param_sources(session),
            "action_params_used": list(actions[0].action_params.keys()) if actions[0].action_params else []
        }

        signature = self._generate_signature("parameter", pattern_data)

        return {
            "type": "parameter",
            "signature": signature,
            "data": pattern_data,
            "confidence": 0.6  # Default confidence for parameter patterns
        }

    def _generalize_params(self, params: Dict) -> Dict:
        """Generalize parameters to create reusable template."""
        template = {}
        for key, value in params.items():
            if isinstance(value, str):
                # Replace specific values with placeholders
                template[key] = "<user_input>"
            elif isinstance(value, (int, float)):
                template[key] = "<numeric>"
            elif isinstance(value, bool):
                template[key] = value
            elif isinstance(value, list):
                template[key] = "<list>"
            elif isinstance(value, dict):
                template[key] = self._generalize_params(value)
            else:
                template[key] = "<value>"
        return template

    def _extract_context_requirements(self, context: Dict) -> Dict:
        """Extract required context for pattern application."""
        requirements = {}

        # Check for required capabilities
        if "available_capabilities" in context:
            requirements["capabilities"] = True

        # Check for user preferences
        if "user_preferences" in context:
            requirements["preferences"] = True

        # Check for previous actions
        if "previous_actions" in context:
            requirements["history"] = True

        return requirements

    def _analyze_param_sources(self, session: ConversationSession) -> Dict:
        """Analyze where parameters came from in the conversation."""
        # Note: This is a simplified version since we can't do async queries here
        # In a real implementation, this would be moved to an async method
        sources = {}
        for param_name in session.extracted_params.keys():
            sources[param_name] = "inferred"  # Simplified
        return sources

    def _generate_signature(self, pattern_type: str, data: Dict) -> str:
        """Generate unique signature for pattern."""
        # Create a stable string representation
        if pattern_type == "intent_action":
            key = f"{data['intent']}:{data['action_type']}"
        elif pattern_type == "sequence":
            intents = [i["intent"] for i in data["intent_sequence"]]
            key = f"{':'.join(intents)}:{data.get('final_action', 'none')}"
        elif pattern_type == "parameter":
            key = f"{data['intent']}:{':'.join(sorted(data['extracted_params']))}"
        else:
            key = json.dumps(data, sort_keys=True)

        # Generate hash
        return hashlib.sha256(f"{pattern_type}:{key}".encode()).hexdigest()[:32]

    async def _store_pattern(
        self,
        pattern: Dict,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> Optional[LearnedPattern]:
        """Store or update a learned pattern with multi-tier scoping.

        Args:
            pattern: Pattern data to store
            user_id: User ID for user-scoped patterns
            organization_id: Organization ID for org-scoped patterns

        Returns:
            Stored or updated pattern
        """
        # Determine scope based on provided IDs
        if user_id:
            scope = "user"
        elif organization_id:
            scope = "organization"
        else:
            scope = "global"

        # Check if pattern already exists for this user/org/global
        stmt = select(LearnedPattern).filter(
            and_(
                LearnedPattern.pattern_signature == pattern["signature"],
                LearnedPattern.scope == scope,
                LearnedPattern.user_id == user_id if scope == "user" else True,
                LearnedPattern.organization_id == organization_id if scope == "organization" else True
            )
        )
        result = await self.db.execute(stmt)
        existing = result.scalar_one_or_none()

        if existing:
            # Update existing pattern
            existing.occurrence_count += 1
            existing.success_count += 1
            existing.last_observed = datetime.utcnow()

            # Update confidence score (weighted average)
            weight = 0.1  # Weight for new observation
            existing.confidence_score = (
                existing.confidence_score * (1 - weight) +
                pattern["confidence"] * weight
            )

            # Check if pattern should be activated
            if (existing.confidence_score >= self.activation_threshold and
                existing.occurrence_count >= 3 and
                not existing.is_active):
                existing.is_active = True
                logger.info(f"Activated pattern {pattern['signature']} at {scope} scope")

            await self.db.commit()
            return existing

        else:
            # Create new pattern if confidence is sufficient
            if pattern["confidence"] < self.min_confidence:
                return None

            new_pattern = LearnedPattern(
                # Multi-tier scoping
                scope=scope,
                user_id=user_id,
                organization_id=organization_id,
                # Pattern identification
                pattern_type=pattern["type"],
                pattern_signature=pattern["signature"],
                pattern_data=pattern["data"],
                # Metrics
                confidence_score=pattern["confidence"],
                occurrence_count=1,
                success_count=1,
                is_active=False,  # Not active until threshold met
                # Privacy defaults
                is_shareable=True,  # Can be changed by user
                contains_sensitive_data=False,  # Auto-detected later
                anonymized=False
            )

            self.db.add(new_pattern)
            await self.db.commit()
            logger.info(f"Stored new pattern {pattern['signature']} at {scope} scope")
            return new_pattern

    async def get_patterns_for_prompt(
        self,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        max_patterns: int = 5,
        max_chars: int = 800
    ) -> str:
        """Format learned patterns as compact text for system prompt injection.

        Calls get_relevant_patterns() and formats the result as a short text block
        suitable for inclusion in a system prompt, respecting a character budget.

        Returns:
            Formatted string of patterns, or "" if none found.
        """
        try:
            patterns = await self.get_relevant_patterns(
                context={},
                user_id=user_id,
                organization_id=organization_id
            )
        except Exception:
            logger.debug("Could not retrieve patterns for prompt")
            return ""

        if not patterns:
            return ""

        lines = []
        total_chars = 0

        for p in patterns[:max_patterns]:
            data = p.pattern_data or {}
            ptype = p.pattern_type

            if ptype == "intent_action":
                line = f"- When intent is \"{data.get('intent', '?')}\", use {data.get('action_type', '?')}"
            elif ptype == "sequence":
                seq = [i.get("intent", "?") for i in data.get("intent_sequence", [])]
                line = f"- Sequence: {' → '.join(seq)} → {data.get('final_action', '?')}"
            elif ptype == "parameter":
                params = data.get("extracted_params", [])
                line = f"- For \"{data.get('intent', '?')}\", extract: {', '.join(params)}"
            else:
                continue

            if total_chars + len(line) + 1 > max_chars:
                break
            lines.append(line)
            total_chars += len(line) + 1

        return "\n".join(lines)

    async def get_relevant_patterns(
        self,
        context: Dict,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None
    ) -> List[LearnedPattern]:
        """Get patterns relevant to current context with multi-tier priority cascade.

        Priority order:
        1. User-specific patterns (highest priority)
        2. Organization patterns (if Enterprise)
        3. Global patterns (lowest priority)

        Args:
            context: Current conversation context
            user_id: User ID for user-scoped patterns
            organization_id: Organization ID for org-scoped patterns

        Returns:
            List of relevant active patterns ordered by priority and confidence
        """
        all_patterns = []

        # Tier 1: Get user-specific patterns (highest priority)
        if user_id:
            stmt = select(LearnedPattern).filter(
                and_(
                    LearnedPattern.is_active == True,
                    LearnedPattern.scope == "user",
                    LearnedPattern.user_id == user_id
                )
            )
            result = await self.db.execute(stmt)
            user_patterns = result.scalars().all()

            for pattern in user_patterns:
                if self._is_pattern_relevant(pattern, context):
                    all_patterns.append({
                        'pattern': pattern,
                        'tier': 1,  # Highest priority
                        'tier_name': 'user',
                        'score': pattern.confidence_score * 1.5  # Boost user patterns
                    })

        # Tier 2: Get organization patterns (medium priority)
        if organization_id:
            stmt = select(LearnedPattern).filter(
                and_(
                    LearnedPattern.is_active == True,
                    LearnedPattern.scope == "organization",
                    LearnedPattern.organization_id == organization_id
                )
            )
            result = await self.db.execute(stmt)
            org_patterns = result.scalars().all()

            for pattern in org_patterns:
                if self._is_pattern_relevant(pattern, context):
                    # Check if we already have this pattern from user tier
                    if not any(p['pattern'].pattern_signature == pattern.pattern_signature
                              for p in all_patterns):
                        all_patterns.append({
                            'pattern': pattern,
                            'tier': 2,  # Medium priority
                            'tier_name': 'organization',
                            'score': pattern.confidence_score * 1.2  # Slight boost
                        })

        # Tier 3: Get global patterns (lowest priority, but still valuable)
        stmt = select(LearnedPattern).filter(
            and_(
                LearnedPattern.is_active == True,
                LearnedPattern.scope == "global"
            )
        )
        result = await self.db.execute(stmt)
        global_patterns = result.scalars().all()

        for pattern in global_patterns:
            if self._is_pattern_relevant(pattern, context):
                # Check if we already have this pattern from higher tiers
                if not any(p['pattern'].pattern_signature == pattern.pattern_signature
                          for p in all_patterns):
                    all_patterns.append({
                        'pattern': pattern,
                        'tier': 3,  # Lowest priority
                        'tier_name': 'global',
                        'score': pattern.confidence_score  # No boost
                    })

        # Sort by tier first (lower is higher priority), then by score
        all_patterns.sort(key=lambda p: (p['tier'], -p['score']))

        # Extract just the pattern objects
        result_patterns = [p['pattern'] for p in all_patterns[:10]]

        logger.info(
            f"Retrieved {len(result_patterns)} relevant patterns "
            f"(user: {sum(1 for p in all_patterns if p['tier'] == 1)}, "
            f"org: {sum(1 for p in all_patterns if p['tier'] == 2)}, "
            f"global: {sum(1 for p in all_patterns if p['tier'] == 3)})"
        )

        return result_patterns

    def _is_pattern_relevant(self, pattern: LearnedPattern, context: Dict) -> bool:
        """Check if pattern is relevant to current context."""
        pattern_data = pattern.pattern_data

        # Check intent match for intent_action patterns
        if pattern.pattern_type == "intent_action":
            current_intent = context.get("primary_intent")
            if current_intent and pattern_data.get("intent") == current_intent:
                return True

        # Check sequence match
        elif pattern.pattern_type == "sequence":
            recent_intents = context.get("recent_intents", [])
            if recent_intents:
                pattern_intents = [i["intent"] for i in pattern_data.get("intent_sequence", [])]
                # Check if recent intents match start of pattern
                if len(recent_intents) >= len(pattern_intents) - 1:
                    if recent_intents[-len(pattern_intents)+1:] == pattern_intents[:-1]:
                        return True

        # Check parameter patterns
        elif pattern.pattern_type == "parameter":
            if pattern_data.get("intent") == context.get("primary_intent"):
                return True

        return False

    async def apply_pattern(self, pattern: LearnedPattern, context: Dict) -> Dict:
        """Apply a learned pattern to generate suggestions.

        Args:
            pattern: The pattern to apply
            context: Current conversation context

        Returns:
            Suggestions based on the pattern
        """
        suggestions = {
            "pattern_id": str(pattern.id),
            "pattern_type": pattern.pattern_type,
            "confidence": pattern.confidence_score
        }

        pattern_data = pattern.pattern_data

        if pattern.pattern_type == "intent_action":
            # Suggest action based on intent
            suggestions["suggested_action"] = pattern_data.get("action_type")
            suggestions["action_params_template"] = pattern_data.get("action_params_template")

        elif pattern.pattern_type == "sequence":
            # Suggest next intent or action
            suggestions["next_intent"] = pattern_data.get("intent_sequence", [])[-1]
            suggestions["expected_action"] = pattern_data.get("final_action")

        elif pattern.pattern_type == "parameter":
            # Suggest parameters to extract
            suggestions["parameters_to_extract"] = pattern_data.get("extracted_params", [])
            suggestions["param_sources"] = pattern_data.get("param_sources", {})

        # Update pattern usage
        pattern.last_applied = datetime.utcnow()
        pattern.application_count += 1
        await self.db.commit()

        return suggestions

    async def get_pattern_statistics(self) -> Dict:
        """Get statistics about learned patterns."""
        stmt = select(func.count(LearnedPattern.id))
        result = await self.db.execute(stmt)
        total_patterns = result.scalar()

        stmt = select(func.count(LearnedPattern.id)).filter(
            LearnedPattern.is_active == True
        )
        result = await self.db.execute(stmt)
        active_patterns = result.scalar()

        # Get patterns by type
        stmt = select(
            LearnedPattern.pattern_type,
            func.count(LearnedPattern.id)
        ).group_by(LearnedPattern.pattern_type)
        result = await self.db.execute(stmt)
        pattern_types = result.all()

        # Get most successful patterns
        stmt = select(LearnedPattern).filter(
            LearnedPattern.is_active == True
        ).order_by(
            LearnedPattern.success_count.desc()
        ).limit(5)
        result = await self.db.execute(stmt)
        top_patterns = result.scalars().all()

        return {
            "total_patterns": total_patterns,
            "active_patterns": active_patterns,
            "patterns_by_type": {pt: count for pt, count in pattern_types},
            "top_patterns": [
                {
                    "id": str(p.id),
                    "type": p.pattern_type,
                    "success_count": p.success_count,
                    "confidence": p.confidence_score
                }
                for p in top_patterns
            ]
        }

    async def cleanup_old_patterns(self, days: int = 90) -> int:
        """Remove patterns that haven't been observed recently.

        Args:
            days: Number of days to consider a pattern stale

        Returns:
            Number of patterns removed
        """
        cutoff_date = datetime.utcnow() - timedelta(days=days)

        # Find stale patterns
        stmt = select(LearnedPattern).filter(
            and_(
                LearnedPattern.last_observed < cutoff_date,
                LearnedPattern.is_active == False,
                LearnedPattern.occurrence_count < 3
            )
        )
        result = await self.db.execute(stmt)
        stale_patterns = result.scalars().all()

        count = len(stale_patterns)

        # Remove stale patterns
        for pattern in stale_patterns:
            await self.db.delete(pattern)

        await self.db.commit()
        logger.info(f"Cleaned up {count} stale patterns")

        return count