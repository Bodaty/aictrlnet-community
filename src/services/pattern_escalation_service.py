"""Pattern Escalation Service for Multi-Tier Pattern Learning.

Handles the promotion of patterns from user → organization → global levels.
Part of the Intelligent Assistant v3 adaptive learning system.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_, func
from sqlalchemy.orm import selectinload
import logging

from models.knowledge import LearnedPattern
from models.user import User

logger = logging.getLogger(__name__)


class PatternEscalationService:
    """Service for managing pattern escalation across user/org/global tiers.

    Escalation Flow:
    1. User patterns that meet criteria become promotion candidates
    2. Promotion candidates can be elevated to org level (Enterprise only)
    3. Org patterns that meet criteria can become global patterns

    Privacy Controls:
    - Only shareable patterns can be promoted
    - Sensitive data is flagged and blocked from promotion
    - Patterns are anonymized before promotion to higher tiers
    """

    # Escalation thresholds
    USER_TO_ORG_THRESHOLD = {
        'min_occurrence_count': 10,
        'min_success_rate': 0.8,
        'min_confidence_score': 0.75,
        'min_users': 1,  # Single user must demonstrate consistency
    }

    ORG_TO_GLOBAL_THRESHOLD = {
        'min_occurrence_count': 50,
        'min_success_rate': 0.85,
        'min_confidence_score': 0.80,
        'min_users': 5,  # Must be validated across multiple users
    }

    def __init__(self, db: AsyncSession):
        self.db = db

    async def identify_promotion_candidates(
        self,
        scope: str = "user",
        min_age_days: int = 7
    ) -> List[LearnedPattern]:
        """Identify patterns that are candidates for promotion to next tier.

        Args:
            scope: Current scope to evaluate (user or organization)
            min_age_days: Minimum age in days before pattern can be promoted

        Returns:
            List of patterns that meet promotion criteria
        """
        threshold = (
            self.USER_TO_ORG_THRESHOLD if scope == "user"
            else self.ORG_TO_GLOBAL_THRESHOLD
        )

        min_date = datetime.utcnow() - timedelta(days=min_age_days)

        # Query patterns that meet base criteria
        query = select(LearnedPattern).where(
            and_(
                LearnedPattern.scope == scope,
                LearnedPattern.is_active == True,
                LearnedPattern.is_shareable == True,
                LearnedPattern.contains_sensitive_data == False,
                LearnedPattern.promotion_candidate == False,  # Not already flagged
                LearnedPattern.first_observed <= min_date,
                LearnedPattern.occurrence_count >= threshold['min_occurrence_count'],
                LearnedPattern.confidence_score >= threshold['min_confidence_score'],
            )
        )

        result = await self.db.execute(query)
        patterns = result.scalars().all()

        # Filter by success rate
        candidates = []
        for pattern in patterns:
            success_rate = (
                pattern.success_count / pattern.occurrence_count
                if pattern.occurrence_count > 0 else 0
            )
            if success_rate >= threshold['min_success_rate']:
                candidates.append(pattern)

        logger.info(
            f"Identified {len(candidates)} promotion candidates from {scope} scope"
        )
        return candidates

    async def mark_as_promotion_candidate(
        self,
        pattern_id: str
    ) -> Optional[LearnedPattern]:
        """Mark a pattern as a candidate for promotion.

        Args:
            pattern_id: ID of the pattern to mark

        Returns:
            Updated pattern or None if not found
        """
        query = select(LearnedPattern).where(LearnedPattern.id == pattern_id)
        result = await self.db.execute(query)
        pattern = result.scalar_one_or_none()

        if not pattern:
            logger.warning(f"Pattern {pattern_id} not found")
            return None

        pattern.promotion_candidate = True
        await self.db.commit()
        await self.db.refresh(pattern)

        logger.info(
            f"Marked pattern {pattern_id} ({pattern.pattern_signature}) as promotion candidate"
        )
        return pattern

    async def promote_pattern(
        self,
        pattern_id: str,
        target_scope: str,
        organization_id: Optional[str] = None,
        validated_by: Optional[str] = None,
        anonymize: bool = True
    ) -> Optional[LearnedPattern]:
        """Promote a pattern to the next tier.

        Args:
            pattern_id: ID of the pattern to promote
            target_scope: Target scope (organization or global)
            organization_id: Organization ID (required for user→org promotion)
            validated_by: User ID who validated the promotion
            anonymize: Whether to anonymize the pattern data

        Returns:
            Newly created promoted pattern or None if promotion failed
        """
        # Get source pattern
        query = select(LearnedPattern).where(LearnedPattern.id == pattern_id)
        result = await self.db.execute(query)
        source_pattern = result.scalar_one_or_none()

        if not source_pattern:
            logger.error(f"Source pattern {pattern_id} not found")
            return None

        # Validate promotion is allowed
        if not source_pattern.is_shareable:
            logger.error(f"Pattern {pattern_id} is not shareable, cannot promote")
            return None

        if source_pattern.contains_sensitive_data:
            logger.error(f"Pattern {pattern_id} contains sensitive data, cannot promote")
            return None

        # Create promoted pattern
        promoted_pattern = LearnedPattern(
            scope=target_scope,
            user_id=None if target_scope == "global" else source_pattern.user_id,
            organization_id=organization_id if target_scope == "organization" else None,
            promoted_from_scope=source_pattern.scope,
            contributing_users_count=1,  # Will be incremented as more users contribute
            promotion_candidate=False,
            is_shareable=True,
            contains_sensitive_data=False,
            anonymized=anonymize,
            # Copy pattern identification
            pattern_type=source_pattern.pattern_type,
            pattern_signature=source_pattern.pattern_signature,
            pattern_data=self._anonymize_pattern_data(source_pattern.pattern_data) if anonymize else source_pattern.pattern_data,
            context_requirements=source_pattern.context_requirements,
            # Initialize metrics
            occurrence_count=source_pattern.occurrence_count,
            success_count=source_pattern.success_count,
            confidence_score=source_pattern.confidence_score,
            is_active=True,
            activation_threshold=source_pattern.activation_threshold,
            application_count=0,
            # Timestamps
            first_observed=datetime.utcnow(),
            last_observed=datetime.utcnow(),
            # Validation
            is_validated=True,
            validated_by=validated_by,
            validated_at=datetime.utcnow(),
        )

        self.db.add(promoted_pattern)
        await self.db.commit()
        await self.db.refresh(promoted_pattern)

        logger.info(
            f"Promoted pattern {pattern_id} from {source_pattern.scope} to {target_scope}: "
            f"{promoted_pattern.id}"
        )

        return promoted_pattern

    async def merge_similar_patterns(
        self,
        scope: str,
        organization_id: Optional[str] = None
    ) -> int:
        """Merge similar patterns at a given scope level.

        When multiple users have similar patterns, consolidate them into a single
        pattern with aggregated metrics.

        Args:
            scope: Scope level to merge (organization or global)
            organization_id: Organization ID (required for org scope)

        Returns:
            Number of patterns merged
        """
        # Get all patterns at this scope
        query = select(LearnedPattern).where(
            and_(
                LearnedPattern.scope == scope,
                LearnedPattern.is_active == True,
            )
        )

        if scope == "organization" and organization_id:
            query = query.where(LearnedPattern.organization_id == organization_id)

        result = await self.db.execute(query)
        patterns = result.scalars().all()

        # Group by pattern_signature
        signature_groups: Dict[str, List[LearnedPattern]] = {}
        for pattern in patterns:
            key = f"{pattern.pattern_type}:{pattern.pattern_signature}"
            if key not in signature_groups:
                signature_groups[key] = []
            signature_groups[key].append(pattern)

        merged_count = 0

        # Merge groups with multiple patterns
        for key, group in signature_groups.items():
            if len(group) <= 1:
                continue

            # Keep the pattern with highest confidence, update metrics
            group.sort(key=lambda p: p.confidence_score, reverse=True)
            primary_pattern = group[0]

            total_occurrences = sum(p.occurrence_count for p in group)
            total_successes = sum(p.success_count for p in group)
            total_applications = sum(p.application_count for p in group)
            contributing_users = len(set(p.user_id for p in group if p.user_id))

            # Update primary pattern
            primary_pattern.occurrence_count = total_occurrences
            primary_pattern.success_count = total_successes
            primary_pattern.application_count = total_applications
            primary_pattern.contributing_users_count = contributing_users
            primary_pattern.confidence_score = total_successes / total_occurrences if total_occurrences > 0 else 0

            # Deactivate other patterns
            for pattern in group[1:]:
                pattern.is_active = False
                merged_count += 1

        await self.db.commit()

        logger.info(f"Merged {merged_count} patterns at {scope} scope")
        return merged_count

    async def get_promotion_statistics(
        self,
        scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get statistics about pattern promotions.

        Args:
            scope: Specific scope to analyze (optional)

        Returns:
            Dictionary with promotion statistics
        """
        stats = {
            'by_scope': {},
            'promotion_candidates': 0,
            'total_promoted': 0,
        }

        scopes = [scope] if scope else ['user', 'organization', 'global']

        for s in scopes:
            query = select(func.count(LearnedPattern.id)).where(
                LearnedPattern.scope == s
            )
            result = await self.db.execute(query)
            total = result.scalar()

            query_candidates = select(func.count(LearnedPattern.id)).where(
                and_(
                    LearnedPattern.scope == s,
                    LearnedPattern.promotion_candidate == True
                )
            )
            result_candidates = await self.db.execute(query_candidates)
            candidates = result_candidates.scalar()

            query_promoted = select(func.count(LearnedPattern.id)).where(
                LearnedPattern.promoted_from_scope == s
            )
            result_promoted = await self.db.execute(query_promoted)
            promoted = result_promoted.scalar()

            stats['by_scope'][s] = {
                'total_patterns': total,
                'promotion_candidates': candidates,
                'promoted_to_next_tier': promoted,
            }

            stats['promotion_candidates'] += candidates
            stats['total_promoted'] += promoted

        return stats

    def _anonymize_pattern_data(self, pattern_data: Dict[str, Any]) -> Dict[str, Any]:
        """Remove personally identifiable information from pattern data.

        Args:
            pattern_data: Original pattern data

        Returns:
            Anonymized pattern data
        """
        # Create a copy to avoid modifying original
        anonymized = pattern_data.copy()

        # Remove common PII fields
        pii_fields = [
            'user_id', 'username', 'email', 'name', 'phone', 'address',
            'ip_address', 'session_id', 'device_id', 'user_name'
        ]

        for field in pii_fields:
            if field in anonymized:
                anonymized[field] = '[ANONYMIZED]'

        # Recursively anonymize nested dictionaries
        for key, value in anonymized.items():
            if isinstance(value, dict):
                anonymized[key] = self._anonymize_pattern_data(value)
            elif isinstance(value, list):
                anonymized[key] = [
                    self._anonymize_pattern_data(item) if isinstance(item, dict) else item
                    for item in value
                ]

        return anonymized

    async def auto_promote_eligible_patterns(
        self,
        scope: str = "user",
        organization_id: Optional[str] = None,
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Automatically promote eligible patterns based on criteria.

        This is designed to run as a background job.

        Args:
            scope: Source scope to evaluate (user or organization)
            organization_id: Organization ID for org→global promotions
            dry_run: If True, only identify candidates without promoting

        Returns:
            Summary of promotion activity
        """
        candidates = await self.identify_promotion_candidates(scope)

        if dry_run:
            return {
                'dry_run': True,
                'candidates_identified': len(candidates),
                'would_promote': [p.id for p in candidates],
            }

        # Mark all as candidates
        for pattern in candidates:
            await self.mark_as_promotion_candidate(pattern.id)

        # Auto-promote patterns with very high confidence
        auto_promote_threshold = 0.95
        promoted = []

        for pattern in candidates:
            if pattern.confidence_score >= auto_promote_threshold:
                target_scope = "organization" if scope == "user" else "global"

                promoted_pattern = await self.promote_pattern(
                    pattern.id,
                    target_scope=target_scope,
                    organization_id=organization_id if target_scope == "organization" else None,
                    validated_by="system",
                    anonymize=True
                )

                if promoted_pattern:
                    promoted.append(promoted_pattern.id)

        return {
            'dry_run': False,
            'candidates_identified': len(candidates),
            'candidates_marked': len(candidates),
            'auto_promoted': len(promoted),
            'promoted_pattern_ids': promoted,
            'pending_validation': len(candidates) - len(promoted),
        }
