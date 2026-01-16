"""Cost and usage tracking for LLM operations."""

import logging
from typing import Dict, Optional, Tuple, List
from datetime import datetime, timedelta
from collections import defaultdict

from .models import ModelProvider, UsageStats

logger = logging.getLogger(__name__)


class CostTracker:
    """
    Track token usage and costs for LLM operations.
    
    This provides visibility into LLM usage patterns and costs,
    helping optimize model selection and budget management.
    """
    
    def __init__(self):
        """Initialize the cost tracker."""
        # In-memory tracking (could be persisted to database)
        self.usage_data = defaultdict(lambda: {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0,
            "by_model": defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0}),
            "by_provider": defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
        })
        
        self.daily_limits = {
            "default": 10.0,  # $10 per day default limit
        }
        
        self.alerts_triggered = set()
    
    async def track_usage(
        self,
        user_id: str,
        model: str,
        provider: ModelProvider,
        tokens: int,
        cost: float
    ):
        """
        Track usage for a user.
        
        Args:
            user_id: User identifier
            model: Model used
            provider: Provider used
            tokens: Tokens consumed
            cost: Cost incurred
        """
        # Update user stats
        user_stats = self.usage_data[user_id]
        user_stats["requests"] += 1
        user_stats["tokens"] += tokens
        user_stats["cost"] += cost
        
        # Update model-specific stats
        user_stats["by_model"][model]["requests"] += 1
        user_stats["by_model"][model]["tokens"] += tokens
        user_stats["by_model"][model]["cost"] += cost
        
        # Update provider-specific stats
        user_stats["by_provider"][provider.value]["requests"] += 1
        user_stats["by_provider"][provider.value]["tokens"] += tokens
        user_stats["by_provider"][provider.value]["cost"] += cost
        
        # Check daily limit
        await self._check_daily_limit(user_id, user_stats["cost"])
        
        logger.debug(
            f"Tracked usage for {user_id}: {model} ({provider.value}), "
            f"{tokens} tokens, ${cost:.4f}"
        )
    
    async def get_stats(
        self,
        user_id: Optional[str] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> UsageStats:
        """
        Get usage statistics.
        
        Args:
            user_id: Optional user ID to filter by
            date_range: Optional date range (not implemented in memory version)
            
        Returns:
            Usage statistics
        """
        if user_id and user_id in self.usage_data:
            user_stats = self.usage_data[user_id]
            
            # Calculate cache hit rate (placeholder)
            cache_hit_rate = 0.0  # Would be calculated from actual cache stats
            
            return UsageStats(
                total_requests=user_stats["requests"],
                total_tokens=user_stats["tokens"],
                total_cost=user_stats["cost"],
                by_model=dict(user_stats["by_model"]),
                by_provider=dict(user_stats["by_provider"]),
                cache_hit_rate=cache_hit_rate
            )
        
        # Aggregate all users
        total_stats = {
            "requests": 0,
            "tokens": 0,
            "cost": 0.0,
            "by_model": defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0}),
            "by_provider": defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
        }
        
        for user_stats in self.usage_data.values():
            total_stats["requests"] += user_stats["requests"]
            total_stats["tokens"] += user_stats["tokens"]
            total_stats["cost"] += user_stats["cost"]
            
            for model, model_stats in user_stats["by_model"].items():
                total_stats["by_model"][model]["requests"] += model_stats["requests"]
                total_stats["by_model"][model]["tokens"] += model_stats["tokens"]
                total_stats["by_model"][model]["cost"] += model_stats["cost"]
            
            for provider, provider_stats in user_stats["by_provider"].items():
                total_stats["by_provider"][provider]["requests"] += provider_stats["requests"]
                total_stats["by_provider"][provider]["tokens"] += provider_stats["tokens"]
                total_stats["by_provider"][provider]["cost"] += provider_stats["cost"]
        
        return UsageStats(
            total_requests=total_stats["requests"],
            total_tokens=total_stats["tokens"],
            total_cost=total_stats["cost"],
            by_model=dict(total_stats["by_model"]),
            by_provider=dict(total_stats["by_provider"]),
            cache_hit_rate=0.0
        )
    
    async def set_daily_limit(self, user_id: str, limit: float):
        """
        Set daily spending limit for a user.
        
        Args:
            user_id: User identifier
            limit: Daily limit in dollars
        """
        self.daily_limits[user_id] = limit
        logger.info(f"Set daily limit for {user_id}: ${limit:.2f}")
    
    async def get_daily_usage(self, user_id: str) -> float:
        """
        Get today's usage for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Today's cost in dollars
        """
        # In this simple implementation, we return total cost
        # In production, this would filter by date
        if user_id in self.usage_data:
            return self.usage_data[user_id]["cost"]
        return 0.0
    
    async def _check_daily_limit(self, user_id: str, total_cost: float):
        """
        Check if user has exceeded daily limit.
        
        Args:
            user_id: User identifier
            total_cost: Total cost so far
        """
        limit = self.daily_limits.get(user_id, self.daily_limits["default"])
        
        if total_cost > limit:
            alert_key = f"{user_id}:{datetime.utcnow().date()}"
            if alert_key not in self.alerts_triggered:
                self.alerts_triggered.add(alert_key)
                logger.warning(
                    f"User {user_id} exceeded daily limit: ${total_cost:.2f} > ${limit:.2f}"
                )
                # In production, this would trigger notifications
    
    def get_top_models(self, limit: int = 5) -> List[Tuple[str, Dict]]:
        """
        Get the most used models.
        
        Args:
            limit: Number of top models to return
            
        Returns:
            List of (model_name, stats) tuples
        """
        model_totals = defaultdict(lambda: {"requests": 0, "tokens": 0, "cost": 0.0})
        
        for user_stats in self.usage_data.values():
            for model, model_stats in user_stats["by_model"].items():
                model_totals[model]["requests"] += model_stats["requests"]
                model_totals[model]["tokens"] += model_stats["tokens"]
                model_totals[model]["cost"] += model_stats["cost"]
        
        # Sort by number of requests
        sorted_models = sorted(
            model_totals.items(),
            key=lambda x: x[1]["requests"],
            reverse=True
        )
        
        return sorted_models[:limit]
    
    def get_cost_by_provider(self) -> Dict[str, float]:
        """
        Get total costs grouped by provider.
        
        Returns:
            Dictionary of provider -> total cost
        """
        provider_costs = defaultdict(float)
        
        for user_stats in self.usage_data.values():
            for provider, provider_stats in user_stats["by_provider"].items():
                provider_costs[provider] += provider_stats["cost"]
        
        return dict(provider_costs)
    
    def reset_daily_stats(self):
        """Reset daily statistics (would be called by a scheduler)."""
        # In production, this would archive current stats and reset counters
        logger.info("Daily stats reset (not implemented in memory version)")