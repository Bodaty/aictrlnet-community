"""Model tier resolution logic for user preferences.

This module provides functions to resolve user model tier preferences with
appropriate fallback logic.
"""

import logging
import os
from typing import Optional, Tuple, List
from llm.models import ModelTier

logger = logging.getLogger(__name__)


# Environment-configured default model (for Cloud Run/non-Ollama deployments)
# Using Vertex AI for enterprise features (SLA, IAM, audit logging)
DEFAULT_LLM_MODEL = os.environ.get('DEFAULT_LLM_MODEL', 'gemini-2.0-flash-vertex')


def get_environment_default_model() -> str:
    """
    Get the environment-configured default LLM model.

    This is the centralized function for getting the DEFAULT_LLM_MODEL,
    which is used when Ollama is not available (e.g., Cloud Run deployments).

    Returns:
        The DEFAULT_LLM_MODEL environment variable value, or 'gemini-2.0-flash-vertex'
        if not set.
    """
    return os.environ.get('DEFAULT_LLM_MODEL', 'gemini-2.0-flash-vertex')


def is_ollama_model(model: str) -> bool:
    """
    Check if a model name indicates an Ollama model (not an API model).

    Ollama models typically have patterns like:
    - llama3.2:1b, llama3.1:8b-instruct-q4_K_M
    - mistral:7b, codellama:13b
    - No provider prefix (gemini-, gpt-, claude-)

    Args:
        model: Model name to check

    Returns:
        True if this appears to be an Ollama model, False if it's an API model
    """
    if not model:
        return False

    model_lower = model.lower()

    # API model prefixes/patterns
    api_patterns = [
        'gemini',
        'gpt-',
        'claude',
        'anthropic',
        'vertex',
        'openai',
        'deepseek-chat', 'deepseek-reasoner',  # DeepSeek API models
        'qwen-turbo', 'qwen-plus', 'qwen-max', 'qwen-long',  # DashScope API models
    ]

    for pattern in api_patterns:
        if pattern in model_lower:
            return False

    # If no API pattern found, it's likely an Ollama model
    return True


def get_model_for_tier(
    tier: ModelTier,
    user_preferences: Optional[dict] = None
) -> Optional[str]:
    """
    Get the user's preferred model for a specific tier.

    Fallback logic:
    1. If user has configured model for requested tier → use it
    2. If user has configured quality tier model → use it (highest quality fallback)
    3. If user has configured ANY tier model → use it (single model for all)
    4. Return None (caller will use system defaults)

    Args:
        tier: The ModelTier to resolve (FAST, BALANCED, or QUALITY)
        user_preferences: User's preferences dict (from User.preferences JSON column)

    Returns:
        Model name if found in user preferences, None otherwise

    Examples:
        >>> # User has all 3 tiers configured
        >>> prefs = {
        ...     'preferredFastModel': 'llama3.2:1b',
        ...     'preferredBalancedModel': 'llama3.2:3b',
        ...     'preferredQualityModel': 'llama3.1:8b-instruct-q4_K_M'
        ... }
        >>> get_model_for_tier(ModelTier.FAST, prefs)
        'llama3.2:1b'
        >>> get_model_for_tier(ModelTier.QUALITY, prefs)
        'llama3.1:8b-instruct-q4_K_M'

        >>> # User has only quality tier configured (use for all)
        >>> prefs = {'preferredQualityModel': 'llama3.1:8b-instruct-q4_K_M'}
        >>> get_model_for_tier(ModelTier.FAST, prefs)
        'llama3.1:8b-instruct-q4_K_M'

        >>> # User configured same model for all tiers (perfectly valid!)
        >>> prefs = {
        ...     'preferredFastModel': 'llama3.1:8b-instruct-q4_K_M',
        ...     'preferredBalancedModel': 'llama3.1:8b-instruct-q4_K_M',
        ...     'preferredQualityModel': 'llama3.1:8b-instruct-q4_K_M'
        ... }
        >>> get_model_for_tier(ModelTier.BALANCED, prefs)
        'llama3.1:8b-instruct-q4_K_M'
    """
    if not user_preferences:
        return None

    # Map tier enum to preference key
    tier_key_map = {
        ModelTier.FAST: 'preferredFastModel',
        ModelTier.BALANCED: 'preferredBalancedModel',
        ModelTier.QUALITY: 'preferredQualityModel',
    }

    # 1. Check if user has configured model for this specific tier
    tier_key = tier_key_map.get(tier)
    if tier_key and tier_key in user_preferences:
        model = user_preferences[tier_key]
        if model:
            logger.debug(f"Resolved {tier.value} tier to user's {tier_key}: {model}")
            return model

    # 2. Fall back to quality tier (user wants quality everywhere)
    quality_model = user_preferences.get('preferredQualityModel')
    if quality_model:
        logger.debug(f"Resolved {tier.value} tier to user's quality model (fallback): {quality_model}")
        return quality_model

    # 3. Check if user has ANY tier configured (single model for all tiers)
    for tier_key in ['preferredBalancedModel', 'preferredFastModel']:
        if tier_key in user_preferences and user_preferences[tier_key]:
            model = user_preferences[tier_key]
            logger.debug(f"Resolved {tier.value} tier to user's single configured model: {model}")
            return model

    # 4. No user preference found - caller will use system defaults
    logger.debug(f"No user preference found for {tier.value} tier, will use system default")
    return None


def get_model_with_tier_fallback(
    tier: ModelTier,
    user_preferences: Optional[dict] = None,
    system_default: str = "llama3.1:8b-instruct-q4_K_M"
) -> Tuple[str, str]:
    """
    Get model for tier with complete fallback to system defaults.

    This is a convenience function that always returns a valid model by falling
    back to system defaults when no user preference is found.

    Args:
        tier: The ModelTier to resolve
        user_preferences: User's preferences dict
        system_default: System default model (if no user preferences)

    Returns:
        Tuple of (model_name, source) where source is one of:
        - "user_tier_specific": User configured this specific tier
        - "user_quality_fallback": Fell back to user's quality tier
        - "user_single_model": User has single model for all tiers
        - "system_default": Using system default

    Examples:
        >>> prefs = {'preferredFastModel': 'llama3.2:1b'}
        >>> get_model_with_tier_fallback(ModelTier.FAST, prefs)
        ('llama3.2:1b', 'user_tier_specific')

        >>> get_model_with_tier_fallback(ModelTier.BALANCED, prefs)
        ('llama3.2:1b', 'user_single_model')

        >>> get_model_with_tier_fallback(ModelTier.QUALITY, None)
        ('llama3.1:8b-instruct-q4_K_M', 'system_default')
    """
    if not user_preferences:
        return (system_default, "system_default")

    # Map tier enum to preference key
    tier_key_map = {
        ModelTier.FAST: 'preferredFastModel',
        ModelTier.BALANCED: 'preferredBalancedModel',
        ModelTier.QUALITY: 'preferredQualityModel',
    }

    # 1. Check tier-specific preference
    tier_key = tier_key_map.get(tier)
    if tier_key and tier_key in user_preferences and user_preferences[tier_key]:
        return (user_preferences[tier_key], "user_tier_specific")

    # 2. Fall back to quality tier
    if 'preferredQualityModel' in user_preferences and user_preferences['preferredQualityModel']:
        return (user_preferences['preferredQualityModel'], "user_quality_fallback")

    # 3. Check for any configured tier (single model)
    for check_key in ['preferredBalancedModel', 'preferredFastModel']:
        if check_key in user_preferences and user_preferences[check_key]:
            return (user_preferences[check_key], "user_single_model")

    # 4. System default
    return (system_default, "system_default")


# System-level defaults for each tier
SYSTEM_TIER_DEFAULTS = {
    ModelTier.FAST: "llama3.2:1b",           # ~1-2s per call
    ModelTier.BALANCED: "llama3.2:3b",       # ~3-5s per call
    ModelTier.QUALITY: "llama3.1:8b-instruct-q4_K_M",  # ~20-25s per call
}


def get_system_default_for_tier(tier: ModelTier) -> str:
    """
    Get the system default model for a specific tier.

    These are the recommended models for each tier based on performance testing.

    Args:
        tier: The ModelTier to get default for

    Returns:
        Model name for the tier
    """
    return SYSTEM_TIER_DEFAULTS.get(tier, SYSTEM_TIER_DEFAULTS[ModelTier.QUALITY])


def get_dynamic_system_default_for_tier(
    tier: ModelTier,
    available_models: List[str],
) -> Optional[str]:
    """Get best available model for tier when hardcoded default isn't available.

    1. Hardcoded default available → use it
    2. Classify all available models by tier → pick from matching tier
    3. No match → None (caller falls through)
    """
    from llm.model_selection import classify_model_tier, _estimate_model_size_billions

    # Try hardcoded default first
    default = SYSTEM_TIER_DEFAULTS.get(tier)
    if default and default in available_models:
        return default

    # Classify available models and find best match for this tier
    candidates = []
    for model in available_models:
        model_tier = classify_model_tier(model)
        if model_tier == tier:
            size = _estimate_model_size_billions(model) or 0.0
            candidates.append((model, size))

    if candidates:
        # Pick largest model in tier (best quality within tier)
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]

    return None
