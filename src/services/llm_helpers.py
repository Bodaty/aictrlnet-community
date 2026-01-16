"""Helper functions for LLM service integration with user preferences.

This module provides utilities for resolving LLM model selection with proper
user preference handling and fallback logic.
"""

import logging
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from models.user import User
from llm import UserLLMSettings
from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def get_user_llm_settings(
    db: AsyncSession,
    user_id: str,
    model_override: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream_responses: bool = False,
    **kwargs
) -> UserLLMSettings:
    """
    Get LLM settings with proper user preference resolution.

    This function implements the correct priority cascade for model selection:
    1. Explicit model_override (if provided by caller)
    2. User's saved preference from database (from Settings UI)
    3. System default (settings.DEFAULT_LLM_MODEL)

    Args:
        db: Database session for fetching user preferences
        user_id: User ID (can be UUID string or "system" for system tasks)
        model_override: Explicit model override (highest priority)
        temperature: Temperature override (optional)
        max_tokens: Max tokens override (optional)
        stream_responses: Whether to stream responses
        **kwargs: Additional UserLLMSettings parameters

    Returns:
        UserLLMSettings with properly resolved model selection

    Example:
        # User-initiated request with preference reading
        settings = await get_user_llm_settings(
            db=db,
            user_id=str(current_user.id),
            temperature=0.7
        )

        # System task (no user preference)
        settings = await get_user_llm_settings(
            db=db,
            user_id="system",
            temperature=0.3
        )

        # Explicit override (ignores user preference)
        settings = await get_user_llm_settings(
            db=db,
            user_id=str(current_user.id),
            model_override="gpt-4"
        )
    """
    # Priority 1: If explicit override, use it
    if model_override:
        logger.info(f"Using explicit model override: {model_override}")
        return UserLLMSettings(
            user_id=user_id,
            selected_model=model_override,
            provider="ollama",  # Will be determined by model name in LLM service
            temperature=temperature or 0.7,
            max_tokens=max_tokens or 1000,
            stream_responses=stream_responses,
            **kwargs
        )

    # Priority 2: Try to get user preference from database
    user_model = None
    preferred_fast = None
    preferred_balanced = None
    preferred_quality = None

    if user_id and user_id != "system":
        try:
            # User.id is defined as String in the model, so keep user_id as string
            result = await db.execute(
                select(User).filter(User.id == user_id)
            )
            user = result.scalar_one_or_none()

            if user and user.preferences:
                # Extract legacy aiModel preference
                user_model = user.preferences.get('aiModel')
                if user_model:
                    logger.info(f"Using user preference for {user_id}: {user_model}")

                    # Map UI-friendly names to backend model names
                    model_mapping = {
                        'llama3.1-local': 'llama3.1:8b-instruct-q4_K_M',
                        'llama3.2-1b': 'llama3.2:1b',
                        'llama3.2-3b': 'llama3.2:3b'
                    }
                    user_model = model_mapping.get(user_model, user_model)

                # Extract tier preferences
                preferred_fast = user.preferences.get('preferredFastModel')
                preferred_balanced = user.preferences.get('preferredBalancedModel')
                preferred_quality = user.preferences.get('preferredQualityModel')

                if preferred_fast or preferred_balanced or preferred_quality:
                    logger.info(f"Using tier preferences for {user_id}: fast={preferred_fast}, balanced={preferred_balanced}, quality={preferred_quality}")

        except Exception as e:
            logger.warning(f"Could not fetch user preferences for {user_id}: {e}")
            # Continue with fallback

    # Priority 3: Fallback to system default
    selected_model = user_model or settings.DEFAULT_LLM_MODEL

    if not user_model:
        logger.info(f"Using system default model: {selected_model}")

    return UserLLMSettings(
        user_id=user_id,
        selected_model=selected_model,
        provider="ollama",
        temperature=temperature or 0.7,
        max_tokens=max_tokens or 1000,
        stream_responses=stream_responses,
        fallback_model=settings.DEFAULT_LLM_MODEL,  # Always use system default as fallback
        preferredFastModel=preferred_fast,
        preferredBalancedModel=preferred_balanced,
        preferredQualityModel=preferred_quality,
        **kwargs
    )


def get_system_llm_settings(
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    **kwargs
) -> UserLLMSettings:
    """
    Get LLM settings for system tasks (no user context).

    This is a convenience wrapper for system-initiated tasks like:
    - Background jobs
    - Cron tasks
    - Webhooks
    - System maintenance

    Args:
        model: Model to use (defaults to system default)
        temperature: Temperature setting
        max_tokens: Max tokens
        **kwargs: Additional UserLLMSettings parameters

    Returns:
        UserLLMSettings configured for system use
    """
    return UserLLMSettings(
        user_id="system",
        selected_model=model or settings.DEFAULT_LLM_MODEL,
        provider="ollama",
        temperature=temperature,
        max_tokens=max_tokens,
        stream_responses=False,
        **kwargs
    )
