"""get_user_llm_settings must not masquerade the system default as a user choice."""
from unittest.mock import AsyncMock, MagicMock

from services.llm_helpers import get_user_llm_settings


def _db_returning(user):
    result = MagicMock()
    result.scalar_one_or_none.return_value = user
    db = MagicMock()
    db.execute = AsyncMock(return_value=result)
    return db


async def test_no_user_pref_leaves_selected_model_none():
    user = MagicMock()
    user.preferences = {}
    settings = await get_user_llm_settings(db=_db_returning(user), user_id="u1")
    assert settings.selected_model is None
    assert settings.fallback_model is None


async def test_real_user_pref_kept():
    user = MagicMock()
    user.preferences = {"aiModel": "mistral:7b"}
    settings = await get_user_llm_settings(db=_db_returning(user), user_id="u1")
    assert settings.selected_model == "mistral:7b"


async def test_explicit_override_still_wins():
    settings = await get_user_llm_settings(db=_db_returning(None), user_id="u1", model_override="gpt-4o")
    assert settings.selected_model == "gpt-4o"
