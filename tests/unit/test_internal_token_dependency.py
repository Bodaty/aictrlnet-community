"""`require_internal_token` fails closed: unset = 503, missing or wrong = 401."""
import pytest


async def test_internal_token_fails_closed(monkeypatch):
    from fastapi import HTTPException

    from core.internal_auth import require_internal_token

    monkeypatch.delenv("INTERNAL_SERVICE_TOKEN", raising=False)
    with pytest.raises(HTTPException) as unset:
        await require_internal_token("anything")
    assert unset.value.status_code == 503

    monkeypatch.setenv("INTERNAL_SERVICE_TOKEN", "s3cret")
    for wrong in (None, "", "nope"):
        with pytest.raises(HTTPException) as bad:
            await require_internal_token(wrong)
        assert bad.value.status_code == 401
    assert await require_internal_token("s3cret") is None
