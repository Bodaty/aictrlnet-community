"""POST /upgrade/trial: server-set length, Business tiers only, one per tenant.

Before this fix the endpoint took trial_days from the client with no cap and
accepted target_edition=enterprise; core/enforcement.py honors an unexpired
FeatureTrial as that edition's features, so a signed-in user could grant
themselves years of Enterprise. Trials were also counted per edition, so
Starter, Growth and Scale could be trialled back to back.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from api.v1.endpoints import upgrade


def _db(existing=None):
    res = MagicMock()
    res.scalar_one_or_none.return_value = existing
    db = MagicMock(execute=AsyncMock(return_value=res), commit=AsyncMock())
    added = []
    db.add = MagicMock(side_effect=added.append)

    async def _refresh(obj):
        obj.id = "trial-1"
        obj.started_at = datetime.utcnow()

    db.refresh = AsyncMock(side_effect=_refresh)
    return db, added


@pytest.mark.asyncio
async def test_feature_trial_refuses_enterprise():
    db, added = _db()
    request = SimpleNamespace(target_edition="enterprise", trial_days=14)
    with pytest.raises(HTTPException) as exc:
        await upgrade.start_feature_trial(request=request, current_user={"tenant_id": "t1"}, db=db)
    assert exc.value.status_code == 400
    assert added == []


@pytest.mark.asyncio
async def test_feature_trial_length_is_set_by_the_server():
    db, added = _db()
    request = SimpleNamespace(target_edition="business_scale", trial_days=3650)
    enforcer = MagicMock(EDITION_FEATURES={}, cache=MagicMock(delete=AsyncMock()))
    with patch.object(upgrade, "get_settings", return_value=SimpleNamespace(TRIAL_DAYS=14)), \
         patch.object(upgrade, "LicenseEnforcer", return_value=enforcer):
        resp = await upgrade.start_feature_trial(request=request, current_user={"tenant_id": "t1"}, db=db)

    trial = next(obj for obj in added if hasattr(obj, "expires_at"))
    assert trial.expires_at <= datetime.utcnow() + timedelta(days=14, minutes=1)
    assert resp.days_remaining <= 14


@pytest.mark.asyncio
async def test_feature_trial_cannot_chain_business_tiers():
    used = SimpleNamespace(
        id="trial-0",
        edition_required="business_starter",
        started_at=datetime.utcnow() - timedelta(days=30),
        expires_at=datetime.utcnow() - timedelta(days=16),
    )
    db, added = _db(existing=used)
    request = SimpleNamespace(target_edition="business_growth", trial_days=14)
    with pytest.raises(HTTPException) as exc:
        await upgrade.start_feature_trial(request=request, current_user={"tenant_id": "t1"}, db=db)
    assert exc.value.status_code == 400
    assert added == []
