"""Tests for trial_code_service — parsing + redemption state machine.

Runs against the container database (same pattern as Business tests).
Each test creates uniquely-named users/subscriptions and deletes them in
teardown, so the suite is safe on a shared dev DB.

Run inside the community container:
    python -m pytest tests/unit/test_trial_code_service.py -v --no-cov
"""

import os
import uuid
import pytest
from datetime import datetime, timedelta
from types import SimpleNamespace

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

from models import User
from models.subscription import (
    Subscription,
    SubscriptionPlan,
    SubscriptionStatus,
    BillingPeriod,
)
import services.trial_code_service as tcs
from services.trial_code_service import (
    parse_trial_codes,
    redeem_trial_code,
    TrialCodeError,
)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@postgres:5432/aictrlnet",
)

CODES = "TESTCODE90:90:40:2099-12-31,TINYCAP:30:1,EXPIREDCODE:30:10:2020-01-01"


# ---------------------------------------------------------------------------
# parse_trial_codes — pure function
# ---------------------------------------------------------------------------

class TestParseTrialCodes:
    def test_empty(self):
        assert parse_trial_codes("") == {}
        assert parse_trial_codes("   ") == {}

    def test_basic(self):
        specs = parse_trial_codes("INSTITUTE90:90:40")
        assert specs["INSTITUTE90"].days == 90
        assert specs["INSTITUTE90"].max_redemptions == 40
        assert specs["INSTITUTE90"].expires is None

    def test_with_expiry_and_multiple(self):
        specs = parse_trial_codes("A:90:40:2026-10-31, b:30:100")
        assert specs["A"].expires.isoformat() == "2026-10-31"
        assert "B" in specs  # lowercased input normalized to uppercase

    def test_malformed_entries_skipped(self):
        specs = parse_trial_codes("GOOD:30:5,BAD:xx:5,:90:5,NEG:-1:5,ALSOGOOD:10:1")
        assert set(specs) == {"GOOD", "ALSOGOOD"}


# ---------------------------------------------------------------------------
# redeem_trial_code — state machine (container DB)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    eng = create_async_engine(DATABASE_URL, echo=False, poolclass=NullPool)
    yield eng


@pytest.fixture
async def db(engine):
    maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with maker() as session:
        yield session


@pytest.fixture(autouse=True)
def trial_codes_config(monkeypatch):
    monkeypatch.setattr(
        tcs, "get_settings", lambda: SimpleNamespace(TRIAL_CODES=CODES)
    )


@pytest.fixture
async def starter_plan(db):
    result = await db.execute(
        select(SubscriptionPlan).where(SubscriptionPlan.name == "business_starter")
    )
    plan = result.scalar_one_or_none()
    created = False
    if plan is None:
        plan = SubscriptionPlan(
            id=str(uuid.uuid4()),
            name="business_starter",
            edition="business",
            display_name="Business Starter (test)",
            price_monthly=599.0,
        )
        db.add(plan)
        await db.commit()
        created = True
    yield plan
    if created:
        await db.execute(delete(SubscriptionPlan).where(SubscriptionPlan.id == plan.id))
        await db.commit()


def _mk_user(**kw):
    uid = str(uuid.uuid4())
    return User(
        id=uid,
        email=f"trialcode-{uid}@test.local",
        username=f"trialcode-{uid[:8]}",
        hashed_password="x",
        edition=kw.get("edition", "business"),
        is_active=True,
    )


def _mk_sub(user, plan, **kw):
    now = datetime.utcnow()
    return Subscription(
        id=str(uuid.uuid4()),
        user_id=user.id,
        tenant_id="default-tenant",
        plan_id=plan.id,
        status=kw.get("status", SubscriptionStatus.TRIALING),
        billing_period=BillingPeriod.MONTHLY,
        started_at=now,
        current_period_start=now,
        current_period_end=kw.get("trial_end", now + timedelta(days=14)),
        trial_end=kw.get("trial_end", now + timedelta(days=14)),
        stripe_subscription_id=kw.get("stripe_subscription_id"),
        meta_data=kw.get("meta_data", {}),
    )


@pytest.fixture
async def cleanup(db):
    created = {"users": [], "subs": []}
    yield created
    for sid in created["subs"]:
        await db.execute(delete(Subscription).where(Subscription.id == sid))
    for uid in created["users"]:
        await db.execute(delete(User).where(User.id == uid))
    await db.commit()


async def _seed(db, cleanup, plan, **sub_kw):
    user = _mk_user(**sub_kw.pop("user_kw", {}))
    db.add(user)
    sub = _mk_sub(user, plan, **sub_kw) if sub_kw.pop("with_sub", True) else None
    if sub is not None:
        db.add(sub)
    await db.commit()
    cleanup["users"].append(user.id)
    if sub is not None:
        cleanup["subs"].append(sub.id)
    return user, sub


class TestRedeemStateMachine:
    async def test_invalid_code(self, db, cleanup, starter_plan):
        user, _ = await _seed(db, cleanup, starter_plan)
        with pytest.raises(TrialCodeError) as exc:
            await redeem_trial_code(db, user.id, "NOPE")
        assert exc.value.status_code == 404

    async def test_expired_code(self, db, cleanup, starter_plan):
        user, _ = await _seed(db, cleanup, starter_plan)
        with pytest.raises(TrialCodeError) as exc:
            await redeem_trial_code(db, user.id, "EXPIREDCODE")
        assert exc.value.status_code == 404

    async def test_happy_path_extends_and_stamps(self, db, cleanup, starter_plan, engine):
        user, sub = await _seed(db, cleanup, starter_plan)
        result = await redeem_trial_code(db, user.id, "testcode90")  # case-insensitive
        assert result["code"] == "TESTCODE90"
        assert 88 <= result["days_remaining"] <= 90

        # Round-trip through a FRESH session: meta_data stamp must persist
        maker = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        async with maker() as fresh:
            row = (await fresh.execute(
                select(Subscription).where(Subscription.id == sub.id)
            )).scalar_one()
            assert row.meta_data.get("trial_code") == "TESTCODE90"
            assert row.trial_end > datetime.utcnow() + timedelta(days=85)
            assert row.current_period_end == row.trial_end
            assert row.status == SubscriptionStatus.TRIALING

    async def test_never_shortens_longer_trial(self, db, cleanup, starter_plan):
        far = datetime.utcnow() + timedelta(days=200)
        user, sub = await _seed(db, cleanup, starter_plan, trial_end=far)
        await redeem_trial_code(db, user.id, "TESTCODE90")
        await db.refresh(sub)
        assert sub.trial_end == far

    async def test_already_redeemed(self, db, cleanup, starter_plan):
        user, _ = await _seed(db, cleanup, starter_plan)
        await redeem_trial_code(db, user.id, "TESTCODE90")
        with pytest.raises(TrialCodeError) as exc:
            await redeem_trial_code(db, user.id, "TESTCODE90")
        assert exc.value.status_code == 400
        assert exc.value.code == "already_redeemed"

    async def test_stripe_managed_rejected(self, db, cleanup, starter_plan):
        user, _ = await _seed(db, cleanup, starter_plan, stripe_subscription_id="sub_x1")
        with pytest.raises(TrialCodeError) as exc:
            await redeem_trial_code(db, user.id, "TESTCODE90")
        assert exc.value.status_code == 409

    async def test_active_paid_rejected(self, db, cleanup, starter_plan):
        user, _ = await _seed(db, cleanup, starter_plan, status=SubscriptionStatus.ACTIVE)
        with pytest.raises(TrialCodeError) as exc:
            await redeem_trial_code(db, user.id, "TESTCODE90")
        assert exc.value.status_code == 400
        assert exc.value.code == "not_eligible"

    async def test_date_expired_trial_extends_and_restores_edition(self, db, cleanup, starter_plan):
        past = datetime.utcnow() - timedelta(days=3)
        user, sub = await _seed(
            db, cleanup, starter_plan,
            trial_end=past, user_kw={"edition": "trial_expired"},
        )
        result = await redeem_trial_code(db, user.id, "TESTCODE90")
        await db.refresh(sub)
        await db.refresh(user)
        assert sub.trial_end > datetime.utcnow() + timedelta(days=85)
        assert user.edition == "business"
        assert result["reactivated"] is False  # status was still TRIALING

    async def test_expired_status_reactivates(self, db, cleanup, starter_plan):
        past = datetime.utcnow() - timedelta(days=30)
        user, sub = await _seed(
            db, cleanup, starter_plan,
            status=SubscriptionStatus.EXPIRED, trial_end=past,
            user_kw={"edition": "trial_expired"},
        )
        result = await redeem_trial_code(db, user.id, "TESTCODE90")
        await db.refresh(sub)
        assert result["reactivated"] is True
        assert sub.status == SubscriptionStatus.TRIALING

    async def test_no_subscription_row_creates_one(self, db, cleanup, starter_plan):
        user, _ = await _seed(db, cleanup, starter_plan, with_sub=False)
        result = await redeem_trial_code(db, user.id, "TESTCODE90")
        row = (await db.execute(
            select(Subscription).where(Subscription.user_id == user.id)
        )).scalar_one()
        cleanup["subs"].append(row.id)
        assert row.status == SubscriptionStatus.TRIALING
        assert result["days"] == 90

    async def test_cap_exhausted(self, db, cleanup, starter_plan):
        user1, _ = await _seed(db, cleanup, starter_plan)
        user2, _ = await _seed(db, cleanup, starter_plan)
        await redeem_trial_code(db, user1.id, "TINYCAP")
        with pytest.raises(TrialCodeError) as exc:
            await redeem_trial_code(db, user2.id, "TINYCAP")
        assert exc.value.status_code == 410
