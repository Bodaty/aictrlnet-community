"""What the seed scripts are allowed to write, and how hard they fail.

Read straight from the environment, normalised the same way core.config's
Settings validator does, because the seed scripts run before (and outside) the
application and several of them never build a Settings object.

- AICTRLNET_SEED_PROFILE: 'demo' (default) seeds dev/test users and
  demonstration rows; 'minimal' seeds only what the product needs to run.
- SEED_STRICT=1: a failed product-required step makes the seed exit non-zero,
  so a caller that records "seeded" (a first-start script) does not. Unset
  keeps the tolerant behaviour `make fresh` and `make seed` rely on.
"""

import os
import sys

SEED_PROFILES = ("demo", "minimal")


def _normalised(name):
    value = (os.environ.get(name) or "").strip().lower()
    return value or None


def seed_profile():
    profile = _normalised("AICTRLNET_SEED_PROFILE") or "demo"
    if profile not in SEED_PROFILES:
        raise SystemExit(
            f"AICTRLNET_SEED_PROFILE={profile!r} is not one of {SEED_PROFILES}"
        )
    return profile


def is_minimal():
    return seed_profile() == "minimal"


def is_strict():
    return (os.environ.get("SEED_STRICT") or "").strip() == "1"


class SeedFailures:
    """Collects failures of product-required steps.

    Outside strict mode a failure is logged and seeding carries on, exactly as
    before. Under strict mode the script finishes the remaining steps, then
    exits 1 so the caller knows the database is not fully seeded.
    """

    def __init__(self, seed_name):
        self.seed_name = seed_name
        self.failures = []

    def record(self, step, error):
        self.failures.append(f"{step}: {error}")
        print(f"  ❌ [{self.seed_name}] required step failed — {step}: {error}")

    def exit_if_strict(self):
        if self.failures and is_strict():
            print(
                f"\n❌ [{self.seed_name}] {len(self.failures)} required step(s) failed "
                f"under SEED_STRICT=1:"
            )
            for failure in self.failures:
                print(f"   - {failure}")
            sys.exit(1)


# ENVIRONMENT values under which accounts with published passwords may be
# seeded. Read straight from the environment like the rest of this module; an
# unset value counts as production, the same fail-safe core.config applies.
_PRODUCTION_ENVIRONMENTS = ("production", "prod")
# Hosted test environments get them only on an explicit opt-in
# (SEED_ALLOW_KNOWN_PASSWORDS=1; `make seed-staging` sets it).
_STAGING_ENVIRONMENTS = ("staging", "stage")


def known_password_accounts_allowed():
    """False wherever dev@, admin@ and test@ must not be created.

    Production (or unset): never. Staging: only with SEED_ALLOW_KNOWN_PASSWORDS=1.

    Their passwords are in the repo (README, seeds, test fixtures), so on a
    deployment those logins are a published admin credential. On 25 Nov 2026
    dev@aictrlnet.com signed in to production with testpass123: the hosted seed
    job ran with ENVIRONMENT=development and nothing here looked at it.
    """
    environment = _normalised("ENVIRONMENT")
    if environment is None or environment in _PRODUCTION_ENVIRONMENTS:
        return False
    if environment in _STAGING_ENVIRONMENTS:
        return (os.environ.get("SEED_ALLOW_KNOWN_PASSWORDS") or "").strip() == "1"
    return True
