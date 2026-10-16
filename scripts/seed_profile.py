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
