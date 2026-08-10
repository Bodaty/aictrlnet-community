"""Smoke test configuration for Community Edition."""

import os
import sys

# Hard-set, not setdefault: containers preset ENVIRONMENT=development, which
# silently disabled the enforcement middleware's in-process-test guard
# (middleware/enforcement.py:104) — its real-session UsageTracker singleton
# then poisoned later usage-endpoint tests across per-test event loops.
os.environ["ENVIRONMENT"] = "test"
os.environ.setdefault("EDITION", "community")
os.environ.setdefault("AICTRLNET_EDITION", "community")

# Redirect every filesystem-writing path at a throwaway dir BEFORE the app is
# imported. The schema-derived body probe actually executes create handlers, and
# a mocked DB does not stop them writing real files — control-plane components,
# staged files, state entries and personal workflow templates all landed in the
# container (and the repo) and then broke unrelated tests on the next run.
import tempfile as _tempfile

_smoke_data = _tempfile.mkdtemp(prefix="smoke-data-")
os.environ["DATA_PATH"] = _smoke_data
os.environ["STAGED_FILES_DIR"] = os.path.join(_smoke_data, "staged_files")
os.environ["AICTRLNET_TEMPLATE_DIR"] = os.path.join(_smoke_data, "workflow-templates")
os.environ["UPLOAD_DIR"] = os.path.join(_smoke_data, "uploads")

# smoke_common resolution, most-specific first:
# 1. Edition-local tests dir (Community ships its own tests/smoke_common/)
# 2. Repo-root tests/ (where Business/Enterprise pull it from)
# 3. Docker-compose container mount at /workspace/tests
_this_dir = os.path.dirname(os.path.abspath(__file__))           # editions/community/tests/smoke/
_tests_dir = os.path.dirname(_this_dir)                           # editions/community/tests/
_repo_tests = os.path.abspath(os.path.join(_tests_dir, "..", "..", "..", "tests"))

if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
if _repo_tests not in sys.path and os.path.isdir(_repo_tests):
    sys.path.insert(0, _repo_tests)
if "/workspace/tests" not in sys.path and os.path.isdir("/workspace/tests"):
    sys.path.insert(0, "/workspace/tests")

# Block third-party network calls BEFORE the app is imported, for the same
# reason the filesystem redirects above exist: the schema-derived body probe
# executes handlers for real. On 2026-08-07 that sent 28 live SendGrid emails to
# the sales inbox from a container holding a production key. Credentials must be
# cleared pre-import because modules latch them at module scope (contact.py
# computes SENDGRID_ENABLED on import, so clearing later has no effect).
# --- BEGIN outbound-egress guard (identical in every conftest — see tests/smoke_common/egress_guard.py) ---
# Runs before any application import: contact.py computes SENDGRID_ENABLED at
# module scope, so a credential scrub that happens later changes nothing. The
# walk locates tests/smoke_common from any depth, including inside the edition
# containers, where only parts of the repo are mounted.
import os as _egress_os
import sys as _egress_sys

_egress_root = _egress_os.path.dirname(_egress_os.path.abspath(__file__))
while not _egress_os.path.isfile(
    _egress_os.path.join(_egress_root, "tests", "smoke_common", "egress_guard.py")
):
    _egress_parent = _egress_os.path.dirname(_egress_root)
    if _egress_parent == _egress_root:
        raise RuntimeError(
            "tests/smoke_common/egress_guard.py not found above "
            f"{_egress_os.path.abspath(__file__)}. The guard that stops test runs "
            "from reaching live third parties cannot be loaded, so this refuses "
            "to collect rather than running unprotected."
        )
    _egress_root = _egress_parent

if _egress_os.path.join(_egress_root, "tests") not in _egress_sys.path:
    _egress_sys.path.insert(0, _egress_os.path.join(_egress_root, "tests"))

from smoke_common.egress_guard import install_test_egress_guard  # noqa: E402

install_test_egress_guard()
# --- END outbound-egress guard ---
