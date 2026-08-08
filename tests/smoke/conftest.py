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
