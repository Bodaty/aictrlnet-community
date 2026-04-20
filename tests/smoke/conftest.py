"""Smoke test configuration for Community Edition."""

import os
import sys

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("EDITION", "community")
os.environ.setdefault("AICTRLNET_EDITION", "community")

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
