"""Smoke test configuration for Community Edition."""

import os
import sys

os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("EDITION", "community")
os.environ.setdefault("AICTRLNET_EDITION", "community")

# smoke_common is mounted at /app/tests/smoke_common
# This conftest is at /app/tests/smoke/ — parent dir /app/tests/ is what we need
_tests_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _tests_dir not in sys.path:
    sys.path.insert(0, _tests_dir)
