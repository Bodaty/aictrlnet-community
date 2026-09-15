"""Locate tests/smoke_common inside the container and re-export the guard's
exception. business/enterprise mount it at /workspace/tests/smoke_common,
community at /app/tests/smoke_common."""
import sys
from pathlib import Path
# In-container mounts first, then the repo checkout (editions/<ed>/tests/unit -> <repo>/tests) for bare CI runners.
_repo_tests = Path(__file__).resolve().parent.parent.parent.parent.parent / "tests"  # .parent of / is /, so shallow container paths just miss
for _cand in (Path("/workspace/tests"), Path("/app/tests"), _repo_tests):
    if (_cand / "smoke_common" / "egress_guard.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
from smoke_common.egress_guard import SmokeEgressBlocked  # noqa: E402
from smoke_common.runner import synthesize_value  # noqa: E402
__all__ = ["SmokeEgressBlocked", "synthesize_value"]
