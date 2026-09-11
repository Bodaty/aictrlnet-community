"""Locate tests/smoke_common inside the container and re-export the guard's
exception. business/enterprise mount it at /workspace/tests/smoke_common,
community at /app/tests/smoke_common."""
import sys
from pathlib import Path
for _cand in (Path("/workspace/tests"), Path("/app/tests")):
    if (_cand / "smoke_common" / "egress_guard.py").exists() and str(_cand) not in sys.path:
        sys.path.insert(0, str(_cand))
from smoke_common.egress_guard import SmokeEgressBlocked  # noqa: E402
from smoke_common.runner import synthesize_value  # noqa: E402
__all__ = ["SmokeEgressBlocked", "synthesize_value"]
