"""Community-edition rootdir: block live third-party egress.

pytest loads conftest.py from its rootdir down to each test file, so the guard
has to sit at every point a run could be rooted — including inside the edition
containers, which mount only editions/*/tests and tests/smoke_common.
tests/integration/regressions/test_egress_guard_is_default_on.py fails if any
directory holding tests stops being covered.
"""

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
