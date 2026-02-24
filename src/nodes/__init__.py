"""Nodes module.

Supports edition extensions via the NODES_EXTRA_PATHS environment variable.
Higher editions can set this to a colon-separated list of directories that
will be added to this package's search path, enabling plugin discovery
for additional node implementations.
"""
import os as _os

_extra_paths = _os.environ.get('NODES_EXTRA_PATHS', '')
for _p in _extra_paths.split(':'):
    _p = _p.strip()
    if _p and _os.path.isdir(_p) and _p not in __path__:
        __path__.append(_p)
