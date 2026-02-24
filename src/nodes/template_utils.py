"""Template resolution utility for workflow node parameters.

Resolves {{dotted.path}} patterns from a context dict and {{env.VAR}} from
os.environ.  Works recursively on strings, dicts, and lists so it can be
applied to entire parameter trees in one call.
"""

import os
import re
from typing import Any, Dict


def resolve_templates(value: Any, context: Dict[str, Any]) -> Any:
    """Resolve ``{{path.to.var}}`` placeholders in *value*.

    Supported patterns:
    - ``{{env.VAR}}``        – looked up in ``os.environ``
    - ``{{dotted.path}}``    – walked through *context* dict/list
    - ``{{simple_key}}``     – top-level key in *context*

    Unresolved placeholders are left as-is so downstream code can still
    detect missing data.
    """
    if isinstance(value, str):
        def _replacer(match: re.Match) -> str:
            path = match.group(1)

            # Environment variable lookup
            if path.startswith("env."):
                return os.environ.get(path[4:], match.group(0))

            # Walk the dotted path through context
            parts = path.split(".")
            current: Any = context
            for part in parts:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, (list, tuple)) and part.isdigit():
                    idx = int(part)
                    current = current[idx] if idx < len(current) else None
                else:
                    return match.group(0)  # unresolvable
                if current is None:
                    return match.group(0)  # unresolvable
            return str(current)

        return re.sub(r"\{\{(\w+(?:\.\w+)*)\}\}", _replacer, value)

    if isinstance(value, dict):
        return {k: resolve_templates(v, context) for k, v in value.items()}

    if isinstance(value, list):
        return [resolve_templates(item, context) for item in value]

    return value
