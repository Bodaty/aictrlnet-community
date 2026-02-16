"""Prompt Template Loader — loads .md prompt files with caching and variable substitution.

Provides a file-based prompt template system for the Community edition.
Templates are loaded at startup and cached with file-modification-time
invalidation for development convenience.
"""

import os
import re
import logging
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Only these variable names are allowed in {{variable}} substitutions.
# Arbitrary user input never reaches template substitution.
_ALLOWED_VARIABLES = frozenset({
    "edition",
    "tool_names",
    "agent_name",
    "user_name",
    "organization_name",
    "industry",
})

# Resolve prompts directory relative to this file.
_DEFAULT_PROMPTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "prompts",
)


class PromptTemplateLoader:
    """Loads .md prompt template files with in-memory caching.

    Usage::

        loader = PromptTemplateLoader()
        identity = loader.get_section("identity", {"edition": "business"})
        industries = loader.get_industry_sections()
    """

    def __init__(self, prompts_dir: Optional[str] = None):
        self._prompts_dir = prompts_dir or _DEFAULT_PROMPTS_DIR
        # Cache: filename (without .md) -> (mtime, content)
        self._cache: Dict[str, tuple] = {}
        self._load_all()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_section(self, name: str, variables: Optional[Dict[str, str]] = None) -> str:
        """Return the content of a prompt template section.

        *name* is the template key — e.g. ``"identity"`` loads ``identity.md``,
        ``"industries/legal"`` loads ``industries/legal.md``.

        *variables* is an optional dict of ``{{key}}`` substitutions.
        Only keys in ``_ALLOWED_VARIABLES`` are replaced; unknown keys are
        left as-is (not an error, but logged at debug level).

        Falls back to an empty string if the file is missing.
        """
        content = self._get_cached(name)
        if content is None:
            logger.debug("Prompt template '%s' not found, returning empty", name)
            return ""
        if variables:
            content = self._substitute(content, variables)
        return content

    def get_industry_sections(self) -> List[str]:
        """Return a list of available industry template keys.

        Scans the ``industries/`` sub-directory and returns keys like
        ``["legal", "healthcare", "finance", "retail"]``.
        """
        industries_dir = os.path.join(self._prompts_dir, "industries")
        if not os.path.isdir(industries_dir):
            return []
        keys = []
        for fname in sorted(os.listdir(industries_dir)):
            if fname.endswith(".md"):
                keys.append(fname[:-3])
        return keys

    def reload(self) -> None:
        """Force-reload all templates from disk (useful in tests)."""
        self._cache.clear()
        self._load_all()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_all(self) -> None:
        """Walk the prompts directory and cache every .md file."""
        if not os.path.isdir(self._prompts_dir):
            logger.warning("Prompts directory does not exist: %s", self._prompts_dir)
            return

        for root, _dirs, files in os.walk(self._prompts_dir):
            for fname in files:
                if not fname.endswith(".md"):
                    continue
                full_path = os.path.join(root, fname)
                rel = os.path.relpath(full_path, self._prompts_dir)
                # Key = relative path without .md extension
                key = rel[:-3]  # strip ".md"
                # Normalise path separators
                key = key.replace(os.sep, "/")
                try:
                    mtime = os.path.getmtime(full_path)
                    with open(full_path, "r", encoding="utf-8") as fh:
                        content = fh.read().strip()
                    self._cache[key] = (mtime, content)
                except OSError as exc:
                    logger.warning("Could not load prompt template %s: %s", full_path, exc)

        logger.info(
            "PromptTemplateLoader: loaded %d templates from %s",
            len(self._cache),
            self._prompts_dir,
        )

    def _get_cached(self, name: str) -> Optional[str]:
        """Return cached content for *name*, reloading if the file changed."""
        entry = self._cache.get(name)
        full_path = os.path.join(self._prompts_dir, f"{name}.md")

        if not os.path.isfile(full_path):
            # File removed or never existed
            return entry[1] if entry else None

        current_mtime = os.path.getmtime(full_path)

        if entry is None or entry[0] != current_mtime:
            # File is new or changed — reload
            try:
                with open(full_path, "r", encoding="utf-8") as fh:
                    content = fh.read().strip()
                self._cache[name] = (current_mtime, content)
                return content
            except OSError:
                return entry[1] if entry else None

        return entry[1]

    @staticmethod
    def _substitute(content: str, variables: Dict[str, str]) -> str:
        """Replace ``{{key}}`` placeholders with values from *variables*.

        Only whitelisted variable names are substituted. Unknown variable
        names in the template are left as-is.
        """
        def _replacer(match):
            key = match.group(1).strip()
            if key in _ALLOWED_VARIABLES and key in variables:
                return str(variables[key])
            return match.group(0)  # leave as-is

        return re.sub(r"\{\{(\s*\w+\s*)\}\}", _replacer, content)
