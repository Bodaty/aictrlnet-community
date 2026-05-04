"""Regression tests for the NLP edit_mode response contract.

The bug being prevented: ``_build_transparency_response`` is the shared
helper used by AI-complete, template-enhance, template-match, and
fallback paths in ``NLPService.process_natural_language``. Before
``283c551`` it never stamped ``edit_mode`` on the dict it returned,
even though three other paths in the same function set
``"edit_mode": True`` directly. When the LLM happened to land in a
helper-routed path during edit-mode requests, the response was missing
the flag entirely — gate-25b-3 caught it intermittently.

This test pins the caller-level invariant: every call to
``_build_transparency_response`` inside ``process_natural_language``
must pass ``is_edit_mode=`` so the helper can stamp the right value.
A future refactor that adds a new path without threading the flag
will fail this check.

The test parses ``services/nlp.py`` directly via ``ast`` rather than
importing the module — keeps it runnable without sqlalchemy/asyncpg
deps, and means the structural assertion catches drift even before
runtime. The runtime assertion (helper actually stamps edit_mode) is
covered by gate-25b-3 in the E2E suite.
"""

from __future__ import annotations

import ast
from pathlib import Path


def _nlp_source_path() -> Path:
    """Locate Community's NLP service source from this test file."""
    # editions/community/tests/unit/<this file>
    # → editions/community/src/services/nlp.py
    return (
        Path(__file__).resolve().parents[2]
        / "src"
        / "services"
        / "nlp.py"
    )


def _find_function(tree: ast.AST, class_name: str, func_name: str) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    """Find ``ClassName.func_name`` in the AST."""
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef) or cls.name != class_name:
            continue
        for item in cls.body:
            if (
                isinstance(item, (ast.AsyncFunctionDef, ast.FunctionDef))
                and item.name == func_name
            ):
                return item
    return None


def test_every_helper_call_in_process_natural_language_threads_is_edit_mode():
    """Every call to ``_build_transparency_response`` inside
    ``NLPService.process_natural_language`` must pass ``is_edit_mode=``.

    The fix in ``283c551`` updated all 5 existing call sites; this test
    pins the convention so a 6th can't slip in without deliberate intent.
    """
    src_path = _nlp_source_path()
    assert src_path.exists(), (
        f"Could not find NLP service source at {src_path}. Has the file "
        f"been moved? Update _nlp_source_path() to match."
    )

    tree = ast.parse(src_path.read_text())

    func = _find_function(tree, "NLPService", "process_natural_language")
    assert func is not None, (
        "NLPService.process_natural_language no longer exists — has the "
        "method been renamed? Update this test if the contract moved."
    )

    helper_calls: list[ast.Call] = []
    for node in ast.walk(func):
        if not isinstance(node, ast.Call):
            continue
        attr = node.func
        # Match self._build_transparency_response(...)
        if not isinstance(attr, ast.Attribute):
            continue
        if attr.attr != "_build_transparency_response":
            continue
        helper_calls.append(node)

    assert helper_calls, (
        "process_natural_language no longer calls "
        "_build_transparency_response — has the helper been renamed? "
        "Update this test if the contract moved."
    )

    missing = []
    for call in helper_calls:
        kwarg_names = {kw.arg for kw in call.keywords}
        if "is_edit_mode" not in kwarg_names:
            missing.append(call.lineno)

    assert not missing, (
        f"_build_transparency_response calls in process_natural_language "
        f"missing is_edit_mode= at lines {missing}. Every call must "
        f"thread the flag so the helper can stamp response['edit_mode'] "
        f"consistently. See commit 283c551 / gate-25b-3."
    )


def test_helper_signature_accepts_is_edit_mode():
    """``_build_transparency_response`` must declare ``is_edit_mode`` as a
    keyword argument so callers can pass it. Pinning this prevents an
    accidental signature revert that would silently make every call site
    error at runtime instead of at import.
    """
    src_path = _nlp_source_path()
    tree = ast.parse(src_path.read_text())
    func = _find_function(tree, "NLPService", "_build_transparency_response")
    assert func is not None, (
        "_build_transparency_response no longer exists in NLPService."
    )

    arg_names = {a.arg for a in func.args.args} | {a.arg for a in func.args.kwonlyargs}
    assert "is_edit_mode" in arg_names, (
        "_build_transparency_response signature dropped is_edit_mode. "
        "The helper must accept this kwarg so process_natural_language "
        "callers can flag edit-mode requests; otherwise the response "
        "shape silently diverges from the contract gate-25b-3 enforces."
    )
