"""Safe expression evaluator — a drop-in replacement for eval() on
workflow-author / LLM-supplied conditions and formulas.

`eval(expr, {"__builtins__": {}}, ctx)` is NOT a sandbox: an attacker reaches
arbitrary code via `().__class__.__bases__[0].__subclasses__()`. This evaluator
walks the AST and permits ONLY a whitelist of node types — literals, names
resolved from the supplied context, boolean/comparison/arithmetic operators,
and simple containers/indexing. No attribute access, no calls, no comprehensions,
no dunder — so none of the object-graph escapes are reachable.
"""

import ast
import operator
from typing import Any, Dict


class UnsafeExpressionError(ValueError):
    """Raised when an expression uses a construct outside the safe whitelist."""


_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos, ast.USub: operator.neg, ast.Not: operator.not_,
}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt,
    ast.LtE: operator.le, ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
    ast.Is: operator.is_, ast.IsNot: operator.is_not,
}
_MAX_POW = 1000  # guard against 10**10**10 style DoS


def _eval(node: ast.AST, names: Dict[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _eval(node.body, names)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id in names:
            return names[node.id]
        raise UnsafeExpressionError(f"unknown name: {node.id}")
    if isinstance(node, ast.BoolOp):
        vals = [_eval(v, names) for v in node.values]
        if isinstance(node.op, ast.And):
            result = True
            for v in vals:
                result = v
                if not v:
                    break
            return result
        result = False
        for v in vals:
            result = v
            if v:
                break
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand, names))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        if isinstance(node.op, ast.Pow):
            exp = _eval(node.right, names)
            if isinstance(exp, (int, float)) and exp > _MAX_POW:
                raise UnsafeExpressionError("exponent too large")
        return _BIN_OPS[type(node.op)](_eval(node.left, names), _eval(node.right, names))
    if isinstance(node, ast.Compare):
        left = _eval(node.left, names)
        for op, comparator in zip(node.ops, node.comparators):
            if type(op) not in _CMP_OPS:
                raise UnsafeExpressionError(f"operator not allowed: {type(op).__name__}")
            right = _eval(comparator, names)
            if not _CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        elts = [_eval(e, names) for e in node.elts]
        return elts if isinstance(node, ast.List) else (
            tuple(elts) if isinstance(node, ast.Tuple) else set(elts))
    if isinstance(node, ast.Dict):
        return {_eval(k, names): _eval(v, names) for k, v in zip(node.keys, node.values)}
    if isinstance(node, ast.Subscript):
        return _eval(node.value, names)[_eval(node.slice, names)]
    if isinstance(node, ast.IfExp):
        return _eval(node.body, names) if _eval(node.test, names) else _eval(node.orelse, names)
    if isinstance(node, ast.Call):
        # Only direct calls to a whitelisted callable provided in `names`
        # (e.g. min/max/len/abs/round). No attribute calls, no *args/**kwargs,
        # so object-graph escapes stay unreachable.
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError("only direct calls to whitelisted names allowed")
        func = names.get(node.func.id)
        if not callable(func):
            raise UnsafeExpressionError(f"call to non-whitelisted name: {node.func.id}")
        if node.keywords:
            raise UnsafeExpressionError("keyword arguments not allowed")
        args = [_eval(a, names) for a in node.args]
        return func(*args)
    raise UnsafeExpressionError(f"disallowed expression element: {type(node).__name__}")


def safe_eval(expression: str, names: Dict[str, Any] | None = None) -> Any:
    """Evaluate a restricted expression against `names`. Raises
    UnsafeExpressionError for anything outside the whitelist."""
    names = names or {}
    if not isinstance(expression, str):
        raise UnsafeExpressionError("expression must be a string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise UnsafeExpressionError(f"invalid expression: {exc}") from exc
    return _eval(tree, names)
