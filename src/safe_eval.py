"""Safe expression evaluator using Python's AST module.

Replaces raw ``eval()`` with an AST-walking evaluator that only permits:
- Arithmetic operators: +, -, *, /, //, **, %, unary +/-
- Comparison operators (for piecewise): <, <=, >, >=, ==, !=
- Boolean operators: and, or, not
- Numeric literals (int, float)
- Whitelisted function calls (sin, cos, sinc, exp, tanh, etc.)
- Whitelisted variable names (t, u, s, n, g, ph, rnd, beat, pi, e)

This is critical for security: a malicious user could type
``__import__('os').system('rm -rf /')`` into an equation field.
The AST walker rejects any node type not on the whitelist.
"""

from __future__ import annotations
import ast
import operator
from typing import Any, Callable, Dict, Optional
import numpy as np


# ---------------------------------------------------------------------------
# Whitelisted functions available in equations
# ---------------------------------------------------------------------------
SAFE_FUNCTIONS: Dict[str, Callable] = {
    # Trigonometric
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "arctan": np.arctan,
    "cot": lambda x: np.cos(x) / (np.sin(x) + 1e-12),
    "atan2": np.arctan2,
    # Hyperbolic
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    # Exponential / logarithmic
    "exp": np.exp,
    "log": np.log,
    "log2": np.log2,
    "log10": np.log10,
    "sqrt": np.sqrt,
    # Special
    "sinc": np.sinc,  # sin(pi*x) / (pi*x), the normalized sinc
    "abs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "round": np.round,
    "clip": np.clip,
    # Min / max (element-wise)
    "minimum": np.minimum,
    "maximum": np.maximum,
    "min": np.minimum,
    "max": np.maximum,
    # Modulo helper
    "mod": np.mod,
    # Smooth gate (logistic sigmoid)
    "gate": lambda x: 1.0 / (1.0 + np.exp(-8.0 * x)),
    "sigmoid": lambda x: 1.0 / (1.0 + np.exp(-x)),
    # Power
    "pow": np.power,
}

# Whitelisted constants
SAFE_CONSTANTS: Dict[str, float] = {
    "pi": np.pi,
    "e": np.e,
    "tau": 2.0 * np.pi,
}


class SafeEvalError(Exception):
    """Raised when the expression contains disallowed operations."""
    pass


# ---------------------------------------------------------------------------
# Binary / unary operator maps
# ---------------------------------------------------------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CMP_OPS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}

_BOOL_OPS = {
    ast.And: lambda a, b: np.logical_and(a, b),
    ast.Or: lambda a, b: np.logical_or(a, b),
}


class _ASTEvaluator(ast.NodeVisitor):
    """Walks an AST tree and evaluates only safe nodes."""

    def __init__(self, variables: Dict[str, Any]):
        self.variables = variables

    def visit_Expression(self, node: ast.Expression) -> Any:
        return self.visit(node.body)

    def visit_Constant(self, node: ast.Constant) -> Any:
        if isinstance(node.value, (int, float)):
            return node.value
        raise SafeEvalError(f"Disallowed constant type: {type(node.value).__name__}")

    # Python < 3.8 compat
    def visit_Num(self, node: ast.Num) -> Any:
        return node.n

    def visit_Name(self, node: ast.Name) -> Any:
        name = node.id
        if name in self.variables:
            return self.variables[name]
        if name in SAFE_CONSTANTS:
            return SAFE_CONSTANTS[name]
        if name in SAFE_FUNCTIONS:
            return SAFE_FUNCTIONS[name]
        raise SafeEvalError(f"Disallowed variable name: '{name}'")

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        op_type = type(node.op)
        if op_type not in _BIN_OPS:
            raise SafeEvalError(f"Disallowed binary operator: {op_type.__name__}")
        left = self.visit(node.left)
        right = self.visit(node.right)
        return _BIN_OPS[op_type](left, right)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        op_type = type(node.op)
        if op_type == ast.Not:
            return np.logical_not(self.visit(node.operand))
        if op_type not in _UNARY_OPS:
            raise SafeEvalError(f"Disallowed unary operator: {op_type.__name__}")
        return _UNARY_OPS[op_type](self.visit(node.operand))

    def visit_Call(self, node: ast.Call) -> Any:
        func = self.visit(node.func)
        if not callable(func):
            raise SafeEvalError(f"'{func}' is not callable")
        # Verify it's one of our whitelisted functions
        if func not in SAFE_FUNCTIONS.values():
            raise SafeEvalError("Disallowed function call")
        args = [self.visit(arg) for arg in node.args]
        kwargs = {kw.arg: self.visit(kw.value) for kw in node.keywords}
        return func(*args, **kwargs)

    def visit_Compare(self, node: ast.Compare) -> Any:
        left = self.visit(node.left)
        result = None
        for op, comparator in zip(node.ops, node.comparators):
            op_type = type(op)
            if op_type not in _CMP_OPS:
                raise SafeEvalError(f"Disallowed comparison operator: {op_type.__name__}")
            right = self.visit(comparator)
            cmp_result = _CMP_OPS[op_type](left, right)
            if result is None:
                result = cmp_result
            else:
                result = np.logical_and(result, cmp_result)
            left = right
        return result.astype(float) if isinstance(result, np.ndarray) else float(result)

    def visit_BoolOp(self, node: ast.BoolOp) -> Any:
        op_type = type(node.op)
        if op_type not in _BOOL_OPS:
            raise SafeEvalError(f"Disallowed boolean operator: {op_type.__name__}")
        result = self.visit(node.values[0])
        for value in node.values[1:]:
            result = _BOOL_OPS[op_type](result, self.visit(value))
        return result

    def visit_IfExp(self, node: ast.IfExp) -> Any:
        """Support ternary: ``a if condition else b`` → np.where."""
        cond = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return np.where(cond, body, orelse)

    def visit_Tuple(self, node: ast.Tuple) -> Any:
        return tuple(self.visit(elt) for elt in node.elts)

    def visit_Attribute(self, node: ast.Attribute) -> Any:
        raise SafeEvalError(f"Attribute access is not allowed: .{node.attr}")

    def visit_Subscript(self, node: ast.Subscript) -> Any:
        raise SafeEvalError("Subscript/indexing is not allowed")

    def generic_visit(self, node: ast.AST) -> Any:
        raise SafeEvalError(f"Disallowed AST node type: {type(node).__name__}")


def safe_eval(expression: str, variables: Optional[Dict[str, Any]] = None) -> Any:
    """Safely evaluate a mathematical expression string.

    Parameters
    ----------
    expression : str
        The equation string, e.g. ``"sin(ph) * exp(-3*u)"``
    variables : dict, optional
        Mapping of variable names to numpy arrays or scalars.

    Returns
    -------
    numpy.ndarray or scalar
        The evaluated result.

    Raises
    ------
    SafeEvalError
        If the expression contains disallowed operations.
    """
    if variables is None:
        variables = {}
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise SafeEvalError(f"Syntax error in expression: {exc}") from exc
    evaluator = _ASTEvaluator(variables)
    return evaluator.visit(tree)
