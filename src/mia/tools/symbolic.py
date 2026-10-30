"""Tool A: Symbolic computation using SymPy."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import sympy as sp


class SymbolicComputeArgs(BaseModel):
    """Arguments for symbolic mathematical computation."""
    operation: Literal[
        "simplify", "diff", "integrate", "solve", "poly_roots",
        "exact_eval", "limits", "factor", "series", "compare_expressions"
    ] = Field(description="Symbolic operation to execute")
    expression: str = Field(description="Primary mathematical expression (e.g. 'x^2 + 2*x + 1' or 'sin(3*x)')")
    variable: Optional[str] = Field(default="x", description="Primary variable (e.g. 'x', 'n', 't')")
    secondary_expression: Optional[str] = Field(default=None, description="Second expression for comparison or equation solving")
    point: Optional[str] = Field(default=None, description="Limit point or expansion center (e.g. '0', 'oo')")
    order: Optional[int] = Field(default=1, description="Derivative order or series expansion order")
    direction: Optional[Literal["+", "-", "+-"]] = Field(default="+-", description="Direction for limit computation")


def symbolic_compute_handler(
    operation: str,
    expression: str,
    variable: Optional[str] = "x",
    secondary_expression: Optional[str] = None,
    point: Optional[str] = None,
    order: Optional[int] = 1,
    direction: Optional[str] = "+-"
) -> Dict[str, Any]:
    """Execute symbolic mathematical operation via SymPy."""
    try:
        var = sp.Symbol(variable) if variable else sp.Symbol("x")
        # Sanitize common syntax: ^ to **
        expr_str = expression.replace("^", "**")
        parsed_expr = sp.sympify(expr_str, evaluate=False)

        if operation == "simplify":
            simplified = sp.simplify(parsed_expr)
            return {
                "operation": "simplify",
                "original": str(parsed_expr),
                "result": str(simplified),
                "latex": sp.latex(simplified),
                "is_evaluated": simplified != parsed_expr
            }

        elif operation == "diff":
            deriv_order = order or 1
            result = sp.diff(parsed_expr, var, deriv_order)
            simplified = sp.simplify(result)
            return {
                "operation": "diff",
                "variable": str(var),
                "order": deriv_order,
                "result": str(simplified),
                "latex": sp.latex(simplified)
            }

        elif operation == "integrate":
            if point is not None and secondary_expression is not None:
                # Definite integral from point to secondary_expression
                pt_a = sp.sympify(point.replace("^", "**"))
                pt_b = sp.sympify(secondary_expression.replace("^", "**"))
                result = sp.integrate(parsed_expr, (var, pt_a, pt_b))
            else:
                # Indefinite integral
                result = sp.integrate(parsed_expr, var)
            return {
                "operation": "integrate",
                "result": str(result),
                "latex": sp.latex(result),
                "has_unevaluated_integral": isinstance(result, sp.Integral)
            }

        elif operation == "solve":
            if secondary_expression:
                eq = sp.Eq(parsed_expr, sp.sympify(secondary_expression.replace("^", "**")))
            else:
                eq = parsed_expr
            solutions = sp.solve(eq, var)
            return {
                "operation": "solve",
                "equation": str(eq),
                "variable": str(var),
                "solutions": [str(s) for s in solutions],
                "num_solutions": len(solutions)
            }

        elif operation == "poly_roots":
            poly = sp.Poly(parsed_expr, var)
            roots = poly.all_roots()
            return {
                "operation": "poly_roots",
                "degree": poly.degree(),
                "roots": [str(r) for r in roots]
            }

        elif operation == "exact_eval":
            evaluated = sp.sympify(expr_str, evaluate=True)
            return {
                "operation": "exact_eval",
                "original": expr_str,
                "exact_result": str(evaluated),
                "is_integer": evaluated.is_integer,
                "is_rational": evaluated.is_rational
            }

        elif operation == "limits":
            pt_str = point if point is not None else "0"
            if pt_str == "oo" or pt_str == "inf":
                pt = sp.oo
            elif pt_str == "-oo" or pt_str == "-inf":
                pt = -sp.oo
            else:
                pt = sp.sympify(pt_str.replace("^", "**"))

            lim_dir = direction or "+-"
            lim_val = sp.limit(parsed_expr, var, pt, dir=lim_dir)
            return {
                "operation": "limits",
                "point": str(pt),
                "direction": lim_dir,
                "limit": str(lim_val),
                "is_finite": lim_val.is_finite
            }

        elif operation == "factor":
            factored = sp.factor(parsed_expr)
            return {
                "operation": "factor",
                "original": str(parsed_expr),
                "factored": str(factored),
                "is_factored": factored != parsed_expr
            }

        elif operation == "series":
            expansion_order = order or 6
            pt_val = sp.sympify((point or "0").replace("^", "**"))
            expansion = sp.series(parsed_expr, var, pt_val, n=expansion_order)
            return {
                "operation": "series",
                "point": str(pt_val),
                "order": expansion_order,
                "series": str(expansion),
                "truncated": str(expansion.removeO())
            }

        elif operation == "compare_expressions":
            if not secondary_expression:
                raise ValueError("secondary_expression is required for comparison.")
            expr2 = sp.sympify(secondary_expression.replace("^", "**"))
            diff = sp.simplify(parsed_expr - expr2)
            is_identically_zero = bool(diff == 0)
            return {
                "operation": "compare_expressions",
                "expression_1": str(parsed_expr),
                "expression_2": str(expr2),
                "difference": str(diff),
                "identically_equal": is_identically_zero,
                "conclusive": is_identically_zero or (diff.is_constant() and diff != 0)
            }

        else:
            raise ValueError(f"Unsupported symbolic operation: {operation}")

    except Exception as e:
        return {
            "error": f"Symbolic compute failed: {type(e).__name__}: {e}",
            "operation": operation,
            "expression": expression
        }
