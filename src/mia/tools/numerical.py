"""Tool B: High-precision numerical evaluation."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
import mpmath
import sympy as sp


class NumericalEvaluateArgs(BaseModel):
    """Arguments for numerical evaluation."""
    expression: str = Field(description="Mathematical expression to evaluate (e.g. 'sin(x) / x')")
    variable: str = Field(default="x", description="Independent variable name")
    points: List[float] = Field(description="List of numerical points at which to evaluate the expression")
    precision_dps: int = Field(default=50, ge=10, le=500, description="Decimal places of precision")
    tolerance: float = Field(default=1e-7, gt=0.0, description="Numerical comparison tolerance")


def numerical_evaluate_handler(
    expression: str,
    points: List[float],
    variable: str = "x",
    precision_dps: int = 50,
    tolerance: float = 1e-7
) -> Dict[str, Any]:
    """Evaluate mathematical expression at given points with high decimal precision."""
    old_dps = mpmath.mp.dps
    mpmath.mp.dps = precision_dps

    clean_expr = expression.replace("^", "**")
    var = sp.Symbol(variable)

    try:
        sym_expr = sp.sympify(clean_expr)
        lambdified = sp.lambdify(var, sym_expr, modules=["mpmath"])
    except Exception as e:
        mpmath.mp.dps = old_dps
        return {
            "success": False,
            "error": f"Failed to parse/lambdify expression: {e}",
            "expression": expression
        }

    results = []
    anomalies = []

    for pt in points:
        pt_mp = mpmath.mpf(pt)
        entry: Dict[str, Any] = {"x": float(pt), "x_exact": str(pt_mp)}
        try:
            val = lambdified(pt_mp)
            if isinstance(val, (int, float, mpmath.mpf)):
                val_str = str(val)
                is_fin = bool(mpmath.isfinite(val))
                entry["value"] = float(val) if is_fin else str(val)
                entry["value_high_precision"] = val_str
                entry["is_finite"] = is_fin
                if not is_fin:
                    anomalies.append({"point": float(pt), "reason": "Non-finite result"})
            elif isinstance(val, (complex, mpmath.mpc)):
                entry["value_complex"] = str(val)
                entry["is_real"] = abs(val.imag) < tolerance
            else:
                entry["value"] = str(val)
        except ZeroDivisionError:
            entry["error"] = "ZeroDivisionError"
            anomalies.append({"point": float(pt), "reason": "Division by zero"})
        except ValueError as ve:
            entry["error"] = f"DomainViolation: {ve}"
            anomalies.append({"point": float(pt), "reason": f"Domain violation: {ve}"})
        except OverflowError:
            entry["error"] = "OverflowError"
            anomalies.append({"point": float(pt), "reason": "Numerical overflow"})
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            anomalies.append({"point": float(pt), "reason": str(e)})

        results.append(entry)

    mpmath.mp.dps = old_dps

    return {
        "success": True,
        "expression": expression,
        "variable": variable,
        "precision_dps": precision_dps,
        "tolerance": tolerance,
        "evaluations_count": len(results),
        "results": results,
        "anomalies": anomalies,
        "has_anomalies": len(anomalies) > 0
    }
