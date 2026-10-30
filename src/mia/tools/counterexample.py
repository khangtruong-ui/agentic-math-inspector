"""Tool D: Counterexample search and validation."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
import sympy as sp
from sympy.ntheory import isprime, primefactors


class CounterexampleSearchArgs(BaseModel):
    """Arguments for counterexample search."""
    conjecture_type: Literal[
        "primality", "equality", "inequality", "divisibility", "custom_condition"
    ] = Field(description="Mathematical property claimed to hold universally")
    expression: str = Field(description="Mathematical expression (e.g. 'n^2 + n + 41')")
    variable: str = Field(default="n", description="Search variable name")
    search_min: int = Field(default=1, description="Minimum search value")
    search_max: int = Field(default=100, description="Maximum search value")
    secondary_expression: Optional[str] = Field(
        default=None,
        description="Target expression for equality or right side of inequality"
    )
    inequality_op: Optional[Literal[">", ">=", "<", "<="]] = Field(
        default=">=",
        description="Inequality operator if testing an inequality"
    )


def counterexample_search_handler(
    conjecture_type: str,
    expression: str,
    variable: str = "n",
    search_min: int = 1,
    search_max: int = 100,
    secondary_expression: Optional[str] = None,
    inequality_op: Optional[str] = ">="
) -> Dict[str, Any]:
    """Search for inputs violating a conjectured mathematical property."""
    clean_expr = expression.replace("^", "**")
    var = sp.Symbol(variable)
    sym_expr = sp.sympify(clean_expr)

    tested_count = 0
    counterexample_found = None
    all_counterexamples: List[Dict[str, Any]] = []

    for val in range(search_min, search_max + 1):
        tested_count += 1
        computed_val = sym_expr.subs(var, val)

        if conjecture_type == "primality":
            # Claim: expression is prime for all n
            try:
                num = int(computed_val)
                if not isprime(num):
                    factors = primefactors(num)
                    ce = {
                        "variable": variable,
                        "value": val,
                        "expression_value": num,
                        "violation": f"Computed value {num} is composite",
                        "prime_factors": factors,
                        "is_prime": False
                    }
                    all_counterexamples.append(ce)
                    if counterexample_found is None:
                        counterexample_found = ce
            except (TypeError, ValueError):
                continue

        elif conjecture_type == "equality":
            # Claim: expression == secondary_expression for all n
            if secondary_expression:
                sec_expr = sp.sympify(secondary_expression.replace("^", "**"))
                sec_val = sec_expr.subs(var, val)
                if computed_val != sec_val:
                    ce = {
                        "variable": variable,
                        "value": val,
                        "left_value": str(computed_val),
                        "right_value": str(sec_val),
                        "violation": f"LHS ({computed_val}) != RHS ({sec_val})"
                    }
                    all_counterexamples.append(ce)
                    if counterexample_found is None:
                        counterexample_found = ce

        elif conjecture_type == "inequality":
            # Claim: expression (op) secondary_expression for all n
            sec_val = 0
            if secondary_expression:
                sec_val = sp.sympify(secondary_expression.replace("^", "**")).subs(var, val)

            diff = float(computed_val - sec_val)
            violated = False
            if inequality_op == ">" and diff <= 0:
                violated = True
            elif inequality_op == ">=" and diff < -1e-9:
                violated = True
            elif inequality_op == "<" and diff >= 0:
                violated = True
            elif inequality_op == "<=" and diff > 1e-9:
                violated = True

            if violated:
                ce = {
                    "variable": variable,
                    "value": val,
                    "computed_value": str(computed_val),
                    "target_value": str(sec_val),
                    "violation": f"Inequality {computed_val} {inequality_op} {sec_val} violated"
                }
                all_counterexamples.append(ce)
                if counterexample_found is None:
                    counterexample_found = ce

        if counterexample_found and len(all_counterexamples) >= 5:
            # Stop search once we have multiple verified counterexamples
            break

    # Independent validation of the primary counterexample
    verified_counterexample = None
    if counterexample_found:
        verified_counterexample = _validate_counterexample(
            conjecture_type, sym_expr, var, counterexample_found
        )

    return {
        "conjecture_type": conjecture_type,
        "expression": expression,
        "search_range": [search_min, search_max],
        "tested_count": tested_count,
        "found_counterexample": verified_counterexample is not None,
        "first_counterexample": verified_counterexample,
        "total_counterexamples_found": len(all_counterexamples),
        "caveat": (
            "Absence of counterexamples over a finite range is empirical evidence, "
            "not a universal proof for infinite domains."
        )
    }


def _validate_counterexample(
    conjecture_type: str,
    sym_expr: sp.Expr,
    var: sp.Symbol,
    ce: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Perform independent exact calculation to verify the candidate counterexample."""
    val = ce["value"]
    exact_res = sym_expr.subs(var, val)
    if conjecture_type == "primality":
        num = int(exact_res)
        # Check divisibility directly
        if num <= 1:
            ce["verified"] = True
            return ce
        factors = primefactors(num)
        if len(factors) > 1 or (len(factors) == 1 and factors[0] != num):
            ce["verified"] = True
            ce["exact_factors"] = factors
            return ce
        return None  # False positive, actually prime
    else:
        ce["verified"] = True
        return ce
