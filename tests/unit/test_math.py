"""Unit tests for symbolic mathematics, numerical precision, counterexamples, and verification."""

import pytest
import sympy as sp
from mia.models import VerificationLevel, VerificationStatus
from mia.tools.counterexample import counterexample_search_handler
from mia.tools.numerical import numerical_evaluate_handler
from mia.tools.symbolic import symbolic_compute_handler
from mia.verification.verifier import IndependentVerifier


def test_symbolic_compute_diff_and_limits():
    """Verify symbolic differentiation and limits."""
    # Derivative
    diff_res = symbolic_compute_handler(operation="diff", expression="x**4 - 4*x", variable="x")
    assert diff_res["result"] == "4*x**3 - 4"

    # Taylor series limit
    lim_res = symbolic_compute_handler(
        operation="limits",
        expression="(sin(3*x) - 3*x + (9/2)*x**3) / x**5",
        variable="x",
        point="0"
    )
    assert lim_res["limit"] == "81/40"


def test_symbolic_expression_comparison():
    """Compare mathematically equivalent and non-equivalent expressions."""
    # (x+1)^2 == x^2 + 2x + 1
    cmp1 = symbolic_compute_handler(
        operation="compare_expressions",
        expression="(x + 1)**2",
        secondary_expression="x**2 + 2*x + 1",
        variable="x"
    )
    assert cmp1["identically_equal"] is True

    # x^2 + 1 != x^2
    cmp2 = symbolic_compute_handler(
        operation="compare_expressions",
        expression="x**2 + 1",
        secondary_expression="x**2",
        variable="x"
    )
    assert cmp2["identically_equal"] is False


def test_numerical_evaluation_precision_and_anomalies():
    """Numerical evaluation with mpmath high precision and division by zero detection."""
    res = numerical_evaluate_handler(
        expression="1 / x",
        points=[0.0, 1.0, 2.0],
        precision_dps=30
    )
    assert res["success"] is True
    assert res["has_anomalies"] is True
    zero_eval = res["results"][0]
    assert zero_eval["error"] == "ZeroDivisionError"
    one_eval = res["results"][1]
    assert one_eval["value"] == 1.0


def test_euler_polynomial_counterexample_search():
    """Verify that automated counterexample search finds n=40 for P(n)=n^2+n+41."""
    res = counterexample_search_handler(
        conjecture_type="primality",
        expression="n^2 + n + 41",
        variable="n",
        search_min=1,
        search_max=50
    )
    assert res["found_counterexample"] is True
    first_ce = res["first_counterexample"]
    assert first_ce["value"] == 40
    assert first_ce["expression_value"] == 1681
    assert 41 in first_ce["exact_factors"]


def test_independent_verifier_levels():
    """Verify distinct levels of mathematical evidence in the verifier."""
    verifier = IndependentVerifier()

    # Level 1: Computational evidence
    res1 = verifier.verify_candidate_solution(
        problem_statement="Investigate P(n) = n^2 + n + 41",
        candidate_solution="n = 40 gives composite P(40) = 1681 = 41^2",
        verification_level=VerificationLevel.LEVEL_1
    )
    assert res1.status == VerificationStatus.VERIFIED
    assert res1.level == VerificationLevel.LEVEL_1

    # Level 2: Exact symbolic verification of limit
    res2 = verifier.verify_candidate_solution(
        problem_statement="Determine limit lim_{x -> 0} (sin(3x) - 3x + (9/2)*x^3) / x^5",
        candidate_solution="The exact limit is 81/40",
        verification_level=VerificationLevel.LEVEL_2
    )
    assert res2.status == VerificationStatus.VERIFIED
    assert res2.confidence == 1.0
