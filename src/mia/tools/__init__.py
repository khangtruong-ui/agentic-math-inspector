"""MIA Mathematical Tools Registry and Definitions."""

from __future__ import annotations

from mia.tools.registry import ToolRegistry, ToolDefinition
from mia.tools.symbolic import SymbolicComputeArgs, symbolic_compute_handler
from mia.tools.numerical import NumericalEvaluateArgs, numerical_evaluate_handler
from mia.tools.test_cases import GenerateTestCasesArgs, generate_test_cases_handler
from mia.tools.counterexample import CounterexampleSearchArgs, counterexample_search_handler
from mia.tools.experiment import ExecuteExperimentArgs, execute_experiment_handler
from mia.tools.verification import VerifySolutionArgs, verify_solution_handler
from mia.tools.inspection import InspectInvestigationArgs, inspect_investigation_handler


def create_default_registry() -> ToolRegistry:
    """Create and populate standard mathematical tool registry."""
    registry = ToolRegistry()

    registry.register(
        name="symbolic_compute",
        description=(
            "Execute symbolic mathematical operations via SymPy (simplify, differentiate, integrate, "
            "solve equations, find polynomial roots, exact arithmetic, limits, factor, series expansion)."
        ),
        arg_model=SymbolicComputeArgs
    )(symbolic_compute_handler)

    registry.register(
        name="numerical_evaluate",
        description=(
            "Evaluate mathematical expressions numerically at arbitrary precision with mpmath / numpy. "
            "Checks for domain violations, non-finite values, and numerical overflow."
        ),
        arg_model=NumericalEvaluateArgs
    )(numerical_evaluate_handler)

    registry.register(
        name="generate_test_cases",
        description=(
            "Generate reproducible deterministic boundary and randomized test inputs for integers, "
            "reals, polynomials, or coordinate pairs."
        ),
        arg_model=GenerateTestCasesArgs
    )(generate_test_cases_handler)

    registry.register(
        name="search_counterexamples",
        description=(
            "Search for mathematical inputs that violate a candidate conjecture (e.g. primality, "
            "equality, inequality) with independent validation."
        ),
        arg_model=CounterexampleSearchArgs
    )(counterexample_search_handler)

    registry.register(
        name="execute_experiment",
        description=(
            "Execute an arbitrary Python experiment script in an isolated execution sandbox "
            "(Local subprocess or Modal cloud container) with resource and timeout enforcement."
        ),
        arg_model=ExecuteExperimentArgs
    )(execute_experiment_handler)

    registry.register(
        name="verify_solution",
        description=(
            "Perform independent mathematical verification of a candidate solution or formula "
            "across computational (Level 1), symbolic (Level 2), or proof structure (Level 3) tiers."
        ),
        arg_model=VerifySolutionArgs
    )(verify_solution_handler)

    registry.register(
        name="inspect_investigation",
        description=(
            "Structured query of accumulated past experiments, hypotheses, failed attempts, and "
            "counterexamples in the active investigation."
        ),
        arg_model=InspectInvestigationArgs
    )(inspect_investigation_handler)

    return registry


__all__ = [
    "ToolRegistry",
    "ToolDefinition",
    "create_default_registry",
]
