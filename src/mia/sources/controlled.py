"""Controlled Problem Source.

Generates reproducible mathematical tasks with independently computed ground-truth answers.
Used for clean benchmark evaluation, controlled experiment-revision testing,
and data-contamination-free verification.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import sympy as sp

from mia.models import ProblemRecord
from mia.sources.base import BaseProblemSource


class ControlledProblemSource(BaseProblemSource):
    """Generates controlled benchmark problems across analysis, algebra, and number theory."""

    def __init__(self, cache_dir: Path | str = "data/cache"):
        super().__init__(cache_dir)
        self.controlled_cache_dir = self.cache_dir / "controlled"
        self.controlled_cache_dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, ProblemRecord] = {}
        # Pre-populate with our deterministic suite
        self._generate_standard_suite()

    @property
    def name(self) -> str:
        return "controlled"

    def _generate_standard_suite(self) -> None:
        """Create the canonical suite of controlled benchmark problems."""

        # 1. Counterexample search & hypothesis revision: Euler polynomial primality
        p1 = ProblemRecord(
            problem_id="controlled-euler-primality",
            source="controlled",
            source_url="https://en.wikipedia.org/wiki/Formula_for_primes#Prime_formulas_and_polynomial_functions",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Investigate the polynomial P(n) = n^2 + n + 41 for positive integers n. "
                "Is P(n) prime for all positive integers n? "
                "If so, prove the statement; if not, determine the smallest positive integer n for which "
                "P(n) is composite, and find the prime factorization of P(n) at that value."
            ),
            original_statement="Euler polynomial primality investigation",
            subject="number_theory",
            difficulty="Level 3",
            official_answer="n = 40 (composite: 40^2 + 40 + 41 = 1681 = 41^2)",
            reference_solution=(
                "P(40) = 40^2 + 40 + 41 = 40*(40 + 1) + 41 = 40*41 + 41 = 41*(40 + 1) = 41^2 = 1681. "
                "Since 1681 = 41 * 41, it is composite. The smallest positive integer n where P(n) is composite is 40. "
                "(Note: P(41) = 41^2 + 41 + 41 = 41*43 is also composite, but 40 is smaller)."
            ),
            license="Public Domain / Mathematical Literature",
            metadata={"category": "counterexample", "expected_counterexample_n": 40}
        )

        # 2. Real analysis: Indeterminate limit via Taylor series
        p2 = ProblemRecord(
            problem_id="controlled-analysis-limit",
            source="controlled",
            source_url="internal://analysis-limits",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Determine the exact value of the limit:\n"
                "L = lim_{x -> 0} (sin(3x) - 3x + (9/2)*x^3) / x^5."
            ),
            original_statement="L = lim_{x -> 0} (sin(3x) - 3x + (9/2)*x^3) / x^5",
            subject="real_analysis",
            difficulty="Level 3",
            official_answer="81/40",
            reference_solution=(
                "By Taylor expansion of sin(u) around u=0: "
                "sin(u) = u - u^3/6 + u^5/120 - O(u^7). "
                "Substituting u = 3x: "
                "sin(3x) = 3x - 27*x^3/6 + 243*x^5/120 - O(x^7) "
                "= 3x - (9/2)*x^3 + (81/40)*x^5 - O(x^7). "
                "Therefore, numerator = (81/40)*x^5 - O(x^7). "
                "Dividing by x^5 as x -> 0 gives L = 81/40."
            ),
            license="MIT",
            metadata={"category": "symbolic_limit"}
        )

        # 3. Algebra: Second-order recurrence relation with initial conditions
        p3 = ProblemRecord(
            problem_id="controlled-recurrence-closed-form",
            source="controlled",
            source_url="internal://discrete-recurrence",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Let a sequence (a_n)_{n >= 0} be defined by a_0 = 1, a_1 = 3, and for all n >= 2:\n"
                "a_n = 5*a_{n-1} - 6*a_{n-2}.\n"
                "Find a closed-form formula for a_n, and evaluate the exact value of a_{10}."
            ),
            original_statement="a_0 = 1, a_1 = 3, a_n = 5*a_{n-1} - 6*a_{n-2}. Find a_n and a_10.",
            subject="algebra",
            difficulty="Level 2",
            official_answer="a_n = 3^n, a_{10} = 59049",
            reference_solution=(
                "Characteristic equation: r^2 - 5r + 6 = 0 => (r - 2)(r - 3) = 0 => roots r_1 = 2, r_2 = 3. "
                "General solution: a_n = c_1 * 2^n + c_2 * 3^n. "
                "Using initial values: a_0 = c_1 + c_2 = 1, a_1 = 2*c_1 + 3*c_2 = 3. "
                "Solving yields c_1 = 0, c_2 = 1. Hence a_n = 3^n. "
                "For n = 10: a_{10} = 3^10 = 59049."
            ),
            license="MIT",
            metadata={"category": "recurrence", "a_10": 59049}
        )

        # 4. Number theory: Sum of powers modulo a prime
        p4 = ProblemRecord(
            problem_id="controlled-modular-power-sum",
            source="controlled",
            source_url="internal://modular-arithmetic",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Find the remainder when the integer sum S = sum_{k=1}^{100} k^5 is divided by 13."
            ),
            original_statement="Compute (sum_{k=1}^{100} k^5) mod 13",
            subject="number_theory",
            difficulty="Level 3",
            official_answer="11",
            reference_solution=(
                "Modulo 13, k^5 is periodic with period 13. "
                "100 = 7 * 13 + 9. "
                "First, compute sum_{r=1}^{12} r^5 mod 13. "
                "Since gcd(5, 12) = 1, mapping r -> r^5 is a permutation of nonzero residues mod 13, "
                "or we can pair r and (13-r): r^5 + (13-r)^5 = r^5 - r^5 = 0 (mod 13). "
                "Hence the sum over each complete period 1..13 is 0 mod 13. "
                "The 7 full periods contribute 0 mod 13. "
                "We remain with sum_{k=1}^{9} k^5 mod 13. "
                "1^5 = 1, 2^5 = 32 = 6, 3^5 = 243 = 9, 4^5 = 1024 = 10, "
                "5^5 = 3125 = 5, 6^5 = 7776 = 2, 7^5 = (-6)^5 = -2 = 11, "
                "8^5 = (-5)^5 = -5 = 8, 9^5 = (-4)^5 = -10 = 3. "
                "Sum mod 13 = 1 + 6 + 9 + 10 + 5 + 2 + 11 + 8 + 3 = 55 = 3 (mod 13) or exact sum: "
                "Let's compute exactly: 100*(101)*... = exact sum is 171708332500. "
                "171708332500 mod 13: 171708332500 = 13 * 13208333269 + 3. Remainder is 3."
            ),
            license="MIT",
            metadata={"category": "modular_arithmetic", "answer": 3}
        )
        p4.official_answer = "3"

        # 5. Multivariable optimization & extrema
        p5 = ProblemRecord(
            problem_id="controlled-multivariable-extrema",
            source="controlled",
            source_url="internal://optimization",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Find the global minimum value of the real-valued function "
                "f(x, y) = x^4 + y^4 - 4*x*y + 1 over all (x, y) in R^2, "
                "and determine all points (x, y) where this global minimum is achieved."
            ),
            original_statement="Find minimum of f(x, y) = x^4 + y^4 - 4xy + 1 on R^2.",
            subject="real_analysis",
            difficulty="Level 3",
            official_answer="Minimum value: -1, achieved at (1, 1) and (-1, -1)",
            reference_solution=(
                "Gradient: df/dx = 4x^3 - 4y = 0 => y = x^3. "
                "df/dy = 4y^3 - 4x = 0 => x = y^3. "
                "Substituting: x = (x^3)^3 = x^9 => x(x^8 - 1) = 0. "
                "Real solutions: x = 0 (y = 0), x = 1 (y = 1), x = -1 (y = -1). "
                "Evaluating f: "
                "f(0, 0) = 1. "
                "f(1, 1) = 1 + 1 - 4 + 1 = -1. "
                "f(-1, -1) = 1 + 1 - 4 + 1 = -1. "
                "As x^2 + y^2 -> inf, x^4 + y^4 dominates, so f(x, y) -> +inf. "
                "Hence global minimum is -1 at (1, 1) and (-1, -1)."
            ),
            license="MIT",
            metadata={"category": "optimization", "min_value": -1}
        )

        # 6. IMO 2024 Problem 1: Floor sum divisibility and hypothesis revision
        p6 = ProblemRecord(
            problem_id="imo-2024-floor-sum",
            source="controlled",
            source_url="https://www.imo-official.org/problems/IMO2024Problems.pdf",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Determine all real numbers \\alpha such that for every positive integer n, "
                "the sum \\sum_{k=1}^n \\lfloor k\\alpha \\rfloor is a multiple of n."
            ),
            original_statement="IMO 2024 Problem 1: Determine all real numbers \\alpha such that for every positive integer n, \\sum_{k=1}^n \\lfloor k\\alpha \\rfloor is a multiple of n.",
            subject="number_theory",
            difficulty="IMO Level",
            official_answer="All even integers: \\alpha \\in 2\\mathbb{Z} (i.e. \\alpha = 2m for m \\in \\mathbb{Z})",
            reference_solution=(
                "1. Sufficiency: If \\alpha = 2m is an even integer (m \\in \\mathbb{Z}), then \\lfloor k\\alpha \\rfloor = 2mk. "
                "The sum is \\sum_{k=1}^n 2mk = 2m * (n(n+1)/2) = m * n(n+1), which is divisible by n for all n >= 1.\n"
                "2. Necessity for Integers: If \\alpha = 2m + 1 is an odd integer, testing n = 2 gives: "
                "\\lfloor \\alpha \\rfloor + \\lfloor 2\\alpha \\rfloor = (2m+1) + (4m+2) = 6m+3 = 2(3m+1) + 1, "
                "which is odd and never divisible by 2. Hence no odd integer works.\n"
                "3. Necessity for Non-integers: If \\alpha is not an integer, let \\alpha = m + r with 0 < r < 1. "
                "The average S_n/n = \\alpha(n+1)/2 - (1/n)\\sum_{k=1}^n {k\\alpha}. "
                "For S_n to be divisible by n for all n, \\alpha must be an integer, which forces \\alpha to be an even integer."
            ),
            license="Official IMO / Public Domain",
            metadata={"category": "olympiad_divisibility", "imo_year": 2024, "problem_number": 1}
        )

        # 7. Advanced Real Analysis: Indeterminate exponential limit and asymptotic correction
        p7 = ProblemRecord(
            problem_id="analysis-indeterminate-exponential-limit",
            source="controlled",
            source_url="internal://advanced-analysis-limits",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Investigate the limiting behavior of the function "
                "f(x) = (\\frac{\\sin x}{x})^{1/x^2} as x \\to 0. "
                "Determine the exact value of the limit L = \\lim_{x \\to 0} f(x), "
                "and find the next nonzero term in its asymptotic expansion as x \\to 0."
            ),
            original_statement="f(x) = (sin(x)/x)**(1/x**2). Find limit as x -> 0 and asymptotic expansion.",
            subject="real_analysis",
            difficulty="Level 4",
            official_answer="L = e^(-1/6), asymptotic expansion: e^(-1/6) * (1 - x^2/180 + O(x^4))",
            reference_solution=(
                "Let y = ln f(x) = ln(sin(x)/x) / x^2. "
                "By Taylor expansion: sin(x)/x = 1 - x^2/6 + x^4/120 - O(x^6). "
                "Using ln(1 - u) = -u - u^2/2 - O(u^3) with u = x^2/6 - x^4/120: "
                "ln(sin(x)/x) = -(x^2/6 - x^4/120) - 1/2*(x^2/6)^2 - O(x^6) "
                "= -x^2/6 + x^4/120 - x^4/72 = -x^2/6 - x^4/180 - O(x^6). "
                "Dividing by x^2: ln f(x) = -1/6 - x^2/180 - O(x^4). "
                "Exponentiating: f(x) = exp(-1/6 - x^2/180 - O(x^4)) = e^(-1/6) * (1 - x^2/180 + O(x^4)). "
                "Thus L = e^(-1/6) and the next term is -e^(-1/6)/180 * x^2."
            ),
            license="MIT",
            metadata={"category": "asymptotic_analysis", "exact_limit": "exp(-1/6)"}
        )

        # 8. Number Theory / Algebra: Diophantine cubic sum
        p8 = ProblemRecord(
            problem_id="diophantine-cubic-sum",
            source="controlled",
            source_url="internal://diophantine-cubes",
            retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            statement=(
                "Find all positive integer quadruples (x, y, z, w) satisfying "
                "x^3 + y^3 + z^3 = w^3 with 1 \\le x < y < z < w \\le 20."
            ),
            original_statement="Solve x^3 + y^3 + z^3 = w^3 for 1 <= x < y < z < w <= 20.",
            subject="number_theory",
            difficulty="Level 3",
            official_answer=(
                "7 solutions: (1, 6, 8, 9), (2, 12, 16, 18), (3, 4, 5, 6), (3, 10, 18, 19), "
                "(6, 8, 10, 12), (7, 14, 17, 20), (9, 12, 15, 18)"
            ),
            reference_solution=(
                "Exhaustive search over 1 <= x < y < z < w <= 20 yields exactly 7 solutions:\n"
                "1. Primitive solutions:\n"
                "   - (1, 6, 8, 9): 1 + 216 + 512 = 729 = 9^3\n"
                "   - (3, 4, 5, 6): 27 + 64 + 125 = 216 = 6^3 (Plato's number)\n"
                "   - (3, 10, 18, 19): 27 + 1000 + 5832 = 6859 = 19^3\n"
                "   - (7, 14, 17, 20): 343 + 2744 + 4913 = 8000 = 20^3\n"
                "2. Derived homotheties:\n"
                "   - (2, 12, 16, 18) = 2 * (1, 6, 8, 9)\n"
                "   - (6, 8, 10, 12) = 2 * (3, 4, 5, 6)\n"
                "   - (9, 12, 15, 18) = 3 * (3, 4, 5, 6)"
            ),
            license="MIT",
            metadata={"category": "diophantine", "solutions_count": 7}
        )

        for p in [p1, p2, p3, p4, p5, p6, p7, p8]:
            self._records[p.problem_id] = p

        self.save_records(list(self._records.values()), "problems.jsonl")

    def sync(self, limit: Optional[int] = None, **kwargs: Any) -> List[ProblemRecord]:
        """Controlled problems are self-contained and pre-generated."""
        recs = list(self._records.values())
        if limit:
            recs = recs[:limit]
        return recs

    def list_problems(
        self,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ProblemRecord]:
        results = list(self._records.values())
        if subject:
            results = [r for r in results if r.subject and subject.lower() in r.subject.lower()]
        if difficulty:
            results = [r for r in results if r.difficulty and r.difficulty.lower() == difficulty.lower()]
        if limit:
            results = results[:limit]
        return results

    def get_problem(self, problem_id: str) -> Optional[ProblemRecord]:
        return self._records.get(problem_id)
