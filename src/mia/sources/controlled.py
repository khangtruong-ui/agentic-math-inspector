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

        for p in [p1, p2, p3, p4, p5]:
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
