"""Evaluation metrics calculation and data structures for MIA."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mia.models import InvestigationReport, InvestigationStatus, ProblemRecord, VerificationStatus


class ProblemEvaluationRecord(BaseModel):
    """Detailed result for an individual problem evaluated under a specific configuration."""
    problem_id: str
    config_name: str
    source: str
    status: InvestigationStatus
    is_correct: bool
    is_verified: bool
    counterexamples_found: int
    experiments_count: int
    failed_tool_calls: int
    repeated_actions_count: int
    model_calls: int
    wall_clock_seconds: float
    remote_execution_seconds: float
    unresolved_claims_count: int
    official_answer: Optional[str] = None
    candidate_answer_snippet: Optional[str] = None


class AggregateMetrics(BaseModel):
    """Summary metrics aggregated across a benchmark evaluation run."""
    config_name: str
    total_problems: int = 0
    solved_count: int = 0
    correct_count: int = 0
    verified_count: int = 0
    total_counterexamples: int = 0
    total_experiments: int = 0
    total_failed_tool_calls: int = 0
    total_repeated_actions: int = 0
    total_model_calls: int = 0
    total_wall_clock_seconds: float = 0.0
    total_remote_execution_seconds: float = 0.0
    problems_with_unresolved_claims: int = 0

    @property
    def accuracy_rate(self) -> float:
        return round(self.correct_count / self.total_problems, 4) if self.total_problems else 0.0

    @property
    def verification_rate(self) -> float:
        return round(self.verified_count / self.total_problems, 4) if self.total_problems else 0.0

    @property
    def completion_rate(self) -> float:
        return round(self.solved_count / self.total_problems, 4) if self.total_problems else 0.0

    @property
    def unresolved_rate(self) -> float:
        return round(self.problems_with_unresolved_claims / self.total_problems, 4) if self.total_problems else 0.0

    @property
    def avg_experiments_per_problem(self) -> float:
        return round(self.total_experiments / self.total_problems, 2) if self.total_problems else 0.0

    @property
    def avg_duration_seconds(self) -> float:
        return round(self.total_wall_clock_seconds / self.total_problems, 2) if self.total_problems else 0.0


def check_answer_correctness(
    problem: ProblemRecord,
    report: InvestigationReport
) -> bool:
    """Evaluate whether candidate solution matches official ground-truth answer."""
    if not problem.official_answer or not report.candidate_solution:
        # If no official answer exists (e.g. IMO open proof problem), verify by verification status
        if report.verification and report.verification.status == VerificationStatus.VERIFIED:
            return True
        return False

    ref = problem.official_answer.strip().lower()
    cand = report.candidate_solution.strip().lower()

    # Exact or sub-string match
    if ref in cand:
        return True

    # Numeric extraction (e.g. 40, 81/40, -1, 3)
    ref_numbers = re.findall(r"[-+]?\d*\.?\d+(?:/\d+)?", ref)
    if ref_numbers:
        # Check if the primary reference number appears in the candidate text
        primary_ref = ref_numbers[0]
        if primary_ref in cand:
            return True

    return False
