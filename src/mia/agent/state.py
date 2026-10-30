"""Investigation state management."""

from __future__ import annotations

import datetime
import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mia.models import (
    AgentProblemView,
    ExperimentExecutionResult,
    HypothesisRecord,
    HypothesisStatus,
    InvestigationReport,
    InvestigationStatus,
    VerificationResult,
)


class ActiveInvestigationState(BaseModel):
    """Runtime mutable state for an ongoing investigation."""
    investigation_id: str
    problem: AgentProblemView
    status: InvestigationStatus = InvestigationStatus.INITIALIZING
    current_goal: str = "Understand and formalize mathematical problem"
    active_hypothesis: Optional[str] = None
    hypotheses_history: List[HypothesisRecord] = Field(default_factory=list)
    experiments_history: List[ExperimentExecutionResult] = Field(default_factory=list)
    step_count: int = 0
    model_calls_count: int = 0
    tool_calls_count: int = 0
    failed_tool_calls_count: int = 0
    consecutive_nonproductive_steps: int = 0
    start_mono_time: float = Field(default_factory=time.monotonic)
    candidate_solution: Optional[str] = None
    verification_result: Optional[VerificationResult] = None
    proven_results: List[str] = Field(default_factory=list)
    computational_evidence: List[str] = Field(default_factory=list)
    assumptions_made: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    termination_reason: Optional[str] = None
    action_hashes: List[str] = Field(default_factory=list)

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.start_mono_time

    def record_hypothesis(self, text: str, status: HypothesisStatus = HypothesisStatus.ACTIVE) -> HypothesisRecord:
        hyp_id = f"hyp-{len(self.hypotheses_history) + 1}"
        # Archive any prior active hypothesis
        if status == HypothesisStatus.ACTIVE:
            for h in self.hypotheses_history:
                if h.status == HypothesisStatus.ACTIVE:
                    h.status = HypothesisStatus.REVISED

        rec = HypothesisRecord(
            hypothesis_id=hyp_id,
            text=text,
            status=status,
            created_step=self.step_count
        )
        self.hypotheses_history.append(rec)
        self.active_hypothesis = text
        return rec

    def is_action_repeated(self, action_signature: str, max_allowed: int = 3) -> bool:
        """Check if an identical action has been repeated consecutively or excessively."""
        recent = self.action_hashes[-max_allowed:]
        return len(recent) == max_allowed and all(a == action_signature for a in recent)

    def to_report(self) -> InvestigationReport:
        return InvestigationReport(
            investigation_id=self.investigation_id,
            problem_id=self.problem.problem_id,
            problem_statement=self.problem.statement,
            status=self.status,
            termination_reason=self.termination_reason or "Completed",
            proven_results=self.proven_results,
            computational_evidence=self.computational_evidence,
            assumptions_made=self.assumptions_made,
            unresolved_questions=self.unresolved_questions,
            candidate_solution=self.candidate_solution,
            verification=self.verification_result,
            hypotheses_evaluated=self.hypotheses_history,
            total_steps=self.step_count,
            total_experiments=len(self.experiments_history),
            total_tool_calls=self.tool_calls_count,
            total_model_calls=self.model_calls_count,
            wall_clock_seconds=round(self.elapsed_seconds(), 2)
        )
