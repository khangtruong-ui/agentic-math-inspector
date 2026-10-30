"""Core data models and schemas for MIA."""

from __future__ import annotations

import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Base model enforcing validation and clean serialization."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AgentProblemView(StrictModel):
    """Restricted view of a problem presented to the solving agent.

    CRITICAL: Never contains `official_answer` or `reference_solution` to prevent data leakage.
    """
    problem_id: str
    source: str
    statement: str
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProblemRecord(StrictModel):
    """Complete problem record as ingested and stored."""
    problem_id: str
    source: str
    source_url: Optional[str] = None
    retrieved_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    statement: str
    original_statement: str
    subject: Optional[str] = None
    difficulty: Optional[str] = None
    official_answer: Optional[str] = None
    reference_solution: Optional[str] = None
    license: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_agent_view(self) -> AgentProblemView:
        """Create a sanitized view of the problem for the solving agent."""
        return AgentProblemView(
            problem_id=self.problem_id,
            source=self.source,
            statement=self.statement,
            subject=self.subject,
            difficulty=self.difficulty,
            metadata=self.metadata.copy()
        )


class HypothesisStatus(str, Enum):
    ACTIVE = "ACTIVE"
    REVISED = "REVISED"
    REFUTED = "REFUTED"
    CONFIRMED = "CONFIRMED"


class HypothesisRecord(StrictModel):
    """A candidate mathematical conjecture, strategy, or approach."""
    hypothesis_id: str
    text: str
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    created_step: int = 0
    updated_at: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class ExperimentExecutionResult(StrictModel):
    """Structured result produced by experiment execution (Local or Modal)."""
    action_id: str
    experiment_id: str
    input_parameters: Dict[str, Any] = Field(default_factory=dict)
    execution_backend: str = "local"
    start_time: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    finish_time: str = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )
    exit_status: int = 0
    stdout: str = ""
    stderr: str = ""
    numerical_results: Optional[Dict[str, Any]] = None
    symbolic_results: Optional[Dict[str, Any]] = None
    runtime_seconds: float = 0.0
    resource_usage: Optional[Dict[str, Any]] = None
    error_classification: Optional[str] = None
    artifact_references: List[str] = Field(default_factory=list)
    is_interrupted: bool = False


class VerificationLevel(str, Enum):
    LEVEL_1 = "LEVEL_1"  # Computational / empirical evidence
    LEVEL_2 = "LEVEL_2"  # Symbolic or exact arithmetic identity check
    LEVEL_3 = "LEVEL_3"  # Formal or structured proof step validation


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    REFUTED = "REFUTED"
    PARTIAL = "PARTIAL"
    INCONCLUSIVE = "INCONCLUSIVE"


class VerificationResult(StrictModel):
    """Independent mathematical verification output."""
    level: VerificationLevel
    status: VerificationStatus
    evidence: List[str] = Field(default_factory=list)
    unverified_steps: List[str] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    counterexample: Optional[Dict[str, Any]] = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    summary: str = ""


class InvestigationStatus(str, Enum):
    INITIALIZING = "INITIALIZING"
    EXPLORING = "EXPLORING"
    REVISING = "REVISING"
    VERIFYING = "VERIFYING"
    SOLVED = "SOLVED"
    PARTIALLY_SOLVED = "PARTIALLY_SOLVED"
    REFUTED = "REFUTED"
    EXHAUSTED = "EXHAUSTED"
    FAILED = "FAILED"


class InvestigationReport(StrictModel):
    """Comprehensive final report produced by an investigation."""
    investigation_id: str
    problem_id: str
    problem_statement: str
    status: InvestigationStatus
    termination_reason: str
    proven_results: List[str] = Field(default_factory=list)
    computational_evidence: List[str] = Field(default_factory=list)
    assumptions_made: List[str] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)
    candidate_solution: Optional[str] = None
    verification: Optional[VerificationResult] = None
    hypotheses_evaluated: List[HypothesisRecord] = Field(default_factory=list)
    total_steps: int = 0
    total_experiments: int = 0
    total_tool_calls: int = 0
    total_model_calls: int = 0
    wall_clock_seconds: float = 0.0
    estimated_cost_usd: Optional[float] = None
