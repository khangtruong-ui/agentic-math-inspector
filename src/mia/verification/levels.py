"""Mathematical Verification Levels and Evidence Definitions."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from mia.models import VerificationLevel, VerificationStatus, VerificationResult


class StepVerificationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    UNVERIFIED = "UNVERIFIED"
    REFUTED = "REFUTED"
    HEURISTIC = "HEURISTIC"


class ProofStep(BaseModel):
    """An individual deduction or transformation step in a mathematical proof."""
    step_number: int
    statement: str
    justification: str
    status: StepVerificationStatus = StepVerificationStatus.UNVERIFIED
    notes: Optional[str] = None
