"""Tool F: Solution and hypothesis verification."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional
from pydantic import BaseModel, Field

from mia.models import VerificationLevel
from mia.verification.verifier import IndependentVerifier


class VerifySolutionArgs(BaseModel):
    """Arguments for independent mathematical verification."""
    problem_statement: str = Field(description="Statement of the mathematical problem being solved")
    candidate_solution: str = Field(description="The proposed solution, conjecture, or answer to verify")
    verification_level: Literal["LEVEL_1", "LEVEL_2", "LEVEL_3"] = Field(
        default="LEVEL_2",
        description="Verification rigour: LEVEL_1 (computational), LEVEL_2 (symbolic/exact), LEVEL_3 (proof structure)"
    )


def verify_solution_handler(
    problem_statement: str,
    candidate_solution: str,
    verification_level: str = "LEVEL_2",
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute independent verification checks against candidate solution."""
    verifier = IndependentVerifier()
    lvl = VerificationLevel(verification_level)
    res = verifier.verify_candidate_solution(
        problem_statement=problem_statement,
        candidate_solution=candidate_solution,
        verification_level=lvl
    )
    return {
        "verification_level": res.level.value,
        "status": res.status.value,
        "is_verified": res.status.value == "VERIFIED",
        "evidence": res.evidence,
        "unverified_steps": res.unverified_steps,
        "assumptions": res.assumptions,
        "counterexample": res.counterexample,
        "confidence": res.confidence,
        "summary": res.summary
    }
