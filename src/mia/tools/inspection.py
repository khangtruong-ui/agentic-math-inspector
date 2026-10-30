"""Tool G: Structured inspection of accumulated investigation records."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class InspectInvestigationArgs(BaseModel):
    """Arguments for querying prior investigation state and experiments."""
    query_type: Literal[
        "summary", "recent_experiments", "failed_experiments", "active_hypotheses", "counterexamples"
    ] = Field(
        default="summary",
        description="Type of records to query from current investigation"
    )
    limit: int = Field(default=5, ge=1, le=20, description="Max number of items to return")


def inspect_investigation_handler(
    query_type: str = "summary",
    limit: int = 5,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Retrieve structured past experiments, hypotheses, or observations from the investigation context."""
    if not context or "state" not in context:
        return {
            "query_type": query_type,
            "status": "No active investigation state attached in execution context."
        }

    state = context["state"]
    experiments = context.get("experiments", [])
    hypotheses = context.get("hypotheses", [])

    if query_type == "summary":
        return {
            "investigation_id": state.get("investigation_id"),
            "problem_id": state.get("problem_id"),
            "status": state.get("status"),
            "active_hypothesis": state.get("active_hypothesis"),
            "total_experiments": len(experiments),
            "total_hypotheses": len(hypotheses),
            "step_count": state.get("step_count", 0)
        }

    elif query_type == "recent_experiments":
        recent = experiments[-limit:] if experiments else []
        return {
            "query_type": "recent_experiments",
            "count": len(recent),
            "experiments": [
                {
                    "experiment_id": exp.get("experiment_id"),
                    "tool": exp.get("tool_name"),
                    "exit_status": exp.get("exit_status"),
                    "stdout_summary": (exp.get("stdout") or "")[:200],
                    "runtime_seconds": exp.get("runtime_seconds")
                }
                for exp in recent
            ]
        }

    elif query_type == "failed_experiments":
        failed = [exp for exp in experiments if exp.get("exit_status", 0) != 0][-limit:]
        return {
            "query_type": "failed_experiments",
            "count": len(failed),
            "failures": [
                {
                    "experiment_id": exp.get("experiment_id"),
                    "tool": exp.get("tool_name"),
                    "error_classification": exp.get("error_classification"),
                    "stderr": (exp.get("stderr") or "")[:200]
                }
                for exp in failed
            ]
        }

    elif query_type == "active_hypotheses":
        active = [h for h in hypotheses if h.get("status") == "ACTIVE"]
        all_h = hypotheses[-limit:]
        return {
            "query_type": "active_hypotheses",
            "active": active,
            "recent_hypotheses": all_h
        }

    elif query_type == "counterexamples":
        ces = []
        for exp in experiments:
            num_res = exp.get("numerical_results") or {}
            if "counterexample" in str(exp) or "violation" in str(exp):
                ces.append({
                    "experiment_id": exp.get("experiment_id"),
                    "evidence": exp.get("stdout", "")[:300]
                })
        return {
            "query_type": "counterexamples",
            "found_count": len(ces),
            "counterexamples": ces[:limit]
        }

    return {"error": f"Unknown query type '{query_type}'"}
