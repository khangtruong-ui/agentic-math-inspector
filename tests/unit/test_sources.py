"""Unit tests for problem sources, normalization, and reference answer separation."""

from pathlib import Path
import pytest
from mia.models import AgentProblemView, ProblemRecord
from mia.sources import get_source
from mia.sources.controlled import ControlledProblemSource
from mia.sources.imo import IMOSource


def test_controlled_problem_source(tmp_path):
    """Controlled source should generate canonical problems with ground truth."""
    source = ControlledProblemSource(cache_dir=tmp_path)
    problems = source.list_problems()
    assert len(problems) >= 5

    euler = source.get_problem("controlled-euler-primality")
    assert euler is not None
    assert "n^2 + n + 41" in euler.statement
    assert euler.official_answer is not None
    assert "40" in euler.official_answer


def test_reference_answer_separation():
    """CRITICAL: AgentProblemView must NEVER expose official answers or reference solutions."""
    record = ProblemRecord(
        problem_id="test-prob-1",
        source="test",
        statement="Prove that sqrt(2) is irrational.",
        original_statement="Prove that sqrt(2) is irrational.",
        official_answer="sqrt(2) is irrational",
        reference_solution="Assume p/q in lowest terms, then 2q^2 = p^2, so p is even..."
    )

    agent_view = record.to_agent_view()
    assert isinstance(agent_view, AgentProblemView)
    assert not hasattr(agent_view, "official_answer")
    assert not hasattr(agent_view, "reference_solution")
    assert "official_answer" not in agent_view.model_dump()
    assert "reference_solution" not in agent_view.model_dump()


def test_imo_text_normalization():
    """IMO statement normalization removes noise and headers."""
    raw = "Tuesday, 16. July 2024\nLanguage: English\n\nProblem 1. Determine all real numbers.\n\n"
    clean = IMOSource._normalize_statement(raw)
    assert "Tuesday" not in clean
    assert "Language: English" not in clean
    assert "Determine all real numbers." in clean


def test_imo_subject_inference():
    """Verify keyword-based mathematical subject heuristics."""
    assert IMOSource._infer_subject("Let ABC be a triangle with circumcircle Omega") == "geometry"
    assert IMOSource._infer_subject("Determine all composite integers n such that d divides n") == "number_theory"
    assert IMOSource._infer_subject("Let f: R -> R be a polynomial satisfying an equation") == "algebra"
    assert IMOSource._infer_subject("Turbo the snail plays a game on a board with rows and columns") == "combinatorics"
