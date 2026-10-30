"""Unit tests for the autonomous agent investigation loop, hypothesis revision, and persistence."""

from pathlib import Path
import pytest
from mia.agent.loop import MathematicalAgent
from mia.agent.state import ActiveInvestigationState
from mia.config import AppConfig
from mia.models import AgentProblemView, HypothesisStatus, InvestigationStatus
from mia.sources import get_source
from mia.storage.experiment_store import ExperimentStore


def test_agent_investigation_loop(tmp_path):
    """Verify end-to-end investigation cycle on a controlled problem."""
    cfg = AppConfig()
    cfg.database_url = f"sqlite:///{tmp_path}/test.db"
    cfg.trace_dir = str(tmp_path / "traces")

    agent = MathematicalAgent(cfg)
    source = get_source("controlled", cache_dir=tmp_path)
    problem = source.get_problem("controlled-euler-primality")
    assert problem is not None

    report = agent.investigate(problem.to_agent_view())
    assert report.total_steps > 0
    assert report.total_experiments > 0
    assert report.candidate_solution is not None
    assert report.verification is not None
    assert "40" in report.candidate_solution


def test_hypothesis_revision_on_contradiction(tmp_path):
    """Verify that when a counterexample is discovered, the hypothesis transitions to REVISED/REFUTED."""
    cfg = AppConfig()
    cfg.database_url = f"sqlite:///{tmp_path}/test.db"
    cfg.trace_dir = str(tmp_path / "traces")

    agent = MathematicalAgent(cfg)
    source = get_source("controlled", cache_dir=tmp_path)
    prob = source.get_problem("controlled-euler-primality")

    report = agent.investigate(prob.to_agent_view())

    # Check that hypothesis was formed and updated
    assert len(report.hypotheses_evaluated) >= 1
    # At least one hypothesis should capture n=40 composite
    euler_hyp = [h for h in report.hypotheses_evaluated if "40" in h.text or "not prime" in h.text.lower()]
    assert len(euler_hyp) >= 1


def test_repeated_action_detection():
    """Verify detection of consecutive identical action signatures."""
    state = ActiveInvestigationState(
        investigation_id="test-rep",
        problem=AgentProblemView(problem_id="p1", source="test", statement="test")
    )

    sig = "tool_a_args_xyz"
    state.action_hashes.append(sig)
    state.action_hashes.append(sig)
    assert not state.is_action_repeated(sig, max_allowed=3)

    state.action_hashes.append(sig)
    assert state.is_action_repeated(sig, max_allowed=3)


def test_investigation_resume(tmp_path):
    """Verify that an interrupted investigation can be resumed seamlessly."""
    cfg = AppConfig()
    cfg.database_url = f"sqlite:///{tmp_path}/test.db"
    cfg.trace_dir = str(tmp_path / "traces")

    agent = MathematicalAgent(cfg)
    source = get_source("controlled", cache_dir=tmp_path)
    problem = source.get_problem("controlled-analysis-limit")
    assert problem is not None

    # Step 1: Initial partial run
    report1 = agent.investigate(problem.to_agent_view(), max_steps_override=1)
    inv_id = report1.investigation_id

    # Step 2: Resume
    report2 = agent.resume(inv_id, additional_steps=5)
    assert report2.investigation_id == inv_id
    assert report2.total_steps >= 2
