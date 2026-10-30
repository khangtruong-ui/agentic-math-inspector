"""Unit tests for the evaluation runner, baselines, and report generation."""

from pathlib import Path
import pytest
from mia.config import AppConfig
from mia.evaluation.baselines import build_agent_for_config
from mia.evaluation.metrics import check_answer_correctness
from mia.evaluation.runner import EvaluationRunner
from mia.models import InvestigationReport, InvestigationStatus, ProblemRecord


def test_answer_correctness_checker():
    """Verify ground-truth answer verification."""
    prob = ProblemRecord(
        problem_id="p1",
        source="test",
        statement="What is 20 + 20?",
        original_statement="What is 20 + 20?",
        official_answer="40"
    )

    # Correct candidate report
    rep_correct = InvestigationReport(
        investigation_id="inv1",
        problem_id="p1",
        problem_statement="What is 20 + 20?",
        status=InvestigationStatus.SOLVED,
        termination_reason="Success",
        candidate_solution="The answer is 40."
    )
    assert check_answer_correctness(prob, rep_correct) is True

    # Incorrect candidate report
    rep_incorrect = InvestigationReport(
        investigation_id="inv2",
        problem_id="p1",
        problem_statement="What is 20 + 20?",
        status=InvestigationStatus.SOLVED,
        termination_reason="Success",
        candidate_solution="The answer is 99."
    )
    assert check_answer_correctness(prob, rep_incorrect) is False


def test_evaluation_runner_pipeline(tmp_path):
    """Verify evaluation runner generates results.jsonl, summary.csv, and report.md."""
    cfg = AppConfig()
    cfg.database_url = f"sqlite:///{tmp_path}/test.db"
    cfg.trace_dir = str(tmp_path / "traces")
    cfg.evaluation.output_dir = str(tmp_path / "evaluations")

    runner = EvaluationRunner(cfg)
    res = runner.run_evaluation(
        source_name="controlled",
        configs=["A", "C"],
        num_problems=2,
        seed=123
    )

    assert Path(res["results_jsonl"]).is_file()
    assert Path(res["summary_csv"]).is_file()
    assert Path(res["report_md"]).is_file()

    # Check aggregate contents
    aggs = res["aggregates"]
    assert "A" in aggs
    assert "C" in aggs
    assert aggs["A"]["total_problems"] == 2
    assert aggs["C"]["total_problems"] == 2
