"""SQLite Experiment Store for persistent investigation state."""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

from mia.models import (
    AgentProblemView,
    ExperimentExecutionResult,
    HypothesisRecord,
    HypothesisStatus,
    InvestigationStatus,
    VerificationResult,
)


class ExperimentStore:
    """Manages SQLite storage for investigations, hypotheses, experiments, and verifications."""

    def __init__(self, db_path: str = "runs/mia.db"):
        self.db_path = db_path
        if db_path.startswith("sqlite:///"):
            self.file_path = Path(db_path.replace("sqlite:///", ""))
        else:
            self.file_path = Path(db_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_tables()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.file_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS investigations (
                    id TEXT PRIMARY KEY,
                    problem_id TEXT NOT NULL,
                    problem_view_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    active_hypothesis TEXT,
                    candidate_solution TEXT,
                    termination_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hypotheses (
                    id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    text TEXT NOT NULL,
                    status TEXT NOT NULL,
                    supporting_evidence_json TEXT,
                    contradicting_evidence_json TEXT,
                    created_step INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (investigation_id) REFERENCES investigations(id)
                );

                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    parameters_json TEXT,
                    backend TEXT,
                    exit_status INTEGER,
                    stdout TEXT,
                    stderr TEXT,
                    runtime_seconds REAL,
                    error_classification TEXT,
                    result_json TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (investigation_id) REFERENCES investigations(id)
                );

                CREATE TABLE IF NOT EXISTS verifications (
                    id TEXT PRIMARY KEY,
                    investigation_id TEXT NOT NULL,
                    level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    confidence REAL,
                    evidence_json TEXT,
                    unverified_steps_json TEXT,
                    summary TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (investigation_id) REFERENCES investigations(id)
                );
            """)

    def create_investigation(self, investigation_id: str, problem: AgentProblemView) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO investigations
                (id, problem_id, problem_view_json, status, active_hypothesis, candidate_solution, termination_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    investigation_id,
                    problem.problem_id,
                    problem.model_dump_json(),
                    InvestigationStatus.INITIALIZING.value,
                    None,
                    None,
                    None,
                    now,
                    now
                )
            )

    def update_investigation(
        self,
        investigation_id: str,
        status: Optional[str] = None,
        active_hypothesis: Optional[str] = None,
        candidate_solution: Optional[str] = None,
        termination_reason: Optional[str] = None
    ) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        updates = ["updated_at = ?"]
        params: List[Any] = [now]

        if status is not None:
            updates.append("status = ?")
            params.append(status)
        if active_hypothesis is not None:
            updates.append("active_hypothesis = ?")
            params.append(active_hypothesis)
        if candidate_solution is not None:
            updates.append("candidate_solution = ?")
            params.append(candidate_solution)
        if termination_reason is not None:
            updates.append("termination_reason = ?")
            params.append(termination_reason)

        params.append(investigation_id)
        sql = f"UPDATE investigations SET {', '.join(updates)} WHERE id = ?"
        with self._get_connection() as conn:
            conn.execute(sql, tuple(params))

    def record_hypothesis(self, investigation_id: str, hypothesis: HypothesisRecord) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO hypotheses
                (id, investigation_id, text, status, supporting_evidence_json, contradicting_evidence_json, created_step, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    hypothesis.hypothesis_id,
                    investigation_id,
                    hypothesis.text,
                    hypothesis.status.value,
                    json.dumps(hypothesis.supporting_evidence),
                    json.dumps(hypothesis.contradicting_evidence),
                    hypothesis.created_step,
                    now
                )
            )

    def record_experiment(
        self,
        investigation_id: str,
        tool_name: str,
        parameters: Dict[str, Any],
        result: ExperimentExecutionResult
    ) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments
                (id, investigation_id, tool_name, parameters_json, backend, exit_status, stdout, stderr, runtime_seconds, error_classification, result_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.experiment_id,
                    investigation_id,
                    tool_name,
                    json.dumps(parameters),
                    result.execution_backend,
                    result.exit_status,
                    result.stdout,
                    result.stderr,
                    result.runtime_seconds,
                    result.error_classification,
                    result.model_dump_json(),
                    now
                )
            )

    def record_verification(self, investigation_id: str, result: VerificationResult) -> None:
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        ver_id = f"ver-{uuid.uuid4().hex[:8]}"
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO verifications
                (id, investigation_id, level, status, confidence, evidence_json, unverified_steps_json, summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ver_id,
                    investigation_id,
                    result.level.value,
                    result.status.value,
                    result.confidence,
                    json.dumps(result.evidence),
                    json.dumps(result.unverified_steps),
                    result.summary,
                    now
                )
            )

    def get_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            row = conn.execute("SELECT * FROM investigations WHERE id = ?", (investigation_id,)).fetchone()
            if not row:
                return None
            data = dict(row)
            data["problem"] = AgentProblemView.model_validate_json(data["problem_view_json"])
            return data

    def list_experiments(self, investigation_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            if investigation_id:
                rows = conn.execute(
                    "SELECT * FROM experiments WHERE investigation_id = ? ORDER BY created_at DESC LIMIT ?",
                    (investigation_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM experiments ORDER BY created_at DESC LIMIT ?",
                    (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_hypotheses(self, investigation_id: str) -> List[HypothesisRecord]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM hypotheses WHERE investigation_id = ? ORDER BY created_step ASC",
                (investigation_id,)
            ).fetchall()
            recs = []
            for r in rows:
                recs.append(HypothesisRecord(
                    hypothesis_id=r["id"],
                    text=r["text"],
                    status=HypothesisStatus(r["status"]),
                    supporting_evidence=json.loads(r["supporting_evidence_json"] or "[]"),
                    contradicting_evidence=json.loads(r["contradicting_evidence_json"] or "[]"),
                    created_step=r["created_step"],
                    updated_at=r["created_at"]
                ))
            return recs
