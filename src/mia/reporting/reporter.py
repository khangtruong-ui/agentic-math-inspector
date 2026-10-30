"""Report generation and formatting from stored investigations and evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from mia.models import InvestigationReport
from mia.storage.experiment_store import ExperimentStore
from mia.storage.traces import TraceLogger


class Reporter:
    """Formats and exports investigation traces, reports, and benchmark evaluations."""

    def __init__(self, db_path: str = "runs/mia.db", trace_dir: str = "runs/traces"):
        self.store = ExperimentStore(db_path)
        self.traces = TraceLogger(trace_dir)

    def generate_investigation_markdown(self, investigation_id: str) -> str:
        """Produce human-readable Markdown investigation report from stored database records."""
        inv_data = self.store.get_investigation(investigation_id)
        if not inv_data:
            return f"# Investigation `{investigation_id}` not found."

        experiments = self.store.list_experiments(investigation_id)
        hypotheses = self.store.get_hypotheses(investigation_id)
        problem = inv_data["problem"]

        lines = [
            f"# Mathematical Investigation Report: `{investigation_id}`",
            "",
            f"- **Problem ID**: `{problem.problem_id}`",
            f"- **Source**: `{problem.source}`",
            f"- **Status**: `{inv_data['status']}`",
            f"- **Termination Reason**: {inv_data.get('termination_reason') or 'N/A'}",
            f"- **Created**: {inv_data['created_at']}",
            "",
            "## 1. Mathematical Problem Statement",
            "",
            f"> {problem.statement}",
            "",
            "## 2. Hypothesis Evolution",
            ""
        ]

        if hypotheses:
            for idx, h in enumerate(hypotheses, 1):
                status_badge = f"**[{h.status.value}]**"
                lines.append(f"{idx}. {status_badge} `{h.text}`")
                if h.contradicting_evidence:
                    lines.append(f"   - Contradiction: {'; '.join(h.contradicting_evidence)}")
        else:
            lines.append("No explicit hypotheses recorded.")

        lines.extend([
            "",
            "## 3. Executed Experiments and Observations",
            "",
            "| Exp ID | Tool | Backend | Status | Runtime (s) | Error Classification |",
            "| :--- | :--- | :---: | :---: | :---: | :--- |"
        ])

        for exp in reversed(experiments):
            exit_code = exp.get("exit_status", 0)
            status_symbol = "✓" if exit_code == 0 else f"✗ ({exit_code})"
            lines.append(
                f"| `{exp.get('id')}` | `{exp.get('tool_name')}` | `{exp.get('backend')}` | {status_symbol} | {exp.get('runtime_seconds', 0.0):.3f}s | {exp.get('error_classification') or 'None'} |"
            )

        lines.extend([
            "",
            "## 4. Final Mathematical Solution & Findings",
            "",
            inv_data.get("candidate_solution") or "_No final solution produced._",
            ""
        ])

        return "\n".join(lines)

    def print_trace_summary(self, investigation_id: str) -> str:
        """Summarize execution trace events sequentially."""
        events = self.traces.read_trace(investigation_id)
        if not events:
            return f"No trace found for investigation '{investigation_id}'."

        lines = [f"=== Investigation Trace: {investigation_id} ({len(events)} events) ==="]
        for e in events:
            ts = e.get("timestamp", "").split("T")[-1][:8]
            etype = e.get("event_type", "")
            payload = e.get("payload", {})

            if etype == "investigation_started":
                lines.append(f"[{ts}] START: Problem '{payload.get('problem_id')}'")
            elif etype == "hypothesis_formed":
                lines.append(f"[{ts}] HYPOTHESIS: {payload.get('hypothesis')}")
            elif etype == "hypothesis_revised":
                lines.append(f"[{ts}] REVISION: {payload.get('revised')}")
            elif etype == "counterexample_discovered":
                lines.append(f"[{ts}] CONTRADICTION: Found counterexample {payload.get('counterexample')}")
            elif etype == "tool_success":
                lines.append(f"[{ts}] TOOL '{payload.get('tool')}': Succeeded")
            elif etype == "tool_failure":
                lines.append(f"[{ts}] TOOL '{payload.get('tool')}': FAILED ({payload.get('error')})")
            elif etype == "investigation_completed":
                lines.append(f"[{ts}] COMPLETE: Status '{payload.get('status')}'")
            else:
                lines.append(f"[{ts}] EVENT: {etype}")

        return "\n".join(lines)
