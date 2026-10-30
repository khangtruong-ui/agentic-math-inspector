"""Append-only JSONL execution traces for investigation observability."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


class TraceLogger:
    """Logs structured investigation events to append-only JSONL files."""

    def __init__(self, trace_dir: str = "runs/traces"):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)

    def log_event(
        self,
        investigation_id: str,
        event_type: str,
        payload: Dict[str, Any]
    ) -> None:
        """Append an event to the investigation's trace file."""
        file_path = self.trace_dir / f"{investigation_id}.jsonl"
        entry = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "investigation_id": investigation_id,
            "event_type": event_type,
            "payload": payload
        }
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def read_trace(self, investigation_id: str) -> List[Dict[str, Any]]:
        """Read all events from an investigation trace."""
        file_path = self.trace_dir / f"{investigation_id}.jsonl"
        if not file_path.is_file():
            return []
        events = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        pass
        return events
