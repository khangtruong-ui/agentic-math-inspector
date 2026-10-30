"""Base abstractions for mathematical problem acquisition."""

from __future__ import annotations

import abc
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from mia.models import ProblemRecord


class ExtractionFailure(Exception):
    """Raised or recorded when a problem document cannot be parsed."""
    def __init__(self, source: str, identifier: str, reason: str):
        super().__init__(f"Extraction failure in {source} ({identifier}): {reason}")
        self.source = source
        self.identifier = identifier
        self.reason = reason


class BaseProblemSource(abc.ABC):
    """Abstract interface for benchmark and problem acquisition."""

    def __init__(self, cache_dir: Path | str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.failures: List[Dict[str, Any]] = []

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the problem source."""
        pass

    @abc.abstractmethod
    def sync(self, limit: Optional[int] = None, **kwargs) -> List[ProblemRecord]:
        """Fetch or update benchmark data from source, returning newly acquired records."""
        pass

    @abc.abstractmethod
    def list_problems(
        self,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ProblemRecord]:
        """List locally available/cached problems matching criteria."""
        pass

    @abc.abstractmethod
    def get_problem(self, problem_id: str) -> Optional[ProblemRecord]:
        """Retrieve a specific problem by its unique identifier."""
        pass

    def record_failure(self, identifier: str, reason: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Explicitly record an extraction or retrieval failure."""
        entry = {
            "source": self.name,
            "identifier": identifier,
            "reason": reason,
            "metadata": metadata or {}
        }
        self.failures.append(entry)

    def save_records(self, records: List[ProblemRecord], filename: str) -> Path:
        """Save normalized records to JSONL in the cache directory."""
        target = self.cache_dir / self.name / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            for r in records:
                f.write(r.model_dump_json() + "\n")
        return target

    def load_cached_records(self, filename: str) -> List[ProblemRecord]:
        """Load records from a cached JSONL file."""
        target = self.cache_dir / self.name / filename
        if not target.is_file():
            return []
        records = []
        with open(target, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(ProblemRecord.model_validate_json(line))
                    except Exception:
                        pass
        return records
