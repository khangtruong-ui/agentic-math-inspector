"""Hendrycks MATH Dataset Source Adapter.

Retrieves and normalizes problems from the Hugging Face dataset:
https://huggingface.co/datasets/EleutherAI/hendrycks_math
Supports bounded pagination and streaming retrieval without downloading full multi-gigabyte archives.
"""

from __future__ import annotations

import datetime
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from mia.models import ProblemRecord
from mia.sources.base import BaseProblemSource


class HendrycksMathSource(BaseProblemSource):
    """Adapter for EleutherAI/hendrycks_math via HuggingFace datasets-server API."""

    BASE_API = "https://datasets-server.huggingface.co/rows"
    DATASET_NAME = "EleutherAI/hendrycks_math"
    DEFAULT_CONFIGS = [
        "algebra",
        "number_theory",
        "precalculus",
        "intermediate_algebra",
        "geometry",
        "counting_and_probability",
        "prealgebra"
    ]

    def __init__(self, cache_dir: Path | str = "data/cache"):
        super().__init__(cache_dir)
        self.math_cache_dir = self.cache_dir / "hendrycks_math"
        self.math_cache_dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, ProblemRecord] = {}
        self._load_all_cached()

    @property
    def name(self) -> str:
        return "hendrycks_math"

    def _load_all_cached(self) -> None:
        for rec in self.load_cached_records("problems.jsonl"):
            self._records[rec.problem_id] = rec

    def sync(
        self,
        configs: Optional[List[str]] = None,
        split: str = "train",
        limit_per_config: int = 5,
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> List[ProblemRecord]:
        """Fetch bounded subset of problems from Hugging Face datasets-server."""
        target_configs = configs or self.DEFAULT_CONFIGS
        newly_acquired: List[ProblemRecord] = []

        for config in target_configs:
            if limit and len(newly_acquired) >= limit:
                break

            params = {
                "dataset": self.DATASET_NAME,
                "config": config,
                "split": split,
                "offset": 0,
                "length": limit_per_config
            }
            url = f"{self.BASE_API}?{urllib.parse.urlencode(params)}"

            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "MIA-Math-Agent/1.0 (Research)"}
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
            except Exception as e:
                self.record_failure(
                    identifier=f"hendrycks-{config}-{split}",
                    reason=f"Failed to query HF datasets API at {url}: {e}",
                    metadata={"config": config, "split": split}
                )
                continue

            rows = data.get("rows", [])
            for row_entry in rows:
                row = row_entry.get("row", {})
                raw_prob = row.get("problem", "")
                if not raw_prob:
                    continue

                row_idx = row_entry.get("row_idx", 0)
                prob_id = f"hendrycks-{config}-{split}-{row_idx}"
                solution = row.get("solution", "")
                official_answer = self._extract_boxed_answer(solution)
                level = row.get("level", "")
                subject = row.get("type", config)

                rec = ProblemRecord(
                    problem_id=prob_id,
                    source="hendrycks_math",
                    source_url=f"https://huggingface.co/datasets/{self.DATASET_NAME}",
                    retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    statement=raw_prob.strip(),
                    original_statement=raw_prob,
                    subject=subject,
                    difficulty=level,
                    official_answer=official_answer,
                    reference_solution=solution,
                    license="MIT",
                    metadata={
                        "config": config,
                        "split": split,
                        "row_idx": row_idx,
                        "level": level
                    }
                )

                if prob_id not in self._records:
                    self._records[prob_id] = rec
                    newly_acquired.append(rec)

                if limit and len(newly_acquired) >= limit:
                    break

        self.save_records(list(self._records.values()), "problems.jsonl")
        return newly_acquired

    def list_problems(
        self,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[ProblemRecord]:
        results = list(self._records.values())
        if subject:
            results = [r for r in results if r.subject and subject.lower() in r.subject.lower()]
        if difficulty:
            results = [r for r in results if r.difficulty and r.difficulty.lower() == difficulty.lower()]
        if limit:
            results = results[:limit]
        return results

    def get_problem(self, problem_id: str) -> Optional[ProblemRecord]:
        return self._records.get(problem_id)

    @staticmethod
    def _extract_boxed_answer(solution_text: str) -> Optional[str]:
        """Extract answer from LaTeX \\boxed{...} if present."""
        if not solution_text:
            return None
        matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", solution_text)
        if matches:
            return matches[-1].strip()
        return None
