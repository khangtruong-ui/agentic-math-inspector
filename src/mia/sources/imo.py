"""IMO Problem Source Adapter.

Retrieves, caches, parses, and normalizes problems from the official IMO site:
https://www.imo-official.org/
"""

from __future__ import annotations

import datetime
import io
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional
import pypdf

from mia.models import ProblemRecord
from mia.sources.base import BaseProblemSource


class IMOSource(BaseProblemSource):
    """Adapter for acquiring problems from the International Mathematical Olympiad."""

    BASE_URL = "https://www.imo-official.org"
    DEFAULT_YEARS = [2024, 2023, 2022, 2021, 2020]

    def __init__(self, cache_dir: Path | str = "data/cache"):
        super().__init__(cache_dir)
        self.imo_cache_dir = self.cache_dir / "imo"
        self.imo_cache_dir.mkdir(parents=True, exist_ok=True)
        self._records: Dict[str, ProblemRecord] = {}
        self._load_all_cached()

    @property
    def name(self) -> str:
        return "imo"

    def _load_all_cached(self) -> None:
        cached_file = self.imo_cache_dir / "problems.jsonl"
        for rec in self.load_cached_records("problems.jsonl"):
            self._records[rec.problem_id] = rec

    def sync(
        self,
        years: Optional[List[int]] = None,
        limit: Optional[int] = None,
        **kwargs: Any
    ) -> List[ProblemRecord]:
        """Fetch official IMO PDFs, parse problem statements, and cache them."""
        target_years = years or self.DEFAULT_YEARS
        newly_acquired: List[ProblemRecord] = []

        for year in target_years:
            if limit and len(newly_acquired) >= limit:
                break

            pdf_url = f"{self.BASE_URL}/assets/documents/problems/{year}/{year}_eng.pdf"
            pdf_path = self.imo_cache_dir / f"{year}_eng.pdf"

            # 1. Download if not already cached
            if not pdf_path.is_file():
                try:
                    req = urllib.request.Request(
                        pdf_url,
                        headers={"User-Agent": "MIA-Math-Agent/1.0 (Research)"}
                    )
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        pdf_data = resp.read()
                    with open(pdf_path, "wb") as f:
                        f.write(pdf_data)
                except Exception as e:
                    self.record_failure(
                        identifier=f"imo-{year}",
                        reason=f"Failed to download PDF from {pdf_url}: {e}",
                        metadata={"year": year, "url": pdf_url}
                    )
                    continue

            # 2. Extract text from PDF
            try:
                reader = pypdf.PdfReader(str(pdf_path))
                full_text = "\n".join(page.extract_text() or "" for page in reader.pages)
            except Exception as e:
                self.record_failure(
                    identifier=f"imo-{year}",
                    reason=f"Failed to read PDF for year {year}: {e}",
                    metadata={"path": str(pdf_path)}
                )
                continue

            # 3. Parse individual problems (1-6)
            pattern = re.compile(
                r"Problem\s+([1-6])[\.\:\s](.*?)(?=Problem\s+[1-6][\.\:\s]|Day\s+[12]|\Z)",
                re.DOTALL | re.IGNORECASE
            )
            matches = list(pattern.finditer(full_text))

            if not matches:
                self.record_failure(
                    identifier=f"imo-{year}",
                    reason=f"No problems parsed from PDF text (extracted length: {len(full_text)})",
                    metadata={"year": year}
                )
                continue

            for m in matches:
                p_num = int(m.group(1))
                raw_statement = m.group(2).strip()
                # Clean up repeated whitespace and page headers
                clean_statement = self._normalize_statement(raw_statement)
                prob_id = f"imo-{year}-p{p_num}"

                # Infer primary mathematical subject if detectable
                subject = self._infer_subject(clean_statement)

                rec = ProblemRecord(
                    problem_id=prob_id,
                    source="imo",
                    source_url=pdf_url,
                    retrieved_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    statement=clean_statement,
                    original_statement=raw_statement,
                    subject=subject,
                    difficulty="Olympiad",
                    official_answer=None,  # IMO proof problems do not have trivial one-line answers
                    reference_solution=None,  # Separated from agent
                    license="Official IMO public competition problems",
                    metadata={"year": year, "problem_number": p_num}
                )

                if prob_id not in self._records:
                    self._records[prob_id] = rec
                    newly_acquired.append(rec)
                if limit and len(newly_acquired) >= limit:
                    break

        # Persist all cached records
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
            results = [r for r in results if r.subject and r.subject.lower() == subject.lower()]
        if difficulty:
            results = [r for r in results if r.difficulty and r.difficulty.lower() == difficulty.lower()]
        if limit:
            results = results[:limit]
        return results

    def get_problem(self, problem_id: str) -> Optional[ProblemRecord]:
        return self._records.get(problem_id)

    @staticmethod
    def _normalize_statement(text: str) -> str:
        # Remove date lines or day markers
        text = re.sub(r"^(?:Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Monday).*?\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"Language:\s*English.*?\n", "", text, flags=re.MULTILINE)
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def _infer_subject(statement: str) -> str:
        lower = statement.lower()
        if any(w in lower for w in ["triangle", "circle", "line", "angle", "tangent", "parallel", "polygon"]):
            return "geometry"
        if any(w in lower for w in ["integer", "prime", "divisible", "gcd", "modulo", "composite"]):
            return "number_theory"
        if any(w in lower for w in ["polynomial", "sequence", "function f", "real number", "inequality", "equation"]):
            return "algebra"
        if any(w in lower for w in ["game", "board", "grid", "coins", "subset", "graph", "color"]):
            return "combinatorics"
        return "general"
