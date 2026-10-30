"""Problem sources package for MIA."""

from __future__ import annotations

from typing import Dict, Optional, Type
from mia.sources.base import BaseProblemSource, ExtractionFailure
from mia.sources.imo import IMOSource
from mia.sources.hendrycks_math import HendrycksMathSource
from mia.sources.controlled import ControlledProblemSource

SOURCES: Dict[str, Type[BaseProblemSource]] = {
    "imo": IMOSource,
    "hendrycks_math": HendrycksMathSource,
    "controlled": ControlledProblemSource,
}


def get_source(name: str, cache_dir: str = "data/cache") -> BaseProblemSource:
    """Instantiate a problem source by name."""
    if name not in SOURCES:
        raise ValueError(
            f"Unknown problem source '{name}'. Available sources: {list(SOURCES.keys())}"
        )
    return SOURCES[name](cache_dir=cache_dir)


__all__ = [
    "BaseProblemSource",
    "ExtractionFailure",
    "IMOSource",
    "HendrycksMathSource",
    "ControlledProblemSource",
    "SOURCES",
    "get_source",
]
