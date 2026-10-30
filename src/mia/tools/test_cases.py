"""Tool C: Test-case generation for mathematical hypotheses."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class GenerateTestCasesArgs(BaseModel):
    """Arguments for generating mathematical test cases."""
    domain: Literal["positive_integers", "integers", "reals", "rationals", "polynomials", "pairs"] = Field(
        default="positive_integers",
        description="Domain of test inputs to generate"
    )
    count: int = Field(default=20, ge=1, le=200, description="Total number of test cases to generate")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    include_boundaries: bool = Field(default=True, description="Whether to include canonical boundary values")
    min_val: float = Field(default=1.0, description="Minimum range bound")
    max_val: float = Field(default=100.0, description="Maximum range bound")


def generate_test_cases_handler(
    domain: str = "positive_integers",
    count: int = 20,
    seed: int = 42,
    include_boundaries: bool = True,
    min_val: float = 1.0,
    max_val: float = 100.0
) -> Dict[str, Any]:
    """Generate deterministic boundary and randomized test inputs."""
    rng = random.Random(seed)
    boundary_cases: List[Any] = []
    random_cases: List[Any] = []

    if domain == "positive_integers":
        if include_boundaries:
            boundary_cases = [1, 2, 3, 4, 5, 10, 40, 41, int(max_val)]
        needed = max(0, count - len(boundary_cases))
        random_cases = [rng.randint(max(1, int(min_val)), int(max_val)) for _ in range(needed)]

    elif domain == "integers":
        if include_boundaries:
            boundary_cases = [0, 1, -1, 2, -2, int(min_val), int(max_val)]
        needed = max(0, count - len(boundary_cases))
        random_cases = [rng.randint(int(min_val), int(max_val)) for _ in range(needed)]

    elif domain == "reals":
        if include_boundaries:
            boundary_cases = [0.0, 1.0, -1.0, 0.5, -0.5, 1e-5, -1e-5, min_val, max_val]
        needed = max(0, count - len(boundary_cases))
        random_cases = [round(rng.uniform(min_val, max_val), 6) for _ in range(needed)]

    elif domain == "rationals":
        if include_boundaries:
            boundary_cases = ["0/1", "1/1", "-1/1", "1/2", "-1/2", "2/3", "3/2"]
        needed = max(0, count - len(boundary_cases))
        for _ in range(needed):
            den = rng.randint(1, 20)
            num = rng.randint(int(min_val), int(max_val))
            random_cases.append(f"{num}/{den}")

    elif domain == "pairs":
        if include_boundaries:
            boundary_cases = [(0, 0), (1, 1), (-1, -1), (1, -1), (0, 1), (1, 0), (2, 3), (3, 2)]
        needed = max(0, count - len(boundary_cases))
        for _ in range(needed):
            random_cases.append((
                rng.randint(int(min_val), int(max_val)),
                rng.randint(int(min_val), int(max_val))
            ))

    elif domain == "polynomials":
        # Simple degree 1 to 3 polynomials
        boundary_cases = ["x", "x^2", "x^2 - 1", "x^3 - x"]
        needed = max(0, count - len(boundary_cases))
        for _ in range(needed):
            a = rng.randint(-5, 5)
            b = rng.randint(-10, 10)
            random_cases.append(f"{a}*x^2 + {b}*x + 1")

    # Deduplicate while preserving order
    all_cases = []
    seen = set()
    for item in (boundary_cases + random_cases):
        key = str(item)
        if key not in seen:
            seen.add(key)
            all_cases.append(item)

    return {
        "domain": domain,
        "seed": seed,
        "count_generated": len(all_cases),
        "boundary_cases": boundary_cases,
        "random_cases": random_cases,
        "test_cases": all_cases[:count]
    }
