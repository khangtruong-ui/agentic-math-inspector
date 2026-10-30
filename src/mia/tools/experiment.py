"""Tool E: Arbitrary code experiment execution."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field

from mia.config import ExperimentConfig
from mia.execution.base import BaseExecutionBackend


class ExecuteExperimentArgs(BaseModel):
    """Arguments for executing an arbitrary mathematical experiment script."""
    code: str = Field(description="Executable Python script to run in the experiment sandbox")
    description: str = Field(description="Brief explanation of what hypothesis this experiment tests")
    timeout_seconds: Optional[float] = Field(default=30.0, description="Execution timeout in seconds")


def execute_experiment_handler(
    code: str,
    description: str,
    timeout_seconds: Optional[float] = 30.0,
    context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Execute Python experiment code using the active execution backend (local or Modal)."""
    backend: Optional[BaseExecutionBackend] = None
    if context and "backend" in context:
        backend = context["backend"]

    if backend is None:
        from mia.config import ExecutionConfig
        from mia.execution.local import LocalExecutionBackend
        backend = LocalExecutionBackend(ExecutionConfig(backend="local"))

    exp_cfg = ExperimentConfig()
    if timeout_seconds:
        exp_cfg.max_runtime_seconds = timeout_seconds

    result = backend.execute(code=code, experiment_config=exp_cfg)

    return {
        "description": description,
        "backend": result.execution_backend,
        "exit_status": result.exit_status,
        "runtime_seconds": result.runtime_seconds,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "error_classification": result.error_classification,
        "is_interrupted": result.is_interrupted,
        "numerical_results": result.numerical_results,
        "symbolic_results": result.symbolic_results,
        "experiment_id": result.experiment_id
    }
