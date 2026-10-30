"""Base interface and data contracts for execution backends."""

from __future__ import annotations

import abc
import datetime
import uuid
from typing import Any, Dict, Optional
from mia.config import ExperimentConfig, ExecutionConfig
from mia.models import ExperimentExecutionResult


class BaseExecutionBackend(abc.ABC):
    """Abstract interface for experiment execution backends."""

    def __init__(self, exec_config: ExecutionConfig):
        self.exec_config = exec_config

    @property
    @abc.abstractmethod
    def backend_name(self) -> str:
        """Name of the execution backend."""
        pass

    @abc.abstractmethod
    def execute(
        self,
        code: str,
        experiment_config: Optional[ExperimentConfig] = None,
        experiment_id: Optional[str] = None
    ) -> ExperimentExecutionResult:
        """Execute experiment code and return structured result."""
        pass

    def estimate_cost(self, num_executions: int, avg_duration_seconds: float = 5.0) -> Dict[str, Any]:
        """Estimate execution resource requirements and theoretical compute bounds."""
        total_cpu_seconds = num_executions * avg_duration_seconds * self.exec_config.cpu
        total_gb_seconds = num_executions * avg_duration_seconds * (self.exec_config.memory_mb / 1024.0)
        return {
            "backend": self.backend_name,
            "planned_executions": num_executions,
            "estimated_duration_seconds": num_executions * avg_duration_seconds,
            "total_cpu_seconds": total_cpu_seconds,
            "total_gb_seconds": total_gb_seconds,
            "configured_cpu": self.exec_config.cpu,
            "configured_memory_mb": self.exec_config.memory_mb,
            "gpu": self.exec_config.gpu,
            "estimated_cost_usd": None  # Exact billing depends on user plan / cloud provider rates
        }
