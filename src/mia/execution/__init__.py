"""Execution backend package for MIA."""

from __future__ import annotations

from mia.config import ExecutionConfig
from mia.execution.base import BaseExecutionBackend
from mia.execution.local import LocalExecutionBackend
from mia.execution.modal_backend import ModalExecutionBackend


def get_execution_backend(config: ExecutionConfig) -> BaseExecutionBackend:
    """Instantiate the configured execution backend."""
    if config.backend == "local":
        return LocalExecutionBackend(config)
    elif config.backend == "modal":
        return ModalExecutionBackend(config)
    else:
        raise ValueError(f"Unknown execution backend: '{config.backend}'. Must be 'local' or 'modal'.")


__all__ = [
    "BaseExecutionBackend",
    "LocalExecutionBackend",
    "ModalExecutionBackend",
    "get_execution_backend",
]
