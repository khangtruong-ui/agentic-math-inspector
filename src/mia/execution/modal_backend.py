"""Modal remote execution backend."""

from __future__ import annotations

import datetime
import time
import uuid
from typing import Any, Dict, Optional

from mia.config import ExecutionConfig, ExperimentConfig
from mia.execution.base import BaseExecutionBackend
from mia.models import ExperimentExecutionResult


class ModalExecutionBackend(BaseExecutionBackend):
    """Executes experiments remotely on Modal serverless infrastructure."""

    def __init__(self, exec_config: ExecutionConfig):
        super().__init__(exec_config)
        self._app = None
        self._run_fn = None
        self._smoke_fn = None
        self._init_modal()

    def _init_modal(self) -> None:
        """Dynamically import modal_app to avoid hard dependency during local unit tests."""
        import sys
        from pathlib import Path
        cwd_str = str(Path.cwd().resolve())
        if cwd_str not in sys.path:
            sys.path.insert(0, cwd_str)
        try:
            import modal_app
            self._app = modal_app.app
            self._run_fn = modal_app.run_remote_experiment
            self._smoke_fn = modal_app.smoke_test
        except ImportError:
            self._app = None
            self._run_fn = None
            self._smoke_fn = None

    @property
    def backend_name(self) -> str:
        return "modal"

    def is_available(self) -> bool:
        """Check if Modal SDK and app definitions are accessible."""
        return self._run_fn is not None

    def execute(
        self,
        code: str,
        experiment_config: Optional[ExperimentConfig] = None,
        experiment_id: Optional[str] = None
    ) -> ExperimentExecutionResult:
        exp_id = experiment_id or f"modal-exp-{uuid.uuid4().hex[:8]}"
        action_id = f"act-{uuid.uuid4().hex[:8]}"
        cfg = experiment_config or ExperimentConfig()
        timeout = min(self.exec_config.timeout_seconds, cfg.max_runtime_seconds)

        if not self.is_available():
            raise RuntimeError(
                "Modal execution backend is not available. Ensure 'modal_app.py' is in python path."
            )

        start_time_dt = datetime.datetime.now(datetime.timezone.utc)
        start_mono = time.monotonic()

        try:
            # Execute remote function on Modal
            with self._app.run():
                remote_res = self._run_fn.remote(
                    code=code,
                    timeout_seconds=timeout,
                    max_stdout_bytes=cfg.max_stdout_bytes,
                    max_stderr_bytes=cfg.max_stderr_bytes
                )

            duration = time.monotonic() - start_mono
            finish_time_dt = datetime.datetime.now(datetime.timezone.utc)

            return ExperimentExecutionResult(
                action_id=action_id,
                experiment_id=exp_id,
                input_parameters={
                    "timeout": timeout,
                    "backend": "modal",
                    "cpu": self.exec_config.cpu,
                    "memory_mb": self.exec_config.memory_mb
                },
                execution_backend="modal",
                start_time=start_time_dt.isoformat(),
                finish_time=finish_time_dt.isoformat(),
                exit_status=remote_res.get("exit_status", 0),
                stdout=remote_res.get("stdout", ""),
                stderr=remote_res.get("stderr", ""),
                numerical_results=remote_res.get("numerical_results"),
                symbolic_results=remote_res.get("symbolic_results"),
                runtime_seconds=remote_res.get("runtime_seconds", round(duration, 4)),
                resource_usage={
                    "cpu_cores": self.exec_config.cpu,
                    "memory_mb": self.exec_config.memory_mb,
                    "remote_duration_seconds": remote_res.get("runtime_seconds")
                },
                error_classification=remote_res.get("error_classification"),
                artifact_references=[],
                is_interrupted=remote_res.get("is_interrupted", False)
            )

        except Exception as e:
            duration = time.monotonic() - start_mono
            finish_time_dt = datetime.datetime.now(datetime.timezone.utc)
            return ExperimentExecutionResult(
                action_id=action_id,
                experiment_id=exp_id,
                input_parameters={"timeout": timeout, "backend": "modal"},
                execution_backend="modal",
                start_time=start_time_dt.isoformat(),
                finish_time=finish_time_dt.isoformat(),
                exit_status=1,
                stdout="",
                stderr=f"Modal remote execution failure: {e}",
                numerical_results=None,
                symbolic_results=None,
                runtime_seconds=round(duration, 4),
                resource_usage=None,
                error_classification="ModalExecutionError",
                artifact_references=[],
                is_interrupted=True
            )

    def smoke_test(self) -> Dict[str, Any]:
        """Perform a remote smoke test against Modal servers."""
        if not self.is_available() or self._smoke_fn is None:
            raise RuntimeError("Modal backend is not configured or available.")

        with self._app.run():
            res = self._smoke_fn.remote()
        return res
