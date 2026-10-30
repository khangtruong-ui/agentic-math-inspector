"""Local execution backend using isolated subprocesses."""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from mia.config import ExecutionConfig, ExperimentConfig
from mia.execution.base import BaseExecutionBackend
from mia.models import ExperimentExecutionResult


class LocalExecutionBackend(BaseExecutionBackend):
    """Executes untrusted mathematical code in an isolated local Python subprocess."""

    SENSITIVE_ENV_KEYS = {
        "API_KEY", "SECRET", "TOKEN", "PASSWORD", "AUTH", "CREDENTIAL",
        "MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET", "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY", "GEMINI_API_KEY"
    }

    @property
    def backend_name(self) -> str:
        return "local"

    def execute(
        self,
        code: str,
        experiment_config: Optional[ExperimentConfig] = None,
        experiment_id: Optional[str] = None
    ) -> ExperimentExecutionResult:
        exp_id = experiment_id or f"exp-{uuid.uuid4().hex[:8]}"
        action_id = f"act-{uuid.uuid4().hex[:8]}"
        cfg = experiment_config or ExperimentConfig()
        timeout = min(self.exec_config.timeout_seconds, cfg.max_runtime_seconds)

        start_time_dt = datetime.datetime.now(datetime.timezone.utc)
        start_mono = time.monotonic()

        # Sanitize environment if isolate_environment is true
        env = self._build_sanitized_env()

        # Create temporary working directory for this experiment
        with tempfile.TemporaryDirectory(prefix="mia_local_exp_") as tmp_dir:
            script_path = Path(tmp_dir) / "experiment.py"
            script_path.write_text(code, encoding="utf-8")

            # Execute via current Python executable with isolated flags:
            # -I : isolate from user's site-packages and environment variables
            # -s : do not add user site directory to sys.path
            cmd = [sys.executable, "-s", str(script_path)]

            try:
                proc = subprocess.run(
                    cmd,
                    cwd=tmp_dir,
                    env=env,
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )
                duration = time.monotonic() - start_mono
                finish_time_dt = datetime.datetime.now(datetime.timezone.utc)

                stdout = self._truncate(proc.stdout, cfg.max_stdout_bytes)
                stderr = self._truncate(proc.stderr, cfg.max_stderr_bytes)
                exit_code = proc.returncode

                # Parse any structured result if printed
                numerical_res, symbolic_res = self._extract_structured_outputs(stdout)

                error_class = None
                if exit_code != 0:
                    error_class = self._classify_error(stderr or stdout)

                return ExperimentExecutionResult(
                    action_id=action_id,
                    experiment_id=exp_id,
                    input_parameters={"timeout": timeout, "backend": "local"},
                    execution_backend="local",
                    start_time=start_time_dt.isoformat(),
                    finish_time=finish_time_dt.isoformat(),
                    exit_status=exit_code,
                    stdout=stdout,
                    stderr=stderr,
                    numerical_results=numerical_res,
                    symbolic_results=symbolic_res,
                    runtime_seconds=round(duration, 4),
                    resource_usage={"cpu_seconds": round(duration, 4), "memory_mb": None},
                    error_classification=error_class,
                    artifact_references=[],
                    is_interrupted=False
                )

            except subprocess.TimeoutExpired as e:
                duration = time.monotonic() - start_mono
                finish_time_dt = datetime.datetime.now(datetime.timezone.utc)
                raw_out = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
                raw_err = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")

                return ExperimentExecutionResult(
                    action_id=action_id,
                    experiment_id=exp_id,
                    input_parameters={"timeout": timeout, "backend": "local"},
                    execution_backend="local",
                    start_time=start_time_dt.isoformat(),
                    finish_time=finish_time_dt.isoformat(),
                    exit_status=124,  # Standard UNIX timeout exit code
                    stdout=self._truncate(raw_out, cfg.max_stdout_bytes),
                    stderr=(raw_err + f"\n[MIA: Execution timed out after {timeout:.1f} seconds]").strip(),
                    numerical_results=None,
                    symbolic_results=None,
                    runtime_seconds=round(duration, 4),
                    resource_usage={"cpu_seconds": round(duration, 4)},
                    error_classification="TimeoutError",
                    artifact_references=[],
                    is_interrupted=True
                )
            except Exception as e:
                duration = time.monotonic() - start_mono
                finish_time_dt = datetime.datetime.now(datetime.timezone.utc)
                return ExperimentExecutionResult(
                    action_id=action_id,
                    experiment_id=exp_id,
                    input_parameters={"timeout": timeout, "backend": "local"},
                    execution_backend="local",
                    start_time=start_time_dt.isoformat(),
                    finish_time=finish_time_dt.isoformat(),
                    exit_status=1,
                    stdout="",
                    stderr=f"Subprocess spawn error: {e}",
                    numerical_results=None,
                    symbolic_results=None,
                    runtime_seconds=round(duration, 4),
                    resource_usage=None,
                    error_classification="SubprocessError",
                    artifact_references=[],
                    is_interrupted=True
                )

    def _build_sanitized_env(self) -> Dict[str, str]:
        if not self.exec_config.isolate_environment:
            return dict(os.environ)

        clean_env = {}
        # Keep non-sensitive environment variables
        allowed_exact = {"PATH", "PYTHONPATH", "LANG", "LC_ALL", "TERM"}
        for k, v in os.environ.items():
            if k in allowed_exact:
                clean_env[k] = v
            elif any(s in k.upper() for s in self.SENSITIVE_ENV_KEYS):
                continue
            else:
                clean_env[k] = v
        # Ensure Python does not write .pyc files into sandbox
        clean_env["PYTHONDONTWRITEBYTECODE"] = "1"
        clean_env["PYTHONUNBUFFERED"] = "1"
        return clean_env

    @staticmethod
    def _truncate(text: str, max_bytes: int) -> str:
        encoded = text.encode("utf-8")
        if len(encoded) <= max_bytes:
            return text
        truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
        return truncated + f"\n\n[MIA: Output truncated after exceeding {max_bytes} bytes]"

    @staticmethod
    def _extract_structured_outputs(stdout: str) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        num_res = None
        sym_res = None
        match = re.search(r"__MIA_RESULT__\s*=\s*(\{.*?\})", stdout, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                num_res = data.get("numerical")
                sym_res = data.get("symbolic")
            except Exception:
                pass
        return num_res, sym_res

    @staticmethod
    def _classify_error(err_text: str) -> str:
        if "ZeroDivisionError" in err_text:
            return "ZeroDivisionError"
        if "OverflowError" in err_text:
            return "OverflowError"
        if "MemoryError" in err_text:
            return "MemoryError"
        if "RecursionError" in err_text:
            return "RecursionError"
        if "SyntaxError" in err_text:
            return "SyntaxError"
        if "NameError" in err_text:
            return "NameError"
        if "TypeError" in err_text:
            return "TypeError"
        if "ValueError" in err_text:
            return "ValueError"
        return "RuntimeError"
