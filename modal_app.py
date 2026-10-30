"""Modal Application for Mathematical Investigation Agent (MIA).

Defines remote compute functions for isolated, scalable execution of
model-generated mathematical experiments and verification routines.
"""

from __future__ import annotations

import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, Optional
import modal

# Define standard scientific container image for mathematical experiments
mia_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sympy>=1.12",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "mpmath>=1.3.0"
    )
)

app = modal.App("mia-executor")


@app.function(
    image=mia_image,
    cpu=1.0,
    memory=1024,
    timeout=60
)
def run_remote_experiment(
    code: str,
    timeout_seconds: float = 30.0,
    max_stdout_bytes: int = 65536,
    max_stderr_bytes: int = 65536
) -> Dict[str, Any]:
    """Execute untrusted Python code inside a dedicated Modal container sandbox."""
    start_time = time.time()
    start_mono = time.monotonic()

    # Isolate environment inside container
    clean_env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1"
    }

    with tempfile.TemporaryDirectory(prefix="mia_modal_exp_") as tmp_dir:
        script_file = os.path.join(tmp_dir, "experiment.py")
        with open(script_file, "w", encoding="utf-8") as f:
            f.write(code)

        cmd = [sys.executable, "-s", script_file]

        try:
            proc = subprocess.run(
                cmd,
                cwd=tmp_dir,
                env=clean_env,
                capture_output=True,
                timeout=timeout_seconds,
                text=True
            )
            duration = time.monotonic() - start_mono
            finish_time = time.time()

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            exit_code = proc.returncode

            # Truncate if exceeding limits
            if len(stdout.encode("utf-8")) > max_stdout_bytes:
                stdout = stdout.encode("utf-8")[:max_stdout_bytes].decode("utf-8", errors="ignore") + "\n[MIA: Truncated stdout]"
            if len(stderr.encode("utf-8")) > max_stderr_bytes:
                stderr = stderr.encode("utf-8")[:max_stderr_bytes].decode("utf-8", errors="ignore") + "\n[MIA: Truncated stderr]"

            # Parse structured outputs if present
            num_res, sym_res = _extract_structured_outputs(stdout)

            error_class = None
            if exit_code != 0:
                error_class = _classify_error(stderr or stdout)

            return {
                "exit_status": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "runtime_seconds": round(duration, 4),
                "start_time": start_time,
                "finish_time": finish_time,
                "numerical_results": num_res,
                "symbolic_results": sym_res,
                "error_classification": error_class,
                "is_interrupted": False
            }

        except subprocess.TimeoutExpired as e:
            duration = time.monotonic() - start_mono
            finish_time = time.time()
            raw_out = e.stdout if isinstance(e.stdout, str) else (e.stdout or b"").decode("utf-8", errors="replace")
            raw_err = e.stderr if isinstance(e.stderr, str) else (e.stderr or b"").decode("utf-8", errors="replace")
            return {
                "exit_status": 124,
                "stdout": raw_out,
                "stderr": raw_err + f"\n[MIA: Remote execution timed out after {timeout_seconds}s]",
                "runtime_seconds": round(duration, 4),
                "start_time": start_time,
                "finish_time": finish_time,
                "numerical_results": None,
                "symbolic_results": None,
                "error_classification": "TimeoutError",
                "is_interrupted": True
            }
        except Exception as e:
            duration = time.monotonic() - start_mono
            finish_time = time.time()
            return {
                "exit_status": 1,
                "stdout": "",
                "stderr": f"Remote container execution error: {e}",
                "runtime_seconds": round(duration, 4),
                "start_time": start_time,
                "finish_time": finish_time,
                "numerical_results": None,
                "symbolic_results": None,
                "error_classification": "ContainerRuntimeError",
                "is_interrupted": True
            }


@app.function(
    image=mia_image,
    cpu=0.5,
    memory=512,
    timeout=30
)
def smoke_test() -> Dict[str, Any]:
    """Lightweight health check and dependency verification on Modal cloud."""
    import numpy as np
    import scipy as sp
    import sympy as sym

    # Compute a symbolic and numerical verification check
    x = sym.Symbol("x")
    expr = sym.diff(sym.sin(x**2), x)

    return {
        "status": "healthy",
        "python_version": sys.version,
        "sympy_version": sym.__version__,
        "numpy_version": np.__version__,
        "scipy_version": sp.__version__,
        "test_derivative": str(expr)
    }


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


@app.local_entrypoint()
def main():
    """CLI entrypoint when running directly via `modal run modal_app.py`."""
    print("Executing Modal smoke test...")
    res = smoke_test.remote()
    print("Smoke test result:", res)
