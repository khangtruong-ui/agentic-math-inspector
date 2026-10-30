"""Unit tests for local execution backend, timeouts, truncation, and isolation."""

import time
import pytest
from mia.config import ExecutionConfig, ExperimentConfig
from mia.execution.local import LocalExecutionBackend


def test_local_execution_success():
    """Verify normal Python script execution in isolated subprocess."""
    backend = LocalExecutionBackend(ExecutionConfig(backend="local"))
    res = backend.execute("x = 10; y = 20; print('SUM:', x + y)")
    assert res.exit_status == 0
    assert "SUM: 30" in res.stdout
    assert res.is_interrupted is False
    assert res.runtime_seconds >= 0.0


def test_local_execution_timeout():
    """Verify that infinite loops or slow code are terminated at configured timeout."""
    backend = LocalExecutionBackend(ExecutionConfig(backend="local", timeout_seconds=1.0))
    cfg = ExperimentConfig(max_runtime_seconds=1.0)
    res = backend.execute("import time; time.sleep(10)", experiment_config=cfg)

    assert res.exit_status == 124
    assert res.is_interrupted is True
    assert res.error_classification == "TimeoutError"
    assert "timed out" in res.stderr.lower()


def test_local_execution_error_classification():
    """Verify error classification for division by zero and syntax errors."""
    backend = LocalExecutionBackend(ExecutionConfig(backend="local"))

    # ZeroDivisionError
    res1 = backend.execute("print(1 / 0)")
    assert res1.exit_status != 0
    assert res1.error_classification == "ZeroDivisionError"

    # SyntaxError
    res2 = backend.execute("def invalid_syntax(:")
    assert res2.exit_status != 0
    assert res2.error_classification == "SyntaxError"


def test_output_truncation():
    """Subprocess stdout exceeding max bytes must be safely truncated."""
    backend = LocalExecutionBackend(ExecutionConfig(backend="local"))
    cfg = ExperimentConfig(max_stdout_bytes=200)

    res = backend.execute("print('A' * 5000)", experiment_config=cfg)
    assert len(res.stdout) < 1000
    assert "truncated" in res.stdout.lower()


def test_environment_isolation(monkeypatch):
    """Subprocess environment must not inherit sensitive API keys or tokens."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-123456789")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "modal-secret-abc")

    backend = LocalExecutionBackend(ExecutionConfig(backend="local", isolate_environment=True))
    code = (
        "import os\n"
        "keys = [k for k in os.environ if 'API_KEY' in k or 'MODAL' in k or 'SECRET' in k]\n"
        "print('LEAKED_KEYS:', keys)\n"
    )
    res = backend.execute(code)
    assert res.exit_status == 0
    assert "LEAKED_KEYS: []" in res.stdout
