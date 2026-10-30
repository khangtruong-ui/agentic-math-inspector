"""Integration test for actual remote Modal execution."""

import pytest
from mia.config import ExecutionConfig, ExperimentConfig
from mia.execution.modal_backend import ModalExecutionBackend


@pytest.mark.integration
@pytest.mark.modal
def test_live_modal_remote_execution():
    """Verify actual remote execution on Modal serverless infrastructure."""
    backend = ModalExecutionBackend(ExecutionConfig(backend="modal"))
    if not backend.is_available():
        pytest.skip("Modal is not installed or configured in the environment.")

    # 1. Execute remote smoke test
    smoke_res = backend.smoke_test()
    assert smoke_res.get("status") == "healthy"
    assert "sympy_version" in smoke_res

    # 2. Execute remote experiment
    exp_res = backend.execute(
        code="import math; print('MODAL_MATH_RESULT:', math.factorial(6))",
        experiment_config=ExperimentConfig(max_runtime_seconds=30.0)
    )
    assert exp_res.exit_status == 0
    assert "MODAL_MATH_RESULT: 720" in exp_res.stdout
    assert exp_res.execution_backend == "modal"
