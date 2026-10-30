# Execution Security and Sandboxing

Treat all model-generated Python code and experiment scripts as **untrusted**.

MIA implements multi-layered security controls across both local and remote backends:

## 1. Process and Memory Isolation
- Model-generated experiment scripts are **never** executed inside the main MIA orchestrator process.
- Local executions run in dedicated temporary directories (`tempfile.TemporaryDirectory`) using an isolated subprocess: `python3 -s <script.py>`.
- Remote executions run in isolated serverless Debian containers managed by Modal (`modal_app.py`).

## 2. Environment Variable Sanitization
- By default (`isolate_environment: true`), the execution environment is stripped of all API credentials, Modal tokens, cloud keys, and sensitive environment variables matching `API_KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `AUTH`, or `CREDENTIAL`.
- Subprocesses inherit only standard system runtime paths (`PATH`, `PYTHONPATH`, `LANG`, `LC_ALL`, `TERM`).

## 3. Resource and Time Limits
- **Execution Timeouts**: Enforced via `subprocess.TimeoutExpired` locally and container timeouts remotely (default: 30.0s). Timed out runs receive exit code 124.
- **Output Flooding Protection**: Standard output and standard error are capped to `max_stdout_bytes` and `max_stderr_bytes` (default: 64 KB). Any exceeding output is truncated to prevent memory exhaustion.
- **Resource Constraints**: CPU cores and memory allocations are bounded strictly in configuration.

## 4. Network Policy
- By default, experiment code runs without internet access (`network_enabled: false`).
- All dataset acquisitions occur through controlled source adapters (`src/mia/sources/`), not within the untrusted experiment sandbox.

## Security Boundaries & Limitations
- While local subprocess execution provides isolation from the parent process, OS-level virtualization or root jail boundaries are not enforced for local execution.
- For untrusted model generation in multi-tenant environments, using the `modal` execution backend is recommended, which runs each experiment in an ephemeral, containerized VM.
