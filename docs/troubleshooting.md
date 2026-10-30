# Troubleshooting and FAQ

## Common Issues and Solutions

### 1. `Missing required API key for provider ...`
- **Cause**: An external model provider (`openai`, `anthropic`, or `gemini`) was configured, but the corresponding environment variable (e.g. `OPENAI_API_KEY`) is not set.
- **Solution**: Set the environment variable in your shell:
  ```bash
  export OPENAI_API_KEY="sk-..."
  ```
  Or run offline demonstration mode using `--demo` or setting `model.provider: "mock"`.

### 2. `Modal backend is not configured or available`
- **Cause**: Modal credentials or token are not configured locally.
- **Solution**: Ensure the `modal` package is installed and set your token:
  ```bash
  modal token set --token-id <ID> --token-secret <SECRET>
  ```
  Test remote connectivity with:
  ```bash
  mia modal smoke-test
  ```

### 3. `Execution timed out after N seconds`
- **Cause**: A generated experiment script ran an expensive computation or infinite loop.
- **Solution**: MIA automatically catches timeouts, assigns exit code 124, and informs the agent to refine or bound its algorithm. You can adjust limits in `config.yaml` under `experiment.max_runtime_seconds`.

### 4. `Repetition detected: Tool ... invoked with identical arguments`
- **Cause**: The agent repeated the same failing action multiple times.
- **Solution**: MIA detects repeated identical actions (`agent.max_repeated_actions`) and forces the agent to pivot to alternative mathematical strategies.

### 5. `ModuleNotFoundError: No module named 'mia'`
- **Cause**: Package not installed in editable mode.
- **Solution**: Run:
  ```bash
  pip install -e .
  ```
