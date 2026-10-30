# Configuration Guide

MIA uses a validated hierarchical configuration system built on Pydantic v2.

## Precedence Order

Settings are resolved in the following order of priority:
1. **Explicit CLI overrides** (e.g. `--backend modal`, `--max-steps 10`)
2. **Environment variables** (e.g. `MIA__MODEL__PROVIDER="openai"`)
3. **YAML configuration file** (e.g. `config.yaml` or specified via `--config-file`)
4. **Documented defaults**

Unknown configuration keys are rejected strictly by default to catch typos.

## Configuration Schema

### 1. Model Settings (`model`)
- `provider`: `"mock" | "modal-huggingface" | "huggingface" | "openai" | "anthropic" | "gemini"`
- `model`: Model name or Hugging Face repository identifier (e.g. `"Qwen/Qwen2.5-Math-1.5B-Instruct"`, `"Qwen/Qwen2.5-7B-Instruct"`, `"gpt-4o"`, `"mock-investigator"`)
- `gpu`: GPU accelerator type on Modal when using `modal-huggingface` (default: `"T4"`, also supports `"L4"`, `"A10G"`)
- `api_endpoint`: Optional custom API endpoint (e.g. for Ollama or vLLM proxy)
- `api_key_env_var`: Name of environment variable holding the API key (default: `"OPENAI_API_KEY"`)
- `temperature`: Sampling temperature (0.0 to 2.0, default 0.2)
- `max_tokens`: Completion token budget (default 4096)
- `reasoning_budget`: Optional reasoning/thinking token budget for supported models
- `timeout_seconds`: HTTP request timeout (default 60.0)
- `max_retries`: Max retry attempts on rate limits (default 3)

### 2. Agent Settings (`agent`)
- `max_steps`: Total investigation steps budget (default 15)
- `max_model_calls`: Hard limit on LLM invocations (default 20)
- `max_tool_calls`: Hard limit on tool calls (default 30)
- `max_failed_tool_calls`: Consecutive failed tool calls before halting (default 5)
- `max_wall_clock_seconds`: Max wall-clock execution time (default 300.0)
- `max_repeated_actions`: Consecutive identical actions before forced strategy pivot (default 3)
- `max_experiments`: Max executable code experiments (default 10)
- `max_nonproductive_steps`: Steps without new mathematical evidence before halting (default 4)
- `persistent_memory`: Persist state to SQLite (default true)
- `allow_hypothesis_revision`: Allow agent to revise refuted hypotheses (default true)
- `required_verification_level`: Target verification tier (`"LEVEL_1"`, `"LEVEL_2"`, `"LEVEL_3"`)

### 3. Execution Settings (`execution`)
- `backend`: `"local"` (isolated subprocess) or `"modal"` (remote serverless function)
- `modal_app_name`: Modal App name (default `"mia-executor"`)
- `modal_function_name`: Remote experiment function name (default `"run_remote_experiment"`)
- `cpu`: CPU cores allocated per experiment (default 1.0)
- `memory_mb`: Memory in MB (default 1024)
- `gpu`: GPU accelerator type (default null; CPU is standard for ordinary mathematics)
- `timeout_seconds`: Per-experiment timeout (default 30.0)
- `isolate_environment`: Strip API keys and secrets from subprocess environment (default true)

### 4. Experiment Settings (`experiment`)
- `random_seed`: Seed for reproducible test-case generation (default 42)
- `numerical_precision`: Decimal places of precision for mpmath (default 50)
- `numerical_tolerance`: Numerical equality tolerance (default 1e-7)
- `max_stdout_bytes`: Standard output cap before truncation (default 65536)
- `max_stderr_bytes`: Standard error cap before truncation (default 65536)

## Validating Configuration

To validate your configuration and view effective values with secrets redacted:

```bash
mia config validate
```
