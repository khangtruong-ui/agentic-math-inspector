"""MIA Configuration System.

Provides validated Pydantic models, hierarchical loading (Defaults -> YAML -> Env -> CLI),
strict key checking (rejects unknown keys), and secret redaction.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictBaseModel(BaseModel):
    """Base model that forbids unexpected fields to catch typos."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(StrictBaseModel):
    """Configuration for LLM reasoning and generation."""
    provider: Literal["openai", "anthropic", "gemini", "mock", "modal-huggingface", "huggingface"] = Field(
        default="mock",
        description="LLM provider: 'openai', 'anthropic', 'gemini', 'mock', 'modal-huggingface', or 'huggingface'"
    )
    model: str = Field(
        default="mock-investigator",
        description="Model identifier, e.g. 'Qwen/Qwen2.5-Math-1.5B-Instruct', 'gpt-4o', etc."
    )
    gpu: Optional[str] = Field(
        default="T4",
        description="GPU type for Modal Hugging Face inference (e.g. 'T4', 'L4', 'A10G')"
    )
    api_endpoint: Optional[str] = Field(
        default=None,
        description="Custom API endpoint URL for OpenAI-compatible proxies or local endpoints"
    )
    api_key_env_var: str = Field(
        default="OPENAI_API_KEY",
        description="Environment variable containing the API key"
    )
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, gt=0)
    reasoning_budget: Optional[int] = Field(
        default=None,
        description="Reasoning / thinking token budget where supported"
    )
    timeout_seconds: float = Field(default=60.0, gt=0.0)
    max_retries: int = Field(default=3, ge=0)
    max_concurrency: int = Field(default=1, gt=0)
    structured_output: bool = Field(default=True)


class AgentConfig(StrictBaseModel):
    """Configuration for the agent investigation loop."""
    max_steps: int = Field(default=15, gt=0, description="Hard cap on total investigation steps")
    max_model_calls: int = Field(default=20, gt=0, description="Hard cap on model calls")
    max_tool_calls: int = Field(default=30, gt=0, description="Hard cap on tool calls")
    max_failed_tool_calls: int = Field(default=5, gt=0, description="Max allowable consecutive failed tool calls")
    max_wall_clock_seconds: float = Field(default=300.0, gt=0.0, description="Max wall-clock investigation duration")
    max_repeated_actions: int = Field(default=3, gt=0, description="Max identical actions before forced pivot")
    max_experiments: int = Field(default=10, gt=0, description="Max code execution experiments")
    max_nonproductive_steps: int = Field(default=4, gt=0, description="Steps with no new information before termination")
    max_history_messages: int = Field(default=50, gt=0)
    state_summarization_threshold: int = Field(default=10, gt=0)
    persistent_memory: bool = Field(default=True, description="Enable SQLite investigation persistence")
    parallel_experiments: bool = Field(default=False)
    allow_hypothesis_revision: bool = Field(default=True)
    required_verification_level: Literal["LEVEL_1", "LEVEL_2", "LEVEL_3"] = Field(
        default="LEVEL_2",
        description="Target verification level before concluding success"
    )


class ExperimentConfig(StrictBaseModel):
    """Configuration for mathematical experiment execution."""
    random_seed: int = Field(default=42)
    max_runtime_seconds: float = Field(default=30.0, gt=0.0)
    max_test_cases: int = Field(default=100, gt=0)
    numerical_precision: int = Field(default=50, gt=0, description="Precision digits for exact/mpmath computations")
    numerical_tolerance: float = Field(default=1e-7, gt=0.0)
    max_array_elements: int = Field(default=10000, gt=0)
    max_stdout_bytes: int = Field(default=65536, gt=0)
    max_stderr_bytes: int = Field(default=65536, gt=0)
    max_artifact_bytes: int = Field(default=1048576, gt=0)
    enable_plotting: bool = Field(default=False)
    enable_symbolic: bool = Field(default=True)
    allow_optional_dependencies: bool = Field(default=False)


class ExecutionConfig(StrictBaseModel):
    """Configuration for execution backend (local or Modal)."""
    backend: Literal["local", "modal"] = Field(
        default="local",
        description="Execution backend: 'local' (isolated subprocess) or 'modal' (remote cloud function)"
    )
    modal_app_name: str = Field(default="mia-executor")
    modal_function_name: str = Field(default="run_remote_experiment")
    cpu: float = Field(default=1.0, gt=0.0, description="CPU allocation cores")
    memory_mb: int = Field(default=1024, ge=128, description="Memory in megabytes")
    gpu: Optional[str] = Field(default=None, description="GPU type if needed; None for CPU execution")
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_parallel: int = Field(default=4, gt=0)
    dependency_policy: Literal["preconfigured", "strict", "custom"] = Field(default="preconfigured")
    artifact_storage_dir: str = Field(default="runs/artifacts")
    network_enabled: bool = Field(default=False, description="Disable network during experiment execution")
    isolate_environment: bool = Field(default=True, description="Strip sensitive credentials from subprocess environment")


class EvaluationConfig(StrictBaseModel):
    """Configuration for benchmark evaluations."""
    benchmark_source: Literal["imo", "hendrycks_math", "controlled"] = Field(
        default="controlled",
        description="Benchmark source to evaluate on"
    )
    split: Optional[str] = Field(default=None)
    num_problems: int = Field(default=5, gt=0)
    random_seed: int = Field(default=42)
    time_budget_per_problem: float = Field(default=120.0, gt=0.0)
    max_total_budget: float = Field(default=600.0, gt=0.0)
    verification_requirements: Literal["LEVEL_1", "LEVEL_2", "LEVEL_3"] = Field(default="LEVEL_1")
    reference_answers_for_evaluator: bool = Field(default=True)
    output_dir: str = Field(default="runs/evaluations")
    baseline_configs: List[str] = Field(default_factory=lambda: ["A", "B", "C"])
    retries_allowed: bool = Field(default=False)


class AppConfig(StrictBaseModel):
    """Root configuration object for MIA."""
    model: ModelConfig = Field(default_factory=ModelConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    database_url: str = Field(default="sqlite:///runs/mia.db")
    trace_dir: str = Field(default="runs/traces")

    def redact_secrets(self) -> Dict[str, Any]:
        """Produce a dictionary representation with sensitive secrets redacted."""
        raw = self.model_dump()
        return _redact_dict(raw)


def _redact_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    secret_substrings = ("key", "secret", "token", "password", "auth", "credential")
    redacted = {}
    for k, v in d.items():
        if isinstance(v, dict):
            redacted[k] = _redact_dict(v)
        elif isinstance(k, str) and any(s in k.lower() for s in secret_substrings) and isinstance(v, str):
            # Check if this is the name of an env var rather than the secret itself
            if k.endswith("_env_var"):
                redacted[k] = v
            else:
                redacted[k] = "***REDACTED***"
        else:
            redacted[k] = v
    return redacted


def load_config(
    config_path: Optional[str | Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None
) -> AppConfig:
    """Load effective configuration with precedence: CLI > Env > YAML > Defaults.

    Precedence order:
    1. CLI overrides
    2. Environment variables (e.g. MIA__MODEL__PROVIDER or MIA_MODEL__PROVIDER)
    3. YAML config file
    4. Documented defaults
    """
    raw_data: Dict[str, Any] = {}

    # 1. Load from YAML file if exists
    path_to_try = config_path or os.environ.get("MIA_CONFIG_FILE") or "config.yaml"
    p = Path(path_to_try)
    if p.is_file():
        with open(p, "r", encoding="utf-8") as f:
            yaml_content = yaml.safe_load(f)
            if isinstance(yaml_content, dict):
                raw_data = yaml_content
            elif yaml_content is not None:
                raise ValueError(f"Configuration file {p} must contain a YAML mapping/dictionary.")

    # 2. Apply environment variable overrides
    # Matches MIA__<SECTION>__<KEY> or MIA_<SECTION>_<KEY>
    for env_k, env_v in os.environ.items():
        if env_k.startswith("MIA__"):
            parts = env_k[5:].lower().split("__")
            _deep_set(raw_data, parts, _coerce_env_val(env_v))
        elif env_k.startswith("MIA_") and "__" not in env_k:
            # Single underscore format e.g. MIA_DATABASE_URL
            key = env_k[4:].lower()
            if key in ("database_url", "trace_dir"):
                raw_data[key] = env_v

    # 3. Apply CLI overrides
    if cli_overrides:
        for k, v in cli_overrides.items():
            if v is not None:
                if "." in k:
                    parts = k.lower().split(".")
                    _deep_set(raw_data, parts, v)
                elif "__" in k:
                    parts = k.lower().split("__")
                    _deep_set(raw_data, parts, v)
                else:
                    raw_data[k] = v

    # 4. Validate and construct AppConfig
    return AppConfig.model_validate(raw_data)


def _deep_set(d: Dict[str, Any], path: List[str], val: Any) -> None:
    current = d
    for step in path[:-1]:
        if step not in current or not isinstance(current[step], dict):
            current[step] = {}
        current = current[step]
    current[path[-1]] = val


def _coerce_env_val(val: str) -> Any:
    if val.lower() in ("true", "yes", "1"):
        return True
    if val.lower() in ("false", "no", "0"):
        return False
    try:
        if "." in val:
            return float(val)
        return int(val)
    except ValueError:
        return val
