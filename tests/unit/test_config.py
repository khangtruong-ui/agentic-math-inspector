"""Unit tests for configuration validation, precedence, and secret redaction."""

import os
from pathlib import Path
import pytest
from pydantic import ValidationError
import yaml

from mia.config import AppConfig, load_config, _redact_dict


def test_default_config():
    """Verify default configuration values."""
    config = AppConfig()
    assert config.model.provider == "mock"
    assert config.agent.max_steps == 15
    assert config.agent.allow_hypothesis_revision is True
    assert config.execution.backend == "local"
    assert config.experiment.numerical_precision == 50
    assert config.evaluation.benchmark_source == "controlled"


def test_unknown_keys_rejected():
    """Configuration must strictly forbid unknown keys to catch typos."""
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"unexpected_key_123": True})

    with pytest.raises(ValidationError):
        AppConfig.model_validate({"agent": {"unknown_agent_setting": 999}})


def test_secret_redaction():
    """Verify that credentials and secrets are masked in dict representation."""
    raw = {
        "model": {
            "api_key_env_var": "OPENAI_API_KEY",
            "secret_token": "super_secret_12345",
            "normal_setting": "safe_value"
        },
        "database_url": "sqlite:///runs/mia.db"
    }
    redacted = _redact_dict(raw)
    assert redacted["model"]["api_key_env_var"] == "OPENAI_API_KEY"  # env var name preserved
    assert redacted["model"]["secret_token"] == "***REDACTED***"
    assert redacted["model"]["normal_setting"] == "safe_value"


def test_environment_variable_overrides(monkeypatch):
    """Environment variables prefixed with MIA__ should override defaults."""
    monkeypatch.setenv("MIA__MODEL__PROVIDER", "openai")
    monkeypatch.setenv("MIA__AGENT__MAX_STEPS", "42")
    monkeypatch.setenv("MIA__EXECUTION__BACKEND", "modal")

    cfg = load_config()
    assert cfg.model.provider == "openai"
    assert cfg.agent.max_steps == 42
    assert cfg.execution.backend == "modal"


def test_cli_overrides():
    """Explicit CLI overrides must take highest precedence."""
    cli_overrides = {
        "model.provider": "gemini",
        "agent.max_steps": 7,
        "execution.backend": "local"
    }
    cfg = load_config(cli_overrides=cli_overrides)
    assert cfg.model.provider == "gemini"
    assert cfg.agent.max_steps == 7
    assert cfg.execution.backend == "local"


def test_yaml_config_loading(tmp_path):
    """YAML file configuration should be correctly parsed and validated."""
    cfg_file = tmp_path / "custom_config.yaml"
    data = {
        "model": {"provider": "anthropic", "temperature": 0.7},
        "agent": {"max_steps": 25}
    }
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")

    cfg = load_config(config_path=cfg_file)
    assert cfg.model.provider == "anthropic"
    assert cfg.model.temperature == 0.7
    assert cfg.agent.max_steps == 25
    assert cfg.execution.backend == "local"  # Default remains
