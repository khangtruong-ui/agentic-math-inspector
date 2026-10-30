"""Baseline configurations for fair experimental comparison."""

from __future__ import annotations

import copy
from typing import Dict, Tuple
from mia.agent.loop import MathematicalAgent
from mia.config import AppConfig
from mia.execution import get_execution_backend
from mia.llm.client import get_model_client
from mia.storage.experiment_store import ExperimentStore
from mia.storage.traces import TraceLogger
from mia.tools.registry import ToolRegistry
from mia.tools import create_default_registry


def build_agent_for_config(
    config_name: str,
    base_config: AppConfig
) -> MathematicalAgent:
    """Instantiate agent configured according to evaluation experimental conditions A, B, or C."""
    cfg = copy.deepcopy(base_config)
    client = get_model_client(cfg.model)
    backend = get_execution_backend(cfg.execution)
    store = ExperimentStore(cfg.database_url)
    traces = TraceLogger(cfg.trace_dir)

    if config_name == "A":
        # Configuration A: Reasoning-only baseline (no executable investigation tools)
        empty_registry = ToolRegistry()
        cfg.agent.max_steps = 1
        return MathematicalAgent(
            config=cfg,
            model_client=client,
            tool_registry=empty_registry,
            execution_backend=backend,
            store=store,
            trace_logger=traces
        )

    elif config_name == "B":
        # Configuration B: Tool-enabled agent (stateless, persistent memory disabled)
        cfg.agent.persistent_memory = False
        full_registry = create_default_registry()
        return MathematicalAgent(
            config=cfg,
            model_client=client,
            tool_registry=full_registry,
            execution_backend=backend,
            store=store,
            trace_logger=traces
        )

    elif config_name == "C":
        # Configuration C: Full tool-enabled agent with persistent state and adaptive revision
        cfg.agent.persistent_memory = True
        cfg.agent.allow_hypothesis_revision = True
        full_registry = create_default_registry()
        return MathematicalAgent(
            config=cfg,
            model_client=client,
            tool_registry=full_registry,
            execution_backend=backend,
            store=store,
            trace_logger=traces
        )

    else:
        raise ValueError(f"Unknown configuration '{config_name}'. Supported: 'A', 'B', 'C'.")
