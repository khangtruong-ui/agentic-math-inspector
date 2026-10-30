# Mathematical Investigation Agent (MIA)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Modal](https://img.shields.io/badge/Execution-Local%20%7C%20Modal-green.svg)](https://modal.com)

**MIA** is a configurable research agent that investigates mathematical problems using LLM reasoning, executable experiments, counterexample search, and mathematical verification.

Rather than relying purely on LLM generation or fixed pipelines, MIA implements an adaptive state machine where the agent tests hypotheses, gathers computational and symbolic evidence, recovers from contradictions, and independently verifies its conclusions.

---

## The Investigation Cycle

For each mathematical task, MIA executes an explicit 10-phase cycle:

```
  1. Receive Problem Statement (Sanitized Agent View)
         ↓
  2. Formalize Domain, Goals & Definitions
         ↓
  3. Formulate Candidate Hypothesis / Conjecture
         ↓
  4. Identify Potential Failure Modes & Uncertainties
         ↓
  5. Select & Execute Tool / Experiment (Local Subprocess or Modal Cloud)
         ↓
  6. Observe Actual Structured Execution Results
         ↓
  7. Detect Contradictions & Adaptively Revise Hypothesis
         ↓
  8. Synthesize Final Mathematical Solution
         ↓
  9. Independently Verify Solution (Level 1 / 2 / 3)
         ↓
 10. Generate Report Distinguishing Proven Facts, Evidence & Limitations
```

---

## Key Features

- **Adaptive Control Loop**: Dynamic tool selection and hypothesis revision based on empirical feedback and counterexample discovery.
- **Typed Tool Registry**: Strongly-typed tools with Pydantic validation:
  - `symbolic_compute`: SymPy differentiation, limits, exact arithmetic, equation solving, polynomial roots.
  - `numerical_evaluate`: High-precision evaluation (`mpmath`) with anomaly and domain violation detection.
  - `generate_test_cases`: Boundary values and deterministic pseudorandom test suites.
  - `search_counterexamples`: Automated counterexample search with independent validation.
  - `execute_experiment`: Arbitrary Python sandbox execution.
  - `verify_solution`: Independent mathematical verification tier.
  - `inspect_investigation`: Structured inspection of past experiment results.
- **Open-Weights Hugging Face Models on Modal GPU**: Pull and serve any open-source mathematical LLM from Hugging Face (such as `Qwen/Qwen2.5-Math-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, or `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`) running on dedicated serverless GPUs (`T4`, `L4`, `A10G`) with persistent model weight caching.
- **Safe Dual-Backend Execution**:
  - **Local Subprocess**: Subprocess sandbox with memory/time capping and environment sanitization (no secret leakage).
  - **Modal Cloud**: Remote serverless execution on Debian Slim scientific containers with CPU/memory resource bounding.
- **Three-Tier Mathematical Verification**:
  - *Level 1*: Computational evidence (numerical and discrete testing).
  - *Level 2*: Symbolic and exact verification (exact arithmetic, identity checks).
  - *Level 3*: Rigorous proof step validation and heuristic leap detection.
- **Multi-Source Problem Ingestion**:
  - **IMO Official**: Automated retrieval and parsing of public PDFs from [imo-official.org](https://www.imo-official.org/).
  - **Hugging Face MATH**: Bounded streaming retrieval from `EleutherAI/hendrycks_math`.
  - **Controlled Suite**: Canonical synthetic tasks with verified ground truth to avoid data contamination.
- **Separation of Concerns**: Ground truth reference solutions are isolated from the solving agent and accessed only by the evaluation harness.
- **Persistent State & Tracing**: SQLite investigation storage (`runs/mia.db`) and append-only JSONL event logging (`runs/traces/`). Resumable from any step.

---

## Repository Structure

```
├── pyproject.toml              # Project dependencies and packaging
├── config.example.yaml         # Fully documented example configuration
├── modal_app.py                # Modal serverless execution application
├── src/mia/                    # Core Python package
│   ├── config.py               # Hierarchical Pydantic configuration
│   ├── models.py               # Normalized problem and experiment schemas
│   ├── cli.py                  # Typer and Rich command line interface
│   ├── sources/                # Problem sources (IMO, HF MATH, Controlled)
│   ├── agent/                  # Autonomous investigation state machine
│   ├── tools/                  # Typed mathematical tools registry
│   ├── llm/                    # Providers (OpenAI, Anthropic, Gemini, Mock)
│   ├── execution/              # Local subprocess & Modal execution backends
│   ├── verification/           # Three-tier independent mathematical verifier
│   ├── storage/                # SQLite persistent store & JSONL trace logger
│   ├── evaluation/             # Comparative evaluation suite & metric runners
│   └── reporting/              # Markdown and CSV report generation
├── tests/
│   ├── unit/                   # Unit test suite (Config, Math, Agent, CLI)
│   ├── integration/            # Live external integration tests (IMO, Modal)
│   └── fixtures/               # Test fixtures and sample configs
├── scripts/
│   ├── demo.py                 # Fast reproducible CPU demonstration
│   └── run_evaluation.sh       # Benchmark evaluation script
└── docs/                       # Comprehensive documentation
    ├── architecture.md
    ├── configuration.md
    ├── benchmark_sources.md
    ├── security.md
    ├── evaluation.md
    └── troubleshooting.md
```

---

## Installation

### Prerequisites
- Python 3.11 or 3.12
- `uv` (recommended) or `pip`

```bash
git clone https://github.com/khangtruong-ui/agentic-math-inspector.git
cd agentic-math-inspector

# Create virtual environment and install in editable mode
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

---

## Quickstart & User-Facing Demonstration

Run the reproducible demonstration on CPU without requiring external API keys:

```bash
python3 scripts/demo.py
# Or via CLI
mia solve controlled-euler-primality --demo
```

This demonstration shows:
1. Ingestion of Euler's polynomial $P(n) = n^2 + n + 41$.
2. Initial candidate hypothesis ($P(n)$ is prime for all positive integers $n$).
3. Automated counterexample search discovering that $P(40) = 1681 = 41^2$ (composite).
4. Immediate hypothesis revision refuting universal primality.
5. Independent symbolic and exact arithmetic verification.
6. Generation of final report distinguishing proven facts, empirical evidence, and limits.

---

## Configuration

MIA supports configuration via YAML files, environment variables (`MIA__<SECTION>__<KEY>`), and CLI flags.

```bash
# Validate your configuration and print effective values with secrets redacted
mia config validate
```

### Environment Variables
- `OPENAI_API_KEY`: API key for OpenAI or OpenAI-compatible models.
- `ANTHROPIC_API_KEY`: API key for Anthropic Claude.
- `GEMINI_API_KEY`: API key for Google Gemini.
- `MODAL_TOKEN_ID` & `MODAL_TOKEN_SECRET`: Credentials for Modal cloud execution.

---

## Command Line Interface (CLI)

```bash
# Browse registered problem sources
mia sources list

# Synchronize official IMO problems
mia sources sync --source imo --year 2024

# Browse problems
mia problems list --source controlled
mia problems show controlled-euler-primality

# Run inference with open-weights LLM from Hugging Face on Modal GPU
mia modal run-llm "Solve for x: x^2 - 5x + 6 = 0." --model "Qwen/Qwen2.5-Math-1.5B-Instruct"

# Solve a problem locally or via Modal cloud
mia solve controlled-euler-primality
mia solve controlled-analysis-limit --backend modal
mia solve controlled-analysis-limit --provider modal-huggingface --model "Qwen/Qwen2.5-Math-1.5B-Instruct" --backend modal

# Resume an interrupted investigation
mia resume <investigation_id> --steps 5

# List recorded experiments
mia experiments --limit 10

# View execution traces and Markdown report
mia report --investigation-id <investigation_id>
mia report --investigation-id <investigation_id> --trace

# Run comparative evaluation across configurations A, B, and C
mia evaluate --source controlled --configs A,B,C --num-problems 5 --seed 42

# Run remote Modal smoke test
mia modal smoke-test
```

---

## Comparative Evaluation Framework

MIA supports comparative benchmarking across three experimental conditions:
- **Configuration A**: Pure reasoning baseline (LLM without executable tools).
- **Configuration B**: Tool-enabled agent (stateless tools).
- **Configuration C**: Tool-enabled agent with persistent SQLite memory & adaptive hypothesis revision.

```bash
./scripts/run_evaluation.sh controlled 5 42
```

Outputs are saved in `runs/evaluations/<eval_id>/`:
- `results.jsonl`: Complete per-problem logs and metrics.
- `summary.csv`: Tabular comparison of configurations.
- `report.md`: Markdown report detailing accuracy, verification rates, and resource usage.

---

## Running the Test Suite

```bash
# Run all unit tests
pytest tests/unit -v

# Run integration tests (requires network & Modal credentials)
pytest tests/integration -v
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
