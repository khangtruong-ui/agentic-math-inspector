# Mathematical Problem & Benchmark Sources

MIA implements automated data acquisition mechanisms with clear separation between reference solutions and the solving agent.

## Supported Sources

### 1. International Mathematical Olympiad (IMO)
- **Source**: Official IMO website: `https://www.imo-official.org/`
- **Adapter**: `IMOSource` (`src/mia/sources/imo.py`)
- **Acquisition Procedure**:
  - Targets official public English problem documents: `/assets/documents/problems/{year}/{year}_eng.pdf`
  - Downloads and caches PDFs in `data/cache/imo/{year}_eng.pdf`
  - Parses text using `pypdf` and extracts Problems 1 to 6
  - Normalizes statements and tags subjects (geometry, number theory, algebra, combinatorics)
  - Records extraction failures rather than silently discarding problems

To synchronize IMO problems:
```bash
mia sources sync --source imo --year 2024
```

### 2. Hugging Face MATH (`EleutherAI/hendrycks_math`)
- **Source**: `https://huggingface.co/datasets/EleutherAI/hendrycks_math`
- **Adapter**: `HendrycksMathSource` (`src/mia/sources/hendrycks_math.py`)
- **Acquisition Procedure**:
  - Uses streaming pagination via Hugging Face `datasets-server` API (`https://datasets-server.huggingface.co/rows`)
  - Supports bounded retrieval without downloading the entire multi-gigabyte dataset
  - Extracts problems across categories: `algebra`, `number_theory`, `precalculus`, `intermediate_algebra`, `geometry`, etc.
  - Separates LaTeX `\boxed{...}` answers for evaluator use only

To synchronize a sample of MATH problems:
```bash
mia sources sync --source hendrycks_math --limit 10
```

### 3. Controlled Problem Suite (`controlled`)
- **Source**: Deterministic synthetic and canonical mathematical tasks with independently computed ground truth.
- **Adapter**: `ControlledProblemSource` (`src/mia/sources/controlled.py`)
- **Purpose**:
  - Avoids benchmark contamination from pretraining data
  - Includes known hypothesis-failure tasks (e.g. Euler polynomial primality $P(n) = n^2 + n + 41$, where counterexample $n=40$ triggers hypothesis revision)
  - Canonical indeterminate limits (Taylor expansions)
  - Second-order linear recurrence relations with characteristic roots
  - Multivariable optimization extrema

To list controlled problems:
```bash
mia problems list --source controlled
```

## Data Contamination & Solution Separation

Reference solutions and ground-truth answers are strictly excluded from the `AgentProblemView` presented to the solving agent. The solving agent receives only the problem statement, subject, and identifier. The evaluator accesses reference answers only after the agent's investigation is completed.
