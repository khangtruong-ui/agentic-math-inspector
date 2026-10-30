# Evaluation Framework and Metrics

MIA contains a reproducible comparative evaluation system that benchmarks agents under controlled experimental conditions.

## Experimental Configurations

1. **Configuration A (Reasoning-Only Baseline)**:
   - Uses the configured LLM without access to executable tools.
   - Evaluates pure pre-trained mathematical reasoning capabilities.
2. **Configuration B (Tool-Enabled Agent, Stateless)**:
   - Enables symbolic computation, numerical evaluation, counterexample search, and code execution.
   - Disables persistent SQLite memory across investigation steps.
3. **Configuration C (Full Adaptive Agent with Persistent Memory)**:
   - Full agent with tools, persistent SQLite investigation memory, and adaptive hypothesis revision.
   - Contradictory evidence triggers hypothesis invalidation and strategy pivoting.

## Fair Comparison Requirements
- **Constant Variables**: Evaluated on identical problem sets using reproducible random seeds.
- **Answer Separation**: Ground truth answers are withheld from solving agents during investigation and evaluated post-hoc.
- **Failures Retained**: Unsuccessful attempts, timeouts, and nonproductive loops are recorded and counted in failure metrics.

## Metric Definitions

1. **Accuracy Rate**: Fraction of attempted problems where final candidate solution matches official ground-truth.
2. **Verification Rate**: Fraction of problems where independent mathematical verification succeeded at Level 1, 2, or 3.
3. **Completion Rate**: Fraction of problems concluding with status `SOLVED`.
4. **Counterexamples Found**: Total count of mathematically verified counterexamples discovered.
5. **Average Experiments**: Mean count of code experiments executed per problem.
6. **Average Wall-Clock Duration**: Average elapsed time per problem in seconds.
7. **Unresolved Rate**: Fraction of completed tasks retaining unresolved mathematical assertions or unverified steps.

## Running an Evaluation

Run comparative evaluation across configurations A, B, and C:

```bash
mia evaluate --source controlled --configs A,B,C --num-problems 5 --seed 42
```

Outputs generated in `runs/evaluations/<eval_run_id>/`:
- `results.jsonl`: Complete per-problem execution details.
- `summary.csv`: Tabular comparison of configurations.
- `report.md`: Formatted Markdown report with metrics and findings.
