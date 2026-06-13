# Benchmark Matrix 4x5 Confirmation

## Run

Repository:

```text
openevolve_original
```

Command:

```bash
PYTHONPATH="$PWD" CODESTRAL_API_KEY=... conda run -n openevolve_test \
  python scripts/run_benchmark_matrix.py \
  --runs 5 \
  --iterations 1 \
  --benchmarks func,cp,sp,kmod \
  --strategies baseline,mcts \
  --output-root output/benchmark_matrix_4x5_confirm \
  --max-total-tokens 1000000 \
  --fail-fast
```

Matrix:

- 4 problems: function minimization, circle packing, signal processing, k-module problem.
- 2 strategies: baseline OpenEvolve (`island_map_elites`) and new MCTS framework (`tree_qd_mcts`).
- 5 runs per problem per strategy.
- 40 total real Codestral runs.
- 1 evolution iteration per run for integration confirmation.

## Result

All runs completed successfully:

```text
Completed 40 runs with 0 failures
```

Outputs:

```text
output/benchmark_matrix_4x5_confirm/results.jsonl
output/benchmark_matrix_4x5_confirm/summary.csv
output/benchmark_matrix_4x5_confirm/aggregate.csv
```

The MCTS logs contain MCTS selection/backpropagation markers. Baseline logs do not contain MCTS markers.

## Aggregate Scores

The benchmark runner uses score priority:

```text
combined_score -> overall_score -> composite_score -> score -> fitness -> accuracy -> numeric average
```

This matters for signal processing because that evaluator does not emit `combined_score`; the runner was corrected to prefer `overall_score` before falling back to a raw numeric average.

| Problem | Strategy | Runs | Successes | Mean | Median | Best |
|---|---:|---:|---:|---:|---:|---:|
| function_minimization | baseline | 5 | 5 | 1.4415 | 1.4252 | 1.4924 |
| function_minimization | mcts | 5 | 5 | 1.4455 | 1.4190 | 1.4995 |
| circle_packing | baseline | 5 | 5 | 0.3846 | 0.3642 | 0.4381 |
| circle_packing | mcts | 5 | 5 | 0.4654 | 0.3642 | 0.6318 |
| signal_processing | baseline | 5 | 5 | 0.2856 | 0.3286 | 0.3899 |
| signal_processing | mcts | 5 | 5 | 0.3886 | 0.3899 | 0.3899 |
| k_module_problem | baseline | 5 | 5 | 0.0000 | 0.0000 | 0.0000 |
| k_module_problem | mcts | 5 | 5 | 0.1500 | 0.0000 | 0.7500 |

## Notes

- This was a functionality confirmation run, not a performance study. One iteration per run is enough to verify the new framework, baseline path, provider configuration, evaluator execution, checkpointing, token tracking, and result aggregation.
- No runtime bugs were found in the framework paths during the 40-run matrix.
- One reporting bug was fixed in `scripts/run_benchmark_matrix.py`: score extraction now prefers `overall_score` and `composite_score` before averaging all numeric metrics.

