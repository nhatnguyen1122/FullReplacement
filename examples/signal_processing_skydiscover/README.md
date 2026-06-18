# Native SkyDiscover Signal Processing

This benchmark is a FullReplacementMCTS/OpenEvolve wrapper for SkyDiscover's native signal-processing benchmark.

Use this when you want the native SkyDiscover signal-processing evaluator. Do not confuse it with:

- `examples/signal_processing`, the OpenEvolve-parity signal-processing benchmark
- SkyDiscover's `signal_processing_openevolve`, the cross-framework OpenEvolve-parity benchmark

Contract:

```python
run_signal_processing(noisy_signal=..., window_size=20)
```

Expected filtered output length:

```python
len(noisy_signal) - window_size + 1
```

Matrix alias:

```bash
python scripts/run_benchmark_matrix.py --benchmarks sp_sky --strategies baseline
```
