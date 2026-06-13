# Benchmark Matrix Runner Review and External Benchmark Setup

Date reviewed: 2026-05-11

## Current `scripts/run_benchmark_matrix.py` status

The runner is usable for the currently included Python-only benchmark subset:

- `func` -> `examples/function_minimization`
- `cp` -> `examples/circle_packing`
- `cp_art` -> `examples/circle_packing_with_artifacts`
- `sp` -> `examples/signal_processing`
- `kmod` -> `examples/k_module_problem`

It is not a general-purpose benchmark runner for all examples yet.

Main issues:

1. External-dependency examples are hard-skipped inside `discover_benchmarks()`.
   There is no CLI flag to include them after the environment is prepared.

2. Directory-based initial programs are unsupported.
   `tsp_tour_minimization` uses `initial_program/` as a project directory. The
   generic OpenEvolve CLI path in this runner expects one initial file, so TSP
   needs its own launcher (`examples/tsp_tour_minimization/start_evolution.py`)
   or a runner extension.

3. Importing the script can write files.
   `BENCHMARKS = discover_benchmarks()` runs at import time, and discovery can
   auto-generate missing `config_baseline.yaml` / `config_mcts.yaml`. That side
   effect is surprising for imports and tests.

4. `score_from_metrics()` is too generic for minimization-style metrics.
   It works for examples with `combined_score`, `score`, `fitness`, or
   `accuracy`, but examples like `attention_optimization` return `error` where
   lower is better. If that benchmark is included, the runner should either
   require a benchmark-specific score key/direction or the evaluator should
   return a maximized `combined_score`.

5. Failed runs can still be recorded as `ok` if the OpenEvolve process returns
   zero but no useful metrics are produced. The runner should treat missing
   metrics or missing primary score as a failed/incomplete run.

6. The generated config pair uses a shallow top-level copy. It is fine for the
   current operation because nested sections are re-bound before mutation, but a
   deep copy would be safer if future generation mutates nested data.

7. Seeds are fair only within one matrix invocation. If baseline and MCTS runs
   are produced in separate output roots, the user must ensure matching
   `--base-seed` and `--runs`.

Recommended runner changes before using excluded benchmarks:

- Move external skip metadata to a module-level dict and add
  `--include-external-deps`.
- Add `--list-benchmarks` that reports included/skipped status and reason.
- Add per-benchmark metadata: `primary_metric`, `maximize`, optional
  `runner_type`.
- Mark records failed when `returncode != 0`, metrics are empty, or primary
  score is `None`.
- Avoid config generation during import. Do it only in `main()`.
- Add a special runner for `tsp`, or keep TSP explicitly outside this matrix.

## Local capability check on this machine

Found:

- `Rscript`: installed
- `g++`: installed

Missing from the plain `python3` environment used for this check:

- `mlir-opt`, `mlir-translate`
- `cargo`, `rustc`
- `kaggle` CLI
- Python modules: `mlx`, `mlx_lm`, `lxml`, `bs4`, `torch`

Your `openevolve_test` conda environment may differ. Recheck inside that env:

```bash
conda run -n openevolve_test which Rscript
conda run -n openevolve_test which cargo
conda run -n openevolve_test which rustc
conda run -n openevolve_test which mlir-opt
conda run -n openevolve_test python -c "import mlx, mlx_lm, lxml, bs4, torch"
```

## Excluded benchmark setup

### `attention_optimization` / `attn`

Can be run after setup: yes, if MLIR tools are installed.

What it needs:

- `mlir-opt` in `PATH`
- Optionally `mlir-translate` for deeper real-execution workflows
- Python dependencies already used by OpenEvolve

Important code detail:

- `examples/attention_optimization/evaluator.py` constructs
  `Path("mlir/self_attn_with_consts_linalg_dialect.mlir")`, so it expects the
  current working directory to be `examples/attention_optimization`.
- `run_benchmark_matrix.py` runs from repo root, so this evaluator will not find
  its MLIR file unless patched to use `Path(__file__).parent / "mlir/..."`.

Setup:

```bash
# macOS Homebrew option, if available
brew install llvm
export PATH="$(brew --prefix llvm)/bin:$PATH"

mlir-opt --version
mlir-translate --version
```

Smoke test:

```bash
cd examples/attention_optimization
python evaluator.py initial_program.py
```

Run directly:

```bash
cd examples/attention_optimization
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config_baseline.yaml \
  --iterations 10 \
  --output output_baseline_smoke
```

Before adding to `run_benchmark_matrix.py`, patch the evaluator path handling or
use a benchmark-specific working directory.

### `mlx_metal_kernel_opt` / `mlx`

Can be run after setup: yes, but only on Apple Silicon for the intended Metal
benchmark.

What it needs:

- Apple Silicon Mac for the intended `mlx` Metal path
- `mlx`, `mlx-lm`, `numpy`, `pyyaml`, `psutil`, optionally `scipy`
- Model download access for `mlx-community/Qwen3-0.6B-bf16`
- LLM API key for evolution

Setup:

```bash
pip install -r examples/mlx_metal_kernel_opt/requirements.txt
python -c "import mlx.core as mx; import mlx_lm; print('mlx ok')"
```

Smoke test:

```bash
cd examples/mlx_metal_kernel_opt
python -c "from evaluator import BulletproofMetalEvaluator; e = BulletproofMetalEvaluator(); print('evaluator ok')"
```

Run with the example launcher:

```bash
export CODESTRAL_API_KEY="..."
cd examples/mlx_metal_kernel_opt
bash run_evolve_experiment.sh \
  --config config_baseline.yaml \
  --iterations 5 \
  --run-name baseline_smoke \
  --foreground
```

Notes:

- The example's shell launcher checks `OPENAI_API_KEY` / `GEMINI_API_KEY`, while
  the generated benchmark configs in this repo use Codestral. For Codestral,
  prefer the generic OpenEvolve CLI or update the script's key check.
- The evaluator loads a model and can be slow on first run due to model download.

### `arc_benchmark` / `arc`

Can be run after setup: yes.

What it needs:

- ARC Prize 2025 data from Kaggle
- A selected task via `ARC_TASK_FILE` and `TASK_NUM`
- `numpy`
- LLM API key

Kaggle setup:

```bash
pip install kaggle
kaggle auth login
```

Alternative Kaggle auth methods include `KAGGLE_API_TOKEN` or token files. The
Kaggle CLI docs currently describe OAuth login, environment variable token, and
token-file options.

Download data:

```bash
mkdir -p data/arc-prize-2025
kaggle competitions download -c arc-prize-2025 -p data/arc-prize-2025
unzip data/arc-prize-2025/arc-prize-2025.zip -d data/arc-prize-2025
```

Expected files include:

- `arc-agi_training_challenges.json`
- `arc-agi_evaluation_challenges.json`
- possibly matching solutions files depending on split

Run:

```bash
cd examples/arc_benchmark
export DATA_ROOT="../../data/arc-prize-2025"
export ARC_TASK_FILE="evaluation"
export TASK_NUM="0"
export CODESTRAL_API_KEY="..."
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config_baseline.yaml \
  --iterations 5 \
  --output outputs/evaluation_task_0_baseline_smoke
```

Notes:

- `run_evolution.sh` overwrites `OPENAI_API_KEY` with a placeholder. Edit it
  before use, or run the CLI command above.
- ARC is not naturally a matrix over one benchmark: each `TASK_NUM` is a
  different problem instance. For fair comparison, include the task id in the
  benchmark key or output name.

### `web_scraper_optillm` / `webscrape`

Can be run after setup: yes, but the intended setup requires an optillm proxy.

What it needs:

- `beautifulsoup4`, `requests`, `lxml`
- optillm running on `http://localhost:8000/v1`
- Either an external provider key routed through optillm, or optillm local
  inference configured

Setup dependencies:

```bash
pip install -r examples/web_scraper_optillm/requirements.txt
pip install optillm
```

Start optillm in terminal 1:

```bash
export OPTILLM_API_KEY="optillm"
optillm --port 8000
```

Run evolution in terminal 2:

```bash
export OPENAI_API_KEY="optillm"
python openevolve-run.py examples/web_scraper_optillm/initial_program.py \
  examples/web_scraper_optillm/evaluator.py \
  --config examples/web_scraper_optillm/config.yaml \
  --iterations 10 \
  --output output/webscrape_optillm_smoke
```

Notes:

- `config_baseline.yaml` and `config_mcts.yaml` in this repo are Codestral
  configs and do not exercise optillm's `readurls` / `moa` behavior. Use
  `config.yaml` if the goal is testing optillm.
- If you want this in `run_benchmark_matrix.py`, add an optillm-specific
  strategy/config path or accept that the matrix uses Codestral rather than the
  intended local proxy behavior.

### `online_judge_programming` / `ojp`

Can be run after setup: yes, but it submits live programs to Kattis.

What it needs:

- Kattis account
- Kattis `.kattisrc` credentials
- `lxml`, `requests`
- Network access to `open.kattis.com`

Setup:

```bash
pip install -r examples/online_judge_programming/requirements.txt
```

Create credentials:

1. Log in to Kattis.
2. Download `kattisrc` from `https://open.kattis.com/download/kattisrc`.
3. Save it as either `~/.kattisrc` or
   `examples/online_judge_programming/.kattisrc`.

The repo README says `.kittisrc`, but `submit.py` looks for `.kattisrc`; use
`.kattisrc`.

Smoke test one submission:

```bash
cd examples/online_judge_programming
python submit.py initial_program.py -p alphabet -l "Python 3" -f
```

Run:

```bash
cd examples/online_judge_programming
export CODESTRAL_API_KEY="..."
python ../../openevolve-run.py initial_program.py evaluator.py \
  --config config_baseline.yaml \
  --iterations 5 \
  --output output_ojp_baseline_smoke
```

Notes:

- This benchmark is not fully reproducible: Kattis service availability,
  submission limits, account state, and judge queue latency can affect results.
- Use small iteration counts first to avoid excessive live submissions.

### `r_robust_regression` / `r_reg`

Can be run after setup: yes, but the current evaluator needs a small code fix.

What it needs:

- `Rscript` in `PATH`
- R package `jsonlite`
- Python `numpy`

Setup:

```bash
Rscript -e 'install.packages("jsonlite", repos="https://cloud.r-project.org")'
pip install -r examples/r_robust_regression/requirements.txt
Rscript --version
```

Current bug:

- `examples/r_robust_regression/evaluator.py` defines `async def evaluate(...)`.
  OpenEvolve's subprocess evaluator calls evaluator functions synchronously.
  That returns a coroutine object, which is not a valid pickled evaluation
  result.

Patch pattern:

```python
async def _evaluate(program_path: str) -> EvaluationResult:
    ...

def evaluate(program_path: str) -> EvaluationResult:
    return asyncio.run(_evaluate(program_path))
```

After patching:

```bash
cd examples/r_robust_regression
python evaluator.py initial_program.r
export CODESTRAL_API_KEY="..."
python ../../openevolve-run.py initial_program.r evaluator.py \
  --config config_baseline.yaml \
  --iterations 10 \
  --output output_r_reg_baseline_smoke
```

Also consider setting `evaluator.cascade_evaluation: false`; the evaluator does
not provide `evaluate_stage1` functions, so cascade falls back to direct
evaluation anyway.

### `rust_adaptive_sort` / `rust_sort`

Can be run after setup: yes.

What it needs:

- Rust toolchain: `rustc`, `cargo`
- Python `numpy`

Setup:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
rustc --version
cargo --version
pip install -r examples/rust_adaptive_sort/requirements.txt
```

Smoke test:

```bash
cd examples/rust_adaptive_sort
python evaluator.py initial_program.rs
```

Run:

```bash
cd examples/rust_adaptive_sort
export CODESTRAL_API_KEY="..."
python ../../openevolve-run.py initial_program.rs evaluator.py \
  --config config_baseline.yaml \
  --iterations 10 \
  --output output_rust_sort_baseline_smoke
```

Notes:

- This can be included in the matrix after Rust is installed because it has a
  single-file `initial_program.rs` and compatible baseline/MCTS configs.
- First Cargo builds may spend time downloading crates if the lockfile is not
  already cached.

### `tsp_tour_minimization` / `tsp`

Can be run after setup: yes, but not through the current generic matrix runner.

What it needs:

- `g++` with C++17 support
- Python packages: `numpy`, `torch`, `click`
- Enough CPU time; each evaluation can run long
- Optional GPU only if using heat-map training paths

Setup:

```bash
pip install -r examples/tsp_tour_minimization/requirements.txt
g++ --version
python -c "import numpy, torch, click; print('tsp deps ok')"
```

Smoke test the custom launcher:

```bash
cd examples/tsp_tour_minimization
export CODESTRAL_API_KEY="..."
python start_evolution.py \
  --initial_program_dir initial_program \
  --openevolve_output_dir output_tsp_smoke
```

Notes:

- `start_evolution.py` converts the source directory into
  `initial_program.txt` and then calls `OpenEvolve` directly.
- It currently uses `config.yaml`, not `config_baseline.yaml` /
  `config_mcts.yaml`. To compare baseline and MCTS, add a `--config` option to
  `start_evolution.py` or create separate launchers that load each config.
- `run_benchmark_matrix.py` cannot run this example as-is because its discovery
  rejects directory-based initial programs.

## Source references

- Kaggle CLI authentication docs:
  https://github.com/Kaggle/kaggle-api/blob/main/docs/README.md
- Rust official installation:
  https://www.rust-lang.org/tools/install
- MLX installation:
  https://github.com/ml-explore/mlx
- optillm quick start:
  https://github.com/codelion/optillm
- MLIR `mlir-opt` docs:
  https://mlir.llvm.org/docs/Tutorials/MlirOpt/
- R project:
  https://www.r-project.org/
- PyTorch local install selector:
  https://docs.pytorch.org/get-started/locally/
