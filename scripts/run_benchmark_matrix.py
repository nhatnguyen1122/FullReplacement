#!/usr/bin/env python3
"""Run baseline OpenEvolve vs TreeQD-MCTS benchmark matrices."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
import time
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Benchmark:
    key: str
    name: str
    example_dir: str
    initial_program: str = "initial_program.py"
    evaluator: str = "evaluator.py"
    artifact_variant: bool = False


# ---------------------------------------------------------------------------
# Auto-discover benchmarks from examples/ directory
# ---------------------------------------------------------------------------

# Short aliases for well-known examples (kept for backward compatibility)
_KEY_ALIASES: Dict[str, str] = {
    "function_minimization": "func",
    "circle_packing": "cp",
    "circle_packing_n32": "cp32",
    "circle_packing_rect": "cprect",
    "erdos_min_overlap": "erdos",
    "first_autocorr_ineq": "auto1",
    "heilbronn_convex/13": "hconv13",
    "heilbronn_convex/14": "hconv14",
    "heilbronn_triangle": "heil",
    "hexagon_packing/11": "hex11",
    "hexagon_packing/12": "hex12",
    "matmul": "matmul",
    "minimizing_max_min_dist/2": "mmd2",
    "minimizing_max_min_dist/3": "mmd3",
    "second_autocorr_ineq": "auto2",
    "sums_diffs_finite_sets": "sumsdiffs",
    "third_autocorr_ineq": "auto3",
    "uncertainty_ineq": "uncert",
    "signal_processing": "sp",
    "k_module_problem": "kmod",
    "circle_packing_with_artifacts": "cp_art",
    "tsp_tour_minimization": "tsp",
    "attention_optimization": "attn",
    "online_judge_programming": "ojp",
    "r_robust_regression": "r_reg",
    "rust_adaptive_sort": "rust_sort",
    "mlx_metal_kernel_opt": "mlx",
    "arc_benchmark": "arc",
    "web_scraper_optillm": "webscrape",
    "llm_prompt_optimization": "llm_prompt",
    "alphaevolve_math_problems": "math",
    "symbolic_regression": "symreg",
}


def _find_initial_program(example_dir: Path) -> Optional[str]:
    """Return the relative filename of the initial program, or None if not found.

    Note: directory-based initial programs (e.g. TSP with multiple C++ files)
    are not supported by the OpenEvolve CLI which expects a single file.
    Returns None for directories so they are excluded from benchmarking.
    """
    # Check for single-file initial programs with various extensions
    for ext in (".py", ".r", ".rs", ".cpp", ".c", ".js", ".ts", ".java"):
        candidate = example_dir / f"initial_program{ext}"
        if candidate.exists():
            return f"initial_program{ext}"

    # Directory-based initial programs are NOT supported by the CLI
    dir_path = example_dir / "initial_program"
    if dir_path.is_dir():
        return None  # Skip — would crash with IsADirectoryError

    return None


def _find_evaluator(example_dir: Path) -> Optional[str]:
    """Return the relative filename of the evaluator, or None if not found."""
    for name in ("evaluator.py", "evaluator_stub.py"):
        if (example_dir / name).exists():
            return name
    return None


def _tool_path(tool: str, extra_dirs: Iterable[Path] = ()) -> Optional[str]:
    """Find a tool on PATH or in known benchmark-specific install locations."""
    found = shutil.which(tool)
    if found:
        return found
    for directory in extra_dirs:
        candidate = directory / tool
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _attention_mlir_bin_dir() -> Optional[Path]:
    mlir_opt = _tool_path("mlir-opt", [Path("/opt/homebrew/opt/llvm/bin"), Path("/usr/local/opt/llvm/bin")])
    if not mlir_opt:
        return None
    return Path(mlir_opt).resolve().parent


def _python_import_available(module: str) -> bool:
    command = [sys.executable, "-c", f"import {module}"]
    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def _mlx_available() -> bool:
    return _python_import_available("mlx.core") and _python_import_available("mlx_lm")


def _optillm_available() -> bool:
    """Return whether the local OptiLLM OpenAI-compatible proxy is reachable."""
    request = urllib.request.Request(
        "http://localhost:8000/v1/models",
        headers={"Authorization": f"Bearer {os.environ.get('OPTILLM_API_KEY', 'optillm')}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return 200 <= response.status < 500
    except Exception:
        return False


def _generate_config_pair(example_dir: Path) -> None:
    """Auto-generate config_baseline.yaml and config_mcts.yaml from config.yaml.

    The generated configs:
    - Override the LLM provider to codestral (for uniform benchmarking)
    - Set search.strategy accordingly
    - Cap max_iterations at 50 for reasonable run time
    - Set max_total_tokens to 5_000_000
    """
    source = example_dir / "config.yaml"
    if not source.exists():
        source = example_dir / "config.yml"
    if not source.exists():
        return

    with source.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f) or {}

    # Normalise LLM section to use codestral provider
    llm = dict(base_config.get("llm") or {})
    llm["provider"] = "codestral"
    llm["primary_model"] = "codestral-latest"
    llm["primary_model_weight"] = 1.0
    llm["api_base"] = "https://api.mistral.ai/v1"
    llm.setdefault("max_total_tokens", 5_000_000)
    # Remove hardcoded API keys
    llm.pop("api_key", None)
    # Remove secondary model (keep single-model for fair comparison)
    llm.pop("secondary_model", None)
    llm.pop("secondary_model_weight", None)
    # Remove models list (let provider default handle it)
    llm.pop("models", None)
    llm.pop("evaluator_models", None)

    base_config["llm"] = llm
    base_config.setdefault("max_iterations", 50)
    if "max_program_length" in base_config and "max_code_length" not in base_config:
        base_config["max_code_length"] = base_config.pop("max_program_length")

    for strategy, filename in [
        ("island_map_elites", "config_baseline.yaml"),
        ("tree_qd_mcts", "config_mcts.yaml"),
    ]:
        dest = example_dir / filename
        if dest.exists():
            continue
        cfg = dict(base_config)
        search = dict(cfg.get("search") or {})
        search["strategy"] = strategy
        if strategy == "tree_qd_mcts":
            search.setdefault("exploration_constant", 0.7)
            search.setdefault("qd_weight", 0.5)
            search.setdefault("max_tree_depth", 15)
            search.setdefault("backup", "mixed")
            search.setdefault("backup_alpha", 0.5)
            search.setdefault("widening_alpha", 0.5)
        cfg["search"] = search
        with dest.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        print(f"  Auto-generated {dest}")


def discover_benchmarks() -> Dict[str, Benchmark]:
    """Scan examples/ for all directories that have an initial_program and evaluator."""
    examples_root = REPO_ROOT / "examples"
    benchmarks: Dict[str, Benchmark] = {}

    if not examples_root.is_dir():
        return benchmarks

    # Examples with hard external dependencies that will fail unless the user
    # has the required toolchain / service installed.
    _SKIP_EXTERNAL_DEPS: set = {
        "arc_benchmark",            # requires specific puzzle data + long runs
        "online_judge_programming", # requires Kattis account + submit.py credentials
        "r_robust_regression",      # evaluator crashes with cascade eval on .r files
        "rust_adaptive_sort",       # requires rustc compiler
    }

    for entry in sorted(examples_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue

        if entry.name in _SKIP_EXTERNAL_DEPS:
            continue

        if entry.name == "attention_optimization" and _attention_mlir_bin_dir() is None:
            continue

        if entry.name == "mlx_metal_kernel_opt" and not _mlx_available():
            continue

        if entry.name == "web_scraper_optillm" and not _optillm_available():
            continue

        init_prog = _find_initial_program(entry)
        evaluator = _find_evaluator(entry)
        if init_prog is None or evaluator is None:
            continue

        # Derive a short key
        key = _KEY_ALIASES.get(entry.name, entry.name)

        # Auto-generate config pair if needed
        _generate_config_pair(entry)

        # Only include if both config_baseline.yaml and config_mcts.yaml exist
        # (either pre-existing or just auto-generated)
        if not (entry / "config_baseline.yaml").exists():
            continue
        if not (entry / "config_mcts.yaml").exists():
            continue

        benchmarks[key] = Benchmark(
            key=key,
            name=entry.name,
            example_dir=f"examples/{entry.name}",
            initial_program=init_prog,
            evaluator=evaluator,
        )

    return benchmarks


BENCHMARKS: Dict[str, Benchmark] = discover_benchmarks()
for name, key in {
    "heilbronn_convex/13": "hconv13",
    "heilbronn_convex/14": "hconv14",
    "hexagon_packing/11": "hex11",
    "hexagon_packing/12": "hex12",
    "minimizing_max_min_dist/2": "mmd2",
    "minimizing_max_min_dist/3": "mmd3",
}.items():
    example_dir = REPO_ROOT / "examples" / name
    if (
        (example_dir / "initial_program.py").exists()
        and (example_dir / "evaluator.py").exists()
        and (example_dir / "config_baseline.yaml").exists()
        and (example_dir / "config_mcts.yaml").exists()
    ):
        BENCHMARKS[key] = Benchmark(
            key=key,
            name=name,
            example_dir=f"examples/{name}",
            initial_program="initial_program.py",
            evaluator="evaluator.py",
        )
for base_key, art_key in {
    "func": "func_art",
    "kmod": "kmod_art",
    "sp": "sp_art",
}.items():
    if base_key in BENCHMARKS:
        base = BENCHMARKS[base_key]
        BENCHMARKS[art_key] = Benchmark(
            key=art_key,
            name=f"{base.name}_artifacts",
            example_dir=base.example_dir,
            initial_program=base.initial_program,
            evaluator=base.evaluator,
            artifact_variant=True,
        )

STRATEGIES = ("baseline", "mcts")


def parse_csv_list(value: str, valid: Iterable[str], label: str) -> List[str]:
    valid_set = set(valid)
    items = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(items) - valid_set)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown {label}: {', '.join(unknown)}; valid values: {', '.join(sorted(valid_set))}"
        )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible benchmark matrix for baseline OpenEvolve and TreeQD-MCTS."
    )
    parser.add_argument("--runs", type=int, default=10, help="Runs per strategy per benchmark.")
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Evolution iterations per run. Defaults to each config's max_iterations.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one seed and one iteration for each selected benchmark/strategy.",
    )
    parser.add_argument(
        "--benchmarks",
        default=",".join(BENCHMARKS),
        help=f"Comma-separated benchmark keys. Available: {','.join(sorted(BENCHMARKS))}.",
    )
    parser.add_argument(
        "--strategies",
        default="baseline,mcts",
        help="Comma-separated strategies: baseline,mcts.",
    )
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument(
        "--output-root",
        default=None,
        help="Output root. Defaults to output/benchmark_matrix_<timestamp>.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Override llm.max_total_tokens for each generated run config.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate per-run configs and summary skeleton without running OpenEvolve.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed records in an existing output root and recover completed run dirs.",
    )
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    args.benchmarks = parse_csv_list(args.benchmarks, BENCHMARKS, "benchmarks")
    args.strategies = parse_csv_list(args.strategies, STRATEGIES, "strategies")
    if args.smoke:
        args.runs = 1
        if args.iterations is None:
            args.iterations = 1
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    if args.iterations is not None and args.iterations < 0:
        parser.error("--iterations must be >= 0")
    return args


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def make_run_config(
    source_config: Path,
    dest_config: Path,
    seed: int,
    iterations: Optional[int],
    max_total_tokens: Optional[int],
    artifact_variant: bool = False,
) -> Dict[str, Any]:
    config = load_yaml(source_config)
    if "max_program_length" in config and "max_code_length" not in config:
        config["max_code_length"] = config.pop("max_program_length")
    config["random_seed"] = seed
    if iterations is not None:
        config["max_iterations"] = iterations
        config["checkpoint_interval"] = max(1, iterations)
    elif config.get("max_iterations"):
        config["checkpoint_interval"] = max(1, int(config["max_iterations"]))

    database = dict(config.get("database") or {})
    database["random_seed"] = seed
    database.pop("db_path", None)
    database.pop("artifacts_base_path", None)
    config["database"] = database

    llm = dict(config.get("llm") or {})
    llm["random_seed"] = seed
    if max_total_tokens is not None:
        llm["max_total_tokens"] = max_total_tokens
    config["llm"] = llm

    for models_key in ("models", "evaluator_models"):
        if models_key in llm and llm[models_key]:
            llm[models_key] = [dict(model, random_seed=seed) for model in llm[models_key]]

    if artifact_variant:
        prompt = dict(config.get("prompt") or {})
        prompt["include_artifacts"] = True
        prompt.setdefault("max_artifact_bytes", 20 * 1024)
        config["prompt"] = prompt

        evaluator = dict(config.get("evaluator") or {})
        evaluator["enable_artifacts"] = True
        config["evaluator"] = evaluator

    write_yaml(dest_config, config)
    return config


def read_best_info(run_dir: Path) -> Dict[str, Any]:
    info_path = run_dir / "best" / "best_program_info.json"
    if not info_path.exists():
        checkpoint_infos = sorted((run_dir / "checkpoints").glob("checkpoint_*/best_program_info.json"))
        if not checkpoint_infos:
            return {}
        info_path = checkpoint_infos[-1]
    with info_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def score_from_metrics(metrics: Dict[str, Any]) -> Optional[float]:
    for key in ("combined_score", "overall_score", "composite_score", "score", "fitness", "accuracy"):
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    numeric = [v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def parse_token_usage(log_text: str) -> Dict[str, int]:
    usage: Dict[str, int] = {}
    marker = "Final token usage:"
    for line in reversed(log_text.splitlines()):
        if marker not in line:
            continue
        tail = line.split(marker, 1)[1]
        for part in tail.strip().split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            try:
                usage[key] = int(value.strip(","))
            except ValueError:
                pass
        break
    return usage


def record_key(record: Dict[str, Any]) -> tuple:
    return (record["benchmark"], record["strategy"], int(record["run"]))


def load_existing_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def read_run_log_text(run_dir: Path) -> tuple[str, Path]:
    run_log = run_dir / "run.log"
    if run_log.exists():
        return run_log.read_text(encoding="utf-8", errors="replace"), run_log

    logs = sorted((run_dir / "logs").glob("openevolve_*.log"))
    if logs:
        log_path = logs[-1]
        return log_path.read_text(encoding="utf-8", errors="replace"), log_path

    return "", run_log


def recover_completed_run(
    benchmark: Benchmark,
    strategy: str,
    run_index: int,
    seed: int,
    run_dir: Path,
    config_path: Path,
) -> Optional[Dict[str, Any]]:
    best_info = read_best_info(run_dir)
    metrics = best_info.get("metrics") or {}
    if not metrics:
        return None

    log_text, log_path = read_run_log_text(run_dir)
    return {
        "benchmark": benchmark.key,
        "benchmark_name": benchmark.name,
        "strategy": strategy,
        "run": run_index,
        "seed": seed,
        "output_dir": str(run_dir),
        "config": str(config_path),
        "log": str(log_path),
        "command": [],
        "status": "ok",
        "returncode": 0,
        "duration_sec": None,
        "metrics": metrics,
        "primary_score": score_from_metrics(metrics),
        "token_usage": parse_token_usage(log_text),
    }


def run_one(
    benchmark: Benchmark,
    strategy: str,
    run_index: int,
    seed: int,
    run_dir: Path,
    config_path: Path,
    dry_run: bool,
    iterations: Optional[int],
) -> Dict[str, Any]:
    initial_path = REPO_ROOT / benchmark.example_dir / benchmark.initial_program
    evaluator_path = REPO_ROOT / benchmark.example_dir / benchmark.evaluator
    log_path = run_dir / "run.log"
    command = [
        sys.executable,
        "-m",
        "openevolve.cli",
        str(initial_path),
        str(evaluator_path),
        "--config",
        str(config_path),
        "--output",
        str(run_dir),
    ]
    if iterations is not None:
        command.extend(["--iterations", str(iterations)])

    record: Dict[str, Any] = {
        "benchmark": benchmark.key,
        "benchmark_name": benchmark.name,
        "strategy": strategy,
        "run": run_index,
        "seed": seed,
        "output_dir": str(run_dir),
        "config": str(config_path),
        "log": str(log_path),
        "command": command,
        "status": "dry_run" if dry_run else "pending",
        "returncode": None,
        "duration_sec": 0.0,
        "metrics": {},
        "primary_score": None,
        "token_usage": {},
    }
    if dry_run:
        return record

    run_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if benchmark.name == "attention_optimization":
        mlir_bin = _attention_mlir_bin_dir()
        if mlir_bin is not None:
            env["PATH"] = f"{mlir_bin}{os.pathsep}{env.get('PATH', '')}"
    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    duration = time.time() - start
    log_text = log_path.read_text(encoding="utf-8", errors="replace")

    best_info = read_best_info(run_dir)
    metrics = best_info.get("metrics") or {}
    record.update(
        {
            "status": "ok" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "duration_sec": round(duration, 3),
            "metrics": metrics,
            "primary_score": score_from_metrics(metrics),
            "token_usage": parse_token_usage(log_text),
        }
    )
    return record


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_summary_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "benchmark",
        "strategy",
        "run",
        "seed",
        "status",
        "returncode",
        "primary_score",
        "duration_sec",
        "total_tokens",
        "output_dir",
        "log",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            usage = record.get("token_usage") or {}
            writer.writerow(
                {
                    "benchmark": record["benchmark"],
                    "strategy": record["strategy"],
                    "run": record["run"],
                    "seed": record["seed"],
                    "status": record["status"],
                    "returncode": record["returncode"],
                    "primary_score": record["primary_score"],
                    "duration_sec": record["duration_sec"],
                    "total_tokens": usage.get("total", usage.get("total_tokens")),
                    "output_dir": record["output_dir"],
                    "log": record["log"],
                }
            )


def write_aggregate_csv(path: Path, records: List[Dict[str, Any]]) -> None:
    rows = []
    for benchmark in sorted({record["benchmark"] for record in records}):
        for strategy in sorted({record["strategy"] for record in records}):
            group = [
                record
                for record in records
                if record["benchmark"] == benchmark and record["strategy"] == strategy
            ]
            scores = [
                record["primary_score"]
                for record in group
                if record.get("status") == "ok" and record.get("primary_score") is not None
            ]
            rows.append(
                {
                    "benchmark": benchmark,
                    "strategy": strategy,
                    "runs": len(group),
                    "successes": sum(1 for record in group if record.get("status") == "ok"),
                    "mean_score": statistics.fmean(scores) if scores else None,
                    "median_score": statistics.median(scores) if scores else None,
                    "stdev_score": statistics.stdev(scores) if len(scores) > 1 else None,
                    "best_score": max(scores) if scores else None,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "benchmark",
                "strategy",
                "runs",
                "successes",
                "mean_score",
                "median_score",
                "stdev_score",
                "best_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def require_api_key(dry_run: bool, benchmarks: Iterable[str]) -> None:
    if dry_run:
        return
    selected = set(benchmarks)
    non_optillm = selected - {"webscrape"}
    if not non_optillm:
        if os.environ.get("OPTILLM_API_KEY") or os.environ.get("OPENAI_API_KEY"):
            return
        raise SystemExit("OPTILLM_API_KEY or OPENAI_API_KEY must be set for the webscrape benchmark.")

    if (
        os.environ.get("CODESTRAL_API_KEY")
        or os.environ.get("MISTRAL_API_KEY")
        or os.environ.get("NVIDIA_API_KEY")
    ):
        return
    raise SystemExit(
        "CODESTRAL_API_KEY, MISTRAL_API_KEY, or NVIDIA_API_KEY must be set for these benchmark configs."
    )


def main() -> int:
    args = parse_args()
    require_api_key(args.dry_run, args.benchmarks)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else REPO_ROOT / "output" / f"benchmark_matrix_{timestamp}"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    configs_dir = output_root / "configs"
    results_jsonl = output_root / "results.jsonl"
    summary_csv = output_root / "summary.csv"
    aggregate_csv = output_root / "aggregate.csv"

    records: List[Dict[str, Any]] = load_existing_records(results_jsonl) if args.resume else []
    seen_records = {record_key(record) for record in records}
    total = len(args.benchmarks) * len(args.strategies) * args.runs
    print(f"Benchmark matrix: {total} runs")
    print(f"Output root: {output_root}")
    if args.resume and records:
        print(f"Loaded {len(records)} existing result records")

    completed_count = 0
    for bench_key in args.benchmarks:
        benchmark = BENCHMARKS[bench_key]
        for strategy in args.strategies:
            source_config = REPO_ROOT / benchmark.example_dir / f"config_{strategy}.yaml"
            if not source_config.exists():
                raise SystemExit(f"missing config: {source_config}")
            for run_index in range(1, args.runs + 1):
                seed = args.base_seed + run_index - 1
                run_name = f"{benchmark.key}_{strategy}_run_{run_index:02d}"
                run_dir = output_root / run_name
                config_path = configs_dir / f"{run_name}.yaml"
                key = (benchmark.key, strategy, run_index)
                make_run_config(
                    source_config=source_config,
                    dest_config=config_path,
                    seed=seed,
                    iterations=args.iterations,
                    max_total_tokens=args.max_total_tokens,
                    artifact_variant=benchmark.artifact_variant,
                )
                completed_count += 1
                print(f"[{completed_count}/{total}] {benchmark.key} {strategy} run={run_index} seed={seed}")

                if args.resume and key in seen_records:
                    print("  skipped existing result")
                    continue

                if args.resume:
                    recovered = recover_completed_run(
                        benchmark=benchmark,
                        strategy=strategy,
                        run_index=run_index,
                        seed=seed,
                        run_dir=run_dir,
                        config_path=config_path,
                    )
                    if recovered is not None:
                        records.append(recovered)
                        seen_records.add(key)
                        append_jsonl(results_jsonl, recovered)
                        write_summary_csv(summary_csv, records)
                        write_aggregate_csv(aggregate_csv, records)
                        print(f"  recovered score={recovered['primary_score']}")
                        continue

                record = run_one(
                    benchmark=benchmark,
                    strategy=strategy,
                    run_index=run_index,
                    seed=seed,
                    run_dir=run_dir,
                    config_path=config_path,
                    dry_run=args.dry_run,
                    iterations=args.iterations,
                )
                records.append(record)
                seen_records.add(key)
                append_jsonl(results_jsonl, record)
                write_summary_csv(summary_csv, records)
                write_aggregate_csv(aggregate_csv, records)

                if record["status"] == "ok":
                    print(f"  ok score={record['primary_score']} duration={record['duration_sec']}s")
                elif record["status"] == "dry_run":
                    print("  dry-run")
                else:
                    print(f"  failed returncode={record['returncode']} log={record['log']}")
                    if args.fail_fast:
                        return int(record["returncode"] or 1)

    failed = [record for record in records if record["status"] == "failed"]
    print(f"Completed {len(records)} runs with {len(failed)} failures")
    print(f"Summary CSV: {summary_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
