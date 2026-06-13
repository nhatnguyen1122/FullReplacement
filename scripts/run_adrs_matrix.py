#!/usr/bin/env python3
"""Run ADRS benchmarks with FullReplacementMCTS/OpenEvolve."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
SOURCE_ROOT = WORKSPACE / "AdaEvolveMCTS_outer_level_mcts" / "benchmarks" / "ADRS"
BENCHMARKS = {
    "cloudcast": "cloudcast",
    "eplb": "eplb",
    "llm_sql": "llm_sql",
    "prism": "prism",
    "txn_scheduling": "txn_scheduling",
}
ALIASES = {"txn": "txn_scheduling", "sql": "llm_sql"}
STRATEGIES = {"mcts": "tree_qd_mcts", "baseline": "island_map_elites"}


def parse_csv(value: str, valid: set[str], aliases: dict[str, str] | None = None) -> list[str]:
    aliases = aliases or {}
    items = [aliases.get(x.strip(), x.strip()) for x in value.split(",") if x.strip()]
    unknown = sorted(set(items) - valid)
    if unknown:
        raise ValueError(f"Unknown value(s): {', '.join(unknown)}")
    return items


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_last_jsonl(path: Path) -> dict[str, Any]:
    last: dict[str, Any] = {}
    if not path.exists():
        return last
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                last = json.loads(line)
    return last


def prepare_config(benchmark: str, strategy: str, output_root: Path, iterations: int) -> Path:
    with (SOURCE_ROOT / benchmark / "config.yaml").open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    cfg = {
        "language": "python",
        "diff_based_evolution": bool(base.get("diff_based_generation", True)),
        "max_iterations": iterations,
        "checkpoint_interval": min(25, max(1, iterations)) if iterations else 1,
        "max_code_length": int(base.get("max_solution_length", 60000)),
        "llm": {
            "api_base": "https://integrate.api.nvidia.com/v1",
            "api_key": "${NVIDIA_API_KEY}",
            "models": [{"name": "openai/gpt-oss-120b", "weight": 1.0}],
            "evaluator_models": [{"name": "openai/gpt-oss-120b", "weight": 1.0}],
            "temperature": 0.2,
            "max_tokens": 4096,
            "timeout": 600,
            "retries": 3,
            "retry_delay": 5,
            "max_total_tokens": 5_000_000,
        },
        "prompt": {
            "system_message": (base.get("prompt") or {}).get("system_message", ""),
            "num_top_programs": 3,
            "use_template_stochasticity": True,
        },
        "database": {"population_size": 80, "archive_size": 30, "num_islands": 4},
        "evaluator": {
            "timeout": int((base.get("evaluator") or {}).get("timeout", 600)),
            "max_retries": 3,
            "parallel_evaluations": 1,
            "cascade_evaluation": bool((base.get("evaluator") or {}).get("cascade_evaluation", True)),
            "cascade_thresholds": (base.get("evaluator") or {}).get("cascade_thresholds", [0.5, 0.75]),
        },
        "search": {
            "strategy": STRATEGIES[strategy],
            "exploration_constant": 0.7,
            "qd_weight": 0.5,
            "max_tree_depth": 15,
            "backup": "mixed",
            "backup_alpha": 0.5,
            "widening_alpha": 0.5,
        },
    }
    cfg_dir = output_root / "_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    path = cfg_dir / f"{benchmark}_{strategy}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def summarize(benchmark: str, strategy: str, run: int, run_dir: Path, returncode: int, elapsed: float) -> dict[str, Any]:
    best = read_json(run_dir / "best" / "best_program_info.json")
    conv = read_last_jsonl(run_dir / "convergence.jsonl")
    metrics = best.get("metrics") or conv.get("metrics") or {}
    return {
        "benchmark": benchmark,
        "strategy": strategy,
        "run": run,
        "status": "ok" if returncode == 0 else "failed",
        "returncode": returncode,
        "output_dir": str(run_dir),
        "elapsed_seconds": round(elapsed, 3),
        "primary_score": metrics.get("combined_score", conv.get("best_score")),
        "metrics": metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    parser.add_argument("--strategies", default="mcts")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-root", default="output/adrs_fullreplacement")
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    benchmarks = parse_csv(args.benchmarks, set(BENCHMARKS), ALIASES)
    strategies = parse_csv(args.strategies, set(STRATEGIES))
    if not args.dry_run and not os.environ.get("NVIDIA_API_KEY"):
        parser.error("NVIDIA_API_KEY is not set")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for benchmark in benchmarks:
        for strategy in strategies:
            cfg = prepare_config(benchmark, strategy, output_root, args.iterations)
            for run_idx in range(1, args.runs + 1):
                run_dir = output_root / benchmark / strategy / f"run_{run_idx:02d}"
                if args.skip_complete and (run_dir / "best" / "best_program_info.json").exists():
                    row = summarize(benchmark, strategy, run_idx, run_dir, 0, 0.0)
                    row["status"] = "skipped"
                    rows.append(row)
                    continue
                cmd = [
                    sys.executable, "-m", "openevolve.cli",
                    str(SOURCE_ROOT / benchmark / "initial_program.py"),
                    str(SOURCE_ROOT / benchmark / "evaluator" / "evaluator.py"),
                    "--config", str(cfg),
                    "--output", str(run_dir),
                    "--iterations", str(args.iterations),
                ]
                print("+ " + " ".join(cmd))
                if args.dry_run:
                    continue
                run_dir.mkdir(parents=True, exist_ok=True)
                start = time.time()
                with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                    proc = subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
                row = summarize(benchmark, strategy, run_idx, run_dir, proc.returncode, time.time() - start)
                rows.append(row)
                with (output_root / "results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                print(f"[{benchmark} {strategy} run {run_idx}] {row['status']} score={row['primary_score']}")
                if args.fail_fast and proc.returncode != 0:
                    return proc.returncode or 1
    if rows:
        keys = sorted({k for row in rows for k in row if k != "metrics"})
        with (output_root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows([{k: v for k, v in row.items() if k != "metrics"} for row in rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
