#!/usr/bin/env python3
"""Run Frontier-CS problems with FullReplacementMCTS/OpenEvolve."""

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
SOURCE_ROOT = WORKSPACE / "AdaEvolveMCTS_outer_level_mcts" / "benchmarks" / "frontier-cs-eval"
FRONTIER_DIR = SOURCE_ROOT / "Frontier-CS"
STRATEGIES = {"mcts": "tree_qd_mcts", "baseline": "island_map_elites"}


def available_problems() -> list[str]:
    problems = FRONTIER_DIR / "algorithmic" / "problems"
    if not problems.exists():
        return []
    return sorted((p.name for p in problems.iterdir() if p.is_dir() and p.name.isdigit()), key=int)


def parse_csv(value: str, valid: set[str] | None = None) -> list[str]:
    if value == "all" and valid is None:
        problems = available_problems()
        if not problems:
            raise ValueError("No Frontier-CS problems found. Clone Frontier-CS first.")
        return problems
    items = [x.strip() for x in value.split(",") if x.strip()]
    if valid is not None:
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


def problem_prompt(problem_id: str) -> str:
    statement = FRONTIER_DIR / "algorithmic" / "problems" / problem_id / "statement.txt"
    config = FRONTIER_DIR / "algorithmic" / "problems" / problem_id / "config.yaml"
    statement_text = statement.read_text(encoding="utf-8") if statement.exists() else f"Frontier-CS problem {problem_id}"
    config_text = config.read_text(encoding="utf-8") if config.exists() else ""
    return f"""You are an expert competitive programmer specializing in algorithmic optimization.

FRONTIER-CS PROBLEM ID: {problem_id}

PROBLEM STATEMENT:
{statement_text}

CONSTRAINTS AND JUDGE CONFIG:
{config_text}

OBJECTIVE: Maximize the score returned by the Frontier-CS judge. Higher is better.
Your solution must be valid C++ code with main(), reading from stdin and writing to stdout.
Return complete C++ code only.
"""


def prepare_config(problem_id: str, strategy: str, output_root: Path, iterations: int) -> Path:
    with (SOURCE_ROOT / "config.yaml").open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    cfg = {
        "language": "cpp",
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
            "system_message": problem_prompt(problem_id),
            "num_top_programs": 3,
            "use_template_stochasticity": True,
        },
        "database": {"population_size": 80, "archive_size": 30, "num_islands": 4},
        "evaluator": {
            "timeout": int((base.get("evaluator") or {}).get("timeout", 600)),
            "max_retries": 3,
            "parallel_evaluations": 1,
            "cascade_evaluation": False,
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
    path = cfg_dir / f"problem_{problem_id}_{strategy}.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def summarize(problem_id: str, strategy: str, run: int, run_dir: Path, returncode: int, elapsed: float) -> dict[str, Any]:
    best = read_json(run_dir / "best" / "best_program_info.json")
    conv = read_last_jsonl(run_dir / "convergence.jsonl")
    metrics = best.get("metrics") or conv.get("metrics") or {}
    return {
        "problem": problem_id,
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
    parser.add_argument("--problems", default="0", help="Comma list of problem IDs, or 'all'.")
    parser.add_argument("--strategies", default="mcts")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-root", default="output/frontiercs_fullreplacement")
    parser.add_argument("--judge-urls", default=os.environ.get("JUDGE_URLS", "http://localhost:8081"))
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()
    problems = parse_csv(args.problems)
    strategies = parse_csv(args.strategies, set(STRATEGIES))
    if not args.dry_run and not os.environ.get("NVIDIA_API_KEY"):
        parser.error("NVIDIA_API_KEY is not set")
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for problem_id in problems:
        for strategy in strategies:
            cfg = prepare_config(problem_id, strategy, output_root, args.iterations)
            for run_idx in range(1, args.runs + 1):
                run_dir = output_root / f"problem_{problem_id}" / strategy / f"run_{run_idx:02d}"
                if args.skip_complete and (run_dir / "best" / "best_program_info.json").exists():
                    row = summarize(problem_id, strategy, run_idx, run_dir, 0, 0.0)
                    row["status"] = "skipped"
                    rows.append(row)
                    continue
                cmd = [
                    sys.executable,
                    "-m",
                    "openevolve.cli",
                    str(SOURCE_ROOT / "initial_program.cpp"),
                    str(SOURCE_ROOT / "evaluator.py"),
                    "--config",
                    str(cfg),
                    "--output",
                    str(run_dir),
                    "--iterations",
                    str(args.iterations),
                ]
                print("+ " + " ".join(cmd))
                if args.dry_run:
                    continue
                env = os.environ.copy()
                env["FRONTIER_CS_PROBLEM"] = str(problem_id)
                env["JUDGE_URLS"] = args.judge_urls
                run_dir.mkdir(parents=True, exist_ok=True)
                start = time.time()
                with (run_dir / "run.log").open("w", encoding="utf-8") as log:
                    proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                row = summarize(problem_id, strategy, run_idx, run_dir, proc.returncode, time.time() - start)
                rows.append(row)
                with (output_root / "results.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, default=str) + "\n")
                print(f"[problem {problem_id} {strategy} run {run_idx}] {row['status']} score={row['primary_score']}")
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
