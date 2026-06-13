#!/usr/bin/env python3
"""Plot benchmark matrix results produced by run_benchmark_matrix.py.

Usage
-----
  python scripts/plot_benchmark_results.py output/benchmark_matrix_<timestamp>

The script reads ``results.jsonl`` from the specified output root and produces:

1. **bar_chart.png**       – Mean primary score ± std per benchmark/strategy.
2. **box_plot.png**        – Box-and-whisker score distributions.
3. **convergence.png**     – Per-benchmark best-so-far curves over iterations when
                             ``convergence.jsonl`` files are available, otherwise
                             over completed runs.
4. **summary_table.png**   – Tabular summary rendered as an image.
5. **duration_chart.png**  – Mean wall-clock duration per benchmark/strategy.

All figures are saved inside ``<output_root>/plots/``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


# ---------------------------------------------------------------------------
# Colour palette (accessible, print-friendly)
# ---------------------------------------------------------------------------

STRATEGY_COLORS = {
    "baseline": "#5A9BD5",   # soft blue
    "mcts": "#ED7D31",       # orange
}

STRATEGY_LABELS = {
    "baseline": "Island MAP-Elites (Baseline)",
    "mcts": "TreeQD-MCTS (Ours)",
}

FITNESS_METRIC_PRIORITY = (
    "combined_score",
    "overall_score",
    "composite_score",
    "score",
    "fitness",
    "accuracy",
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_records(results_path: Path) -> List[Dict[str, Any]]:
    """Load JSON-lines records from ``results.jsonl``."""
    records: List[Dict[str, Any]] = []
    with results_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_convergence_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Load per-run convergence traces, falling back to old run.log parsing."""
    traces: List[Dict[str, Any]] = []
    for record in records:
        run_dir = Path(record.get("output_dir") or "")
        trace_path = run_dir / "convergence.jsonl"
        if trace_path.exists():
            try:
                with trace_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        row["benchmark"] = record.get("benchmark")
                        row["benchmark_name"] = record.get("benchmark_name")
                        row["strategy"] = record.get("strategy")
                        row["run"] = record.get("run")
                        row["seed"] = record.get("seed")
                        traces.append(row)
                continue
            except (OSError, json.JSONDecodeError) as exc:
                print(f"WARNING: failed to load {trace_path}: {exc}", file=sys.stderr)

        traces.extend(parse_log_convergence_records(record))
    return traces


def _parse_metric_value(value: str) -> Any:
    value = value.strip()
    try:
        return float(value)
    except ValueError:
        return value


def _parse_metrics_text(text: str) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*)=([^,]+)", text):
        metrics[match.group(1)] = _parse_metric_value(match.group(2))
    return metrics


def _score_from_metrics(metrics: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    for key in FITNESS_METRIC_PRIORITY:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return key, float(value)
    numeric = [v for v in metrics.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if not numeric:
        return None, None
    return None, float(sum(numeric) / len(numeric))


def parse_log_convergence_records(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Reconstruct iteration-level convergence from old textual run logs."""
    log_path = Path(record.get("log") or "")
    if not log_path.exists():
        run_dir = Path(record.get("output_dir") or "")
        candidate = run_dir / "run.log"
        if candidate.exists():
            log_path = candidate
        else:
            return []

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError as exc:
        print(f"WARNING: failed to read {log_path}: {exc}", file=sys.stderr)
        return []

    rows: List[Dict[str, Any]] = []
    best_score: Optional[float] = None
    best_program_id: Optional[str] = None
    pending_iteration: Optional[int] = None
    pending_program_id: Optional[str] = None
    pending_parent_id: Optional[str] = None
    pending_iteration_time: Optional[float] = None
    search_strategy = None
    if "mcts" in str(record.get("strategy")):
        search_strategy = "tree_qd_mcts"
    elif "baseline" in str(record.get("strategy")):
        search_strategy = "island_map_elites"

    for line in lines:
        if "Evaluated program " in line and " in " in line and ": " in line:
            match = re.search(r"Evaluated program ([0-9a-f-]+) in ([0-9.]+)s: (.*)$", line)
            if match:
                metrics = _parse_metrics_text(match.group(3))
                primary_metric, score = _score_from_metrics(metrics)
                if score is not None:
                    best_score = score if best_score is None else max(best_score, score)
                    if best_score == score:
                        best_program_id = match.group(1)
                rows.append(
                    {
                        "benchmark": record.get("benchmark"),
                        "benchmark_name": record.get("benchmark_name"),
                        "strategy": record.get("strategy"),
                        "run": record.get("run"),
                        "seed": record.get("seed"),
                        "iteration": 0,
                        "event": "initial",
                        "status": "ok",
                        "program_id": match.group(1),
                        "parent_id": None,
                        "score": score,
                        "best_score": best_score,
                        "best_program_id": best_program_id,
                        "primary_metric": primary_metric,
                        "metrics": metrics,
                        "iteration_time": float(match.group(2)),
                        "search_strategy": search_strategy,
                    }
                )
                continue

        iteration_match = re.search(
            r"Iteration (\d+): Program ([0-9a-f-]+) \(parent: ([0-9a-f-]+|None)\) completed in ([0-9.]+)s",
            line,
        )
        if iteration_match:
            pending_iteration = int(iteration_match.group(1))
            pending_program_id = iteration_match.group(2)
            pending_parent_id = iteration_match.group(3)
            if pending_parent_id == "None":
                pending_parent_id = None
            pending_iteration_time = float(iteration_match.group(4))
            continue

        if pending_iteration is not None and "Metrics:" in line:
            metrics = _parse_metrics_text(line.split("Metrics:", 1)[1])
            primary_metric, score = _score_from_metrics(metrics)
            if score is not None:
                best_score = score if best_score is None else max(best_score, score)
                if best_score == score:
                    best_program_id = pending_program_id
            rows.append(
                {
                    "benchmark": record.get("benchmark"),
                    "benchmark_name": record.get("benchmark_name"),
                    "strategy": record.get("strategy"),
                    "run": record.get("run"),
                    "seed": record.get("seed"),
                    "iteration": pending_iteration,
                    "event": "candidate",
                    "status": "ok",
                    "program_id": pending_program_id,
                    "parent_id": pending_parent_id,
                    "score": score,
                    "best_score": best_score,
                    "best_program_id": best_program_id,
                    "primary_metric": primary_metric,
                    "metrics": metrics,
                    "iteration_time": pending_iteration_time,
                    "search_strategy": search_strategy,
                    "error": metrics.get("error"),
                }
            )
            pending_iteration = None
            pending_program_id = None
            pending_parent_id = None
            pending_iteration_time = None
            continue

        error_match = re.search(r"Iteration (\d+) error: (.*)$", line)
        if error_match:
            rows.append(
                {
                    "benchmark": record.get("benchmark"),
                    "benchmark_name": record.get("benchmark_name"),
                    "strategy": record.get("strategy"),
                    "run": record.get("run"),
                    "seed": record.get("seed"),
                    "iteration": int(error_match.group(1)),
                    "event": "error",
                    "status": "error",
                    "program_id": None,
                    "parent_id": None,
                    "score": None,
                    "best_score": best_score,
                    "best_program_id": best_program_id,
                    "primary_metric": None,
                    "metrics": {},
                    "iteration_time": None,
                    "search_strategy": search_strategy,
                    "error": error_match.group(2),
                }
            )

    if not rows:
        return []

    seen_iterations = set()
    deduped: List[Dict[str, Any]] = []
    for row in sorted(rows, key=lambda r: (int(r.get("iteration") or 0), r.get("event") != "initial")):
        key = (row.get("iteration"), row.get("program_id"), row.get("event"))
        if key in seen_iterations:
            continue
        seen_iterations.add(key)
        deduped.append(row)
    return deduped


def group_scores(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    """Group primary scores by (benchmark, strategy).

    Returns:
        ``{benchmark_key: {strategy: [score, ...]}}``
    """
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("status") != "ok":
            continue
        score = r.get("primary_score")
        if score is None:
            continue
        grouped[r["benchmark"]][r["strategy"]].append(float(score))
    return grouped


def group_durations(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, List[float]]]:
    """Group wall-clock durations by (benchmark, strategy)."""
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("status") != "ok":
            continue
        dur = r.get("duration_sec")
        if dur is None:
            continue
        grouped[r["benchmark"]][r["strategy"]].append(float(dur))
    return grouped


# ---------------------------------------------------------------------------
# Benchmark display names
# ---------------------------------------------------------------------------

BENCHMARK_NAMES = {
    "func": "Function Minimization",
    "cp": "Circle Packing",
    "sp": "Signal Processing",
    "kmod": "K-Module Problem",
    "cp_art": "Circle Packing (Artifacts)",
    "tsp": "TSP Tour Minimization",
    "attn": "Attention Optimization",
    "ojp": "Online Judge Programming",
    "r_reg": "R Robust Regression",
    "rust_sort": "Rust Adaptive Sort",
    "mlx": "MLX Metal Kernel Opt",
    "arc": "ARC Benchmark",
    "webscrape": "Web Scraper optillm",
}


def _bname(key: str) -> str:
    if key in BENCHMARK_NAMES:
        return BENCHMARK_NAMES[key]
    # Auto-format: replace underscores with spaces and title-case
    return key.replace("_", " ").title()


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _apply_style(ax: plt.Axes) -> None:
    """Apply a clean, publication-quality style to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(str(path), dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_bar_chart(
    grouped: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    """Mean score ± std bar chart."""
    benchmarks = sorted(grouped.keys())
    strategies = sorted({s for scores in grouped.values() for s in scores})

    fig, ax = plt.subplots(figsize=(max(6, len(benchmarks) * 2.5), 5))
    _apply_style(ax)

    x = np.arange(len(benchmarks))
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(strategies))

    for i, strat in enumerate(strategies):
        means, stds = [], []
        for bk in benchmarks:
            scores = grouped[bk].get(strat, [])
            means.append(statistics.fmean(scores) if scores else 0)
            stds.append(statistics.stdev(scores) if len(scores) > 1 else 0)
        bars = ax.bar(
            x + offsets[i],
            means,
            width * 0.9,
            yerr=stds,
            capsize=4,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat, None),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )
        # Annotate mean values
        for bar, m in zip(bars, means):
            if m > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * max(means),
                    f"{m:.3f}",
                    ha="center", va="bottom", fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_bname(b) for b in benchmarks], fontsize=11)
    ax.set_ylabel("Mean Primary Score", fontsize=12)
    ax.set_title("Benchmark Comparison: MCTS vs Baseline", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")

    _save(fig, out_dir / "bar_chart.png")


def plot_box_plot(
    grouped: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    """Box-and-whisker plot of score distributions."""
    benchmarks = sorted(grouped.keys())
    strategies = sorted({s for scores in grouped.values() for s in scores})

    fig, axes = plt.subplots(
        1, len(benchmarks),
        figsize=(max(4, len(benchmarks) * 3.5), 5),
        sharey=False,
    )
    if len(benchmarks) == 1:
        axes = [axes]

    for idx, bk in enumerate(benchmarks):
        ax = axes[idx]
        _apply_style(ax)
        data, labels, colors = [], [], []
        for strat in strategies:
            scores = grouped[bk].get(strat, [])
            if scores:
                data.append(scores)
                labels.append(STRATEGY_LABELS.get(strat, strat))
                colors.append(STRATEGY_COLORS.get(strat, "#888888"))

        if data:
            bp = ax.boxplot(
                data,
                patch_artist=True,
                tick_labels=[l.split("(")[0].strip() for l in labels],
                widths=0.5,
                showmeans=True,
                meanprops=dict(marker="D", markerfacecolor="black", markersize=5),
            )
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(_bname(bk), fontsize=12, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Primary Score", fontsize=11)

    fig.suptitle("Score Distributions per Benchmark", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "box_plot.png")


def plot_convergence(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Best-so-far curves per benchmark."""
    trace_rows = load_convergence_records(records)
    if trace_rows:
        plot_iteration_convergence(trace_rows, out_dir)
    else:
        plot_run_convergence(records, out_dir)


def plot_run_convergence(records: List[Dict[str, Any]], out_dir: Path) -> None:
    """Fallback best-so-far curves over run index when iteration traces are absent."""
    # Group by (benchmark, strategy)
    runs: Dict[str, Dict[str, List[tuple]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("status") != "ok":
            continue
        score = r.get("primary_score")
        if score is None:
            continue
        runs[r["benchmark"]][r["strategy"]].append((int(r["run"]), float(score)))

    benchmarks = sorted(runs.keys())
    if not benchmarks:
        return

    ncols = min(2, len(benchmarks))
    nrows = (len(benchmarks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)

    for idx, bk in enumerate(benchmarks):
        ax = axes[idx // ncols][idx % ncols]
        _apply_style(ax)

        for strat in sorted(runs[bk]):
            pairs = sorted(runs[bk][strat], key=lambda p: p[0])
            run_indices = [p[0] for p in pairs]
            scores = [p[1] for p in pairs]
            # Cumulative best
            best_so_far = []
            current_best = float("-inf")
            for s in scores:
                current_best = max(current_best, s)
                best_so_far.append(current_best)

            ax.plot(
                run_indices,
                best_so_far,
                marker="o",
                markersize=4,
                label=STRATEGY_LABELS.get(strat, strat),
                color=STRATEGY_COLORS.get(strat, None),
                linewidth=2,
            )
            # Also plot individual scores as faded dots
            ax.scatter(
                run_indices,
                scores,
                color=STRATEGY_COLORS.get(strat, None),
                alpha=0.3,
                s=20,
                zorder=1,
            )

        ax.set_xlabel("Run Index", fontsize=10)
        ax.set_ylabel("Score", fontsize=10)
        ax.set_title(_bname(bk), fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # Hide unused subplots
    for idx in range(len(benchmarks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Convergence: Best-so-far per Run", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "convergence.png")


def plot_iteration_convergence(trace_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    """Plot mean best-so-far score over evolution iterations."""
    grouped_runs: Dict[str, Dict[str, Dict[int, Dict[int, float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for row in trace_rows:
        best_score = row.get("best_score")
        iteration = row.get("iteration")
        run = row.get("run")
        if best_score is None or iteration is None or run is None:
            continue
        grouped_runs[row["benchmark"]][row["strategy"]][int(run)][int(iteration)] = float(best_score)

    benchmarks = sorted(grouped_runs.keys())
    if not benchmarks:
        return

    ncols = min(2, len(benchmarks))
    nrows = (len(benchmarks) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)

    for idx, bk in enumerate(benchmarks):
        ax = axes[idx // ncols][idx % ncols]
        _apply_style(ax)

        for strat in sorted(grouped_runs[bk]):
            runs = grouped_runs[bk][strat]
            iterations = sorted({it for run_rows in runs.values() for it in run_rows})
            if not iterations:
                continue

            means = []
            lows = []
            highs = []
            for iteration in iterations:
                values = []
                for run_rows in runs.values():
                    previous = [it for it in run_rows if it <= iteration]
                    if previous:
                        values.append(run_rows[max(previous)])
                means.append(statistics.fmean(values))
                lows.append(min(values))
                highs.append(max(values))

            color = STRATEGY_COLORS.get(strat, None)
            ax.plot(
                iterations,
                means,
                marker="o",
                markersize=3,
                label=STRATEGY_LABELS.get(strat, strat),
                color=color,
                linewidth=2,
            )
            if len(runs) > 1:
                ax.fill_between(iterations, lows, highs, color=color, alpha=0.15)

        ax.set_xlabel("Iteration", fontsize=10)
        ax.set_ylabel("Best Score So Far", fontsize=10)
        ax.set_title(_bname(bk), fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    for idx in range(len(benchmarks), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Convergence: Best-so-far per Iteration", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "convergence.png")


def plot_duration_chart(
    durations: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    """Mean duration bar chart."""
    benchmarks = sorted(durations.keys())
    strategies = sorted({s for d in durations.values() for s in d})

    fig, ax = plt.subplots(figsize=(max(6, len(benchmarks) * 2.5), 5))
    _apply_style(ax)

    x = np.arange(len(benchmarks))
    width = 0.35
    offsets = np.linspace(-width / 2, width / 2, len(strategies))

    for i, strat in enumerate(strategies):
        means = []
        for bk in benchmarks:
            durs = durations[bk].get(strat, [])
            means.append(statistics.fmean(durs) if durs else 0)
        ax.bar(
            x + offsets[i],
            means,
            width * 0.9,
            label=STRATEGY_LABELS.get(strat, strat),
            color=STRATEGY_COLORS.get(strat, None),
            edgecolor="white",
            linewidth=0.5,
            alpha=0.9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([_bname(b) for b in benchmarks], fontsize=11)
    ax.set_ylabel("Mean Duration (seconds)", fontsize=12)
    ax.set_title("Wall-Clock Duration per Benchmark", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="best")

    _save(fig, out_dir / "duration_chart.png")


def plot_summary_table(
    grouped: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
) -> None:
    """Render aggregate statistics as a table image."""
    benchmarks = sorted(grouped.keys())
    strategies = sorted({s for scores in grouped.values() for s in scores})

    headers = ["Benchmark", "Strategy", "Runs", "Mean", "Std", "Median", "Best"]
    rows = []
    for bk in benchmarks:
        for strat in strategies:
            scores = grouped[bk].get(strat, [])
            n = len(scores)
            if n == 0:
                rows.append([_bname(bk), STRATEGY_LABELS.get(strat, strat), "0", "-", "-", "-", "-"])
            else:
                rows.append([
                    _bname(bk),
                    STRATEGY_LABELS.get(strat, strat).split("(")[0].strip(),
                    str(n),
                    f"{statistics.fmean(scores):.4f}",
                    f"{statistics.stdev(scores):.4f}" if n > 1 else "-",
                    f"{statistics.median(scores):.4f}",
                    f"{max(scores):.4f}",
                ])

    fig, ax = plt.subplots(figsize=(12, 1 + 0.4 * len(rows)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    # Style header
    for j, header in enumerate(headers):
        cell = table[0, j]
        cell.set_facecolor("#4472C4")
        cell.set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(len(rows)):
        color = "#F2F2F2" if i % 2 == 0 else "white"
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor(color)

    ax.set_title("Aggregate Results Summary", fontsize=14, fontweight="bold", pad=20)
    _save(fig, out_dir / "summary_table.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Plot benchmark matrix results.")
    parser.add_argument(
        "output_root",
        help="Path to benchmark matrix output directory (contains results.jsonl).",
    )
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output image format.",
    )
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    results_path = output_root / "results.jsonl"

    if not results_path.exists():
        print(f"ERROR: {results_path} not found.", file=sys.stderr)
        return 1

    records = load_records(results_path)
    if not records:
        print("ERROR: No records found in results.jsonl.", file=sys.stderr)
        return 1

    ok_records = [r for r in records if r.get("status") == "ok"]
    print(f"Loaded {len(records)} records ({len(ok_records)} successful)")

    plots_dir = output_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    grouped = group_scores(records)
    durations = group_durations(records)

    print("Generating plots...")
    plot_bar_chart(grouped, plots_dir)
    plot_box_plot(grouped, plots_dir)
    plot_convergence(records, plots_dir)
    plot_duration_chart(durations, plots_dir)
    plot_summary_table(grouped, plots_dir)

    print(f"\nAll plots saved to: {plots_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
