# Benchmark Scoring Mechanism

This document explains how the currently used FullReplacementMCTS/OpenEvolve
benchmarks turn a candidate program into scalar scores for evolution and
plotting. It covers the four benchmarks currently used for parity runs:

- Circle packing
- Function minimization
- K-module pipeline configuration
- Signal processing

The short version is:

1. Each benchmark has an `evaluator.py`.
2. The evaluator executes the candidate program and returns a dictionary of
   metrics, or an `EvaluationResult(metrics=..., artifacts=...)`.
3. The search algorithm and comparison scripts need one scalar score to decide
   whether a candidate is better.
4. Higher scores are always better.
5. `combined_score` is the preferred main score when present, but not every
   benchmark uses that name.

## Overall Scoring Flow

For every generated candidate program, FullReplacementMCTS/OpenEvolve calls the
benchmark evaluator. The evaluator is problem-specific. It knows what function
the candidate must provide, how to test that function, and how to convert the
test result into metrics.

A typical evaluator returns something like this:

```python
{
    "some_raw_metric": 123.0,
    "some_derived_score": 0.82,
    "combined_score": 0.82,
}
```

Some evaluators return:

```python
EvaluationResult(
    metrics={
        "combined_score": 0.82,
        "other_metric": 0.4,
    },
    artifacts={
        "human_readable_feedback": "...",
    },
)
```

The important distinction is:

- Metrics are numeric values used for scoring, logging, plotting, or feature
  tracking.
- Artifacts are extra feedback text/data for the LLM or for inspection. They
  are not normally the scalar optimization target.

## Primary Score Extraction

The benchmark matrix and plotting scripts use a fixed priority order to extract
one scalar `primary_score` from the evaluator metrics:

```text
combined_score
overall_score
composite_score
score
fitness
accuracy
```

The first numeric metric found in that order becomes the `primary_score`.

If none of those keys exist, the script falls back to the average of all numeric
metrics. For the four benchmarks in this document, a named score exists, so the
fallback should not matter.

This means:

- `primary_score` is not a separate benchmark concept.
- `primary_score` is the comparison/logging name for "the scalar score selected
  from this benchmark's metrics."
- For circle packing, function minimization, and k-module, `primary_score` is
  usually `combined_score`.
- For signal processing, `primary_score` is usually `overall_score`, because
  that evaluator returns `overall_score` and `composite_score`, but not
  `combined_score`.

## Combined Score vs Overall Score vs Composite Score

These names are conventions, not universal mathematical objects.

`combined_score` usually means "the main scalar objective for this benchmark."
OpenEvolve examples often recommend always returning this key.

`overall_score` usually means "a top-level aggregate score" when the benchmark
has several sub-scores. In the signal processing benchmark, this is the primary
selection metric.

`composite_score` usually means "a score built from several penalty or quality
terms." In the signal processing benchmark, it measures filtering behavior on
each test signal, then gets averaged. However, the final primary score is
`overall_score`, not `composite_score`.

`primary_score` is added by the matrix/export scripts. It is whichever score was
selected using the priority order above.

`best_score` in convergence logs is the cumulative maximum of `primary_score`
seen so far in that run.

## Cascade Evaluation

Some benchmarks define both:

- `evaluate_stage1`
- `evaluate_stage2`

Stage 1 is a cheaper or quicker validation/scoring pass. Stage 2 is the full
evaluation. Depending on config, OpenEvolve can use cascade evaluation to avoid
spending full evaluation cost on clearly broken candidates.

The exact formulas can differ slightly between stage 1 and full evaluation.
When comparing final benchmark performance, focus on the full evaluator's final
metrics and the logged `primary_score`.

## Benchmark 1: Circle Packing

Path:

```text
FullReplacementMCTS/examples/circle_packing/evaluator.py
```

### Candidate Interface

The candidate program must define:

```python
run_packing()
```

It should return:

```python
centers, radii, sum_radii
```

Expected shapes:

- `centers`: NumPy-compatible array with shape `(26, 2)`
- `radii`: NumPy-compatible array with shape `(26,)`
- `sum_radii`: reported sum of the radii

The benchmark is for packing 26 circles inside the unit square.

### Raw Objective

The raw problem-specific objective is:

```text
sum_radii = sum of all 26 circle radii
```

Higher `sum_radii` is better, but only if the packing is valid.

### Validity Checks

A candidate receives no useful score unless the packing is valid. The evaluator
checks that:

- The returned centers and radii can be converted to arrays.
- There are no NaN values.
- Shapes are exactly `(26, 2)` and `(26,)`.
- Radii are valid.
- Each circle is inside the unit square.
- Circles do not overlap.

If these checks fail, the evaluator sets:

```text
sum_radii = 0
target_ratio = 0
validity = 0
combined_score = 0
```

### Target Normalization

The evaluator uses:

```text
TARGET_VALUE = 2.635
```

This is described in the code as the AlphaEvolve result for `n = 26`.

For a valid packing:

```text
target_ratio = sum_radii / 2.635
validity = 1
combined_score = target_ratio * validity
```

Because `validity` is either `0` or `1`, this is effectively:

```text
combined_score = sum_radii / 2.635   if valid
combined_score = 0                   if invalid
```

### How To Interpret Scores

Examples:

```text
sum_radii = 2.000  -> combined_score = 2.000 / 2.635 = 0.759
sum_radii = 2.635  -> combined_score = 1.000
sum_radii = 2.700  -> combined_score = 1.025
invalid packing    -> combined_score = 0.000
```

So yes, for circle packing, the score is directly related to the
problem-specific metric `sum_radii`. The score is the valid normalized
`sum_radii`.

Primary score:

```text
primary_score = combined_score
```

## Benchmark 2: Function Minimization

Path:

```text
FullReplacementMCTS/examples/function_minimization/evaluator.py
```

### Candidate Interface

The candidate program must define:

```python
run_search()
```

It should return either:

```python
(x, y, value)
```

or:

```python
(x, y)
```

If only `(x, y)` is returned, the evaluator computes the function value itself.

### Raw Objective

The mathematical objective is to minimize:

```text
f(x, y) = sin(x) * cos(y) + sin(x * y) + (x^2 + y^2) / 20
```

The evaluator uses this approximate known global minimum:

```text
GLOBAL_MIN_X = -1.704
GLOBAL_MIN_Y = 0.678
GLOBAL_MIN_VALUE = -1.519
```

Lower function value is better in the raw mathematical problem. However, the
benchmark converts everything into scores where higher is better.

### Trial Structure

Full evaluation runs:

```text
num_trials = 10
timeout per trial = 5 seconds
```

For each successful trial, the evaluator records:

- `x`
- `y`
- `value`
- distance from `(x, y)` to the known minimum location
- runtime

Invalid returns, NaN/inf values, timeouts, and crashes are skipped.

If all trials fail:

```text
combined_score = 0
value_score = 0
distance_score = 0
reliability_score = 0
```

### Derived Metrics

After successful trials, the evaluator computes:

```text
avg_value = mean(successful returned function values)
avg_distance = mean(distance to known minimum)
reliability_score = successful_trials / 10
```

Then it converts raw error into maximization scores:

```text
value_score = 1 / (1 + abs(avg_value - GLOBAL_MIN_VALUE))
distance_score = 1 / (1 + avg_distance)
```

Interpretation:

- `value_score` is high when the average function value is close to `-1.519`.
- `distance_score` is high when the average `(x, y)` is close to
  `(-1.704, 0.678)`.
- `reliability_score` is high when the program succeeds consistently.

### Solution Quality Multiplier

The evaluator then applies a multiplier based on `avg_distance`:

```text
avg_distance < 0.5  -> multiplier = 1.5
avg_distance < 1.5  -> multiplier = 1.2
avg_distance < 3.0  -> multiplier = 1.0
otherwise           -> multiplier = 0.7
```

This explicitly rewards candidates that are in the correct region of the
search space.

### Final Score

The base score is:

```text
base_score =
    0.5 * value_score
  + 0.3 * distance_score
  + 0.2 * reliability_score
```

The final score is:

```text
combined_score = base_score * solution_quality_multiplier
```

Primary score:

```text
primary_score = combined_score
```

### How To Interpret Scores

This benchmark does not directly optimize the raw function value alone. It
optimizes a weighted score that combines:

- closeness of function value to the known optimum
- closeness of coordinates to the known optimum
- reliability across repeated calls
- a bonus/penalty for being near or far from the known optimum location

Because the multiplier can be `1.5`, `combined_score` can exceed `1.0`.

For example, a candidate with excellent value, excellent location, and perfect
reliability can get a score near:

```text
base_score near 1.0
combined_score near 1.5
```

So for function minimization, the score is related to the problem-specific
metric `f(x, y)`, but it is not simply `-f(x, y)` or `1 / f(x, y)`. It also uses
distance to the known minimizer and reliability.

## Benchmark 3: K-Module Pipeline Configuration

Path:

```text
FullReplacementMCTS/examples/k_module_problem/evaluator.py
```

### Candidate Interface

The candidate program must define one of:

```python
run_pipeline()
```

or:

```python
configure_pipeline()
```

It must return a dictionary describing a four-module pipeline.

### Search Space

The four required modules are:

```text
loader
preprocess
algorithm
formatter
```

Each module has five valid options:

```text
loader:
  csv_reader, json_reader, xml_reader, parquet_reader, sql_reader

preprocess:
  normalize, standardize, minmax, scale, none

algorithm:
  quicksort, mergesort, heapsort, bubblesort, insertion

formatter:
  json, xml, csv, yaml, protobuf
```

Total search space:

```text
5^4 = 625 configurations
```

### Hidden Correct Configuration

The evaluator's target configuration is:

```python
{
    "loader": "csv_reader",
    "preprocess": "normalize",
    "algorithm": "quicksort",
    "formatter": "json",
}
```

### Validation

The evaluator checks:

- The return value is a dictionary.
- All four required modules are present.
- Each module value is one of the valid options.

If validation fails:

```text
correct_modules = 0
accuracy = 0
combined_score = 0
```

### Final Score

The evaluator counts how many module choices match the hidden correct
configuration:

```text
correct_modules = number of correct module choices
total_modules = 4
accuracy = correct_modules / 4
combined_score = accuracy
```

Possible scores are discrete:

```text
0 correct -> combined_score = 0.00
1 correct -> combined_score = 0.25
2 correct -> combined_score = 0.50
3 correct -> combined_score = 0.75
4 correct -> combined_score = 1.00
```

Primary score:

```text
primary_score = combined_score
```

### Feedback Mode

By default, artifacts reveal how many modules are correct, but not which
specific modules are correct. This creates a harder credit-assignment problem.

If the environment variable is set:

```text
RICH_FEEDBACK=1
```

then artifacts can reveal which modules are correct or incorrect. That makes
the problem easier for iterative refinement, but it changes the feedback
setting and should not be mixed with normal parity runs unless intentional.

## Benchmark 4: Signal Processing

Path:

```text
FullReplacementMCTS/examples/signal_processing/evaluator.py
```

### Candidate Interface

The candidate program must define:

```python
run_signal_processing(signal_length, noise_level, window_size)
```

It must return a dictionary containing:

```python
{
    "filtered_signal": ...
}
```

The filtered signal must be non-empty.

### Test Signals

The evaluator generates five synthetic noisy signals. They have different
lengths and noise levels:

```text
length = 500 + i * 100
noise_level = 0.2 + i * 0.1
seed = 42 + i
```

The five clean signal types are:

1. Smooth sinusoidal signal with trend
2. Multiple frequency components
3. Non-stationary signal with changing frequency
4. Step changes
5. Random walk with trend

For each test signal, the evaluator calls the candidate with:

```text
signal_length = len(noisy_signal)
noise_level = 0.3
window_size = 20
timeout = 10 seconds
```

### Per-Signal Penalty Metrics

For each successful signal, the evaluator computes four penalty terms.
Lower penalty is better.

`S`: slope changes

```text
Number of sign changes in the first difference of the filtered signal.
Too many slope changes usually means the output is not smooth enough.
```

`L_recent`: recent lag error

```text
Absolute difference between the last filtered value and the corresponding
recent noisy-signal value.
```

`L_avg`: average tracking error

```text
Average absolute difference between the filtered signal and the aligned noisy
signal.
```

`R`: false reversal penalty

```text
Number of trend reversals in the filtered signal that are not present in the
clean signal.
```

### Per-Signal Composite Score

The four penalties are normalized:

```text
S_norm        = min(S / 50, 2)
L_recent_norm = min(L_recent, 2)
L_avg_norm    = min(L_avg, 2)
R_norm        = min(R / 25, 2)
```

Then the evaluator computes a weighted penalty:

```text
penalty =
    0.3 * S_norm
  + 0.2 * L_recent_norm
  + 0.2 * L_avg_norm
  + 0.3 * R_norm
```

The penalty is converted into a maximization score:

```text
composite_score = 1 / (1 + penalty)
```

Higher `composite_score` is better.

Because the normalized penalties are capped at `2`, the weighted penalty is
roughly in `[0, 2]`, so the per-signal composite score is roughly in:

```text
1 / (1 + 2) = 0.333  up to  1 / (1 + 0) = 1.000
```

Broken candidates can still receive `0` if all test signals fail.

### Additional Per-Signal Metrics

The evaluator also computes:

- `correlation`: Pearson correlation between filtered signal and aligned clean
  signal
- `noise_reduction`: reduction in noise variance compared with the noisy input
- `execution_time`: runtime for the candidate call

These are used in the aggregate score or reported for diagnostics.

### Aggregate Metrics

After all five signals, the evaluator computes averages:

```text
avg_composite_score = mean(per-signal composite scores)
avg_slope_changes
avg_lag_error
avg_avg_error
avg_false_reversals
avg_correlation
avg_noise_reduction
avg_execution_time
success_rate = successful_runs / 5
```

Then it computes derived scores:

```text
smoothness_score = 1 / (1 + avg_slope_changes / 20)
responsiveness_score = 1 / (1 + avg_lag_error)
accuracy_score = max(0, avg_correlation)
efficiency_score = min(1, 1 / max(0.001, avg_execution_time))
```

Important: `responsiveness_score` and `efficiency_score` are returned as
metrics, but they are not included in the final `overall_score` formula.

### Final Overall Score

The final score is:

```text
overall_score =
    0.4 * avg_composite_score
  + 0.2 * smoothness_score
  + 0.2 * accuracy_score
  + 0.1 * avg_noise_reduction
  + 0.1 * success_rate
```

The evaluator returns both:

```text
composite_score = avg_composite_score
overall_score = weighted aggregate score above
```

It does not return `combined_score`.

Therefore, using the primary score extraction order:

```text
primary_score = overall_score
```

### How To Interpret Scores

For signal processing:

- `composite_score` measures the average quality of the filter according to
  smoothness, lag/tracking, and false-reversal penalties.
- `overall_score` adds other concerns: smoothness, correlation with clean
  signal, noise reduction, and reliability.
- The plotted/default comparison score should be `overall_score`.

So the signal benchmark is related to problem-specific metrics like lag,
smoothness, false reversals, correlation, and noise reduction. It is not a
single raw metric.

## How Scores Relate To Plots

For convergence plots, each iteration/run usually records:

```text
iteration
primary_score
best_score
metrics
```

`primary_score` is the score of the evaluated candidate or checkpoint at that
point.

`best_score` is:

```text
best_score_at_iteration_i =
    max(primary_score from iterations 0..i)
```

So if a plot shows `best_score`, it should be monotonic non-decreasing within a
single run. If it shows raw `primary_score`, it can go up and down because new
candidates can be worse than previous candidates.

When comparing multiple algorithms or baselines, use the same benchmark,
iteration budget, model, token budget, and score extraction rule. Otherwise the
curves may not be measuring the same thing.

## Summary Table

| Benchmark | Raw Problem Goal | Main Returned Score | Primary Score Used For Comparison | Can Exceed 1? |
| --- | --- | --- | --- | --- |
| Circle packing | Maximize valid `sum_radii` | `combined_score = sum_radii / 2.635` if valid | `combined_score` | Yes, if beating target |
| Function minimization | Minimize `f(x, y)` near known optimum | Weighted value/distance/reliability score with multiplier | `combined_score` | Yes, due to multiplier |
| K-module | Match hidden 4-module config | `combined_score = correct_modules / 4` | `combined_score` | No |
| Signal processing | Filter noisy signals well | `overall_score` weighted from composite/filter metrics | `overall_score` | Usually no, formula components are mostly bounded around 0..1 |

## Practical Reading Guide

When looking at a result file:

1. Find `primary_score` first. That is the scalar used for parity plotting.
2. Look at `metrics` to understand why that score happened.
3. For circle packing, inspect `sum_radii`, `validity`, and `target_ratio`.
4. For function minimization, inspect `value_score`, `distance_score`, and
   `reliability_score`.
5. For k-module, inspect `correct_modules` and `accuracy`.
6. For signal processing, inspect `overall_score`, `composite_score`,
   `correlation`, `noise_reduction`, `success_rate`, and penalty metrics.

The most common source of confusion is assuming that all benchmarks use
`combined_score`. They do not. The benchmark matrix handles this by converting
each result into a common `primary_score`.
