# Why Use TreeQD-MCTS Instead of Island-Based MAP-Elites?

This note explains the motivation for replacing OpenEvolve's original island-based MAP-Elites population manager with the TreeQD-MCTS approach in this codebase. The short version is:

> Island-based MAP-Elites is a strong diversity mechanism, but it treats LLM-generated program evolution mostly as population maintenance. TreeQD-MCTS treats it as a sequential search problem where each expensive LLM/evaluator call should be allocated to the most promising part of the search tree.

That difference matters because OpenEvolve spends most of its cost on LLM calls and program evaluation. A search controller should therefore learn where additional attempts are likely to pay off, not only preserve a diverse population.

## Baseline: Island-Based MAP-Elites

The baseline combines two evolutionary ideas:

- **Island model:** the population is split into semi-independent subpopulations. Each island evolves locally, and migration occasionally copies strong programs between islands.
- **MAP-Elites:** programs are placed into cells based on behavioral or feature dimensions. Each cell keeps an elite program for that region of the feature space.

This is useful because it preserves diversity. Instead of converging too quickly to one style of solution, the system can keep different families of programs alive. MAP-Elites is especially good when we care about *illumination*: understanding which kinds of solutions work across a feature space, not just finding one best solution.

However, in LLM program search, this baseline has weaknesses:

- **Budget is spread broadly.** Island scheduling and MAP-Elites cells can spend calls on regions that are diverse but not currently promising.
- **Credit assignment is weak.** If a program led to several good descendants, the island/MAP-Elites database keeps the descendants, but it does not explicitly model that lineage as a high-value branch.
- **Parent choice is local.** Sampling is mainly from islands, archives, or fitness-weighted pools. It does not maintain a search tree with visit counts and backed-up rewards.
- **Migration is coarse.** Migration helps share good programs, but it is periodic and population-level. It is not a fine-grained policy for deciding which exact branch should receive the next LLM call.
- **One-program-per-cell can discard useful stepping stones.** MAP-Elites is excellent for retaining elites per cell, but LLM evolution often benefits from intermediate programs that are not elites yet but may be valuable parents.

The baseline is still good when diversity itself is the central objective, or when evaluation is cheap enough that broad exploration is acceptable.

## Replacement: TreeQD-MCTS

The MCTS version in this codebase adds a tree controller around program evolution:

- Each node represents a program.
- Parent-child edges represent evolutionary steps.
- Selection uses MCTS-style tree statistics.
- Results are backpropagated through the selected branch.
- Rewards combine quality and quality-diversity signals.

Concretely, `openevolve/search/tree_qd.py` implements a `TreeQDController` with:

- visit counts,
- value sums,
- best reward,
- best fitness,
- pending rollout counts,
- UCT-style selection,
- configurable backup behavior,
- reward weights for quality, improvement over parent, new cells, improved cells, novelty, invalid programs, and cost.

The database still computes feature keys and novelty signals, but in MCTS mode it bypasses island-local MAP-Elites replacement. The active population is tree-managed, while legacy island fields remain mostly compatibility metadata for prompts and checkpoints.

## Why MCTS Fits LLM Program Evolution

MCTS is a good fit because LLM program search is not just random mutation. It is a sequential decision process:

1. Pick a parent program.
2. Ask the LLM to modify it.
3. Evaluate the child.
4. Decide whether that branch deserves more attempts.

That loop is exactly where MCTS is useful. MCTS does not need a perfect model of the search space. It incrementally builds a tree from sampled outcomes and uses the accumulated statistics to decide what to try next.

The core advantage is the exploration/exploitation tradeoff:

- **Exploit:** revisit branches that have produced high reward.
- **Explore:** still try less-visited branches because they may hide better solutions.

In this implementation, the reward is not only final fitness. It can include:

- absolute program quality,
- improvement over the parent,
- filling a new feature cell,
- improving a feature cell,
- novelty,
- penalties for invalid or rejected candidates,
- optional cost penalties.

That makes the search more aligned with OpenEvolve's objective: find high-quality programs while maintaining enough diversity to avoid premature convergence.

## Why This Can Be Better Than Island-Based Search

### 1. Better use of expensive LLM calls

Each LLM call is expensive. Island-based search distributes attempts across islands and feature cells, but it does not explicitly learn a value estimate for each lineage. TreeQD-MCTS records which selected branch produced reward and uses that information for future scheduling.

This should improve sample efficiency when only a limited number of iterations are affordable.

### 2. Explicit lineage credit assignment

A strong program is often valuable not only because of its current score, but because it is a good parent. MCTS captures this by backing up rewards through the tree. If a branch repeatedly produces good children, its ancestors become more attractive for future selection.

Island-based MAP-Elites can keep the best child, but it does not naturally represent "this path is productive."

### 3. Principled exploration/exploitation

Island-based methods use heuristic sampling, migration, archives, and population balancing. These are useful, but they are not a direct estimate of uncertainty or opportunity.

MCTS gives a more explicit policy: selected nodes balance reward estimates against visit counts. This helps prevent both extremes:

- spending everything on the current best program,
- spreading effort too evenly over weak regions.

### 4. Search tree structure matches program evolution

LLM evolution is path-dependent. A good solution may require several coordinated edits. The tree structure preserves ancestry and lets the controller reason over multi-step improvement paths.

This is especially important when individual mutations are noisy. A mediocre child may be a useful stepping stone if its branch has historically led to improvements.

### 5. QD reward keeps diversity pressure without making MAP-Elites the population manager

A pure MCTS optimizer might over-focus on score and lose diversity. The TreeQD reward avoids that by still rewarding new feature cells, cell improvement, and novelty.

So the goal is not to abandon quality-diversity. The goal is to move from:

> "MAP-Elites decides what survives, islands decide where to sample"

to:

> "MCTS decides where to spend search budget, while QD signals shape the reward."

This is a better match for high-cost black-box program search.

## Why Not Always Use Island-Based MAP-Elites?

Island-based MAP-Elites has real strengths:

- It is simple and robust.
- It preserves diversity well.
- It parallelizes naturally.
- It is easy to inspect through occupied cells and island statistics.
- It is less sensitive to reward design than an MCTS controller.

But those strengths are not always the limiting factor. In OpenEvolve, the limiting factor is often not "can we maintain enough programs?" It is "which parent should receive the next expensive LLM attempt?"

Island-based MAP-Elites answers that indirectly. TreeQD-MCTS answers it directly.

## Expected Benefits

TreeQD-MCTS should be especially useful when:

- LLM calls are expensive or rate-limited.
- Evaluations are expensive.
- Good solutions require multi-step refinement.
- The search space has many invalid or low-quality edits.
- Some lineages are much more productive than others.
- We care about both best score and diversity.

In those settings, MCTS can reduce wasted evaluations by learning which branches are worth expanding.

## Risks and Tradeoffs

TreeQD-MCTS is not automatically better in every benchmark. Important risks:

- **Reward design matters.** Bad weights can overvalue novelty, small deltas, or current quality.
- **More moving parts.** The tree controller, backup policy, and reward computation add complexity.
- **Early noise can bias the tree.** If initial evaluations are noisy, MCTS may over-prioritize misleading branches unless exploration is strong enough.
- **Parallel rollouts are harder.** MCTS statistics are easiest in sequential search. Parallel workers need pending-count handling to avoid oversampling the same node.
- **Short deterministic benchmarks may not show much difference.** If the evaluator is coarse or the first good edit reaches a plateau, MCTS and baseline can look similar.

This means TreeQD-MCTS should be evaluated empirically with multiple seeds, not assumed superior from one run.

## When the Baseline May Still Be Better

The original island-based MAP-Elites approach may be preferable when:

- the main goal is broad illumination of the feature space,
- evaluation is cheap and broad exploration is affordable,
- reward shaping is unclear,
- many diverse solutions are more important than the single best program,
- the benchmark is too noisy for reliable tree backpropagation,
- implementation simplicity is more important than search efficiency.

## Practical Interpretation for This Codebase

The MCTS replacement is good because it makes OpenEvolve's population manager more decision-theoretic:

- It remembers which program lineages were productive.
- It uses backed-up rewards to allocate future LLM calls.
- It keeps QD pressure through reward terms instead of relying on island/MAP-Elites mechanics alone.
- It treats invalid or rejected generations as useful negative feedback, not just discarded noise.
- It can be compared directly to the baseline using convergence curves from `convergence.jsonl`.

The strongest argument for using it is not that islands or MAP-Elites are bad. They are good tools. The argument is that for LLM-driven program evolution, the scarce resource is the next generation/evaluation attempt, and MCTS is a better mechanism for allocating that scarce resource across competing search branches.

## Sources

- Kocsis and Szepesvari introduced UCT, applying bandit ideas to Monte Carlo planning and tree search: https://cris.technion.ac.il/en/publications/bandit-based-monte-carlo-planning/
- A recent MCTS review summarizes MCTS as an intelligent tree search method that balances exploration and exploitation: https://arxiv.org/abs/2103.04931
- Mouret and Clune introduced MAP-Elites as an illumination algorithm for mapping high-performing solutions across feature dimensions: https://arxiv.org/abs/1504.04909
- Island models divide evolutionary populations into subpopulations with occasional migration, preserving diversity longer than a fully mixed population: https://en.wikipedia.org/wiki/Population_model_%28evolutionary_algorithm%29
- Sudholt discusses benefits of migration in parallel evolutionary algorithms: https://research.birmingham.ac.uk/en/publications/the-benefit-of-migration-in-parallel-evolutionary-algorithms/

