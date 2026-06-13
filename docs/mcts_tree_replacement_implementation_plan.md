# MCTS Tree Replacement Implementation Plan

## Goal

Replace OpenEvolve's active island-based + MAP-Elites search policy with an MCTS tree controller. The old implementation remains available behind `search.strategy: island_map_elites` for compatibility, but the default and primary path becomes `search.strategy: tree_qd_mcts`.

## Current Gap

The existing TreeQD-MCTS implementation changes parent selection, but candidate admission still flows through the island-local MAP-Elites grid:

```text
MCTS selects parent -> worker mutates program -> database.add_with_result()
                                      |
                                      v
                         island feature map admission
                         island population membership
                         migration generation counters
```

That means MCTS is only a scheduler layered over OpenEvolve's original population model. The contribution is weaker because MAP-Elites still decides which candidates survive.

## Replacement Design

The new active path is:

```text
MCTS tree selects parent node
  -> worker mutates selected program
  -> evaluator scores child
  -> tree-only database admission
  -> MCTS reward backpropagation
  -> checkpoint tree + program pool
```

### Tree-Only Database Admission

When `database.search_strategy == "tree_qd_mcts"`:

- Do not use `island_feature_maps` for candidate retention.
- Do not reject a valid child because it fails to improve a MAP-Elites cell.
- Do not migrate programs.
- Keep a single compatibility population bucket (`island: 0`) because existing prompt and artifact code expects that field.
- Store all valid, novel evaluated programs until `population_size`, then prune the lowest-fitness non-protected programs.
- Maintain `archive` only as a global elite list for prompt context, not as MAP-Elites cells.
- Maintain `tree_feature_counts` as a light behavioral signature ledger for reward shaping, not as an admission grid.

### MCTS Reward

`AddResult` remains the bridge from database admission to tree backpropagation. For tree-only search:

- `added_to_new_cell` means "first time this behavioral signature has appeared in the tree run".
- `improved_existing_cell` means "child improves over parent fitness", not MAP-Elites cell replacement.
- `feature_key` is a behavioral signature derived from existing feature extraction.
- invalid, rejected, or evaluator-failed candidates receive the configured invalid penalty.

This keeps quality-diversity pressure without retaining a MAP-Elites archive as the population manager.

### Process Scheduling

For `tree_qd_mcts`, the process controller should:

- Fill workers from MCTS selections instead of distributing work by island.
- Track selected node id per iteration.
- Backpropagate success or failure into the selected tree node.
- Stop submitting new work once the token ledger exceeds `llm.max_total_tokens`.

For `island_map_elites`, the original island scheduling and migration behavior remains unchanged.

### Provider And Token Requirements

The default provider is Codestral:

```yaml
llm:
  provider: codestral
  models:
    - provider: codestral
      name: codestral-latest
  evaluator_models:
    - provider: codestral
      name: codestral-latest
```

OpenAI-compatible clients continue to support:

- Codestral/Mistral through `https://api.mistral.ai/v1`
- Gemini through Google's OpenAI-compatible endpoint
- OpenAI/GPT through `https://api.openai.com/v1`

Token tracking reads provider `usage` fields and enforces `llm.max_total_tokens` through the shared multiprocessing token ledger.

## Implementation Steps

1. Change default search strategy to `tree_qd_mcts`.
2. Add `DatabaseConfig.search_strategy` and synchronize it from `Config.search.strategy`.
3. Add tree-only database metadata: `tree_programs` and `tree_feature_counts`.
4. Route `ProgramDatabase.add_with_result()` to tree-only admission when enabled.
5. Make process scheduling tree-first in MCTS mode and island-balanced only in legacy mode.
6. Disable migration and island generation accounting in MCTS mode.
7. Save/load tree-only metadata in checkpoints.
8. Add tests for tree-only admission and process scheduling behavior.
9. Run focused tests, then the non-integration test suite.

