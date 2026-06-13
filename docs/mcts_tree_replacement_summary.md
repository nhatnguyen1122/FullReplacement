# MCTS Tree Replacement Summary

## What Changed

OpenEvolve now defaults to `search.strategy: tree_qd_mcts` in the run-level configuration path. In this mode, the active population manager is the MCTS tree rather than island scheduling plus island-local MAP-Elites admission.

The old island + MAP-Elites behavior remains available with:

```yaml
search:
  strategy: island_map_elites
database:
  search_strategy: island_map_elites
```

This keeps the existing test surface and older configs usable while making new/default runs use the MCTS controller.

## New Active MCTS Flow

```text
TreeQDController.select()
  -> selected program node
  -> worker builds prompt from selected parent
  -> Codestral/OpenAI-compatible model generates diff
  -> evaluator scores child
  -> ProgramDatabase tree-only admission
  -> TreeQDReward computes reward
  -> TreeQDController.record_result() backpropagates reward
```

In `tree_qd_mcts` mode, database admission no longer uses island-local MAP-Elites replacement. Valid novel children are retained in the tree population until `population_size` pruning is needed.

## Database Behavior

Added tree-only metadata:

- `tree_programs`: ids retained by the MCTS population path.
- `tree_feature_counts`: behavioral-signature counts used for novelty/QD reward shaping.
- `Program.metadata["search_strategy"] = "tree_qd_mcts"`.
- `Program.metadata["tree_feature_key"]`.

`AddResult` semantics in tree mode:

- `added_to_new_cell`: first time this behavioral signature appears.
- `improved_existing_cell`: child fitness improves over parent fitness.
- `feature_key`: behavioral signature, not a MAP-Elites archive cell.
- failed/rejected candidates are not stored and receive invalid reward penalties.

The compatibility `island` metadata is still set to `0` because prompt and artifact code expects it, but MCTS mode does not schedule by islands or migrate programs.

## Provider And Token Support

Codestral remains the default model path:

```yaml
llm:
  provider: codestral
  models:
    - provider: codestral
      name: codestral-latest
```

Provider defaults support:

- Codestral/Mistral: `CODESTRAL_API_KEY`, then `MISTRAL_API_KEY`.
- Gemini: `GEMINI_API_KEY`, then `GOOGLE_API_KEY`.
- OpenAI/GPT: `OPENAI_API_KEY`.

Token tracking is enabled through OpenAI-compatible response `usage` fields and supports `llm.max_total_tokens` with a multiprocessing shared token ledger.

## Validation

Focused tests:

```text
conda run -n openevolve_test pytest tests/test_add_result.py tests/test_tree_qd.py tests/test_provider_defaults.py tests/test_token_budget.py
53 passed
```

Full non-integration suite:

```text
conda run -n openevolve_test pytest tests --ignore=tests/integration
406 passed, 1 skipped, 1 warning
```

Codestral smoke run:

```text
conda run -n openevolve_test python openevolve-run.py \
  examples/function_minimization/initial_program.py \
  examples/function_minimization/evaluator.py \
  --config configs/mcts_smoke_codestral.yaml \
  --output output/mcts_tree_replacement_smoke \
  --iterations 1
```

Result:

- MCTS selected the initial node.
- Codestral generated one child.
- Token usage was tracked: `total=1970`, `prompt=1280`, `completion=690`, `requests=1`.
- MCTS reward backpropagated: `reward=0.7525`, `tree_nodes=3`, `tree_depth=2`, `root_visits=1`.
- Best score improved from `combined_score=1.0586` to `combined_score=1.4356`.
- Checkpoint saved `tree_qd_state.json`.

Smoke output:

```text
output/mcts_tree_replacement_smoke/
  best/best_program.py
  best/best_program_info.json
  checkpoints/checkpoint_1/tree_qd_state.json
```
