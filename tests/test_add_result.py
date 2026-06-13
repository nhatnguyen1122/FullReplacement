"""
Tests for AddResult metadata from ProgramDatabase.add()
"""

import unittest
from unittest.mock import patch

from openevolve.config import Config
from openevolve.database import AddResult, Program, ProgramDatabase


class TestAddResult(unittest.TestCase):
    """Tests for AddResult metadata returned by ProgramDatabase.add()."""

    def setUp(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.search_strategy = "island_map_elites"
        self.db = ProgramDatabase(config.database)

    def test_add_returns_add_result(self):
        """add_with_result() should return an AddResult object."""
        program = Program(
            id="test1",
            code="def test(): pass",
            language="python",
            metrics={"score": 0.5},
        )
        result = self.db.add_with_result(program)
        self.assertIsInstance(result, AddResult)
        self.assertEqual(result.program_id, "test1")

    def test_backward_compat_program_id(self):
        """add() preserves the original program-id return contract."""
        program = Program(
            id="compat-test",
            code="def f(): return 1",
            metrics={"score": 0.6},
        )
        result = self.db.add(program)
        self.assertEqual(result, program.id)

    def test_new_cell_flag(self):
        """First program in a MAP-Elites cell should set added_to_new_cell=True."""
        program = Program(
            id="new-cell-1",
            code="def unique_function(): return 42",
            metrics={"score": 0.7},
        )
        result = self.db.add_with_result(program)
        # First program always occupies a new cell
        self.assertTrue(result.added_to_new_cell)
        self.assertFalse(result.improved_existing_cell)

    def test_improved_cell_flag(self):
        """Replacing a worse program should set improved_existing_cell=True."""
        # Use a custom feature dimension so both programs land in the same cell
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.search_strategy = "island_map_elites"
        config.database.similarity_threshold = 1.0
        config.database.feature_dimensions = ["custom_feature"]
        db = ProgramDatabase(config.database)

        # Add initial program
        prog1 = Program(
            id="improve-1",
            code="def f(): return 1",
            metrics={"score": 0.3, "custom_feature": 0.5},
        )
        r1 = db.add_with_result(prog1, target_island=0)

        # Add better program with same custom_feature (same MAP-Elites cell)
        prog2 = Program(
            id="improve-2",
            code="def g(): return 2",  # different code to avoid novelty issues
            metrics={"score": 0.9, "custom_feature": 0.5},  # same feature value → same cell
        )
        result = db.add_with_result(prog2, target_island=0)

        # Both should have same feature key since custom_feature=0.5 maps to same bin
        self.assertEqual(r1.feature_key, result.feature_key)
        self.assertTrue(result.improved_existing_cell)
        self.assertEqual(result.replaced_program_id, "improve-1")

    def test_fitness_computed(self):
        """AddResult should contain the child's fitness."""
        program = Program(
            id="fitness-test",
            code="def f(): return 1",
            metrics={"combined_score": 0.85},
        )
        result = self.db.add_with_result(program)
        self.assertAlmostEqual(result.fitness, 0.85)

    def test_parent_fitness_tracked(self):
        """AddResult should track parent's fitness for delta computation."""
        parent = Program(
            id="parent-1",
            code="def parent(): return 1",
            metrics={"combined_score": 0.6},
        )
        self.db.add(parent, target_island=0)

        child = Program(
            id="child-1",
            code="def child(): return 2",
            parent_id="parent-1",
            metrics={"combined_score": 0.8},
        )
        result = self.db.add_with_result(child, target_island=0)
        self.assertAlmostEqual(result.parent_fitness, 0.6)
        self.assertAlmostEqual(result.fitness, 0.8)

    def test_accepted_flag_default(self):
        """Normal additions should have accepted=True."""
        program = Program(
            id="accepted-test",
            code="def f(): return 1",
            metrics={"score": 0.5},
        )
        result = self.db.add_with_result(program)
        self.assertTrue(result.accepted)
        self.assertTrue(result.evaluation_success)

    def test_evaluation_success_flag_false_for_failed_metrics(self):
        """AddResult should flag evaluator-failed candidates for TreeQD penalties."""
        program = Program(
            id="failed-eval-test",
            code="def f(): return 1",
            metrics={"runs_successfully": 0.0, "combined_score": 0.0, "error": "failed"},
        )
        result = self.db.add_with_result(program)

        self.assertFalse(result.accepted)
        self.assertFalse(result.evaluation_success)
        self.assertNotIn(program.id, self.db.programs)

    def test_evaluation_success_false_for_nested_failure_artifacts(self):
        """Nested evaluator failures should not be treated as successful zero-score programs."""
        program = Program(
            id="nested-failed-eval-test",
            code="def f(): return 1",
            metrics={
                "metrics": {"combined_score": 0.0, "accuracy": 0.0},
                "artifacts": {"status": "VALIDATION_ERROR"},
            },
        )
        result = self.db.add_with_result(program)

        self.assertFalse(result.accepted)
        self.assertFalse(result.evaluation_success)
        self.assertNotIn(program.id, self.db.programs)

    def test_evaluation_success_false_for_failure_status_metric(self):
        """Evaluator failure markers in metrics should be enough to penalize TreeQD."""
        program = Program(
            id="status-failed-eval-test",
            code="def f(): return 1",
            metrics={"combined_score": 0.0, "evaluation_status": "VALIDATION_ERROR"},
        )
        result = self.db.add_with_result(program)

        self.assertFalse(result.accepted)
        self.assertFalse(result.evaluation_success)
        self.assertNotIn(program.id, self.db.programs)

    def test_evaluation_success_true_for_valid_zero_score(self):
        """A valid but poor candidate may legitimately have zero fitness."""
        program = Program(
            id="valid-zero-eval-test",
            code="def f(): return 1",
            metrics={"combined_score": 0.0, "accuracy": 0.0},
        )
        result = self.db.add_with_result(program)

        self.assertTrue(result.accepted)
        self.assertTrue(result.evaluation_success)

    def test_rejected_program_is_not_stored(self):
        """Novelty-rejected programs should not remain selectable in self.programs."""
        program = Program(
            id="rejected-test",
            code="def f(): return 1",
            metrics={"score": 0.5},
        )
        with patch.object(self.db, "_is_novel", return_value=False):
            result = self.db.add_with_result(program)

        self.assertFalse(result.accepted)
        self.assertNotIn(program.id, self.db.programs)

    def test_island_idx_set(self):
        """AddResult should report the island the program was added to."""
        program = Program(
            id="island-test",
            code="def f(): return 1",
            metrics={"score": 0.5},
        )
        result = self.db.add_with_result(program, target_island=2)
        self.assertEqual(result.island_idx, 2)

    def test_feature_key_set(self):
        """AddResult should contain the MAP-Elites feature key."""
        program = Program(
            id="feature-key-test",
            code="def f(): return 1",
            metrics={"score": 0.5},
        )
        result = self.db.add_with_result(program)
        self.assertIsNotNone(result.feature_key)


class TestTreeOnlyAddResult(unittest.TestCase):
    """Tests for MCTS tree-only database admission."""

    def setUp(self):
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        config.database.search_strategy = "tree_qd_mcts"
        config.database.feature_dimensions = ["custom_feature"]
        self.db = ProgramDatabase(config.database)

    def test_tree_only_adds_valid_program_without_map_elites_replacement(self):
        first = Program(
            id="tree-first",
            code="def f(): return 1",
            metrics={"combined_score": 0.9, "custom_feature": 0.5},
        )
        second = Program(
            id="tree-second",
            code="def g(): return 2",
            metrics={"combined_score": 0.1, "custom_feature": 0.5},
        )

        r1 = self.db.add_with_result(first)
        r2 = self.db.add_with_result(second)

        self.assertTrue(r1.accepted)
        self.assertTrue(r2.accepted)
        self.assertIn(first.id, self.db.programs)
        self.assertIn(second.id, self.db.programs)
        self.assertIn(first.id, self.db.tree_programs)
        self.assertIn(second.id, self.db.tree_programs)
        self.assertEqual(self.db.island_feature_maps, [{}, {}, {}])
        self.assertEqual(first.metadata["island"], 0)
        self.assertEqual(second.metadata["island"], 0)

    def test_tree_only_uses_behavioral_feature_counts_for_new_cell_flag(self):
        parent = Program(
            id="tree-parent",
            code="def parent(): return 1",
            metrics={"combined_score": 0.4, "custom_feature": 0.5},
        )
        child = Program(
            id="tree-child",
            code="def child(): return 2",
            parent_id="tree-parent",
            metrics={"combined_score": 0.8, "custom_feature": 0.5},
        )

        r1 = self.db.add_with_result(parent)
        r2 = self.db.add_with_result(child)

        self.assertTrue(r1.added_to_new_cell)
        self.assertFalse(r2.added_to_new_cell)
        self.assertTrue(r2.improved_existing_cell)
        self.assertEqual(r1.feature_key, r2.feature_key)
        self.assertEqual(self.db.tree_feature_counts[r1.feature_key], 2)


if __name__ == "__main__":
    unittest.main()
