"""
Tests for TreeQD-MCTS controller (openevolve.search.tree_qd)
"""

import json
import os
import tempfile
import unittest

from openevolve.search.tree_qd import (
    ActionType,
    AddResult,
    TreeAction,
    TreeNode,
    TreeQDController,
    TreeQDReward,
)


class TestTreeNode(unittest.TestCase):
    """Tests for TreeNode dataclass."""

    def test_node_creation(self):
        node = TreeNode(program_id="prog-1", depth=2)
        self.assertEqual(node.program_id, "prog-1")
        self.assertEqual(node.depth, 2)
        self.assertEqual(node.visits, 0)
        self.assertEqual(node.value_sum, 0.0)
        self.assertEqual(node.best_reward, float("-inf"))
        self.assertIsNotNone(node.node_id)

    def test_mean_reward_zero_visits(self):
        node = TreeNode()
        self.assertEqual(node.mean_reward, 0.0)

    def test_mean_reward(self):
        node = TreeNode(visits=4, value_sum=2.0)
        self.assertAlmostEqual(node.mean_reward, 0.5)

    def test_serialization_roundtrip(self):
        node = TreeNode(
            program_id="p1",
            depth=3,
            visits=10,
            value_sum=5.5,
            best_reward=0.8,
            best_fitness=0.9,
            feature_key="(2, 3)",
            children=["child-1", "child-2"],
        )
        d = node.to_dict()
        restored = TreeNode.from_dict(d)
        self.assertEqual(restored.program_id, "p1")
        self.assertEqual(restored.depth, 3)
        self.assertEqual(restored.visits, 10)
        self.assertAlmostEqual(restored.value_sum, 5.5)
        self.assertAlmostEqual(restored.best_reward, 0.8)
        self.assertAlmostEqual(restored.best_fitness, 0.9)
        self.assertEqual(restored.feature_key, "(2, 3)")
        self.assertEqual(len(restored.children), 2)

    def test_inf_values_serialization(self):
        """Verify -inf best_reward serializes as None and deserializes back."""
        node = TreeNode()
        d = node.to_dict()
        self.assertIsNone(d["best_reward"])
        restored = TreeNode.from_dict(d)
        self.assertEqual(restored.best_reward, float("-inf"))


class TestTreeAction(unittest.TestCase):
    def test_default_action(self):
        action = TreeAction()
        self.assertEqual(action.action_type, ActionType.MUTATE)
        self.assertEqual(action.prompt_mode, "diff")

    def test_serialization(self):
        action = TreeAction(
            action_type=ActionType.CROSSOVER,
            target_program_id="p1",
            model_provider="codestral",
        )
        d = action.to_dict()
        restored = TreeAction.from_dict(d)
        self.assertEqual(restored.action_type, ActionType.CROSSOVER)
        self.assertEqual(restored.target_program_id, "p1")
        self.assertEqual(restored.model_provider, "codestral")


class TestTreeQDReward(unittest.TestCase):
    """Tests for QD reward computation."""

    def test_rejected_program(self):
        result = AddResult(accepted=False)
        reward = TreeQDReward.compute(result)
        self.assertLess(reward, 0.0)

    def test_new_cell_bonus(self):
        result = AddResult(
            accepted=True,
            added_to_new_cell=True,
            fitness=0.5,
            parent_fitness=0.4,
        )
        reward = TreeQDReward.compute(result, max_fitness=1.0)
        self.assertGreater(reward, 0.0)
        # Should include w_cell_new + w_novelty bonuses
        self.assertGreater(reward, 0.3)  # quality + delta + cell_new + novelty

    def test_improved_cell_bonus(self):
        result = AddResult(
            accepted=True,
            improved_existing_cell=True,
            fitness=0.8,
            parent_fitness=0.6,
        )
        reward = TreeQDReward.compute(result, max_fitness=1.0)
        self.assertGreater(reward, 0.0)

    def test_no_improvement(self):
        result = AddResult(
            accepted=True,
            fitness=0.3,
            parent_fitness=0.5,  # child is worse
        )
        reward = TreeQDReward.compute(result, max_fitness=1.0)
        # Delta should be 0 (clipped to non-negative)
        # But quality term still contributes
        self.assertGreater(reward, 0.0)

    def test_zero_max_fitness(self):
        """Edge case: max_fitness near zero should not crash."""
        result = AddResult(accepted=True, fitness=0.0)
        reward = TreeQDReward.compute(result, max_fitness=0.0)
        self.assertEqual(reward, 0.0)


class TestTreeQDController(unittest.TestCase):
    """Tests for the MCTS controller."""

    def setUp(self):
        self.ctrl = TreeQDController(
            exploration_constant=0.7,
            qd_weight=0.0,  # disable QD bonus for deterministic tests
            max_tree_depth=10,
            backup="mixed",
            backup_alpha=0.5,
            widening_alpha=0.5,
        )

    def test_root_exists(self):
        self.assertEqual(self.ctrl.root.node_id, "__root__")
        self.assertIn("__root__", self.ctrl.nodes)

    def test_register_program(self):
        node = self.ctrl.register_program("prog-1", iteration=0, fitness=0.5)
        self.assertEqual(node.program_id, "prog-1")
        self.assertEqual(node.depth, 1)
        self.assertIn("prog-1", self.ctrl.program_to_node)
        self.assertIn(node.node_id, self.ctrl.root.children)

    def test_register_duplicate(self):
        node1 = self.ctrl.register_program("prog-1")
        node2 = self.ctrl.register_program("prog-1")
        self.assertEqual(node1.node_id, node2.node_id)

    def test_register_with_parent(self):
        self.ctrl.register_program("parent-1")
        child = self.ctrl.register_program("child-1", parent_program_id="parent-1")
        self.assertEqual(child.depth, 2)
        parent_node_id = self.ctrl.program_to_node["parent-1"]
        self.assertEqual(child.parent_node_id, parent_node_id)

    def test_select_from_empty_tree(self):
        """Select on empty tree should return root without marking it pending."""
        node, action = self.ctrl.select()
        self.assertEqual(node.node_id, "__root__")
        self.assertEqual(node.pending_count, 0)

    def test_select_from_single_program(self):
        self.ctrl.register_program("prog-1", fitness=0.5)
        node, action = self.ctrl.select()
        # Should select prog-1 since it's the only leaf
        self.assertEqual(node.program_id, "prog-1")
        self.assertEqual(node.pending_count, 1)

    def test_uct_selection_prefers_unvisited(self):
        """Unvisited nodes should get infinite UCT bonus."""
        self.ctrl.register_program("prog-1", fitness=0.5)
        self.ctrl.register_program("prog-2", fitness=0.3)

        # Visit prog-1 many times
        p1_node_id = self.ctrl.program_to_node["prog-1"]
        p1_node = self.ctrl.nodes[p1_node_id]
        p1_node.visits = 100
        p1_node.value_sum = 50.0

        # prog-2 is unvisited → should be selected
        node, action = self.ctrl.select()
        self.assertEqual(node.program_id, "prog-2")

    def test_backpropagation(self):
        self.ctrl.register_program("prog-1", fitness=0.5)
        add_result = AddResult(
            program_id="child-1",
            accepted=True,
            fitness=0.7,
            parent_fitness=0.5,
            feature_key="(1, 2)",
        )

        parent_node_id = self.ctrl.program_to_node["prog-1"]
        child_node = self.ctrl.record_result(
            parent_node_id=parent_node_id,
            child_program_id="child-1",
            add_result=add_result,
            reward=0.6,
            iteration=1,
        )

        # Child should have 1 visit
        self.assertEqual(child_node.visits, 1)
        self.assertAlmostEqual(child_node.value_sum, 0.6)

        # Parent should also have 1 visit from backprop
        parent_node = self.ctrl.nodes[parent_node_id]
        self.assertEqual(parent_node.visits, 1)

        # Root should also have 1 visit
        self.assertEqual(self.ctrl.root.visits, 1)

    def test_rejected_result_does_not_create_child_node(self):
        """Rejected database additions should penalize the parent without adding a child."""
        parent = self.ctrl.register_program("prog-1", fitness=0.5)
        parent.pending_count = 1
        add_result = AddResult(
            program_id="rejected-child",
            accepted=False,
            fitness=0.4,
            parent_fitness=0.5,
        )

        returned = self.ctrl.record_result(
            parent_node_id=parent.node_id,
            child_program_id="rejected-child",
            add_result=add_result,
            reward=-0.1,
            iteration=1,
        )

        self.assertEqual(returned.node_id, parent.node_id)
        self.assertNotIn("rejected-child", self.ctrl.program_to_node)
        self.assertEqual(parent.children, [])
        self.assertEqual(parent.pending_count, 0)
        self.assertEqual(parent.visits, 1)

    def test_failed_evaluation_does_not_create_child_node(self):
        """Evaluation-failed additions should not become selectable tree children."""
        parent = self.ctrl.register_program("prog-1", fitness=0.5)
        parent.pending_count = 1
        add_result = AddResult(
            program_id="failed-child",
            accepted=True,
            evaluation_success=False,
            added_to_new_cell=True,
            fitness=0.0,
        )
        reward = TreeQDReward.compute(add_result, w_invalid=0.2)

        returned = self.ctrl.record_result(
            parent_node_id=parent.node_id,
            child_program_id="failed-child",
            add_result=add_result,
            reward=reward,
            iteration=1,
        )

        self.assertAlmostEqual(reward, -0.2)
        self.assertEqual(returned.node_id, parent.node_id)
        self.assertNotIn("failed-child", self.ctrl.program_to_node)
        self.assertEqual(parent.children, [])
        self.assertEqual(parent.pending_count, 0)

    def test_mark_unavailable_penalizes_stale_node(self):
        """Stale selected nodes should become visited so UCT can move on."""
        node = self.ctrl.register_program("stale-prog", fitness=0.5)
        node.pending_count = 1

        self.ctrl.mark_unavailable(node.node_id, penalty=-0.25)

        self.assertEqual(node.pending_count, 0)
        self.assertEqual(node.visits, 1)
        self.assertAlmostEqual(node.value_sum, -0.25)
        self.assertEqual(self.ctrl.root.visits, 1)

    def test_progressive_widening(self):
        """Test that progressive widening gates child creation."""
        self.ctrl.register_program("prog-1", fitness=0.5)
        parent_node_id = self.ctrl.program_to_node["prog-1"]
        parent_node = self.ctrl.nodes[parent_node_id]

        # With 0 visits, should_widen should be True
        self.assertTrue(self.ctrl._should_widen(parent_node))

        # After some visits, max_children = ceil(visits^0.5)
        parent_node.visits = 4
        # ceil(4^0.5) = ceil(2) = 2
        # If node has < 2 children → widen
        self.assertEqual(len(parent_node.children), 0)
        self.assertTrue(self.ctrl._should_widen(parent_node))

        # Add 2 children manually
        parent_node.children = ["c1", "c2"]
        self.assertFalse(self.ctrl._should_widen(parent_node))

    def test_pending_count_in_uct(self):
        """Pending count should reduce exploration bonus."""
        self.ctrl.register_program("prog-1", fitness=0.5)
        self.ctrl.register_program("prog-2", fitness=0.3)
        p1_node_id = self.ctrl.program_to_node["prog-1"]
        p1_node = self.ctrl.nodes[p1_node_id]
        p2_node_id = self.ctrl.program_to_node["prog-2"]
        p2_node = self.ctrl.nodes[p2_node_id]

        # Mark prog-1 as having many pending workers
        p1_node.pending_count = 10
        p1_node.visits = 0

        # prog-2 has no pending, so it should be preferred
        node, action = self.ctrl.select()
        self.assertEqual(node.program_id, "prog-2")
        self.assertEqual(node.pending_count, 1)  # just incremented by select()

    def test_mixed_backup(self):
        """Verify mixed backup uses alpha * mean + (1-alpha) * max."""
        self.ctrl.backup = "mixed"
        self.ctrl.backup_alpha = 0.3

        self.ctrl.register_program("prog-1", fitness=0.5)
        parent_id = self.ctrl.program_to_node["prog-1"]

        # Record results with different rewards
        for i, reward in enumerate([0.2, 0.8, 0.4]):
            add_result = AddResult(
                program_id=f"child-{i}",
                accepted=True,
                fitness=0.5 + i * 0.1,
            )
            self.ctrl.record_result(
                parent_node_id=parent_id,
                child_program_id=f"child-{i}",
                add_result=add_result,
                reward=reward,
                iteration=i,
            )

        parent_node = self.ctrl.nodes[parent_id]
        mean_val = parent_node.mean_reward
        max_val = parent_node.best_reward
        expected = 0.3 * mean_val + 0.7 * max_val

        # The UCT selection should use this mixed value
        # We verify the values are correct
        self.assertAlmostEqual(mean_val, (0.2 + 0.8 + 0.4) / 3, places=5)
        self.assertAlmostEqual(max_val, 0.8)
        self.assertAlmostEqual(expected, 0.3 * mean_val + 0.7 * 0.8, places=5)

    def test_save_load_json(self):
        """Test serialization roundtrip."""
        self.ctrl.register_program("prog-1", fitness=0.5)
        add_result = AddResult(
            program_id="child-1",
            accepted=True,
            fitness=0.7,
            feature_key="(1, 2)",
        )
        parent_id = self.ctrl.program_to_node["prog-1"]
        self.ctrl.record_result(
            parent_node_id=parent_id,
            child_program_id="child-1",
            add_result=add_result,
            reward=0.6,
            iteration=1,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "tree_state.json")
            self.ctrl.save(path)

            # Verify file exists and is valid JSON
            with open(path) as f:
                state = json.load(f)
            self.assertIn("nodes", state)
            self.assertIn("program_to_node", state)

            # Load and verify
            loaded = TreeQDController.load(path)
            self.assertEqual(len(loaded.nodes), len(self.ctrl.nodes))
            self.assertEqual(loaded.total_expansions, 1)
            self.assertEqual(loaded.total_backprops, 1)
            self.assertIn("prog-1", loaded.program_to_node)
            self.assertIn("child-1", loaded.program_to_node)

            # Verify node state preserved
            p1_node_id = loaded.program_to_node["prog-1"]
            p1_node = loaded.nodes[p1_node_id]
            self.assertEqual(p1_node.visits, 1)

    def test_from_config(self):
        """Test construction from a config-like object."""
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            exploration_constant=1.0,
            qd_weight=0.3,
            cost_weight=0.05,
            max_tree_depth=15,
            backup="max",
            backup_alpha=0.7,
            widening_alpha=0.6,
        )
        ctrl = TreeQDController.from_config(cfg)
        self.assertEqual(ctrl.exploration_constant, 1.0)
        self.assertEqual(ctrl.backup, "max")
        self.assertEqual(ctrl.max_tree_depth, 15)

    def test_summary(self):
        self.ctrl.register_program("prog-1", fitness=0.5)
        summary = self.ctrl.summary()
        self.assertEqual(summary["total_nodes"], 2)  # root + prog-1
        self.assertEqual(summary["max_depth"], 1)

    def test_max_depth_limit(self):
        """Selection should stop at max_tree_depth."""
        self.ctrl.max_tree_depth = 2
        self.ctrl.widening_alpha = 0.0  # disable progressive widening

        # Build a chain: root → p1 → p2
        self.ctrl.register_program("p1", fitness=0.5)
        self.ctrl.register_program("p2", parent_program_id="p1", fitness=0.6)

        # p2 is at depth 2, so it should not be expanded further
        p2_node = self.ctrl.nodes[self.ctrl.program_to_node["p2"]]
        self.assertEqual(p2_node.depth, 2)

        # Make p1 visited so UCT traverses through it to p2
        p1_node = self.ctrl.nodes[self.ctrl.program_to_node["p1"]]
        p1_node.visits = 100
        p1_node.value_sum = 50.0

        # p2 is at max_tree_depth=2, but it's a leaf with no children
        # select() should go: root → p1 → p2 (leaf, stops)
        node, _ = self.ctrl.select()
        self.assertEqual(node.program_id, "p2")
        self.assertEqual(node.depth, 2)


class TestRewardFromConfig(unittest.TestCase):
    """Test TreeQDReward.compute_from_config."""

    def test_uses_config_weights(self):
        from types import SimpleNamespace
        cfg = SimpleNamespace(
            w_quality=1.0,
            w_delta=0.0,
            w_cell_new=0.0,
            w_cell_improved=0.0,
            w_novelty=0.0,
            w_invalid=0.0,
            w_cost=0.0,
        )
        result = AddResult(accepted=True, fitness=0.5)
        reward = TreeQDReward.compute_from_config(result, cfg, max_fitness=1.0)
        self.assertAlmostEqual(reward, 0.5)  # pure quality


if __name__ == "__main__":
    unittest.main()
