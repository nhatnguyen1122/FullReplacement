"""
TreeQD-MCTS controller for OpenEvolve.

A tree-structured quality-diversity controller that treats LLM-driven program
evolution as a sequential decision problem, using UCT selection augmented with
QD reward signals (archive coverage, cell improvement, novelty) to decide where
to allocate LLM and evaluator budget.
"""

from __future__ import annotations

import json
import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action types
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    """Types of MCTS actions (edges in the search tree)."""
    MUTATE = "mutate"
    CROSSOVER = "crossover"
    SIMPLIFY = "simplify"
    EXPLORE = "explore"  # exploratory / random direction


@dataclass
class TreeAction:
    """Represents an MCTS edge — the decision taken to produce a child."""
    action_type: ActionType = ActionType.MUTATE
    target_program_id: Optional[str] = None
    prompt_mode: str = "diff"  # diff, rewrite, simplify, exploratory
    model_provider: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "action_type": self.action_type.value,
            "target_program_id": self.target_program_id,
            "prompt_mode": self.prompt_mode,
            "model_provider": self.model_provider,
        }
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TreeAction":
        return cls(
            action_type=ActionType(d.get("action_type", "mutate")),
            target_program_id=d.get("target_program_id"),
            prompt_mode=d.get("prompt_mode", "diff"),
            model_provider=d.get("model_provider"),
        )


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class TreeNode:
    """A node in the MCTS search tree.

    Each node represents a program (or the virtual root which has no program).
    Children are created via expansion when the node is selected.
    """
    node_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_node_id: Optional[str] = None
    program_id: Optional[str] = None  # None for virtual root

    # Children
    children: List[str] = field(default_factory=list)  # child node_ids

    # MCTS statistics
    visits: int = 0
    value_sum: float = 0.0
    best_reward: float = float("-inf")
    best_fitness: float = float("-inf")
    pending_count: int = 0  # in-flight workers targeting this node

    # Tree structure
    depth: int = 0
    feature_key: Optional[str] = None  # MAP-Elites cell key

    # Action that produced this node (edge from parent)
    action: Optional[Dict[str, Any]] = None

    # Timestamps
    creation_iteration: int = 0
    last_expanded_iteration: int = 0

    @property
    def mean_reward(self) -> float:
        """Average reward across visits."""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_node_id": self.parent_node_id,
            "program_id": self.program_id,
            "children": list(self.children),
            "visits": self.visits,
            "value_sum": self.value_sum,
            "best_reward": self.best_reward if self.best_reward != float("-inf") else None,
            "best_fitness": self.best_fitness if self.best_fitness != float("-inf") else None,
            "pending_count": self.pending_count,
            "depth": self.depth,
            "feature_key": self.feature_key,
            "action": self.action,
            "creation_iteration": self.creation_iteration,
            "last_expanded_iteration": self.last_expanded_iteration,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TreeNode":
        best_reward = d.get("best_reward")
        best_fitness = d.get("best_fitness")
        return cls(
            node_id=d["node_id"],
            parent_node_id=d.get("parent_node_id"),
            program_id=d.get("program_id"),
            children=list(d.get("children", [])),
            visits=d.get("visits", 0),
            value_sum=d.get("value_sum", 0.0),
            best_reward=best_reward if best_reward is not None else float("-inf"),
            best_fitness=best_fitness if best_fitness is not None else float("-inf"),
            pending_count=d.get("pending_count", 0),
            depth=d.get("depth", 0),
            feature_key=d.get("feature_key"),
            action=d.get("action"),
            creation_iteration=d.get("creation_iteration", 0),
            last_expanded_iteration=d.get("last_expanded_iteration", 0),
        )


# ---------------------------------------------------------------------------
# QD Reward computation
# ---------------------------------------------------------------------------

@dataclass
class AddResult:
    """Metadata returned by ProgramDatabase.add() for reward computation.

    This is also defined in database.py for import convenience; here we keep a
    mirror so tree_qd.py stays self-contained for testing.
    """
    program_id: str = ""
    island_idx: int = 0
    feature_key: Optional[str] = None
    added_to_new_cell: bool = False
    replaced_program_id: Optional[str] = None
    improved_existing_cell: bool = False
    accepted: bool = True  # False if not retained in the population
    evaluation_success: bool = True
    fitness: float = 0.0
    parent_fitness: float = 0.0


class TreeQDReward:
    """Compute composite QD reward from AddResult and search config."""

    @staticmethod
    def compute(
        add_result: "AddResult",
        *,
        w_quality: float = 0.3,
        w_delta: float = 0.2,
        w_cell_new: float = 0.2,
        w_cell_improved: float = 0.15,
        w_novelty: float = 0.05,
        w_invalid: float = 0.1,
        w_cost: float = 0.0,
        token_cost: float = 0.0,
        max_fitness: float = 1.0,
    ) -> float:
        """Return a scalar reward in roughly [0, 1].

        Args:
            add_result: insertion metadata from database.add()
            w_*: reward component weights
            token_cost: normalised token cost in [0, 1]
            max_fitness: scaling constant for fitness normalisation
        """
        if not add_result.accepted or not add_result.evaluation_success:
            return -w_invalid

        # Quality: normalised child fitness
        quality = add_result.fitness / max(max_fitness, 1e-8)
        quality = max(0.0, min(1.0, quality))

        # Delta: positive fitness improvement over parent
        delta = max(0.0, add_result.fitness - add_result.parent_fitness)
        delta = delta / max(max_fitness, 1e-8)
        delta = min(1.0, delta)

        # Cell bonuses
        cell_new = 1.0 if add_result.added_to_new_cell else 0.0
        cell_improved = 1.0 if add_result.improved_existing_cell else 0.0

        # Novelty proxy: new cell counts as novelty too
        novelty = 1.0 if add_result.added_to_new_cell else 0.0

        reward = (
            w_quality * quality
            + w_delta * delta
            + w_cell_new * cell_new
            + w_cell_improved * cell_improved
            + w_novelty * novelty
            - w_cost * token_cost
        )

        return reward

    @staticmethod
    def compute_from_config(
        add_result: "AddResult",
        search_config: Any,
        token_cost: float = 0.0,
        max_fitness: float = 1.0,
    ) -> float:
        """Convenience: pull weights from a SearchConfig object."""
        return TreeQDReward.compute(
            add_result,
            w_quality=search_config.w_quality,
            w_delta=search_config.w_delta,
            w_cell_new=search_config.w_cell_new,
            w_cell_improved=search_config.w_cell_improved,
            w_novelty=search_config.w_novelty,
            w_invalid=search_config.w_invalid,
            w_cost=search_config.w_cost,
            token_cost=token_cost,
            max_fitness=max_fitness,
        )


# ---------------------------------------------------------------------------
# TreeQD Controller
# ---------------------------------------------------------------------------

class TreeQDController:
    """MCTS controller with QD-aware selection and backpropagation.

    Usage::

        ctrl = TreeQDController(search_config)
        ctrl.register_program(program_id, parent_program_id=None, iteration=0)
        node, action = ctrl.select()
        # ... run LLM + evaluator ...
        ctrl.record_result(node.node_id, child_program_id, add_result, iteration)
    """

    def __init__(
        self,
        exploration_constant: float = 0.7,
        qd_weight: float = 0.5,
        cost_weight: float = 0.1,
        max_tree_depth: int = 20,
        backup: str = "mixed",
        backup_alpha: float = 0.5,
        widening_alpha: float = 0.5,
    ):
        self.exploration_constant = exploration_constant
        self.qd_weight = qd_weight
        self.cost_weight = cost_weight
        self.max_tree_depth = max_tree_depth
        self.backup = backup
        self.backup_alpha = backup_alpha
        self.widening_alpha = widening_alpha

        # Virtual root
        self.root = TreeNode(node_id="__root__", depth=0)
        self.nodes: Dict[str, TreeNode] = {"__root__": self.root}

        # program_id → node_id mapping
        self.program_to_node: Dict[str, str] = {}

        # Stats
        self.total_expansions: int = 0
        self.total_backprops: int = 0

    # ------------------------------------------------------------------
    # Program registration
    # ------------------------------------------------------------------

    def register_program(
        self,
        program_id: str,
        parent_program_id: Optional[str] = None,
        iteration: int = 0,
        feature_key: Optional[str] = None,
        fitness: float = 0.0,
    ) -> TreeNode:
        """Register a program in the tree (e.g. the initial program or a seed).

        If *parent_program_id* maps to an existing node, the new node becomes
        its child; otherwise the new node is attached to the root.
        """
        if program_id in self.program_to_node:
            return self.nodes[self.program_to_node[program_id]]

        parent_node_id = "__root__"
        parent_depth = 0
        if parent_program_id and parent_program_id in self.program_to_node:
            parent_node_id = self.program_to_node[parent_program_id]
            parent_depth = self.nodes[parent_node_id].depth

        node = TreeNode(
            parent_node_id=parent_node_id,
            program_id=program_id,
            depth=parent_depth + 1,
            feature_key=feature_key,
            creation_iteration=iteration,
            best_fitness=fitness,
        )
        self.nodes[node.node_id] = node
        self.nodes[parent_node_id].children.append(node.node_id)
        self.program_to_node[program_id] = node.node_id

        logger.debug(
            "Registered program %s as tree node %s (parent_node=%s, depth=%d)",
            program_id, node.node_id, parent_node_id, node.depth,
        )
        return node

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(self) -> Tuple[TreeNode, TreeAction]:
        """Select a node for expansion using UCT + QD potential.

        Returns (selected_node, action). Real program nodes are marked pending
        before return. The virtual root can be returned only when the tree has
        no real program yet; in that case no pending count is added.
        """
        node = self.root

        while True:
            if not node.children:
                # Leaf — select this node
                break

            if node.depth >= self.max_tree_depth:
                break

            # For non-root nodes, check progressive widening
            if node.program_id is not None and self._should_widen(node):
                # Progressive widening — create new child later (select this node)
                break

            # Pick best child by UCT
            best_child_id = self._best_child_uct(node)
            if best_child_id is None:
                break
            node = self.nodes[best_child_id]

        # Build a default action
        action = TreeAction(
            action_type=ActionType.MUTATE,
            target_program_id=node.program_id,
            prompt_mode="diff",
        )

        # Mark pending only for real program nodes. The caller may use a root
        # return as a signal to seed the tree from the database.
        if node.program_id is not None:
            node.pending_count += 1

        return node, action

    def _best_child_uct(self, parent: TreeNode) -> Optional[str]:
        """Select the child with the highest UCT score."""
        best_score = float("-inf")
        best_child_id = None
        parent_visits = parent.visits + parent.pending_count

        for child_id in parent.children:
            child = self.nodes.get(child_id)
            if child is None:
                continue

            child_effective_visits = child.visits + child.pending_count

            # Exploitation: backup value
            if self.backup == "max":
                exploit = child.best_reward if child.best_reward != float("-inf") else 0.0
            elif self.backup == "mean":
                exploit = child.mean_reward
            else:  # mixed
                mean_val = child.mean_reward
                max_val = child.best_reward if child.best_reward != float("-inf") else mean_val
                exploit = self.backup_alpha * mean_val + (1.0 - self.backup_alpha) * max_val

            # Exploration: UCT bonus
            if child_effective_visits == 0:
                explore = float("inf")
            else:
                explore = self.exploration_constant * math.sqrt(
                    math.log(parent_visits + 1) / (child_effective_visits + 1)
                )

            # QD archive potential: prefer feature cells with fewer known tree
            # representatives and less local evaluation pressure.
            qd_bonus = 0.0
            if child.feature_key:
                same_cell_count = sum(
                    1
                    for node in self.nodes.values()
                    if node.program_id is not None and node.feature_key == child.feature_key
                )
                cell_rarity = 1.0 / math.sqrt(max(1, same_cell_count))
                local_uncertainty = 1.0 / math.sqrt(child_effective_visits + 1)
                qd_bonus = 0.5 * cell_rarity + 0.5 * local_uncertainty

            score = exploit + explore + self.qd_weight * qd_bonus

            if score > best_score:
                best_score = score
                best_child_id = child_id

        return best_child_id

    def _should_widen(self, node: TreeNode) -> bool:
        """Progressive widening: allow a new child if len(children) < ceil(visits^α)."""
        if node.visits == 0:
            return True
        max_children = math.ceil(node.visits ** self.widening_alpha)
        return len(node.children) < max_children

    # ------------------------------------------------------------------
    # Expansion + backpropagation
    # ------------------------------------------------------------------

    def record_result(
        self,
        parent_node_id: str,
        child_program_id: str,
        add_result: "AddResult",
        reward: float,
        iteration: int = 0,
    ) -> TreeNode:
        """Create a child node and backpropagate reward.

        Args:
            parent_node_id: the node that was selected for expansion
            child_program_id: the newly evaluated program's id
            add_result: insertion metadata from database.add()
            reward: pre-computed composite QD reward
            iteration: current iteration number

        Returns:
            The newly created child tree node.
        """
        parent_node = self.nodes.get(parent_node_id)
        if parent_node is None:
            logger.warning("Parent node %s not found; attaching to root", parent_node_id)
            parent_node = self.root

        # Decrement pending
        parent_node.pending_count = max(0, parent_node.pending_count - 1)

        if not add_result.accepted or not add_result.evaluation_success:
            self._backpropagate(parent_node.node_id, reward, add_result.fitness)
            logger.debug(
                "Recorded rejected/failed result: parent_node=%s program=%s reward=%.4f",
                parent_node.node_id, child_program_id, reward,
            )
            return parent_node

        # Create child node
        child_node = TreeNode(
            parent_node_id=parent_node_id,
            program_id=child_program_id,
            depth=parent_node.depth + 1,
            feature_key=add_result.feature_key,
            creation_iteration=iteration,
            best_fitness=add_result.fitness,
        )
        self.nodes[child_node.node_id] = child_node
        parent_node.children.append(child_node.node_id)
        parent_node.last_expanded_iteration = iteration
        self.program_to_node[child_program_id] = child_node.node_id
        self.total_expansions += 1

        # Backpropagate
        self._backpropagate(child_node.node_id, reward, add_result.fitness)

        logger.debug(
            "Recorded result: parent_node=%s child_node=%s reward=%.4f fitness=%.4f",
            parent_node_id, child_node.node_id, reward, add_result.fitness,
        )
        return child_node

    def release_pending(self, node_id: str) -> None:
        """Release a pending selection when submission falls back or fails."""
        node = self.nodes.get(node_id)
        if node is not None:
            node.pending_count = max(0, node.pending_count - 1)

    def mark_unavailable(self, node_id: str, penalty: float = -1.0) -> None:
        """Penalize a selected node whose program is no longer in the database."""
        node = self.nodes.get(node_id)
        if node is None:
            return
        node.pending_count = max(0, node.pending_count - 1)
        self._backpropagate(node_id, penalty, float("-inf"))

    def _backpropagate(self, start_node_id: str, reward: float, fitness: float) -> None:
        """Walk from *start_node_id* up to root, updating statistics."""
        node_id: Optional[str] = start_node_id
        while node_id is not None:
            node = self.nodes.get(node_id)
            if node is None:
                break
            node.visits += 1
            node.value_sum += reward
            if reward > node.best_reward:
                node.best_reward = reward
            if fitness > node.best_fitness:
                node.best_fitness = fitness
            node_id = node.parent_node_id
        self.total_backprops += 1

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize tree state to JSON."""
        state = {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "program_to_node": dict(self.program_to_node),
            "root_node_id": self.root.node_id,
            "config": {
                "exploration_constant": self.exploration_constant,
                "qd_weight": self.qd_weight,
                "cost_weight": self.cost_weight,
                "max_tree_depth": self.max_tree_depth,
                "backup": self.backup,
                "backup_alpha": self.backup_alpha,
                "widening_alpha": self.widening_alpha,
            },
            "stats": {
                "total_expansions": self.total_expansions,
                "total_backprops": self.total_backprops,
            },
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        logger.info("Saved TreeQD state to %s (%d nodes)", path, len(self.nodes))

    @classmethod
    def load(cls, path: str) -> "TreeQDController":
        """Deserialize tree state from JSON."""
        with open(path, "r") as f:
            state = json.load(f)

        cfg = state.get("config", {})
        ctrl = cls(
            exploration_constant=cfg.get("exploration_constant", 0.7),
            qd_weight=cfg.get("qd_weight", 0.5),
            cost_weight=cfg.get("cost_weight", 0.1),
            max_tree_depth=cfg.get("max_tree_depth", 20),
            backup=cfg.get("backup", "mixed"),
            backup_alpha=cfg.get("backup_alpha", 0.5),
            widening_alpha=cfg.get("widening_alpha", 0.5),
        )

        # Rebuild nodes
        ctrl.nodes = {}
        for nid, nd in state.get("nodes", {}).items():
            node = TreeNode.from_dict(nd)
            node.pending_count = 0
            ctrl.nodes[nid] = node

        ctrl.root = ctrl.nodes.get(
            state.get("root_node_id", "__root__"),
            ctrl.root,
        )

        ctrl.program_to_node = dict(state.get("program_to_node", {}))

        stats = state.get("stats", {})
        ctrl.total_expansions = stats.get("total_expansions", 0)
        ctrl.total_backprops = stats.get("total_backprops", 0)

        logger.info("Loaded TreeQD state from %s (%d nodes)", path, len(ctrl.nodes))
        return ctrl

    @classmethod
    def from_config(cls, search_config: Any) -> "TreeQDController":
        """Construct from a SearchConfig dataclass."""
        return cls(
            exploration_constant=search_config.exploration_constant,
            qd_weight=search_config.qd_weight,
            cost_weight=search_config.cost_weight,
            max_tree_depth=search_config.max_tree_depth,
            backup=search_config.backup,
            backup_alpha=search_config.backup_alpha,
            widening_alpha=search_config.widening_alpha,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """Return summary stats for logging."""
        depths = [n.depth for n in self.nodes.values()]
        visits = [n.visits for n in self.nodes.values() if n.visits > 0]
        return {
            "total_nodes": len(self.nodes),
            "max_depth": max(depths) if depths else 0,
            "total_expansions": self.total_expansions,
            "total_backprops": self.total_backprops,
            "avg_visits": sum(visits) / len(visits) if visits else 0,
            "root_visits": self.root.visits,
        }
