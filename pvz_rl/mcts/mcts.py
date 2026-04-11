"""Monte Carlo Tree Search with neural network guidance (AlphaZero-style).

Implements PUCT-based MCTS that uses a policy+value network to:
  - Bias tree expansion with learned priors P(s, a)
  - Evaluate leaf nodes with V(s) instead of random rollouts

The search generates an improved policy π_MCTS from visit counts,
which is then used as the training target for the neural network.

Reference:
    Silver et al. (2017): "Mastering the game of Go without human knowledge"
    Silver et al. (2018): "A general reinforcement learning algorithm
                           that masters chess, shogi, and Go through self-play"
"""

from __future__ import annotations

import math
import copy
from typing import Optional

import numpy as np

from pvz_rl.sim.game import PvZEngine
from .simulator_wrapper import SimulatorWrapper


class MCTSNode:
    """A node in the MCTS search tree.

    Each node stores:
        - Visit count N(s, a) for each action
        - Total value W(s, a) for each action
        - Mean value Q(s, a) = W(s, a) / N(s, a)
        - Prior probability P(s, a) from the neural network
        - Children: mapping from action -> MCTSNode
    """

    __slots__ = [
        "state_engine", "parent", "parent_action", "is_terminal",
        "terminal_value", "n_actions", "visit_count", "total_value",
        "prior", "children", "action_mask", "expanded",
    ]

    def __init__(
        self,
        state_engine: PvZEngine,
        parent: Optional[MCTSNode] = None,
        parent_action: int = -1,
        n_actions: int = 316,
    ):
        self.state_engine = state_engine
        self.parent = parent
        self.parent_action = parent_action
        self.n_actions = n_actions
        self.is_terminal = SimulatorWrapper.is_terminal(state_engine.state)
        self.terminal_value = SimulatorWrapper.get_value(state_engine.state)

        # Statistics arrays
        self.visit_count = np.zeros(n_actions, dtype=np.float32)
        self.total_value = np.zeros(n_actions, dtype=np.float32)
        self.prior = np.zeros(n_actions, dtype=np.float32)
        self.children: dict[int, MCTSNode] = {}
        self.action_mask = SimulatorWrapper.get_action_mask(state_engine.state)
        self.expanded = False

    @property
    def total_visits(self) -> float:
        return float(self.visit_count.sum())

    def q_values(self) -> np.ndarray:
        """Mean action values Q(s, a) = W(s, a) / N(s, a)."""
        q = np.zeros(self.n_actions, dtype=np.float32)
        visited = self.visit_count > 0
        q[visited] = self.total_value[visited] / self.visit_count[visited]
        return q


class MCTS:
    """Monte Carlo Tree Search with neural network guidance.

    Args:
        network: A PvZDualNet (or any model implementing .predict()).
        num_simulations: Number of MCTS simulations per search.
        c_puct: Exploration constant for PUCT formula.
        dirichlet_alpha: Alpha for Dirichlet noise at root (0 = no noise).
        dirichlet_epsilon: Mix weight for Dirichlet noise.
        temperature: Temperature for action selection from visit counts.
    """

    def __init__(
        self,
        network,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        temperature: float = 1.0,
    ):
        self.network = network
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.temperature = temperature

    def search(self, root_engine: PvZEngine) -> tuple[np.ndarray, float]:
        """Run MCTS from the given state and return improved policy.

        Args:
            root_engine: The current game engine (will be cloned, not modified).

        Returns:
            policy: (action_dim,) probability distribution from visit counts
            root_value: Estimated value of root state
        """
        # Create root node (clone engine so we don't modify the real one)
        root = MCTSNode(
            state_engine=SimulatorWrapper.clone_engine(root_engine),
            n_actions=1 + 7 * 5 * 9,  # 316
        )

        # Expand root
        self._expand(root)

        # Add Dirichlet noise to root priors for exploration
        if self.dirichlet_alpha > 0:
            noise = np.random.dirichlet(
                [self.dirichlet_alpha] * root.n_actions
            )
            root.prior = (
                (1 - self.dirichlet_epsilon) * root.prior
                + self.dirichlet_epsilon * noise
            )
            # Re-mask invalid actions
            root.prior[~root.action_mask] = 0.0
            total = root.prior.sum()
            if total > 0:
                root.prior /= total

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]

            # --- Selection ---
            while node.expanded and not node.is_terminal:
                action = self._select_action(node)
                if action in node.children:
                    node = node.children[action]
                else:
                    # Expand this action
                    child_engine = SimulatorWrapper.clone_engine(node.state_engine)
                    SimulatorWrapper.simulate_action(child_engine, action)
                    child = MCTSNode(
                        state_engine=child_engine,
                        parent=node,
                        parent_action=action,
                        n_actions=node.n_actions,
                    )
                    node.children[action] = child
                    node = child

                search_path.append(node)

            # --- Evaluation ---
            if node.is_terminal:
                value = node.terminal_value
            else:
                # Expand and evaluate with neural network
                self._expand(node)
                obs = SimulatorWrapper.get_observation(node.state_engine.state)
                mask = node.action_mask
                _, value = self.network.predict(obs, mask)

            # --- Backup ---
            self._backup(search_path, value)

        # --- Extract policy from visit counts ---
        policy = self._get_policy(root)
        root_value = root.q_values()[root.action_mask].mean() if root.action_mask.any() else 0.0

        return policy, root_value

    def _expand(self, node: MCTSNode):
        """Expand a node by computing priors from the neural network."""
        if node.is_terminal:
            return

        obs = SimulatorWrapper.get_observation(node.state_engine.state)
        mask = node.action_mask

        probs, _ = self.network.predict(obs, mask)

        # Set priors (already masked and softmaxed by network.predict)
        node.prior = probs.copy()
        # Ensure invalid actions have zero prior
        node.prior[~mask] = 0.0
        total = node.prior.sum()
        if total > 0:
            node.prior /= total
        else:
            # Fallback: uniform over valid actions
            node.prior[mask] = 1.0 / mask.sum()

        node.expanded = True

    def _select_action(self, node: MCTSNode) -> int:
        """Select action using PUCT formula.

        PUCT(s, a) = Q(s, a) + c_puct * P(s, a) * sqrt(N(s)) / (1 + N(s, a))

        Args:
            node: Current node to select from.

        Returns:
            Selected action index.
        """
        q = node.q_values()
        total_visits = node.total_visits

        # PUCT exploration bonus
        puct = (
            self.c_puct
            * node.prior
            * math.sqrt(total_visits)
            / (1.0 + node.visit_count)
        )

        # Combined score
        scores = q + puct

        # Mask invalid actions
        scores[~node.action_mask] = -np.inf

        return int(np.argmax(scores))

    def _backup(self, search_path: list[MCTSNode], value: float):
        """Backup value through the search path.

        Updates N(s, a) and W(s, a) for each node-action pair.
        Value alternates sign for adversarial games, but PvZ is
        single-player so we use the value directly.
        """
        for i in range(len(search_path) - 1):
            node = search_path[i]
            child = search_path[i + 1]
            action = child.parent_action

            if action >= 0:
                node.visit_count[action] += 1
                node.total_value[action] += value

    def _get_policy(self, root: MCTSNode) -> np.ndarray:
        """Extract policy from root visit counts.

        With temperature τ:
            π(a) ∝ N(s, a)^(1/τ)

        Low temperature → greedy (exploit best actions)
        High temperature → exploratory (more uniform)
        """
        visits = root.visit_count.copy()

        if self.temperature < 0.01:
            # Greedy: pick most visited
            policy = np.zeros_like(visits)
            best = np.argmax(visits)
            policy[best] = 1.0
        else:
            # Softmax with temperature
            visits_temp = visits ** (1.0 / self.temperature)
            total = visits_temp.sum()
            if total > 0:
                policy = visits_temp / total
            else:
                # Fallback
                mask = root.action_mask
                policy = np.zeros_like(visits)
                if mask.any():
                    policy[mask] = 1.0 / mask.sum()

        return policy
