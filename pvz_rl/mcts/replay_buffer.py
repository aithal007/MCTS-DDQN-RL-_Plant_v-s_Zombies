"""Experience replay buffer for MCTS+DDQN training.

Stores trajectories from self-play episodes:
    (observation, mcts_policy, reward, next_observation, done, action_mask)

Supports both:
    - Standard DDQN transitions (s, a, r, s', done)
    - MCTS-enhanced transitions (s, π_MCTS, value_target, action_mask)
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Transition:
    """A single experience transition."""
    obs: np.ndarray           # observation at time t
    action: int               # action taken
    reward: float             # reward received
    next_obs: np.ndarray      # observation at time t+1
    done: bool                # episode terminated?
    action_mask: np.ndarray   # valid action mask at time t
    mcts_policy: Optional[np.ndarray] = None  # MCTS improved policy
    mcts_value: Optional[float] = None         # MCTS value estimate


class ReplayBuffer:
    """Fixed-size experience replay buffer with uniform sampling.

    Args:
        capacity: Maximum number of transitions to store.
        seed: Random seed for reproducibility.
    """

    def __init__(self, capacity: int = 100_000, seed: int = 42):
        self.buffer: deque[Transition] = deque(maxlen=capacity)
        self.rng = random.Random(seed)

    def push(self, transition: Transition):
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def push_many(self, transitions: list[Transition]):
        """Add multiple transitions."""
        for t in transitions:
            self.push(t)

    def sample(self, batch_size: int) -> list[Transition]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            List of Transition objects.
        """
        batch_size = min(batch_size, len(self.buffer))
        return self.rng.sample(list(self.buffer), batch_size)

    def sample_numpy(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample a batch and return as numpy arrays (ready for PyTorch).

        Returns dict with keys:
            obs, actions, rewards, next_obs, dones, action_masks,
            mcts_policies (if available), mcts_values (if available)
        """
        batch = self.sample(batch_size)

        obs = np.array([t.obs for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_obs = np.array([t.next_obs for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        action_masks = np.array([t.action_mask for t in batch], dtype=bool)

        result = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "action_masks": action_masks,
        }

        # MCTS data (may not be present for all transitions)
        # Only include if ALL transitions in the batch have MCTS data
        has_policies = all(t.mcts_policy is not None for t in batch)
        if has_policies:
            result["mcts_policies"] = np.array(
                [t.mcts_policy for t in batch], dtype=np.float32
            )

        has_values = all(t.mcts_value is not None for t in batch)
        if has_values:
            result["mcts_values"] = np.array(
                [t.mcts_value for t in batch], dtype=np.float32
            )

        return result

    def __len__(self) -> int:
        return len(self.buffer)

    @property
    def is_ready(self) -> bool:
        """Check if buffer has enough samples for a meaningful batch."""
        return len(self.buffer) >= 256


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized sampling based on TD error.

    Higher-error transitions are sampled more frequently,
    improving learning efficiency for DDQN.

    Args:
        capacity: Maximum buffer size.
        alpha: Priority exponent (0=uniform, 1=full prioritization).
        beta: Importance sampling correction (annealed from beta to 1.0).
        seed: Random seed.
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        seed: int = 42,
    ):
        super().__init__(capacity, seed)
        self.alpha = alpha
        self.beta = beta
        self.priorities: deque[float] = deque(maxlen=capacity)
        self._max_priority = 1.0

    def push(self, transition: Transition):
        """Add transition with max priority."""
        super().push(transition)
        self.priorities.append(self._max_priority)

    def sample_numpy(self, batch_size: int) -> dict[str, np.ndarray]:
        """Sample with prioritized probabilities."""
        n = len(self.buffer)
        batch_size = min(batch_size, n)

        # Ensure priorities are in sync (defensive)
        while len(self.priorities) < n:
            self.priorities.append(self._max_priority)
        while len(self.priorities) > n:
            self.priorities.popleft()

        # Compute sampling probabilities
        priorities = np.array(list(self.priorities), dtype=np.float32)
        probs = priorities ** self.alpha
        total = probs.sum()
        if total == 0:
            probs = np.ones(n, dtype=np.float32) / n
        else:
            probs /= total

        # Sample indices
        indices = np.random.choice(n, size=batch_size, p=probs, replace=False)

        # Importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]

        obs = np.array([t.obs for t in batch], dtype=np.float32)
        actions = np.array([t.action for t in batch], dtype=np.int64)
        rewards = np.array([t.reward for t in batch], dtype=np.float32)
        next_obs = np.array([t.next_obs for t in batch], dtype=np.float32)
        dones = np.array([t.done for t in batch], dtype=np.float32)
        action_masks = np.array([t.action_mask for t in batch], dtype=bool)

        result = {
            "obs": obs,
            "actions": actions,
            "rewards": rewards,
            "next_obs": next_obs,
            "dones": dones,
            "action_masks": action_masks,
            "weights": weights.astype(np.float32),
            "indices": indices,
        }

        has_policies = all(t.mcts_policy is not None for t in batch)
        if has_policies:
            result["mcts_policies"] = np.array(
                [t.mcts_policy for t in batch], dtype=np.float32
            )

        return result

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, td in zip(indices, td_errors):
            priority = abs(td) + 1e-6
            self.priorities[idx] = priority
            self._max_priority = max(self._max_priority, priority)
