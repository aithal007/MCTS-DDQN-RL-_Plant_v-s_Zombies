"""Dual-head Policy+Value network for MCTS-guided RL.

Architecture (AlphaZero-style):
    Input:   Flattened PvZ observation vector (144 dims)
    Encoder: MLP backbone with residual blocks
    Policy:  Logits over action space (316 actions) with action masking
    Value:   Scalar estimate V(s) ∈ [-1, 1]

Also includes a DDQN network for the hybrid MCTS+DDQN approach:
    DuelingDDQN with separate advantage and value streams.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional


class ResidualBlock(nn.Module):
    """Simple residual MLP block: x + MLP(x)."""

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.net(x))


class PvZDualNet(nn.Module):
    """Dual-head network outputting policy logits and state value.

    Used by MCTS to:
      - Provide prior probabilities P(s, a) for tree expansion
      - Evaluate leaf nodes with V(s)

    Args:
        obs_dim:   Observation vector size (default: 144 for PvZSim)
        action_dim: Number of discrete actions (default: 316)
        hidden_dim: Hidden layer width (default: 256)
        num_res_blocks: Number of residual blocks (default: 4)
    """

    def __init__(
        self,
        obs_dim: int = 144,
        action_dim: int = 316,
        hidden_dim: int = 256,
        num_res_blocks: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared encoder backbone
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)]
        )

        # Policy head: outputs log-probabilities over actions
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

        # Value head: outputs scalar V(s) ∈ [-1, 1]
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(
        self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            obs: (batch, obs_dim) observation tensor
            action_mask: (batch, action_dim) boolean mask (True=valid)

        Returns:
            policy_logits: (batch, action_dim) masked logits
            value: (batch, 1) state value
        """
        x = self.encoder(obs)
        x = self.res_blocks(x)

        # Policy
        logits = self.policy_head(x)
        if action_mask is not None:
            # Set invalid action logits to very large negative
            logits = logits.masked_fill(~action_mask, -1e8)

        # Value
        value = self.value_head(x)

        return logits, value

    def predict(
        self,
        obs_np: np.ndarray,
        action_mask_np: Optional[np.ndarray] = None,
    ) -> tuple[np.ndarray, float]:
        """Predict policy probabilities and value from numpy arrays.

        Args:
            obs_np: (obs_dim,) numpy observation
            action_mask_np: (action_dim,) boolean mask

        Returns:
            policy_probs: (action_dim,) probability distribution
            value: scalar float
        """
        self.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
            mask_t = None
            if action_mask_np is not None:
                mask_t = torch.BoolTensor(action_mask_np).unsqueeze(0)

            logits, value = self(obs_t, mask_t)
            probs = F.softmax(logits, dim=-1).squeeze(0).numpy()
            val = value.item()

        return probs, val


class DuelingDDQN(nn.Module):
    """Dueling Double DQN network.

    Decomposes Q(s,a) = V(s) + A(s,a) - mean(A(s,·))

    Used as the Q-network in the hybrid MCTS+DDQN approach.
    The MCTS value estimates help train better Q-values.

    Args:
        obs_dim:    Observation vector size
        action_dim: Number of discrete actions
        hidden_dim: Hidden layer width
        num_layers: Number of hidden layers
    """

    def __init__(
        self,
        obs_dim: int = 144,
        action_dim: int = 316,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Shared feature encoder
        layers = [nn.Linear(obs_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.features = nn.Sequential(*layers)

        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(s, a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(
        self, obs: torch.Tensor, action_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Q-values.

        Args:
            obs: (batch, obs_dim)
            action_mask: (batch, action_dim) boolean, True=valid

        Returns:
            q_values: (batch, action_dim)
        """
        features = self.features(obs)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)

        # Dueling: Q = V + (A - mean(A))
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        if action_mask is not None:
            q = q.masked_fill(~action_mask, -1e8)

        return q

    def predict_action(
        self,
        obs_np: np.ndarray,
        action_mask_np: Optional[np.ndarray] = None,
        epsilon: float = 0.0,
    ) -> int:
        """Select action using ε-greedy policy.

        Args:
            obs_np: observation vector
            action_mask_np: boolean action mask
            epsilon: exploration rate

        Returns:
            Selected action index
        """
        if np.random.random() < epsilon:
            # Random valid action
            if action_mask_np is not None:
                valid = np.where(action_mask_np)[0]
                return int(np.random.choice(valid))
            return int(np.random.randint(self.action_dim))

        self.eval()
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
            mask_t = None
            if action_mask_np is not None:
                mask_t = torch.BoolTensor(action_mask_np).unsqueeze(0)
            q = self(obs_t, mask_t)
            return int(q.argmax(dim=-1).item())
