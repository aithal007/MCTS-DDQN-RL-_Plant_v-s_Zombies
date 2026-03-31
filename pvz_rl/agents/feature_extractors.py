"""Custom feature extractors for Stable-Baselines3 policies.

- ``PvZSimExtractor`` : processes the flattened dict obs through an MLP
- ``PvZCNNExtractor`` : processes RGB image obs (for Steam env) through a CNN
"""

from __future__ import annotations

import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class PvZSimExtractor(BaseFeaturesExtractor):
    """MLP feature extractor for the simulator's flat observation vector."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        input_dim = int(observation_space.shape[0])
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.net(observations)


class PvZCNNExtractor(BaseFeaturesExtractor):
    """CNN feature extractor for RGB image observations (Steam env).

    Expects input shape (N, H, W, 3) -- the wrapper should already handle
    channel-first conversion if needed.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # SB3 NatureCNN expects (C, H, W)
        n_channels = observation_space.shape[-1] if len(observation_space.shape) == 3 else 1
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the output size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, n_channels, *observation_space.shape[:2])
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Ensure channel-first: (N, H, W, C) -> (N, C, H, W)
        if observations.dim() == 4 and observations.shape[-1] in (1, 3):
            observations = observations.permute(0, 3, 1, 2)
        x = self.cnn(observations.float() / 255.0)
        return self.linear(x)
