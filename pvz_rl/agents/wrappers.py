"""Observation wrappers to flatten the dict obs into a flat vector or image.

Stable-Baselines3 works best with either:
  - A flat Box observation (for MLP policies)
  - An image Box observation (for CNN policies)

These wrappers convert the PvZSim dict obs into those formats.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class FlattenDictObs(gym.ObservationWrapper):
    """Flatten the PvZSim dict observation into a single 1-D float32 vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # Calculate total size by sampling
        sample = env.observation_space.sample()
        self._keys = sorted(sample.keys())
        total = sum(np.prod(sample[k].shape) for k in self._keys)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(int(total),), dtype=np.float32)

    def observation(self, obs: dict) -> np.ndarray:
        parts = [obs[k].flatten().astype(np.float32) for k in self._keys]
        return np.concatenate(parts)


class ActionMaskWrapper(gym.Wrapper):
    """Adds action masking support for the PvZSim environment.

    Invalid actions (can't afford, cell occupied, on cooldown) are masked.
    During training with MaskablePPO from sb3-contrib, the policy respects
    these masks.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # The underlying env must have get_action_mask()
        assert hasattr(env.unwrapped, "get_action_mask"), (
            "ActionMaskWrapper requires the env to have get_action_mask()"
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        info["action_mask"] = self.env.unwrapped.get_action_mask()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if not (terminated or truncated):
            info["action_mask"] = self.env.unwrapped.get_action_mask()
        return obs, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Compatibility method for sb3-contrib MaskablePPO."""
        return self.env.unwrapped.get_action_mask()
