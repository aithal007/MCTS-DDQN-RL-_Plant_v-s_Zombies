"""Configurable reward shaping wrappers for PvZ-RL.

Strictly action-based rewards. The agent is NO LONGER rewarded just for "being alive".
Instead, the agent is directly rewarded for:
  - Hitting zombies with projectiles (forces Shooters)
  - Collecting actual sun (forces Sunflowers)
  - Killing zombies
  ...and severely penalized for letting zombies deal damage to plants.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import gymnasium as gym
import numpy as np


class RewardScheme(str, Enum):
    SPARSE = "sparse"
    KILL_PENALTY = "kill_penalty"
    WAVE_BONUS = "wave_bonus"
    TIME_PENALTY = "time_penalty"
    POTENTIAL = "potential"


@dataclass
class RewardConfig:
    """Hyperparameters for reward shaping."""
    scheme: RewardScheme = RewardScheme.KILL_PENALTY

    # ── Terminal rewards ──
    win_reward: float = 5000.0
    loss_penalty: float = 300.0

    # ── Per-event base rewards ──
    alpha: float = 100.0       # per zombie killed
    beta: float = 10.0        # penalty per plant completely destroyed
    delta: float = 300.0      # penalty on the game-over step
    eta: float = 30.0         # bonus per wave cleared

    # ── Dense ACTION rewards ──
    projectile_hit_reward: float = 5.0  # per projectile hit (Shooters ONLY, cuts out Potato Mine)
    sun_reward: float = 2.0             # per point of sun collected (Sunflowers)
    plant_damage_penalty: float = 0.0   # Set to 0! A Wallnut taking damage is doing its job, don't penalize it!
    lane_wasting_penalty: float = 50.0  # Massive penalty for placing shooters in rows with no zombies

    # ── Time / Potential ──
    tau: float = 0.0
    kappa: float = 1.0
    gamma: float = 0.99


class RewardShaperWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: Optional[RewardConfig] = None):
        super().__init__(env)
        self.config = config or RewardConfig()

        self._prev_zombies_killed: int = 0
        self._prev_plants_lost: int = 0
        self._prev_wave_idx: int = 0
        self._prev_potential: float = 0.0
        self._prev_sun_collected: int = 0
        self._prev_projectile_hits: int = 0
        self._prev_wasted_placements: int = 0
        # For tracking plant HP loss
        self._prev_total_plant_hp: int = 0

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_zombies_killed = 0
        self._prev_plants_lost = 0
        self._prev_wave_idx = 0
        self._prev_potential = self._compute_potential()
        self._prev_sun_collected = 0
        self._prev_projectile_hits = 0
        self._prev_wasted_placements = 0
        
        game_state = self._get_game_state()
        self._prev_total_plant_hp = sum(p.hp for p in game_state.plants if p.hp > 0) if game_state else 0
        
        return obs, info

    def step(self, action):
        obs, _original_reward, terminated, truncated, info = self.env.step(action)
        reward = self._compute_shaped_reward(info, terminated)
        return obs, reward, terminated, truncated, info

    def _compute_shaped_reward(self, info: dict, terminated: bool) -> float:
        cfg = self.config
        game_state = self._get_game_state()

        # Extract current values
        cur_killed = info.get("zombies_killed", 0)
        cur_plants_lost = info.get("plants_lost", 0)
        cur_sun_collected = getattr(game_state, "total_sun_collected", 0) if game_state else 0
        cur_projectile_hits = getattr(game_state, "projectile_hits", 0) if game_state else 0
        cur_wasted_placements = getattr(game_state, "wasted_placements", 0) if game_state else 0
        cur_wave_idx = getattr(game_state, "next_wave_idx", 0) if game_state else 0
        is_game_over = info.get("game_over", False)
        is_victory = info.get("victory", False)
        
        cur_total_plant_hp = sum(p.hp for p in game_state.plants if p.hp > 0) if game_state else 0

        # Compute deltas
        new_kills = cur_killed - self._prev_zombies_killed
        new_losses = cur_plants_lost - self._prev_plants_lost
        new_sun_collected = cur_sun_collected - self._prev_sun_collected
        new_waves = cur_wave_idx - self._prev_wave_idx
        
        # Action tracker deltas
        new_projectile_hits = cur_projectile_hits - self._prev_projectile_hits
        new_wasted_placements = cur_wasted_placements - self._prev_wasted_placements
        # If HP went up (new plant placed), that's not damage taken.
        # Damage taken is only when total HP goes down (excluding when a plant is fully lost, which is handled by beta).
        # Actually a simpler way: just check if HP dropped. Or rely on beta. 
        # We will use HP drop to penalize zombies eating.
        hp_diff = self._prev_total_plant_hp - cur_total_plant_hp
        new_hp_lost = max(0, hp_diff)

        reward = 0.0

        if cfg.scheme == RewardScheme.SPARSE:
            if is_victory:
                reward = cfg.win_reward
            elif is_game_over:
                reward = -cfg.loss_penalty

        elif cfg.scheme == RewardScheme.KILL_PENALTY:
            reward = self._base_reward(
                new_kills, new_losses, new_sun_collected, new_projectile_hits, new_wasted_placements, new_hp_lost, is_game_over, is_victory
            )

        elif cfg.scheme == RewardScheme.WAVE_BONUS:
            reward = self._base_reward(
                new_kills, new_losses, new_sun_collected, new_projectile_hits, new_wasted_placements, new_hp_lost, is_game_over, is_victory
            )
            reward += cfg.eta * max(new_waves, 0)

        elif cfg.scheme == RewardScheme.TIME_PENALTY:
            reward = self._base_reward(
                new_kills, new_losses, new_sun_collected, new_projectile_hits, new_wasted_placements, new_hp_lost, is_game_over, is_victory
            )
            reward -= cfg.tau

        elif cfg.scheme == RewardScheme.POTENTIAL:
            reward = self._base_reward(
                new_kills, new_losses, new_sun_collected, new_projectile_hits, new_wasted_placements, new_hp_lost, is_game_over, is_victory
            )
            cur_potential = self._compute_potential()
            if not terminated:
                shaping = cfg.gamma * cur_potential - self._prev_potential
            else:
                shaping = -self._prev_potential
            reward += shaping
            self._prev_potential = cur_potential

        # Update tracking
        self._prev_zombies_killed = cur_killed
        self._prev_plants_lost = cur_plants_lost
        self._prev_sun_collected = cur_sun_collected
        self._prev_projectile_hits = cur_projectile_hits
        self._prev_wasted_placements = cur_wasted_placements
        self._prev_wave_idx = cur_wave_idx
        self._prev_total_plant_hp = cur_total_plant_hp

        return reward

    def _base_reward(
        self,
        new_kills: int,
        new_losses: int,
        new_sun_collected: int,
        new_projectile_hits: int,
        new_wasted_placements: int,
        new_hp_lost: int,
        is_game_over: bool,
        is_victory: bool,
    ) -> float:
        cfg = self.config
        reward = 0.0
        
        reward += cfg.alpha * new_kills
        reward -= cfg.beta * new_losses
        
        # Dense Action Rewards
        reward += cfg.sun_reward * new_sun_collected
        reward += cfg.projectile_hit_reward * new_projectile_hits
        reward -= cfg.plant_damage_penalty * new_hp_lost
        reward -= cfg.lane_wasting_penalty * new_wasted_placements

        if is_game_over:
            reward -= cfg.delta
        if is_victory:
            reward += cfg.win_reward

        return reward

    def _compute_potential(self) -> float:
        game_state = self._get_game_state()
        if game_state is None:
            return 0.0
        from pvz_rl.sim.constants import COLS
        if len(game_state.zombies) == 0:
            return self.config.kappa * float(COLS)
        distances = [z.x for z in game_state.zombies if z.hp > 0]
        if not distances:
            return self.config.kappa * float(COLS)
        mean_dist = sum(distances) / len(distances)
        return self.config.kappa * mean_dist

    def _get_game_state(self):
        unwrapped = self.env.unwrapped
        if hasattr(unwrapped, "engine"):
            return unwrapped.engine.state
        return None


def make_shaped_env(
    scheme: str = "kill_penalty",
    difficulty: int = 1,
    render_mode: str | None = None,
    use_action_mask: bool = True,
    reward_kwargs: dict | None = None,
) -> gym.Env:
    from pvz_rl.sim.env import PvZSimEnv
    from pvz_rl.agents.wrappers import FlattenDictObs, ActionMaskWrapper
    env = PvZSimEnv(difficulty=difficulty, render_mode=render_mode, reward_shaping=False)
    kwargs = reward_kwargs or {}
    config = RewardConfig(scheme=RewardScheme(scheme), **kwargs)
    env = RewardShaperWrapper(env, config)
    env = FlattenDictObs(env)
    if use_action_mask:
        env = ActionMaskWrapper(env)
    return env
