"""Gymnasium environment wrapping the PvZ Pygame simulator.

Observation space (flat dict):
    - grid         : (ROWS, COLS) int8 -- plant type id at each cell (0 = empty)
    - grid_hp      : (ROWS, COLS) float32 -- normalised plant HP [0,1]
    - zombie_map   : (ROWS, COLS) float32 -- total normalised zombie HP in each cell
    - sun          : float32 scalar -- normalised sun count [0,1]
    - cooldowns    : (NUM_PLANT_TYPES,) float32 -- normalised cooldown per plant type
    - wave_progress: float32 scalar -- fraction of waves spawned

Action space (Discrete):
    0              : do nothing
    1..(R*C*P)     : place plant P at (row, col)

    Encoded as:  action = 1 + plant_idx * (ROWS * COLS) + row * COLS + col
    Total size = 1 + NUM_PLANT_TYPES * ROWS * COLS
"""

from __future__ import annotations

from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .constants import (
    COLS,
    FPS,
    NUM_PLANT_TYPES,
    PLANT_STATS,
    ROWS,
    PlantType,
)
from .game import PvZEngine


# Max values for normalisation
_MAX_SUN = 2000.0
_MAX_HP = 4000.0
_MAX_ZOMBIE_HP = 1500.0
_MAX_COOLDOWN = 1500.0

_PLANT_TYPES = [pt for pt in PlantType if pt != PlantType.NONE]


class PvZSimEnv(gym.Env):
    """Gymnasium environment for the built-in PvZ simulator."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        difficulty: int = 1,
        render_mode: Optional[str] = None,
        reward_shaping: bool = True,
        max_ticks: int = 10_000,
    ):
        super().__init__()
        self.difficulty = difficulty
        self.render_mode = render_mode
        self.reward_shaping = reward_shaping
        self.max_ticks = max_ticks

        self.engine = PvZEngine(difficulty=difficulty)
        self.renderer = None

        # --- Action space ---
        # 0 = noop, then plant_idx * ROWS * COLS + row * COLS + col + 1
        self.n_placement_actions = NUM_PLANT_TYPES * ROWS * COLS
        self.action_space = spaces.Discrete(1 + self.n_placement_actions)

        # --- Observation space ---
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(0, len(_PLANT_TYPES), shape=(ROWS, COLS), dtype=np.int8),
                "grid_hp": spaces.Box(0.0, 1.0, shape=(ROWS, COLS), dtype=np.float32),
                "zombie_map": spaces.Box(0.0, 10.0, shape=(ROWS, COLS), dtype=np.float32),
                "sun": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "cooldowns": spaces.Box(
                    0.0, 1.0, shape=(NUM_PLANT_TYPES,), dtype=np.float32
                ),
                "wave_progress": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

        # Track previous state for reward shaping
        self._prev_zombies_killed = 0
        self._prev_plants_lost = 0
        self._prev_sun = 0
        self._prev_damage = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options:
            self.difficulty = options.get("difficulty", self.difficulty)
            self.engine.difficulty = self.difficulty
        self.engine.reset(seed=seed)
        self._prev_zombies_killed = 0
        self._prev_plants_lost = 0
        self._prev_sun = self.engine.state.sun
        self._prev_damage = 0
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        self._apply_action(action)
        self.engine.step()

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self.engine.state.game_over or self.engine.state.victory
        truncated = self.engine.state.tick >= self.max_ticks
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from .renderer import PvZRenderer

            self.renderer = PvZRenderer(headless=(self.render_mode == "rgb_array"))
        frame = self.renderer.render(self.engine.state, fps=FPS)
        if self.render_mode == "rgb_array":
            return frame
        return None  # human mode already displays

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None

    # ------------------------------------------------------------------
    # Action decoding
    # ------------------------------------------------------------------

    def _apply_action(self, action: int):
        if action == 0:
            self.engine.do_nothing()
            return
        action -= 1  # remove noop offset
        plant_idx = action // (ROWS * COLS)
        remainder = action % (ROWS * COLS)
        row = remainder // COLS
        col = remainder % COLS

        if 0 <= plant_idx < NUM_PLANT_TYPES:
            ptype = _PLANT_TYPES[plant_idx]
            self.engine.try_plant(ptype, row, col)
        # Invalid actions are silently ignored (noop)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> dict[str, np.ndarray]:
        s = self.engine.state

        grid = np.zeros((ROWS, COLS), dtype=np.int8)
        grid_hp = np.zeros((ROWS, COLS), dtype=np.float32)
        for plant in s.plants:
            if plant.hp > 0:
                grid[plant.row, plant.col] = int(plant.ptype)
                max_hp = PLANT_STATS[plant.ptype]["hp"]
                grid_hp[plant.row, plant.col] = plant.hp / max(max_hp, 1)

        zombie_map = np.zeros((ROWS, COLS), dtype=np.float32)
        for z in s.zombies:
            if z.hp > 0:
                c = max(0, min(COLS - 1, int(z.x)))
                zombie_map[z.row, c] += z.hp / _MAX_ZOMBIE_HP

        sun = np.array([min(s.sun / _MAX_SUN, 1.0)], dtype=np.float32)

        cooldowns = np.zeros(NUM_PLANT_TYPES, dtype=np.float32)
        for i, pt in enumerate(_PLANT_TYPES):
            cd = s.plant_cooldowns.get(pt, 0)
            cooldowns[i] = min(cd / _MAX_COOLDOWN, 1.0)

        total_waves = max(len(s.wave_schedule), 1)
        wp = np.array([s.next_wave_idx / total_waves], dtype=np.float32)

        return {
            "grid": grid,
            "grid_hp": grid_hp,
            "zombie_map": zombie_map,
            "sun": sun,
            "cooldowns": cooldowns,
            "wave_progress": wp,
        }

    def _get_info(self) -> dict:
        s = self.engine.state
        return {
            "tick": s.tick,
            "sun": s.sun,
            "score": s.score,
            "zombies_killed": s.zombies_killed,
            "plants_lost": s.plants_lost,
            "game_over": s.game_over,
            "victory": s.victory,
            "wave": f"{s.next_wave_idx}/{len(s.wave_schedule)}",
        }

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self) -> float:
        s = self.engine.state

        reward = 0.0

        # Terminal rewards
        if s.game_over:
            reward -= 100.0
        elif s.victory:
            reward += 200.0

        if not self.reward_shaping:
            # Update tracking vars
            self._prev_zombies_killed = s.zombies_killed
            self._prev_plants_lost = s.plants_lost
            self._prev_sun = s.sun
            self._prev_damage = s.damage_dealt
            return reward

        # Reward shaping
        # +2 per zombie killed this tick
        new_kills = s.zombies_killed - self._prev_zombies_killed
        reward += new_kills * 2.0

        # -1 per plant lost this tick
        new_losses = s.plants_lost - self._prev_plants_lost
        reward -= new_losses * 1.0

        # Small reward for damage dealt
        new_damage = s.damage_dealt - self._prev_damage
        reward += new_damage * 0.005

        # Tiny time penalty to encourage efficiency
        reward -= 0.001

        # Update tracking vars
        self._prev_zombies_killed = s.zombies_killed
        self._prev_plants_lost = s.plants_lost
        self._prev_sun = s.sun
        self._prev_damage = s.damage_dealt

        return reward

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        """Return a boolean mask of valid actions (for masked action spaces)."""
        mask = np.zeros(self.action_space.n, dtype=bool)
        mask[0] = True  # noop always valid
        s = self.engine.state
        for i, pt in enumerate(_PLANT_TYPES):
            stats = PLANT_STATS[pt]
            can_afford = s.sun >= stats["cost"]
            off_cooldown = s.plant_cooldowns.get(pt, 0) <= 0
            if can_afford and off_cooldown:
                for r in range(ROWS):
                    for c in range(COLS):
                        if (r, c) not in s.grid:
                            idx = 1 + i * (ROWS * COLS) + r * COLS + c
                            mask[idx] = True
        return mask
