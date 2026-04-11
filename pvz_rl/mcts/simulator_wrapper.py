"""Simulator wrapper for MCTS tree search.

Allows MCTS to simulate forward from any game state without
affecting the real environment. Uses deep-copy of PvZEngine state.

Key operations:
    - clone_state(): Deep copy the current engine for branching
    - simulate_action(): Apply action and tick forward
    - get_observation(): Extract observation vector from state
    - is_terminal(): Check if game ended
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np

from pvz_rl.sim.constants import (
    COLS,
    NUM_PLANT_TYPES,
    PLANT_STATS,
    ROWS,
    PlantType,
)
from pvz_rl.sim.game import PvZEngine, GameState

# Constants for observation normalization (matching env.py)
_MAX_SUN = 2000.0
_MAX_HP = 4000.0
_MAX_ZOMBIE_HP = 1500.0
_MAX_COOLDOWN = 1500.0
_PLANT_TYPES = [pt for pt in PlantType if pt != PlantType.NONE]


class SimulatorWrapper:
    """Wrapper around PvZEngine for MCTS simulation.

    Maintains an internal engine clone and provides methods
    to fork/simulate states without affecting the real environment.
    """

    def __init__(self, difficulty: int = 1):
        self.difficulty = difficulty
        self.action_dim = 1 + NUM_PLANT_TYPES * ROWS * COLS  # 316

    @staticmethod
    def clone_engine(engine: PvZEngine) -> PvZEngine:
        """Create a deep copy of a PvZEngine for tree search.

        This is the core operation that enables MCTS: we can branch
        the game state at any point and simulate different futures.
        """
        return copy.deepcopy(engine)

    @staticmethod
    def simulate_action(engine: PvZEngine, action: int) -> tuple[GameState, float]:
        """Apply an action to an engine and advance one tick.

        Args:
            engine: A (cloned) PvZEngine instance
            action: Discrete action index (0=noop, 1..315=place plant)

        Returns:
            state: Updated GameState
            reward: Raw step reward (kills * 10 from score delta)
        """
        prev_score = engine.state.score
        prev_killed = engine.state.zombies_killed

        # Decode and apply action
        if action == 0:
            engine.do_nothing()
        else:
            a = action - 1
            plant_idx = a // (ROWS * COLS)
            remainder = a % (ROWS * COLS)
            row = remainder // COLS
            col = remainder % COLS

            if 0 <= plant_idx < NUM_PLANT_TYPES:
                ptype = _PLANT_TYPES[plant_idx]
                engine.try_plant(ptype, row, col)

        # Advance simulation
        engine.step()

        # Simple reward: score delta
        reward = float(engine.state.score - prev_score)

        return engine.state, reward

    @staticmethod
    def get_observation(state: GameState) -> np.ndarray:
        """Extract flattened observation vector from a GameState.

        Matches the observation format of PvZSimEnv + FlattenDictObs wrapper.
        The keys are sorted alphabetically: cooldowns, grid, grid_hp, sun, wave_progress, zombie_map
        """
        # Cooldowns (7,)
        cooldowns = np.zeros(NUM_PLANT_TYPES, dtype=np.float32)
        for i, pt in enumerate(_PLANT_TYPES):
            cd = state.plant_cooldowns.get(pt, 0)
            cooldowns[i] = min(cd / _MAX_COOLDOWN, 1.0)

        # Grid (5, 9) -> flat
        grid = np.zeros((ROWS, COLS), dtype=np.float32)
        for plant in state.plants:
            if plant.hp > 0:
                grid[plant.row, plant.col] = float(int(plant.ptype))

        # Grid HP (5, 9) -> flat
        grid_hp = np.zeros((ROWS, COLS), dtype=np.float32)
        for plant in state.plants:
            if plant.hp > 0:
                max_hp = PLANT_STATS[plant.ptype]["hp"]
                grid_hp[plant.row, plant.col] = plant.hp / max(max_hp, 1)

        # Sun (1,)
        sun = np.array([min(state.sun / _MAX_SUN, 1.0)], dtype=np.float32)

        # Wave progress (1,)
        total_waves = max(len(state.wave_schedule), 1)
        wp = np.array([state.next_wave_idx / total_waves], dtype=np.float32)

        # Zombie map (5, 9) -> flat
        zombie_map = np.zeros((ROWS, COLS), dtype=np.float32)
        for z in state.zombies:
            if z.hp > 0:
                c = max(0, min(COLS - 1, int(z.x)))
                zombie_map[z.row, c] += z.hp / _MAX_ZOMBIE_HP

        # Concatenate in sorted key order: cooldowns, grid, grid_hp, sun, wave_progress, zombie_map
        parts = [
            cooldowns.flatten(),
            grid.flatten(),
            grid_hp.flatten(),
            sun.flatten(),
            wp.flatten(),
            zombie_map.flatten(),
        ]
        return np.concatenate(parts)

    @staticmethod
    def get_action_mask(state: GameState) -> np.ndarray:
        """Compute valid action mask from a GameState.

        Returns:
            Boolean array of shape (action_dim,) where True = valid action.
        """
        action_dim = 1 + NUM_PLANT_TYPES * ROWS * COLS
        mask = np.zeros(action_dim, dtype=bool)
        mask[0] = True  # noop always valid

        for i, pt in enumerate(_PLANT_TYPES):
            stats = PLANT_STATS[pt]
            can_afford = state.sun >= stats["cost"]
            off_cooldown = state.plant_cooldowns.get(pt, 0) <= 0
            if can_afford and off_cooldown:
                for r in range(ROWS):
                    for c in range(COLS):
                        if (r, c) not in state.grid:
                            idx = 1 + i * (ROWS * COLS) + r * COLS + c
                            mask[idx] = True
        return mask

    @staticmethod
    def is_terminal(state: GameState) -> bool:
        """Check if the game has ended."""
        return state.game_over or state.victory

    @staticmethod
    def get_value(state: GameState) -> float:
        """Get terminal value for MCTS backup.

        Returns:
            +1.0 for victory, -1.0 for game over, 0.0 for non-terminal.
        """
        if state.victory:
            return 1.0
        elif state.game_over:
            return -1.0
        return 0.0
