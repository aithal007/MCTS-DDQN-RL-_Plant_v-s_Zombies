"""Gymnasium environment for the original PvZ on Steam.

This wraps the live game via screen capture and mouse automation.  The agent
sees raw pixel observations and acts by selecting seed slots and clicking cells.

Observation: RGB image (resized to 160x120 by default) -- suitable for CNN
             policies from stable-baselines3.

Action space:
    0                     : do nothing (wait one step)
    1..tray_slots         : click seed slot i (selects a plant)
    tray_slots+1 .. end   : click cell (row, col) on the lawn

The agent workflow each step:
    1. (optional) select a seed slot to pick a plant
    2. click a lawn cell to place it  (or click sun to collect it)
    The env auto-collects visible sun each step.

NOTE: This environment is inherently slower than the simulator because it
operates in real-time and depends on screen capture latency.
"""

from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    import cv2

    _CV2 = True
except ImportError:
    _CV2 = False

from .capture import GameGeometry, SteamCapture


_OBS_W = 160
_OBS_H = 120
_DEFAULT_ROWS = 5
_DEFAULT_COLS = 9
_DEFAULT_TRAY_SLOTS = 7


class PvZSteamEnv(gym.Env):
    """Gymnasium env that controls the real PvZ game on Steam."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        render_mode: Optional[str] = "rgb_array",
        obs_width: int = _OBS_W,
        obs_height: int = _OBS_H,
        tray_slots: int = _DEFAULT_TRAY_SLOTS,
        rows: int = _DEFAULT_ROWS,
        cols: int = _DEFAULT_COLS,
        step_delay: float = 0.15,
        auto_collect_sun: bool = True,
        geometry: GameGeometry | None = None,
    ):
        super().__init__()
        if not _CV2:
            raise ImportError("opencv-python is required for the Steam env.")

        self.render_mode = render_mode
        self.obs_w = obs_width
        self.obs_h = obs_height
        self.tray_slots = tray_slots
        self.rows = rows
        self.cols = cols
        self.step_delay = step_delay
        self.auto_collect_sun = auto_collect_sun

        self.capture = SteamCapture(geometry=geometry)
        self._connected = False
        self._step_count = 0
        self._last_frame: np.ndarray | None = None

        # Action: 0=noop, 1..tray_slots=select seed, rest=click cell
        n_cell_actions = rows * cols
        self.n_seed_actions = tray_slots
        self.action_space = spaces.Discrete(1 + tray_slots + n_cell_actions)

        # Observation: downscaled RGB image
        self.observation_space = spaces.Box(
            0, 255, shape=(obs_height, obs_width, 3), dtype=np.uint8
        )

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if not self._connected:
            ok = self.capture.connect()
            if not ok:
                raise RuntimeError(
                    "Could not find the PvZ game window. Make sure the game is running."
                )
            self._connected = True

        # We can't programmatically restart a level easily, so we just
        # capture the current state. The user should start/restart the level
        # manually (or a separate script can automate menu navigation).
        time.sleep(0.5)
        self._step_count = 0
        obs = self._get_obs()
        return obs, {"step": self._step_count}

    def step(self, action: int):
        self._apply_action(action)
        time.sleep(self.step_delay)

        # Auto-collect sun
        if self.auto_collect_sun:
            self._collect_sun()

        self._step_count += 1
        obs = self._get_obs()

        # We can't easily detect game-over from pixels alone in a robust way,
        # so we provide a basic heuristic and rely on the user/training script
        # to handle episode boundaries.
        terminated = False
        truncated = False
        reward = 0.0  # Reward must come from an external signal or vision model

        info = {
            "step": self._step_count,
            "frame_shape": self._last_frame.shape if self._last_frame is not None else None,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self._last_frame is None:
            return None
        if self.render_mode == "rgb_array":
            return self._last_frame.copy()
        return None

    def close(self):
        pass

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _apply_action(self, action: int):
        if action == 0:
            return  # noop

        if 1 <= action <= self.tray_slots:
            slot = action - 1
            self.capture.click_seed_slot(slot)
            return

        cell_action = action - 1 - self.tray_slots
        row = cell_action // self.cols
        col = cell_action % self.cols
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.capture.click_cell(row, col)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        frame = self.capture.capture_frame()
        if frame is None:
            # Return blank frame if capture fails
            return np.zeros((self.obs_h, self.obs_w, 3), dtype=np.uint8)
        self._last_frame = frame
        resized = cv2.resize(frame, (self.obs_w, self.obs_h), interpolation=cv2.INTER_AREA)
        return resized

    # ------------------------------------------------------------------
    # Sun collection
    # ------------------------------------------------------------------

    def _collect_sun(self):
        if self._last_frame is None:
            return
        positions = self.capture.detect_sun_positions(self._last_frame)
        for x, y in positions[:5]:  # click up to 5 suns per step
            self.capture.collect_sun_at(x, y)
            time.sleep(0.02)
