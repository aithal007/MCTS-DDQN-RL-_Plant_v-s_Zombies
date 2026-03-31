"""Screen capture and input automation for the OG Plants vs Zombies on Steam.

This module finds the PvZ game window, captures frames via ``mss``, and sends
mouse/keyboard inputs via ``pyautogui``.  It provides a low-level interface
that the Gymnasium env wrapper (``env.py``) builds on top of.

Requirements (install with ``pip install pvz-rl[steam]``):
    mss, pyautogui, pywin32, opencv-python, pillow
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import cv2
    import mss
    import pyautogui
    import win32gui
    import win32con

    _STEAM_DEPS = True
except ImportError:
    _STEAM_DEPS = False


# Typical PvZ GOTY window titles
_WINDOW_TITLES = [
    "Plants vs. Zombies",
    "Plants vs Zombies",
    "PlantsVsZombies",
]

# Grid geometry in the *game window* (approximate for 800x600 resolution)
# These may need calibration per monitor / resolution -- see calibrate().
@dataclass
class GameGeometry:
    """Pixel coordinates within the game window."""

    # Game area bounding box relative to window top-left
    game_x: int = 0
    game_y: int = 0
    game_w: int = 800
    game_h: int = 600

    # Lawn grid
    grid_left: int = 43
    grid_top: int = 109
    cell_w: int = 80
    cell_h: int = 96
    rows: int = 5
    cols: int = 9

    # Seed tray (plant selection bar at top)
    tray_left: int = 73
    tray_top: int = 8
    tray_slot_w: int = 53
    tray_slot_h: int = 70
    tray_slots: int = 7  # number of seed slots visible

    # Sun counter region (for OCR if needed)
    sun_region: tuple[int, int, int, int] = (23, 64, 63, 18)  # x, y, w, h

    # Shovel
    shovel_x: int = 458
    shovel_y: int = 8


def _find_window() -> int | None:
    """Find the PvZ window handle."""
    if not _STEAM_DEPS:
        return None
    for title in _WINDOW_TITLES:
        hwnd = win32gui.FindWindow(None, title)
        if hwnd:
            return hwnd
    return None


class SteamCapture:
    """Captures frames and sends inputs to the Steam PvZ window."""

    def __init__(self, geometry: GameGeometry | None = None):
        if not _STEAM_DEPS:
            raise ImportError(
                "Steam capture requires extra dependencies. "
                "Install with: pip install pvz-rl[steam]"
            )
        self.geo = geometry or GameGeometry()
        self.hwnd: int | None = None
        self.sct = mss.mss()
        self._monitor: dict | None = None

        # Disable pyautogui safety pause for speed
        pyautogui.PAUSE = 0.02
        pyautogui.FAILSAFE = True  # keep failsafe (move to corner to abort)

    def connect(self) -> bool:
        """Find and focus the PvZ window. Returns True on success."""
        self.hwnd = _find_window()
        if self.hwnd is None:
            return False
        # Bring window to foreground
        try:
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception:
            pass
        time.sleep(0.3)
        self._update_monitor()
        return True

    def capture_frame(self) -> np.ndarray | None:
        """Capture the game window as an RGB numpy array (H, W, 3)."""
        if self._monitor is None:
            if not self.connect():
                return None
        try:
            shot = self.sct.grab(self._monitor)
            frame = np.array(shot, dtype=np.uint8)[:, :, :3]  # drop alpha
            # mss gives BGR, convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        except Exception:
            self._monitor = None
            return None

    def click_cell(self, row: int, col: int):
        """Click the centre of a lawn grid cell."""
        if self.hwnd is None:
            return
        wx, wy = self._window_origin()
        cx = wx + self.geo.grid_left + col * self.geo.cell_w + self.geo.cell_w // 2
        cy = wy + self.geo.grid_top + row * self.geo.cell_h + self.geo.cell_h // 2
        pyautogui.click(cx, cy)

    def click_seed_slot(self, slot: int):
        """Click a seed tray slot (0-indexed)."""
        if self.hwnd is None:
            return
        wx, wy = self._window_origin()
        cx = wx + self.geo.tray_left + slot * self.geo.tray_slot_w + self.geo.tray_slot_w // 2
        cy = wy + self.geo.tray_top + self.geo.tray_slot_h // 2
        pyautogui.click(cx, cy)

    def click_shovel(self):
        """Click the shovel."""
        if self.hwnd is None:
            return
        wx, wy = self._window_origin()
        pyautogui.click(wx + self.geo.shovel_x, wy + self.geo.shovel_y)

    def click_position(self, x: int, y: int):
        """Click an arbitrary position relative to the game window."""
        if self.hwnd is None:
            return
        wx, wy = self._window_origin()
        pyautogui.click(wx + x, wy + y)

    def collect_sun_at(self, x: int, y: int):
        """Click a sun drop at the given window-relative position."""
        self.click_position(x, y)

    def send_key(self, key: str):
        """Send a keyboard key press."""
        pyautogui.press(key)

    # ------------------------------------------------------------------
    # Vision helpers
    # ------------------------------------------------------------------

    def get_lawn_region(self, frame: np.ndarray) -> np.ndarray:
        """Crop just the lawn grid from a full frame."""
        g = self.geo
        return frame[
            g.grid_top : g.grid_top + g.rows * g.cell_h,
            g.grid_left : g.grid_left + g.cols * g.cell_w,
        ]

    def get_cell_image(self, frame: np.ndarray, row: int, col: int) -> np.ndarray:
        """Extract a single cell's image."""
        g = self.geo
        y = g.grid_top + row * g.cell_h
        x = g.grid_left + col * g.cell_w
        return frame[y : y + g.cell_h, x : x + g.cell_w]

    def detect_sun_positions(self, frame: np.ndarray) -> list[tuple[int, int]]:
        """Detect falling/stationary sun drops using colour thresholding.

        Returns a list of (x, y) positions relative to the game window.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        # Sun is bright yellow
        lower = np.array([20, 150, 200])
        upper = np.array([35, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 200 < area < 5000:  # filter noise
                M = cv2.moments(cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    positions.append((cx, cy))
        return positions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _window_origin(self) -> tuple[int, int]:
        if self.hwnd is None:
            return (0, 0)
        rect = win32gui.GetWindowRect(self.hwnd)
        # Account for window border/title bar
        client = win32gui.GetClientRect(self.hwnd)
        border_x = ((rect[2] - rect[0]) - client[2]) // 2
        title_h = (rect[3] - rect[1]) - client[3] - border_x
        return rect[0] + border_x, rect[1] + title_h

    def _update_monitor(self):
        if self.hwnd is None:
            self._monitor = None
            return
        wx, wy = self._window_origin()
        self._monitor = {
            "left": wx,
            "top": wy,
            "width": self.geo.game_w,
            "height": self.geo.game_h,
        }
