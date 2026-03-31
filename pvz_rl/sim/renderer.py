"""Pygame renderer for the PvZ simulator.

This is purely a visualisation layer -- all game logic lives in ``game.py``.
When ``headless=True`` the renderer produces RGB numpy arrays without opening a
window (useful for CNN-based agents).
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pygame
    import pygame.freetype

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

from .constants import (
    CELL_H,
    CELL_W,
    COLS,
    COLOR_BG,
    COLOR_GRID,
    COLOR_PEA,
    COLOR_SNOW_PEA_PROJ,
    COLOR_SUN,
    GRID_ORIGIN_X,
    GRID_ORIGIN_Y,
    PLANT_COLORS,
    ROWS,
    SCREEN_H,
    SCREEN_W,
    ZOMBIE_COLORS,
    PlantType,
    ZombieType,
)
from .game import GameState


class PvZRenderer:
    """Draws the game state with Pygame (or off-screen for headless mode)."""

    def __init__(self, headless: bool = False):
        if not _PYGAME_AVAILABLE:
            raise ImportError("pygame is required for rendering. pip install pygame")

        self.headless = headless
        self._initialised = False
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.font: Optional[pygame.freetype.Font] = None

    def _lazy_init(self):
        if self._initialised:
            return
        if self.headless:
            pygame.init()
            self.screen = pygame.Surface((SCREEN_W, SCREEN_H))
        else:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("PvZ-RL Simulator")
            self.clock = pygame.time.Clock()
        self.font = pygame.freetype.SysFont("Consolas", 14)
        self._initialised = True

    def render(self, state: GameState, fps: int = 30) -> np.ndarray:
        """Draw current state and return an RGB numpy array (H, W, 3)."""
        self._lazy_init()
        assert self.screen is not None

        self.screen.fill(COLOR_BG)
        self._draw_grid()
        self._draw_plants(state)
        self._draw_zombies(state)
        self._draw_projectiles(state)
        self._draw_hud(state)

        if not self.headless:
            pygame.display.flip()
            if self.clock:
                self.clock.tick(fps)

        # Return pixel array
        arr = pygame.surfarray.array3d(self.screen)  # (W, H, 3)
        arr = np.transpose(arr, (1, 0, 2))  # -> (H, W, 3)
        return arr

    def close(self):
        if self._initialised:
            pygame.quit()
            self._initialised = False

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _cell_rect(self, row: int, col: int) -> pygame.Rect:
        x = GRID_ORIGIN_X + col * CELL_W
        y = GRID_ORIGIN_Y + row * CELL_H
        return pygame.Rect(x, y, CELL_W, CELL_H)

    def _draw_grid(self):
        for r in range(ROWS):
            for c in range(COLS):
                rect = self._cell_rect(r, c)
                pygame.draw.rect(self.screen, COLOR_GRID, rect, 1)

    def _draw_plants(self, state: GameState):
        for plant in state.plants:
            if plant.hp <= 0:
                continue
            rect = self._cell_rect(plant.row, plant.col)
            color = PLANT_COLORS.get(plant.ptype, (200, 200, 200))

            # Draw a filled circle for the plant
            centre = rect.center
            radius = min(CELL_W, CELL_H) // 3
            pygame.draw.circle(self.screen, color, centre, radius)

            # Health bar
            hp_frac = plant.hp / max(1, self._max_hp(plant.ptype))
            bar_w = int(CELL_W * 0.8 * hp_frac)
            bar_rect = pygame.Rect(rect.x + 4, rect.bottom - 10, bar_w, 5)
            pygame.draw.rect(self.screen, (0, 255, 0), bar_rect)

            # Label
            if self.font:
                self.font.render_to(
                    self.screen,
                    (rect.x + 2, rect.y + 2),
                    plant.ptype.name[:3],
                    (255, 255, 255),
                    size=10,
                )

    def _draw_zombies(self, state: GameState):
        for z in state.zombies:
            if z.hp <= 0:
                continue
            px = GRID_ORIGIN_X + z.x * CELL_W
            py = GRID_ORIGIN_Y + z.row * CELL_H + CELL_H // 2
            color = ZOMBIE_COLORS.get(z.ztype, (150, 150, 150))
            pygame.draw.circle(self.screen, color, (int(px), int(py)), 14)
            # Small red HP text
            if self.font:
                self.font.render_to(
                    self.screen,
                    (int(px) - 10, int(py) - 22),
                    str(z.hp),
                    (255, 80, 80),
                    size=9,
                )

    def _draw_projectiles(self, state: GameState):
        for p in state.projectiles:
            px = GRID_ORIGIN_X + p.x * CELL_W
            py = GRID_ORIGIN_Y + p.row * CELL_H + CELL_H // 2
            color = COLOR_SNOW_PEA_PROJ if p.is_frozen else COLOR_PEA
            pygame.draw.circle(self.screen, color, (int(px), int(py)), 5)

    def _draw_hud(self, state: GameState):
        if not self.font:
            return
        y = 8
        self.font.render_to(self.screen, (10, y), f"Sun: {state.sun}", COLOR_SUN, size=16)
        self.font.render_to(
            self.screen,
            (150, y),
            f"Score: {state.score}  Killed: {state.zombies_killed}  "
            f"Wave: {state.next_wave_idx}/{len(state.wave_schedule)}",
            (255, 255, 255),
            size=14,
        )
        self.font.render_to(
            self.screen,
            (10, y + 22),
            f"Tick: {state.tick}  Zombies alive: {len(state.zombies)}",
            (200, 200, 200),
            size=12,
        )

        if state.game_over:
            self.font.render_to(
                self.screen, (SCREEN_W // 2 - 80, SCREEN_H // 2), "GAME OVER", (255, 0, 0), size=32
            )
        elif state.victory:
            self.font.render_to(
                self.screen, (SCREEN_W // 2 - 60, SCREEN_H // 2), "VICTORY!", (0, 255, 0), size=32
            )

    @staticmethod
    def _max_hp(ptype: PlantType) -> int:
        from .constants import PLANT_STATS

        return PLANT_STATS[ptype]["hp"]
