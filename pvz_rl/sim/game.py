"""Core PvZ game engine -- pure logic, no rendering dependency."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

from .constants import (
    COLS,
    PLANT_STATS,
    ROWS,
    SUN_AMOUNT,
    SUN_INTERVAL,
    TICKS_PER_SECOND,
    ZOMBIE_STATS,
    PlantType,
    ZombieType,
    build_wave_schedule,
)


# ---------------------------------------------------------------------------
# Entity dataclasses
# ---------------------------------------------------------------------------

@dataclass
class Plant:
    ptype: PlantType
    row: int
    col: int
    hp: int = 0
    cooldown_left: int = 0  # remaining cooldown before this slot can be replanted
    fire_timer: int = 0
    sun_timer: int = 0
    armed: bool = True  # for potato mine
    arm_timer: int = 0

    def __post_init__(self):
        stats = PLANT_STATS[self.ptype]
        if self.hp == 0:
            self.hp = stats["hp"]
        if self.ptype == PlantType.POTATOMINE:
            self.armed = False
            self.arm_timer = stats["arm_time"]


@dataclass
class Zombie:
    ztype: ZombieType
    row: int
    x: float  # continuous x position in *cells* (starts at COLS, reaches 0 = loss)
    hp: int = 0
    speed: float = 0.0
    base_speed: float = 0.0
    damage: int = 0
    bite_rate: int = 15
    bite_timer: int = 0
    slowed_ticks: int = 0
    has_vaulted: bool = False

    def __post_init__(self):
        stats = ZOMBIE_STATS[self.ztype]
        if self.hp == 0:
            self.hp = stats["hp"]
        if self.speed == 0.0:
            self.speed = stats["speed"]
        self.base_speed = stats["speed"]
        self.damage = stats["damage"]
        self.bite_rate = stats["bite_rate"]


@dataclass
class Projectile:
    row: int
    x: float  # cell units
    speed: float = 8.0  # cells/sec
    damage: int = 20
    is_frozen: bool = False


@dataclass
class SunDrop:
    x: float
    y: float  # pixel coords
    amount: int = SUN_AMOUNT
    lifetime: int = 300  # ticks before it disappears


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------

@dataclass
class GameState:
    """Complete mutable game state."""

    tick: int = 0
    sun: int = 1000
    plants: list[Plant] = field(default_factory=list)
    zombies: list[Zombie] = field(default_factory=list)
    projectiles: list[Projectile] = field(default_factory=list)
    sun_drops: list[SunDrop] = field(default_factory=list)
    wave_schedule: list[dict] = field(default_factory=list)
    next_wave_idx: int = 0
    sun_timer: int = SUN_INTERVAL
    game_over: bool = False
    victory: bool = False
    plant_cooldowns: dict[PlantType, int] = field(default_factory=dict)
    # Grid helper: (row, col) -> Plant reference
    grid: dict[tuple[int, int], Plant] = field(default_factory=dict)
    score: int = 0
    zombies_killed: int = 0
    plants_lost: int = 0
    difficulty: int = 1

    # Stats for reward shaping
    total_sun_collected: int = 0
    total_sun_spent: int = 0
    damage_dealt: int = 0
    projectile_hits: int = 0
    wasted_placements: int = 0


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class PvZEngine:
    """Tick-based game engine.  Call ``reset()`` then ``step()`` each tick."""

    def __init__(self, difficulty: int = 1, seed: int | None = None):
        self.difficulty = difficulty
        self.rng = random.Random(seed)
        self.state = GameState()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.rng = random.Random(seed)
        self.state = GameState(
            wave_schedule=build_wave_schedule(self.difficulty),
            difficulty=self.difficulty,
        )
        # init cooldowns to zero (ready)
        for pt in PlantType:
            if pt != PlantType.NONE:
                self.state.plant_cooldowns[pt] = 0
        return self.state

    def step(self) -> GameState:
        """Advance game by one tick. Returns updated state."""
        s = self.state
        if s.game_over or s.victory:
            return s
        s.tick += 1

        self._tick_sun(s)
        self._spawn_waves(s)
        self._tick_plants(s)
        self._tick_projectiles(s)
        self._tick_zombies(s)
        self._tick_cooldowns(s)
        self._check_victory(s)
        return s

    def try_plant(self, ptype: PlantType, row: int, col: int) -> bool:
        """Attempt to place a plant. Returns True on success."""
        s = self.state
        if s.game_over or s.victory:
            return False
        if not (0 <= row < ROWS and 0 <= col < COLS):
            return False
        if (row, col) in s.grid:
            return False
        stats = PLANT_STATS[ptype]
        if s.sun < stats["cost"]:
            return False
        if s.plant_cooldowns.get(ptype, 0) > 0:
            return False

        # Place it
        s.sun -= stats["cost"]
        s.total_sun_spent += stats["cost"]
        plant = Plant(ptype=ptype, row=row, col=col)
        s.plants.append(plant)
        s.grid[(row, col)] = plant
        s.plant_cooldowns[ptype] = stats["cooldown"]

        # Instant-use plants (cherry bomb)
        if stats.get("instant"):
            self._detonate_plant(s, plant)

        # Check for wasted placements (placing shooters in empty rows)
        if ptype in (PlantType.PEASHOOTER, PlantType.SNOWPEA, PlantType.REPEATER):
            zombies_in_row = any(z.row == row and z.hp > 0 for z in s.zombies)
            if not zombies_in_row:
                s.wasted_placements += 1

        return True

    def do_nothing(self):
        """Explicit no-op action."""
        pass

    # ------------------------------------------------------------------
    # Internal tick helpers
    # ------------------------------------------------------------------

    def _tick_sun(self, s: GameState):
        s.sun_timer -= 1
        if s.sun_timer <= 0:
            s.sun_timer = SUN_INTERVAL
            s.sun += SUN_AMOUNT
            s.total_sun_collected += SUN_AMOUNT

        # Remove expired sun drops
        remaining = []
        for sd in s.sun_drops:
            sd.lifetime -= 1
            if sd.lifetime > 0:
                remaining.append(sd)
        s.sun_drops = remaining

    def _spawn_waves(self, s: GameState):
        while s.next_wave_idx < len(s.wave_schedule):
            wave = s.wave_schedule[s.next_wave_idx]
            if s.tick >= wave["tick"]:
                for ztype, row in wave["zombies"]:
                    x = float(COLS) + self.rng.uniform(0, 1.5)
                    z = Zombie(ztype=ztype, row=row, x=x)
                    s.zombies.append(z)
                s.next_wave_idx += 1
            else:
                break

    def _tick_plants(self, s: GameState):
        for plant in list(s.plants):
            if plant.hp <= 0:
                continue
            ptype = plant.ptype
            stats = PLANT_STATS[ptype]

            # Sunflower sun production
            if ptype == PlantType.SUNFLOWER:
                plant.sun_timer += 1
                if plant.sun_timer >= stats["sun_interval"]:
                    plant.sun_timer = 0
                    s.sun += SUN_AMOUNT
                    s.total_sun_collected += SUN_AMOUNT

            # Potato mine arming
            if ptype == PlantType.POTATOMINE and not plant.armed:
                plant.arm_timer -= 1
                if plant.arm_timer <= 0:
                    plant.armed = True

            # Shooting plants
            fire_rate = stats.get("fire_rate", 0)
            if fire_rate > 0:
                # Only fire if there's a zombie in this row ahead of the plant
                has_target = any(
                    z.row == plant.row and z.x >= plant.col and z.hp > 0
                    for z in s.zombies
                )
                if has_target:
                    plant.fire_timer += 1
                    if plant.fire_timer >= fire_rate:
                        plant.fire_timer = 0
                        n_shots = stats.get("shots_per_volley", 1)
                        for i in range(n_shots):
                            proj = Projectile(
                                row=plant.row,
                                x=float(plant.col) + 0.8,
                                damage=stats["damage"],
                                is_frozen=stats.get("slow", False),
                            )
                            s.projectiles.append(proj)

            # Potato mine proximity detonation
            if ptype == PlantType.POTATOMINE and plant.armed:
                for z in s.zombies:
                    if z.row == plant.row and z.hp > 0 and abs(z.x - plant.col) < 0.8:
                        self._detonate_plant(s, plant)
                        break

    def _detonate_plant(self, s: GameState, plant: Plant):
        """Handle instant-kill plants (cherry bomb, potato mine)."""
        stats = PLANT_STATS[plant.ptype]
        radius = stats.get("aoe_radius", 0)
        for z in s.zombies:
            if plant.ptype == PlantType.CHERRYBOMB:
                if abs(z.row - plant.row) <= radius and abs(z.x - plant.col) <= radius + 0.5:
                    dmg = min(stats["damage"], z.hp)
                    z.hp -= stats["damage"]
                    s.damage_dealt += dmg
            elif plant.ptype == PlantType.POTATOMINE:
                if z.row == plant.row and abs(z.x - plant.col) < 1.0:
                    dmg = min(stats["damage"], z.hp)
                    z.hp -= stats["damage"]
                    s.damage_dealt += dmg
        # Plant is consumed
        self._remove_plant(s, plant)

    def _tick_projectiles(self, s: GameState):
        dt = 1.0 / TICKS_PER_SECOND
        remaining = []
        for p in s.projectiles:
            p.x += p.speed * dt
            if p.x > COLS + 1:
                continue  # off-screen
            # Check collision with zombies
            hit = False
            for z in s.zombies:
                if z.row == p.row and z.hp > 0 and abs(z.x - p.x) < 0.5:
                    dmg = min(p.damage, z.hp)
                    z.hp -= p.damage
                    s.damage_dealt += dmg
                    s.projectile_hits += 1
                    if p.is_frozen:
                        z.slowed_ticks = 300  # ~10 s slow
                    hit = True
                    break
            if not hit:
                remaining.append(p)
        s.projectiles = remaining

        # Clean up dead zombies from projectile hits
        self._reap_zombies(s)

    def _tick_zombies(self, s: GameState):
        dt = 1.0 / TICKS_PER_SECOND
        for z in s.zombies:
            if z.hp <= 0:
                continue

            # Apply slow
            if z.slowed_ticks > 0:
                z.slowed_ticks -= 1
                z.speed = z.base_speed * 0.5
            else:
                z.speed = z.base_speed

            # Check if zombie is at a plant
            col_at = int(z.x)
            plant_here = s.grid.get((z.row, col_at))

            # Pole vaulter logic
            if (
                z.ztype == ZombieType.POLE_VAULTER
                and not z.has_vaulted
                and plant_here
                and plant_here.hp > 0
            ):
                # Vault over the plant
                z.x = plant_here.col - 0.5
                z.has_vaulted = True
                z.speed = z.base_speed * 0.5  # slows down after vaulting
                continue

            if plant_here and plant_here.hp > 0 and abs(z.x - col_at) < 0.3:
                # Eating the plant
                z.bite_timer += 1
                if z.bite_timer >= z.bite_rate:
                    z.bite_timer = 0
                    plant_here.hp -= z.damage
                    if plant_here.hp <= 0:
                        self._remove_plant(s, plant_here)
                        s.plants_lost += 1
            else:
                # Move forward
                z.x -= z.speed * dt

            # Check if zombie reached the house
            if z.x <= -0.5:
                s.game_over = True
                return

        self._reap_zombies(s)

    def _tick_cooldowns(self, s: GameState):
        for pt in s.plant_cooldowns:
            if s.plant_cooldowns[pt] > 0:
                s.plant_cooldowns[pt] -= 1

    def _check_victory(self, s: GameState):
        if s.next_wave_idx >= len(s.wave_schedule) and len(s.zombies) == 0:
            s.victory = True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _remove_plant(self, s: GameState, plant: Plant):
        plant.hp = 0
        key = (plant.row, plant.col)
        if key in s.grid and s.grid[key] is plant:
            del s.grid[key]
        if plant in s.plants:
            s.plants.remove(plant)

    def _reap_zombies(self, s: GameState):
        alive = []
        for z in s.zombies:
            if z.hp <= 0:
                s.zombies_killed += 1
                s.score += 10
            else:
                alive.append(z)
        s.zombies = alive
