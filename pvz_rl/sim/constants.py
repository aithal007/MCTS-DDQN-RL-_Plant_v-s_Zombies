"""Game constants and configuration for the PvZ simulator.

All stats sourced from PvZ Wiki / StrategyWiki:
  - Standard pea = 20 damage
  - Basic zombie = 200 HP
  - Recharge: Fast=7.5s, Slow=30s, Very Slow=50s
  - Starting sun = 100 (real PvZ adventure mode)
  - Sun drop interval = 10s, amount = 25
"""

from enum import IntEnum

# ---------------------------------------------------------------------------
# Grid layout
# ---------------------------------------------------------------------------
ROWS = 5
COLS = 9
CELL_W = 80
CELL_H = 100
GRID_ORIGIN_X = 40
GRID_ORIGIN_Y = 80
SCREEN_W = GRID_ORIGIN_X + COLS * CELL_W + 40
SCREEN_H = GRID_ORIGIN_Y + ROWS * CELL_H + 60

# ---------------------------------------------------------------------------
# Timing  (all durations in game ticks; 1 tick = 1 call to step())
# ---------------------------------------------------------------------------
FPS = 30  # target frame rate when rendering
TICKS_PER_SECOND = 30
SUN_INTERVAL = 150  # ticks between free sun drops (~5 s)
SUN_AMOUNT = 50

# ---------------------------------------------------------------------------
# Plant types
# ---------------------------------------------------------------------------

class PlantType(IntEnum):
    NONE = 0
    SUNFLOWER = 1
    PEASHOOTER = 2
    WALLNUT = 3
    SNOWPEA = 4
    CHERRYBOMB = 5
    REPEATER = 6
    POTATOMINE = 7

PLANT_NAMES = {
    PlantType.SUNFLOWER: "Sunflower",
    PlantType.PEASHOOTER: "Peashooter",
    PlantType.WALLNUT: "Wall-nut",
    PlantType.SNOWPEA: "Snow Pea",
    PlantType.CHERRYBOMB: "Cherry Bomb",
    PlantType.REPEATER: "Repeater",
    PlantType.POTATOMINE: "Potato Mine",
}

# -----------------------------------------------------------------------
# PLANT_STATS — Real PvZ values
#
#   cost        : sun cost (wiki-accurate)
#   hp          : toughness in damage points
#   cooldown    : RECHARGE time in ticks before the seed-packet can be
#                 used again after planting.
#                   Fast      =  7.5 s = 225 ticks
#                   Slow      = 30   s = 900 ticks
#                   Very Slow = 50   s = 1500 ticks
#   damage      : per pea / per explosion
#   fire_rate   : ticks between shots (Peashooter = 1.5 s = 45 ticks)
#   sun_interval: (Sunflower only) ticks between sun production
#   arm_time    : (Potato Mine only) 15 s = 450 ticks to arm
#   instant     : (Cherry Bomb) detonates on placement
#   proximity   : (Potato Mine) explodes when zombie steps on it
#   slow        : (Snow Pea) slows zombies
# -----------------------------------------------------------------------
PLANT_STATS: dict[PlantType, dict] = {
    PlantType.SUNFLOWER: {
        "cost": 50,
        "hp": 300,
        "cooldown": 60,      # (2 s)
        "sun_interval": 300,  # produces 50 sun every 10 s (buffed)
        "damage": 0,
        "fire_rate": 0,
    },
    PlantType.PEASHOOTER: {
        "cost": 100,
        "hp": 300,
        "cooldown": 60,      # (2 s)
        "damage": 30,         # buffed from 20 to 30
        "fire_rate": 45,      # fires every 1.5 s
    },
    PlantType.WALLNUT: {
        "cost": 50,
        "hp": 4000,
        "cooldown": 120,      # (4 s)
        "damage": 0,
        "fire_rate": 0,
    },
    PlantType.SNOWPEA: {
        "cost": 175,
        "hp": 300,
        "cooldown": 60,      # (2 s)
        "damage": 30,         # buffed from 20 to 30
        "fire_rate": 45,
        "slow": True,
    },
    PlantType.CHERRYBOMB: {
        "cost": 150,
        "hp": 300,
        "cooldown": 300,     # (10 s)
        "damage": 1800,       # instant AOE
        "fire_rate": 0,
        "instant": True,
        "aoe_radius": 1,      # cells
    },
    PlantType.REPEATER: {
        "cost": 200,
        "hp": 300,
        "cooldown": 60,      # (2 s)
        "damage": 20,
        "fire_rate": 45,
        "shots_per_volley": 2,
    },
    PlantType.POTATOMINE: {
        "cost": 25,
        "hp": 300,
        "cooldown": 120,      # (4 s)
        "damage": 1800,
        "fire_rate": 0,
        "arm_time": 200,      # Buffed from 450 to 200
        "proximity": True,
    },
}

NUM_PLANT_TYPES = len(PLANT_STATS)  # excludes NONE

# ---------------------------------------------------------------------------
# Zombie types
# ---------------------------------------------------------------------------

class ZombieType(IntEnum):
    NORMAL = 0
    CONEHEAD = 1
    BUCKETHEAD = 2
    FLAG = 3
    POLE_VAULTER = 4

ZOMBIE_STATS: dict[ZombieType, dict] = {
    ZombieType.NORMAL: {
        "hp": 200,
        "speed": 0.4,       # cells per second
        "damage": 100,      # per bite (every 0.5 s)
        "bite_rate": 15,    # ticks between bites
    },
    ZombieType.CONEHEAD: {
        "hp": 560,
        "speed": 0.4,
        "damage": 100,
        "bite_rate": 15,
    },
    ZombieType.BUCKETHEAD: {
        "hp": 1300,
        "speed": 0.4,
        "damage": 100,
        "bite_rate": 15,
    },
    ZombieType.FLAG: {
        "hp": 200,
        "speed": 0.53,
        "damage": 100,
        "bite_rate": 15,
    },
    ZombieType.POLE_VAULTER: {
        "hp": 340,
        "speed": 0.8,
        "damage": 100,
        "bite_rate": 15,
        "can_vault": True,
    },
}

# ---------------------------------------------------------------------------
# Wave configuration  (simple escalation for training)
# ---------------------------------------------------------------------------

def build_wave_schedule(difficulty: int = 1) -> list[dict]:
    """Return a list of wave descriptors.

    Each entry: {"tick": int, "zombies": list[(ZombieType, row)]}
    `difficulty` in [1..10] scales count and toughness.
    """
    import random

    waves: list[dict] = []
    tick = 150  # start almost immediately (5 seconds)
    
    # We want ~50 zombies total for a win. 
    num_waves = 5 
    # Zombie counts per wave: 5, 8, 10, 12, 15 (Total = 50)
    wave_counts = [5, 8, 10, 12, 15]
    
    for w in range(num_waves):
        count = wave_counts[w]
        for _ in range(count):
            row = random.randint(0, ROWS - 1)
            # Make random completely random but biased by difficulty
            roll = random.random()
            progress = w / max(num_waves - 1, 1)
            
            if roll < 0.05 + 0.05 * progress * difficulty:
                ztype = ZombieType.BUCKETHEAD
            elif roll < 0.10 + 0.15 * progress * difficulty:
                ztype = ZombieType.CONEHEAD
            elif roll < 0.05 + 0.05 * progress * difficulty:
                ztype = ZombieType.POLE_VAULTER
            else:
                ztype = ZombieType.NORMAL
            
            # Stagger spawn over 10 seconds (300 ticks)
            spawn_tick = tick + random.randint(0, 300)
            waves.append({"tick": spawn_tick, "zombies": [(ztype, row)]})
            
        # flag zombie on the final big wave
        if w == num_waves - 1:
            waves.append({"tick": tick + random.randint(0, 100), "zombies": [(ZombieType.FLAG, random.randint(0, ROWS - 1))]})
            
        tick += 400  # Next wave starts 400 ticks (13.3s) after this wave's base tick

    # Sort waves chronologically since we staggered them
    waves.sort(key=lambda x: x["tick"])
    return waves

# ---------------------------------------------------------------------------
# Colors (for simple rendering without sprites)
# ---------------------------------------------------------------------------
COLOR_BG = (34, 139, 34)       # green lawn
COLOR_GRID = (0, 100, 0)
COLOR_SUNFLOWER = (255, 215, 0)
COLOR_PEASHOOTER = (0, 200, 0)
COLOR_WALLNUT = (160, 82, 45)
COLOR_SNOWPEA = (135, 206, 250)
COLOR_CHERRYBOMB = (220, 20, 60)
COLOR_REPEATER = (0, 160, 0)
COLOR_POTATOMINE = (139, 90, 43)
COLOR_PEA = (0, 255, 0)
COLOR_SNOW_PEA_PROJ = (100, 180, 255)
COLOR_ZOMBIE = (120, 120, 120)
COLOR_ZOMBIE_CONE = (255, 140, 0)
COLOR_ZOMBIE_BUCKET = (180, 180, 180)
COLOR_SUN = (255, 255, 0)

PLANT_COLORS: dict[PlantType, tuple] = {
    PlantType.SUNFLOWER: COLOR_SUNFLOWER,
    PlantType.PEASHOOTER: COLOR_PEASHOOTER,
    PlantType.WALLNUT: COLOR_WALLNUT,
    PlantType.SNOWPEA: COLOR_SNOWPEA,
    PlantType.CHERRYBOMB: COLOR_CHERRYBOMB,
    PlantType.REPEATER: COLOR_REPEATER,
    PlantType.POTATOMINE: COLOR_POTATOMINE,
}

ZOMBIE_COLORS: dict[ZombieType, tuple] = {
    ZombieType.NORMAL: COLOR_ZOMBIE,
    ZombieType.CONEHEAD: COLOR_ZOMBIE_CONE,
    ZombieType.BUCKETHEAD: COLOR_ZOMBIE_BUCKET,
    ZombieType.FLAG: (200, 50, 50),
    ZombieType.POLE_VAULTER: (100, 100, 180),
}
